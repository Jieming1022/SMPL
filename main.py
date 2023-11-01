import random
import warnings
import argparse
import shutil
import os.path as osp
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.data import ForeverDataIterator, ImbalancedDatasetSampler
from utils.logger import CompleteLogger
from utils.analysis import collect_analysis_data, tsne, confusion_matrix, predictions, training_data, roc
from utils.analysis.extractor import FeatureMap
from lib.opa import Backbone, OverlappedLocalAttention, PyramidAttention, Classifier
from utils.metric import accuracy, ConfusionMatrix, precision_recall_f1score, sensitivity_specificity, \
    quadratic_weighted_kappa_score
from utils.meter import AverageMeter, ProgressMeter
from utils.dataset.utils import get_supervised_dataset, get_train_transform, get_val_transform
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    train_transform = get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_vertical_flip=True, random_rotation=90,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        get_supervised_dataset(args.root, args.task, train_transform, val_transform, dx=args.dx,
                                     preprocess=args.preprocess)
    if args.imbalanced:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.workers, drop_last=True,
                                  sampler=ImbalancedDatasetSampler(train_dataset))
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_iter = ForeverDataIterator(train_loader)

    # create model
    resblocks = Backbone(in_channels=3, out_channels=512)
    overlapped_attention = OverlappedLocalAttention(in_channels=resblocks.out_channels, num_classes=num_classes)
    pyramid_attention = PyramidAttention(in_channels=resblocks.out_channels, mid_channels=256,
                                         out_channels=512, num_classes=num_classes)
    classifier = Classifier(resblocks, overlapped_attention, pyramid_attention,
                            num_classes, not args.no_overlapped, not args.no_pyramid).to(device)
    # define optimizer and lr scheduler
    if args.optimizer == 'adam':
        optimizer = Adam(classifier.get_parameters(), args.lr, weight_decay=args.wd)
    elif args.optimizer == 'sgd':
        optimizer = SGD(classifier.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd,
                        nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    if args.phase == 'analysis':
        # extract features
        if not args.no_tsne or not args.no_cm or not args.no_prediction or not args.no_roc:
            feature_extractor = \
                nn.Sequential(classifier.backbone, classifier.avg_pool).to(device)
            feature, logits, y_true, y_pred, paths = \
                collect_analysis_data(test_loader, feature_extractor, classifier, device)
        # plot t-SNE
        if not args.no_tsne:
            tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
            tsne.visualize_single_domain(feature, y_true, args.class_names, tSNE_filename)
            print("Saving t-SNE to", tSNE_filename)
        # plot confusion matrix
        if not args.no_cm:
            confusion_matrix_filename = osp.join(logger.visualize_directory, 'confusion_matrix.pdf')
            confusion_matrix.visualize(y_true, y_pred, filename=confusion_matrix_filename)
            print("Saving confusion matrix to", confusion_matrix_filename)
        # save predictions
        if not args.no_prediction:
            predictions_filename = osp.join(logger.visualize_directory, 'predictions.csv')
            predictions.save(y_true, y_pred, logits, paths, num_classes, predictions_filename)
            print("Saving predictions to", predictions_filename)
        # save training data
        if not args.no_training_data:
            training_data_filename = osp.join(logger.visualize_directory, 'training_data.xlsx')
            training_data.save(args.epochs, logger.tensorboard_directory, training_data_filename)
            print("Saving training data to", training_data_filename)
        # plot roc curve
        if not args.no_roc:
            roc_filename = osp.join(logger.visualize_directory, 'roc.pdf')
            roc.visualize(logits, y_true, args.class_names, roc_filename)
            print("Saving roc curve to", roc_filename)
        # plot feature map
        if args.feature_map:
            feature_map = FeatureMap(resblocks, args.feature_map_layers,
                                     args.feature_map_input, logger.visualize_directory)
            feature_map.plot_filters()
            feature_map.plot_feature_maps()
            print("Saving feature maps to", logger.visualize_directory)
        return

    if args.phase == 'test':
        acc1 = validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # create tensorboard writer
    writer = None
    if args.tensorboard:
        writer = SummaryWriter(log_dir=logger.tensorboard_directory)

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # print(lr_scheduler.get_lr())
        # train for one epoch
        train(train_iter, classifier, optimizer, lr_scheduler, epoch, args, device, writer)

        # evaluate on validation set
        acc1 = validate(val_loader, classifier, args, device, epoch, writer)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def train(train_iter, model, optimizer, lr_scheduler, epoch, args, device, writer=None):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        images, labels, _ = next(train_iter)
        images = images.to(device)
        labels = labels.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        logits = model(images)

        cls_loss = F.cross_entropy(logits, labels)
        loss = cls_loss

        cls_acc = accuracy(logits, labels)[0]

        losses.update(loss.item(), images.size(0))
        cls_accs.update(cls_acc.item(), images.size(0))

        # record in tensorboard
        if writer:
            writer.add_scalar("Loss", loss.item(), epoch * args.iters_per_epoch + i)
            writer.add_scalar("RunningAccuracy", cls_acc.item(), epoch * args.iters_per_epoch + i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)



def validate(val_loader, model, args, device, epoch=None, writer=None) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, (images, target, _) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i == 0:
                y_true = target.cpu()
                outputs = output.cpu()
            else:
                y_true = torch.cat((y_true, target.cpu()), 0)
                outputs = torch.cat((outputs, output.cpu()), 0)

            if i % args.print_freq == 0:
                progress.display(i)

        _, y_pred = torch.max(outputs, 1)

        print_str = ' *   Acc@1 {top1.avg:.3f}\n'.format(top1=top1)
        if confmat:
            print_str += confmat.format(args.class_names) + "\n"

        cm = confusion_matrix(y_true, y_pred)
        if len(args.class_names) == 2:
            sensitivity, specificity = sensitivity_specificity(cm)
            precision, recall, f1score = precision_recall_f1score(y_true, y_pred, average="binary")
            if writer:
                writer.add_scalar("Sensitivity", sensitivity, epoch + 1)
                writer.add_scalar("Specificity", specificity, epoch + 1)
            print_str += " *   Sensitivity: {:.4f} Specificity: {:4f}\n".format(sensitivity, specificity)
        else:
            qwk = quadratic_weighted_kappa_score(y_true, y_pred)
            precision, recall, f1score = precision_recall_f1score(y_true, y_pred, average="macro")
            if writer:
                writer.add_scalar("QuadraticWeightedKappa", qwk, epoch + 1)
            print_str += " *   Quadratic Weighted Kappa Score: {:.4f}\n".format(qwk)

        if writer:
            writer.add_scalar("Accuracy", top1.avg, epoch + 1)
            writer.add_scalar("Precision", precision, epoch + 1)
            writer.add_scalar("Recall", recall, epoch + 1)
            writer.add_scalar("F1score", f1score, epoch + 1)
        print_str += " *   Precision: {:.4f} Recall: {:.4f} F1-score: {:.4f}".format(precision, recall, f1score)
    print(print_str)

    return top1.avg



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Source Only for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-t', '--task', type=str, default='A',
                        help='task dataset')
    parser.add_argument('--train-resizing', type=str, default='res.')
    parser.add_argument('--val-resizing', type=str, default='res.')
    parser.add_argument('--resize-size', type=int, default=256,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    parser.add_argument('--dx', default="GRAD", type=str, choices=['RDR', 'NORM', 'GRAD'],
                        help='diagnosis type, GRAD: DR grading, RDR: referable/non-referable DR,'
                             'NORM: normal/abnormal')
    parser.add_argument('--imbalanced', action='store_true',
                        help='use imbalance dataset sampler')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N',
                        help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help="optimizer. (default: adam)")
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.8, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.00001, type=float,
                        metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=70, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='src_only',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--preprocess', default="cropped", type=str,
                        choices=['original', 'cropped', 'cropped_enhanced'],
                        help='image preprocess (default: cropped)')
    parser.add_argument('--tensorboard', dest='tensorboard', default=True,
                        help='use tensorboard to record results')
    # analysis parameters
    parser.add_argument('--no-roc', action='store_true',
                        help='no roc curve')
    parser.add_argument('--no-cm', action='store_true',
                        help='no confusion matrix')
    parser.add_argument('--no-tsne', action='store_true',
                        help='no t-SNE plot')
    parser.add_argument('--no-prediction', action='store_true',
                        help='no prediction csv')
    parser.add_argument('--no-training-data', action='store_true',
                        help='no training data')
    parser.add_argument('--feature-map', action='store_true',
                        help='plot feature map')
    # feature map parameters
    parser.add_argument('--feature-map-layers', dest='feature_map_layers', nargs='*', type=int, default=None,
                        help='layer numbers to plot feature maps')
    parser.add_argument('--feature-map-input', dest='feature_map_input', type=str, default=None,
                        help='the input images where to plot feature maps')
    # ops parameters
    parser.add_argument('--no-overlapped', action='store_true',
                        help='no overlapped ')
    parser.add_argument('--no-pyramid', action='store_true',
                        help='no catagory attention block')
    args = parser.parse_args()
    main(args)
