import os
import numpy
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import interp
from scipy.special import softmax
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from typing import Optional


def visualize(
        logits,
        y_true,
        label_names,
        file_name,
        figsize: Optional[tuple] = None,
        title: Optional[str] = None,
        palettes: Optional[list] = None,
        mode='analysis'
):
    """
    @param logits: softmax函数之前的值
    @param y_true: 真实标签
    @param label_names: 标签名
    @param file_name: 文件名
    @param figsize: 图像大小
    @param title: 标题
    @param palettes: 调色盘
    @param mode: 模式
    """
    classes = np.unique(y_true)
    n_classes = len(np.unique(y_true))
    if mode == 'analysis':
        y_score = torch.nn.Softmax(dim=1)(logits) if n_classes > 2 else torch.nn.Softmax(dim=1)(logits)[:, 1]
    else:
        y_score = softmax(logits, axis=1) if n_classes > 2 else softmax(logits, axis=1)[:, 1]
    y_true = label_binarize(y_true, classes=classes)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i]) if n_classes > 2 else roc_curve(y_true, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    if n_classes != 2:
        # Compute micro-average ROC curve and ROC area
        # fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.detach().numpy().ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        #
        # fpr["macro"] = all_fpr
        # tpr["macro"] = mean_tpr
        # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure(figsize=figsize)
        # plt.plot(fpr["micro"], tpr["micro"],
        #          label='micro-average ROC curve (area = {0:0.2f})'
        #                ''.format(roc_auc["micro"]),
        #          color='deeppink', linestyle=':', linewidth=4)
        #
        # plt.plot(fpr["macro"], tpr["macro"],
        #          label='macro-average ROC curve (AUC = {0:0.2f})'
        #                ''.format(roc_auc["macro"]),
        #          color='navy', linestyle=':', linewidth=4)
        lw = 2
        if palettes == None:
            colors = cycle(sns.palettes.SEABORN_PALETTES['colorblind6'])
        elif isinstance(palettes, list):
            colors = cycle(palettes)
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (AUC = {1:0.2f})'
                     ''.format(label_names[i], roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        if mode == 'web':
            return fpr, tpr
    else:
        lw = 2
        plt.plot(fpr[0], tpr[0], color='#852427',
                 lw=lw, label='AUC = %0.2f' % roc_auc[0])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        fpr_tpr = np.concatenate((np.expand_dims(fpr[0], axis=1), np.expand_dims(tpr[0], axis=1)), axis=1)
        if mode == 'analysis':
            fpr_tpr_filename = os.path.join(os.path.dirname(file_name), 'fpr_tpr_data.xlsx')
            pd.DataFrame(fpr_tpr, columns=['fpr', 'tpr']).to_excel(fpr_tpr_filename, index=False)


    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if title:
        plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(file_name)
    if mode == 'analysis':
        np.save(os.path.join(os.path.dirname(file_name), 'logits.npy'), logits.cpu().numpy())