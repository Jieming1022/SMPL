
import torch
import matplotlib

matplotlib.use('Agg')
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col

try:
    from tsnecuda import TSNE
except ImportError:
    from sklearn.manifold import TSNE


def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
              filename: str, source_color='r', target_color='b'):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    # visualize using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([target_color, source_color]), s=20)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename)


def visualize_single_domain(
        features,
        y_true,
        class_names,
        file_name,
        palettes=None,
        title=None,
        tsne_plot_count=1200,
        mode='analysis'
):
    """
    @param features: 特征
    @param y_true: 真实标签
    @param class_names: 标签名
    @param file_name: 文件名
    @param title: 标题
    @param tsne_plot_count: 画出特征点的个数，默认：1200
    @param mode: 模式
    """
    if not palettes:
        palettes = ["#303872", "#852427", "#029E73", "#CC78BC", "#ECE133", "#56B4E9", "#DE8F05"]
    if features.shape[0] != len(y_true):
        raise RuntimeError('the number of features ({}) and y_true ({}) is unequal.'
                           .format(features.shape[0], len(y_true)))
    plt.figure(figsize=(10, 6))
    classes = np.unique(y_true)
    if tsne_plot_count > len(y_true):
        tsne_plot_count = len(y_true)
    # sample n = tsne_plot_count features
    sample_list = [i for i in range(len(y_true))]
    sample_list = random.sample(sample_list, tsne_plot_count)
    if mode == 'analysis':
        feature = np.asfarray([features[i].cpu().detach().numpy() for i in sample_list])
    else:
        feature = [features[i] for i in sample_list]
    y_labels = [y_true[i] for i in sample_list]
    # Dimensionality Reduction
    tsne_result = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=350, learning_rate=300).fit_transform(
        feature)
    data = {'x': np.array(tsne_result[:, 0]), 'y': np.array(tsne_result[:, 1]), 'label': np.array(y_labels)}
    if mode == 'web':
        return data
    for c in classes:
        plt.scatter(data['x'][data['label'] == c], data['y'][data['label'] == c],
                    c=palettes[int(c)], marker='o', s=40)
    plt.legend(labels=class_names, loc="lower right")
    plt.axis('off')
    if title:
        plt.title(title)

    plt.savefig(file_name)
    if mode == 'analysis':
        # np.save(os.path.join(os.path.dirname(file_name), 'features.npy'), features.cpu().numpy())
        np.savetxt(os.path.join(os.path.dirname(file_name), 'features.csv'), features.cpu().numpy(), delimiter=',')
