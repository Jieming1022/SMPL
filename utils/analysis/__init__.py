import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm
from typing import List, Tuple


def collect_feature(data_loader: DataLoader, feature_extractor: nn.Module,
                    device: torch.device, max_num_features=None) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return

    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features = []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm.tqdm(data_loader)):
            if max_num_features is not None and i >= max_num_features:
                break
            images = images.to(device)
            feature = feature_extractor(images).cpu()
            all_features.append(feature)
    return torch.cat(all_features, dim=0)


def collect_analysis_data(
        data_loader: DataLoader,
        feature_extractor: nn.Module,
        classifier: nn.Module,
        device: torch.device,
        max_num_samples=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List]:
    """
    Fetch data from `data_loader`, and then use `classifier` to collect features, labels, and logits

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        classifier (torch.nn.Module): A classifier.
        device (torch.device)
        max_num_samples (int): The max number of samples to return

    Returns:
        Features in shape (min(len(data_loader), max_num_samples * mini-batch size), :math:`|\mathcal{F}|`).
        Logits in shape (min(len(data_loader), max_num_samples * mini-batch size).
        Y_true in shape (min(len(data_loader), max_num_samples * mini-batch size).
        Y_pred in shape (min(len(data_loader), max_num_samples * mini-batch size).
    """
    feature_extractor.eval()
    classifier.eval()
    all_features = []
    all_logits = []
    all_y_true = []
    all_y_pred = []
    all_path = []
    with torch.no_grad():
        for i, (images, target, path) in enumerate(tqdm.tqdm(data_loader)):
            if max_num_samples is not None and i >= max_num_samples:
                break
            images = images.to(device)
            feature = feature_extractor(images).cpu()
            logits = classifier(images).cpu()
            _, y_pred = torch.max(logits, 1)

            all_features.append(feature)
            all_logits.append(logits)
            all_y_true.append(torch.tensor(target))
            all_y_pred.append(y_pred.cpu())
            all_path += list(path)
    return torch.cat(all_features, dim=0), torch.cat(all_logits, dim=0), \
           torch.cat(all_y_true, dim=0), torch.cat(all_y_pred, dim=0), all_path
