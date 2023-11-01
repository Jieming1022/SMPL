import torch
import torch.nn.functional as F
import pandas as pd


def save(y_true, y_pred, outputs, pathes, num_classes, file_name):
    labels = y_true.tolist()
    preds = y_pred.tolist()
    probs = torch.max(F.softmax(outputs, dim=1), dim=1).values.tolist() if num_classes > 2 else F.softmax(outputs, dim=1)[:, 1].tolist()
    data_df = pd.DataFrame({'path': pathes, 'label': labels, 'prediction': preds, 'probability': probs})
    data_df.to_csv(file_name, index=False)