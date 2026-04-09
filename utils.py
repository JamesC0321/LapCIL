from sklearn.metrics import roc_auc_score, roc_curve,recall_score,accuracy_score,precision_score,f1_score,confusion_matrix
from sklearn.metrics import auc as calc_auc
import torch
import numpy as np
import torchmetrics
from sklearn.preprocessing import label_binarize


def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps

def roc_threshold(label, prediction):
    fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    c_auc = roc_auc_score(label, prediction,multi_class="ovo")
    return c_auc, threshold_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def eval_metric(oprob, label):

    # auc, threshold = roc_threshold(label.cpu().numpy(), oprob.detach().cpu().numpy())
    # prob = oprob > threshold
    # label = label > threshold
    _, prob = torch.max(oprob,dim=1)
    AUROC = torchmetrics.AUROC(num_classes=4, average='weighted') # issue
    metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes=4,
                                                                   average='weighted',task='multiclass'),
                                             torchmetrics.CohenKappa(num_classes=4,task='multiclass'),
                                             torchmetrics.F1Score(num_classes=4,
                                                                  average='weighted',task='multiclass'),
                                             torchmetrics.Recall(average='weighted',
                                                                 num_classes=4,task='multiclass'),
                                             torchmetrics.Precision(average='weighted',
                                                                    num_classes=4,task='multiclass'),
                                             torchmetrics.Specificity(average='weighted',
                                                                      num_classes=4,task='multiclass')])
    all_hats = torch.tensor(prob,device="cpu")
    all_labels = torch.tensor(label,device="cpu")
    met = metrics(all_hats, all_labels)

    accuracy = met['Accuracy']
    precision = met['Precision']
    recall = met['Recall']
    specificity = met['Specificity']
    F1 = met['F1Score']
    auc = AUROC(oprob, label)

    return accuracy, precision, recall, specificity, F1, auc


def ski_eval_metric(oprob, label,cls=4,auc_average='macro'):

    _, prob = torch.max(oprob,dim=1)
    all_hats = prob.cpu().numpy()
    all_labels =  label.cpu().numpy()

    if cls >2 :
        AUROC = torchmetrics.AUROC(task='multiclass', num_classes=cls, average=auc_average)
        avg_auc = AUROC(oprob, label)
        conf_mat = confusion_matrix(all_labels, all_hats)
        accuracy = accuracy_score(all_labels, all_hats)
        precision = precision_score(all_labels, all_hats, average=auc_average)
        recall = recall_score(all_labels, all_hats, average=auc_average)
        F1 = f1_score(all_labels, all_hats, average=auc_average)
    else:
        AUROC = torchmetrics.AUROC(task='binary',num_classes=cls)
        avg_auc = AUROC(oprob[:,1], label)
        conf_mat = confusion_matrix(all_labels,all_hats)
        accuracy = accuracy_score(all_labels,all_hats)
        precision = precision_score(all_labels,all_hats)
        recall = recall_score(all_labels,all_hats)
        F1 = f1_score(all_labels,all_hats)

    return accuracy, precision, recall, F1, avg_auc,conf_mat