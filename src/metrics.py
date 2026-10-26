import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def detection_metrics(id_scores, ood_scores):
    y = np.r_[np.ones(len(id_scores)), np.zeros(len(ood_scores))]
    s = np.r_[id_scores, ood_scores]
    fpr, tpr, _ = roc_curve(y, s)
    i = np.flatnonzero(tpr >= .95)
    return {"auroc_id_positive": float(roc_auc_score(y, s)), "aupr_id_positive": float(average_precision_score(y, s)), "fpr95_ood": float(fpr[i[0]]) if len(i) else 1.0}

def ece(probs, labels, bins=15):
    conf = probs.max(1); pred = probs.argmax(1); labels = np.asarray(labels)
    edges = np.linspace(0, 1, bins + 1); out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): out += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    return float(out)

def risk_coverage(conf, correct, points=101):
    order = np.argsort(-np.asarray(conf)); correct = np.asarray(correct)[order]
    rows=[]
    for cov in np.linspace(0.01,1,points):
        n=max(1,int(np.ceil(cov*len(order))))
        rows.append((float(n/len(order)), float(1-correct[:n].mean())))
    return rows

