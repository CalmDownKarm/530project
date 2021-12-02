import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve, auc
from argparse import ArgumentParser

def warn(*args, **kwargs): pass
import warnings; warnings.warn = warn


parser = ArgumentParser()
parser.add_argument("-p", "--predicted",
                    type=str,
                    dest="predicted_path",
                    required=True,
                    help="path to model's prediction labels file")
parser.add_argument("-a", "--actual",
                    type=str,
                    dest="actual_path",
                    required=True,
                    help="path to true/actual y labels file")

args = parser.parse_args()

pred = pd.read_csv(args.predicted_path, index_col="id")
actual = pd.read_csv(args.actual_path, index_col="id")

pred.columns = ["predicted"]
actual.columns = ["actual"]

data = actual.join(pred)

f1_score = f1_score(data.actual, data.predicted, average="weighted")
acc = accuracy_score(data.actual, data.predicted)
fpr, tpr, _ = roc_curve(data.actual, data.predicted)
auc = auc(fpr, tpr)

print(f"F1 score: {f1_score}\nAccuracy: {acc}") #\nAUC: {auc}")
