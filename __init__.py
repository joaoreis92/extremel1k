import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier as gpc
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support as metrics
import sys
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.svm import LinearSVC as lsvc
from sklearn.svm import SVC as svc
import time
import pickle