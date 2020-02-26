import pandas as pd
import numpy as np

import bayes
import knn
from commonfun import cal_loss, test_train_df

db3a = pd.read_csv("data\P1a_train_data.txt",header=None)
db3b = pd.read_csv("data\P1b_train_data.txt",header=None)


clf = bayes.Bayes()
clf.fit()
