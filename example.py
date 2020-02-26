import pandas as pd
import numpy as np

import bayes
import neighbors
from commonfun import cal_loss, test_train_dt

db3a = pd.read_csv("data\P1a_train_data.txt",header=None)
db3b = pd.read_csv("data\P1b_train_data.txt",header=None)

x_train,y_train = test_train_dt(db1a)
x_test,y_test = test_train_dt(dt1a)


clf = bayes.Bayes()
clf.fit(x_train,y_train)
print("result for q1 Bayes:", clf.params)

clf = neighbors.Knn()
clf.fit(x_train,y_train)
print("result for q1 knn:", clf.params)
