import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import bayes
import neighbors
from commonfun import cal_loss, test_train_dt,plotdb

db1a = pd.read_csv("data/P1a_train_data_2D.txt",header=None)
dt1a = pd.read_csv("data/P1a_test_data_2D.txt",header=None)

x_train,y_train = test_train_dt(db1a)
x_test,y_test = test_train_dt(dt1a)

plotdb(db1a,"train_1a")

clf = bayes.Bayes()
clf.fit(x_train,y_train)
print("result for q1 Bayes:", clf.params)

clf = neighbors.Knn(3)
clf.fit(x_train,y_train)
print("result for q1 knn:", clf.params)
