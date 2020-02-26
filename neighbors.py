import random as rd
import numpy as np
from commonfun import euclidian, cal_loss

class Knn:
    def __init__(self,k,verb=True):
        self.k = k
        self.verbose = verb
    
    def fit(self,x_train,y_train,dist=euclidian):
        self.dist = euclidian
        self.x_train = x_train
        self.y_train = y_train
        if self.verbose:
            self.zloss()
            print("training loss is:", self.loss)
        
    def test(self,x):
        dist_arr = [self.dist(x,xd) for xd in self.x_train]
        indices = np.argsort(dist_arr)
        tot= sum(self.y_train[indices[:self.k]])
        if tot!=0:
            return tot/abs(tot)
        else:
            return rd.choice([1,-1])
    
    def predict(self,x_test):
        y_pred = [self.test(x) for x in x_test]
        return np.array(y_pred)
        
    def zloss(self):
        self.loss = cal_loss(self.predict(self.x_train),
                            self.y_train)
        if self.verbose:
            print("training loss is:", self.loss)
    
