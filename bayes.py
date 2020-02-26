import functools as ft
import numpy as np
from commonfun import diff,vnorm,cal_loss


class Bayes:
    ## since class conditional densities in every problem is normal
    def __init__(self,num_class=2,verb = True):
        self.k = num_class
        self.density = []
        self.params = []
        self.verbose = verb

    def label(self,n):
        ## 1,-1 as label id in 2 class problem
#         return 2*n-1
        yl = [-1,1]
        return yl[n]
    
    def fit(self,x_train,y_train,prior=vnorm):
        self.ylabels = np.unique(y_train)
        for i,y in enumerate(self.ylabels):
            x_class = diff(x_train,y_train,y)
            mi = np.mean(x_class,axis=0)
#             si = np.mean((x_class-mi)**2,axis=0)
            si = np.mean([np.outer((x-mi),(x-mi)) 
                          for x in x_class],axis=0)
            self.params.append(dict(clas=i,mu=mi,sigma=si))
            self.density.append(ft.partial(prior,
                                          mu=mi,
                                          sigma=si))
        self.loss = cal_loss(self.predict(x_train),y_train)
        if self.verbose:
            print("training loss is", self.loss)
           
    def test(self,x):
        dx = [self.density[i](x) for i in range(self.k)]
        dn = np.argmax(dx)
        return self.label(dn)
    
#     def linear(self,x):
        
    def predict(self,x_train):
        return [self.test(x) for x in x_train]
 
