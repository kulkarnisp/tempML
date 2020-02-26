import numpy as np
import functools as ft


def vnorm(Xv,mu,sigma):
    Si = np.linalg.inv(sigma)
    x = Xv-mu
    z = (np.linalg.det(Si)*(2*np.pi)**len(x))**0.5
    u = 0.5*np.dot(x.T,np.dot(Si,x))
    return np.exp(-(u))*z
    
def fnorm(x,mu,sigma):
    z = 1/(abs(sigma)*(2*np.pi)**0.5)
    u = 0.5*((x-mu)/sigma)**2
    return np.exp(-(u))*z
   
def cal_loss(y_pred,y_test):
    yt = abs(y_pred-y_test)
    return np.average(yt/2)

def cal_eff(y_pred,y_test):
    return 1 - cal_loss(y_pred,y_test)
    
def diff(xx,yy,yi):
    return np.array([x for x,y in zip(xx,yy) if y ==yi])
def euclidian(x1,x2):
    return np.sum((x1-x2)**2)

def manhatten(x1,x2):
    return np.sum(np.abs(x1-x2))


def mahalanobis(x1,x2,sigma):
    ## x1 x2 are mean substracted
    return np.dot(x1.T,np.dot(sigma,x2))

def test_train_dt(databae):
    panda = databae.copy()
    y_train = np.array(panda.pop(panda.keys()[-1]))
    x_train = panda.values
    return x_train,y_train
