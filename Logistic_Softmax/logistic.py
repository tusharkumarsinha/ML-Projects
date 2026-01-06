import pandas as pd
import numpy as np
import sys
import time
import math
from sklearn.preprocessing import StandardScaler

# python3 logistic.py a train1.csv params.txt modelweights.txt
# python3 logistic.py b train1.csv test1.csv modelweights.txt modelpredictions.csv

def makeU(xw):
    u=np.exp(xw-np.max(xw,axis=1,keepdims=True))
    u=u/np.sum(u,axis=1,keepdims=True)
    return u

def loss(x,y,wts,freq_arr):
    n,m=x.shape
    xw=np.matmul(x,wts)
    u=makeU(xw)
    logu=np.log(u)
    yt=y.T
    yt_logu=np.matmul(yt,logu)/freq_arr
    t=np.trace(yt_logu)
    lost=t/((-2)*n)
    return lost

def find_optimal(x,y,w,gradient,nlr,freq_arr):
    nl=0
    nr=nlr
    loss_l=loss(x,y,w,freq_arr)
    while loss_l> loss(x,y,w-gradient*nr,freq_arr):
        nr*=2

    for iterations in range(20):
        n1=(2*nl+nr)/3
        n2=(nl+2*nr)/3
        l1=loss(x,y,w-n1*gradient,freq_arr)
        l2=loss(x,y,w-n2*gradient,freq_arr)
        if l1>l2:
            nl=n1
        elif l1<l2:
            nr=n2
        else:
            nl=n1
            nr=n2
    return (nl+nr)/2

def learn(x,y,nlr,klr,no_of_epochs,batch_size,freq_arr,is_ternary):
    temp=nlr
    n,m=x.shape
    k=len(freq_arr)
    w=np.zeros((m,k),dtype=np.float64)
    no_of_batches=math.ceil(n/batch_size)
    for epoch in range(no_of_epochs):
        print("epoch number:-",epoch+1)
        if klr!=0:
            nlr=temp/(1+klr*(epoch+1))
        for batch in range(no_of_batches):
            start=batch*batch_size
            end=min(n,start+batch_size)
            x_batch=x[start:end]
            y_batch=y[start:end].astype(float)
            x_batch_freq=np.zeros_like(x_batch)

            mask=(y_batch==1)
            classes=np.argmax(mask,axis=1)
            x_batch_freq=x_batch/freq_arr[classes][:,np.newaxis]

            x_w=x_batch @ w
            u=makeU(x_w)
            uy=u-y_batch
            x_batch_freq_T=x_batch_freq.T
            gradient=(x_batch_freq_T @ uy) / (2* (end-start))
            alpha=nlr
            if is_ternary:
                alpha=find_optimal(x_batch,y_batch,w,gradient,nlr,freq_arr)
            w=w-(alpha*gradient)

    return w

def part_a():
    train_file=sys.argv[2]
    params_file=sys.argv[3]
    wts_file=sys.argv[4]
    train_data=pd.read_csv(train_file)
    with open(params_file,"r") as f:
        params=f.readlines()
    learn_type=int(params[0])
    no_of_epochs=int(params[2]) 
    batch_size=int(params[3])

    freq_count=train_data.iloc[:,-1].value_counts()
    k=len(freq_count)
    freq_array=np.zeros(k,dtype=float)
    for value in range(1,k+1):
        freq_array[value-1]=freq_count.get(value,0)
    n,m=train_data.shape
    ones=np.ones(n)
    train_data.insert(0,"intercept",ones)
    y=train_data.iloc[:,-1]
    y=pd.get_dummies(y).to_numpy()
    x=train_data.iloc[:,:-1].to_numpy()

    if learn_type==1:
        nlr=float(params[1])
        klr=0
        w=learn(x,y,nlr,klr,no_of_epochs,batch_size,freq_array,False)
    elif learn_type==2:
        nlr,klr=params[1].split(",")
        nlr=float(nlr)
        klr=float(klr)
        w=learn(x,y,nlr,klr,no_of_epochs,batch_size,freq_array,False)
    elif learn_type==3:
        nlr=float(params[1])
        klr=0
        w=learn(x,y,nlr,klr,no_of_epochs,batch_size,freq_array,True)

    with open(wts_file,"w") as f:
        for i in range(len(w)):
            for j in range(len(w[0])):
                f.write(str(w[i][j])+"\n")
    return 

def part_b():
    train_file=sys.argv[2]
    test_file=sys.argv[3]
    wts_file=sys.argv[4]
    pred_file=sys.argv[5]

    train_data=pd.read_csv(train_file)

    freq_count=train_data.iloc[:,-1].value_counts()
    k=len(freq_count)

    freq_arr=np.zeros(k,dtype=float)
    for value in range(1,k+1):
        freq_arr[value-1]=freq_count.get(value,0)

    y_train=train_data.iloc[:,-1]
    y_train=pd.get_dummies(y_train).to_numpy()
    x_train=train_data.iloc[:,:-1].to_numpy()

    test_data=pd.read_csv(test_file)
    x_test=test_data.to_numpy()

    scalar=StandardScaler().fit(x_train)
    x_train=scalar.transform(x_train)
    x_test=scalar.transform(x_test)

    x_train=np.insert(x_train,0,np.ones(x_train.shape[0]),axis=1)
    x_test=np.insert(x_test,0,np.ones(x_test.shape[0]),axis=1)
    n=x_train.shape[0]

    wts=learn(x_train,y_train,np.float64(1e-5),0,8,n,freq_arr,True)

    x_test_w=np.matmul(x_test,wts)
    predicted_prob=makeU(x_test_w)

    with open(wts_file,"w") as f:
        for i in range(len(wts)):
            for j in range(len(wts[0])):
                f.write(str(wts[i][j])+"\n")

    with open(pred_file,"w") as f:
        for i in range(x_test.shape[0]):
            for j in range(k):
                f.write(str(predicted_prob[i][j]))
                if j!=k-1:
                    f.write(",")
            f.write("\n")

if __name__=="__main__":
    t1=time.time()
    ab=sys.argv[1]
    if ab=="a":
        part_a()
    elif ab=="b":
        part_b()
    t2=time.time()
    print("Time taken is :- {:.2f} seconds".format(t2-t1))
