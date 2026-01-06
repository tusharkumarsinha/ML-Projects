import numpy as np
import pandas as pd
import sys

# python3 linear.py a train.csv test.csv sample_weights1.txt modelpredictions.txt modelweights.txt
# python3 linear.py b train.csv test.csv regularization.txt modelpredictions.txt modelweights.txt bestlambda.txt

ab=sys.argv[1] 
train_file=sys.argv[2] 
test_file=sys.argv[3]

def part_a():
    wts_file=sys.argv[4]
    pred_on_test=sys.argv[5]
    model_wts_file=sys.argv[6]

    train_data=pd.read_csv(train_file)
    n,m= train_data.shape
    ones=np.ones(n)
    train_data.insert(0,"b",ones)
    y=train_data.iloc[:,-1].to_numpy()
    x=train_data.iloc[:,:-1].to_numpy()

    test_data=pd.read_csv(test_file)
    test_size,_=test_data.shape
    ones=np.ones(test_size)
    test_data.insert(0,"b",ones)

    wts=pd.read_csv(wts_file,header=None).iloc[:,-1].to_numpy()

    xu=np.empty(x.shape)
    yu=np.empty(y.shape)
    for i in range(n):
        xu[i]=x[i]*wts[i]
        yu[i]=y[i]*wts[i]

    xu_T=np.transpose(xu)
    x_T=np.transpose(x)
    xu_T_x=np.matmul(xu_T,x)
    xu_T_x_inv=np.linalg.inv(xu_T_x)
    x_T_yu=np.matmul(x_T,yu)
    w=np.matmul(xu_T_x_inv,x_T_yu)

    predictions=np.matmul(test_data,w)

    with open(model_wts_file,"w") as f:
        for i in range(len(w)):
            f.write(str(w[i])+"\n")

    with open(pred_on_test,"w") as f:
        for i in range(test_size):
            f.write(str(predictions[i])+"\n")

def part_b():
    reg_parameters=sys.argv[4]
    pred_on_test=sys.argv[5]
    model_wts_file=sys.argv[6]
    best_lambda_file=sys.argv[7]

    train_data=pd.read_csv(train_file)
    test_data=pd.read_csv(test_file)
    n,m=train_data.shape
    ones=np.ones(n)
    train_data.insert(0,"b",ones)
    train_data.drop(index=train_data.index[10*(n//10):],inplace=True)
    n,_=train_data.shape
    #n should be divisible by 10 now
    x=train_data.iloc[:,:-1]
    y=train_data.iloc[:,-1]
    test_data=pd.read_csv(test_file)
    test_size,_=test_data.shape
    ones=np.ones(test_size)
    test_data.insert(0,"b",ones)

    with open(reg_parameters,"r") as f:
        lambdas=[float(line.strip()) for line in f]

    lambda_list=[]

    for l in lambdas:
        avg_mse=0
        for i in range(10):
            start=i*(n//10)
            end=(i+1)*(n//10)

            x_test=x.iloc[start:end].to_numpy()
            x_train=x.drop(index=x.index[start:end]).to_numpy()
            y_test=y.iloc[start:end].to_numpy()
            y_train=y.drop(index=y.index[start:end]).to_numpy()
            n_test,_=x_test.shape

            x_T=np.transpose(x_train)
            x_T_y=np.matmul(x_T,y_train)
            x_T_x=np.matmul(x_T,x_train)
            for j in range(m):
                x_T_x[j][j]+=l
            x_T_x_inv=np.linalg.inv(x_T_x)

            w=np.matmul(x_T_x_inv,x_T_y)
            pred=np.matmul(x_test,w)
            error=y_test-pred
            mse=np.dot(error,error)
            avg_mse+=mse/n_test
        lambda_list.append((avg_mse,l))

    lambda_list.sort()
    best_lambda=lambda_list[0][1]
    x=x.to_numpy()
    y=y.to_numpy()

    x_T=np.transpose(x)
    x_T_x=np.matmul(x_T,x)
    for j in range(m):
        x_T_x[j][j]+=best_lambda
    x_T_x_inv=np.linalg.inv(x_T_x)
    x_T_y=np.matmul(x_T,y)
    w_model=np.matmul(x_T_x_inv,x_T_y)
    predictions=np.matmul(test_data.to_numpy(),w_model)

    with open(pred_on_test,"w") as f:
        for i in range(test_size):
            f.write(str(predictions[i])+"\n")

    with open(best_lambda_file,"w") as f:
        f.write(str(best_lambda))

    with open(model_wts_file,"w") as f:
        for i in range(len(w_model)):
            f.write(str(w_model[i])+"\n")

if ab=="a":
    part_a()
elif ab=="b":
    part_b()
