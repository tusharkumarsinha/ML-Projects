import pandas as pd
import numpy as np
import argparse
from nltk.stem import PorterStemmer

# python3 nb1_1.py --train train.tsv --test valid.tsv --out out.txt --stop stopwords.txt

parser= argparse.ArgumentParser()
parser.add_argument("--train",required=True,type=str)
parser.add_argument("--test",required=True,type=str)
parser.add_argument("--out",required=True,type=str)
parser.add_argument("--stop",required=True,type=str)
args=parser.parse_args()
train_tsv=args.train
test_tsv=args.test
out_path=args.out
stop_path=args.stop
train_data=pd.read_csv(train_tsv,delimiter="\t",header=None,quoting=3)
test_data=pd.read_csv(test_tsv,delimiter="\t",header=None,quoting=3)
train_data=train_data[[2,1]] #keep only columns 2 and 1 in that order
test_data=test_data[[2,1]]
train_data.rename(columns={2:"info",1:"truth"},inplace=True)
test_data.rename(columns={2:"info",1:"truth"},inplace=True)
with open(stop_path,"r") as f:
    stopwords=f.read().splitlines()
stop_words=set(stopwords)
stemmer=PorterStemmer()

token_train_info=[]
for text in train_data["info"]:
    words=text.lower().split()
    filtered_words=[stemmer.stem(word) for word in words if word not in stop_words]
    token_train_info.append(filtered_words)
train_data["token_info"]=token_train_info

token_test_info=[]
for text in test_data["info"]:
    words=text.lower().split()
    filtered_words=[stemmer.stem(word) for word in words if word not in stop_words]
    token_test_info.append(filtered_words)
test_data["token_info"]=token_test_info

dictionary=set(word for train_example in train_data["token_info"] for word in train_example)
dictionary=list(dictionary)
dic_size=len(dictionary)
indexing={word:i for i,word in enumerate(dictionary)}

def create_fvector(example): 
    #make feature vector according to Bernoulli event model...it being a binary vector of the whole dictionary size
    fvector=np.zeros(dic_size)
    for word in example:
        if word in indexing:
            fvector[indexing[word]]=1
    return fvector

def count_new_words(example):
    count=0
    for word in example:
        if word not in indexing:
            count+=1
    return count

x_train=np.array([create_fvector(example) for example in train_data["token_info"]])
y_train=train_data["truth"].values
x_test=np.array([create_fvector(example) for example in test_data["token_info"]])
y_test=test_data["truth"].values
x_test_new_words=np.array([count_new_words(example) for example in test_data["token_info"]])

truth_values=["pants-fire","false","barely-true","half-true","mostly-true","true"]
prior_log_prob={}
for t in truth_values:
    prior_log_prob[t]=np.log(float(np.sum(y_train==t))/len(y_train))

like_occ={} #like_occur[t] has jth entry as prob. of jth feature being present when y==t
like_not_occ={} #....not bein present when y==t
for t in truth_values:
    x_t=x_train[y_train==t]
    likelihood=(np.sum(x_t,axis=0)+1)/(len(x_t)+2) #Laplace Smoothing
    like_occ[t]=likelihood
    like_not_occ[t]=1-likelihood

def predict(sample,idx=None):
    prob_truth_values=[]
    for t in truth_values:
        log_like_occ=np.sum(sample*np.log(like_occ[t]))
        log_like_not_occ=np.sum((1-sample)*np.log(like_not_occ[t]))
        not_seen_like=0
        if idx!=None: #seeing test_set so new words can come in
            multiplier=1 if x_test_new_words[idx]>0 else 0
            not_seen_like=multiplier*np.log(1/(2*np.sum(y_train==t)))
        prob_truth_values.append(log_like_occ+log_like_not_occ+prior_log_prob[t]+not_seen_like)
    prob_truth_values=np.array(prob_truth_values)
    prob_truth_values=np.exp(prob_truth_values)
    return prob_truth_values

pred_train_prob=np.array([predict(example) for example in x_train])
pred_test_prob=np.array([predict(example,index) for (index,example) in enumerate(x_test)])
train_predictions=np.array([truth_values[index] for index in np.argmax(pred_train_prob,axis=1)])
test_predictions=np.array([truth_values[index] for index in np.argmax(pred_test_prob,axis=1)])

accuracy_train=(np.sum(y_train==train_predictions)/len(y_train))*100
accuracy_test=(np.sum(y_test==test_predictions)/len(y_test))*100
print("Training Accuracy=",accuracy_train)
print("Testing Accuracy=",accuracy_test)

with open(out_path,"w") as f:
    for i in range(len(test_predictions)):
        f.write(test_predictions[i])
        if i!=len(test_predictions)-1:
            f.write("\n")
