import pandas as pd
import numpy as np
import argparse
from nltk.stem import PorterStemmer
from nltk import bigrams
import re

# python3 nb2.py --train train.tsv --test valid.tsv --out out.txt --stop stopwords.txt

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
train_data_og=pd.read_csv(train_tsv,delimiter="\t",header=None,quoting=3)
test_data_og=pd.read_csv(test_tsv,delimiter="\t",header=None,quoting=3)
train_data=train_data_og.loc[:,[2,1]]
test_data=test_data_og.loc[:,[2,1]]
train_data.columns=["info","truth"]
test_data.columns=["info","truth"]
with open(stop_path,"r") as f:
    stopwords=f.read().splitlines()
stop_words=set(stopwords)
stemmer=PorterStemmer()

token_train_info=[]
for text in train_data["info"]:
    # words=text.lower().split()
    words=re.findall(r"\b\w+\b",text.lower())
    filtered_words=[stemmer.stem(word) for word in words if word not in stop_words]
    filtered_words=[word for word in filtered_words if word not in stop_words]
    token_train_info.append(filtered_words)
train_data["token_info"]=token_train_info

token_test_info=[]
for text in test_data["info"]:
    # words=text.lower().split()
    words=re.findall(r"\b\w+\b",text.lower())
    filtered_words=[stemmer.stem(word) for word in words if word not in stop_words]
    filtered_words=[word for word in filtered_words if word not in stop_words]
    token_test_info.append(filtered_words)
test_data["token_info"]=token_test_info

train_data["subjects"]=train_data_og[3].apply(lambda x: x.split(","))
test_data["subjects"]=test_data_og[3].apply(lambda x: x.split(","))
train_data["speaker"]=train_data_og[4]
test_data["speaker"]=test_data_og[4]
train_data["job"] = train_data_og[5]
test_data["job"] = test_data_og[5]
train_data["state"] = train_data_og[6]
test_data["state"] = test_data_og[6]
train_data["party"] = train_data_og[7]
test_data["party"] = test_data_og[7]
train_data["barely-true-counts"] = train_data_og[8]
test_data["barely-true-counts"] = test_data_og[8]
train_data["false-counts"] = train_data_og[9]
test_data["false-counts"] = test_data_og[9]
train_data["half-true-counts"] = train_data_og[10]
test_data["half-true-counts"] = test_data_og[10]
train_data["mostly-true-counts"] = train_data_og[11]
test_data["mostly-true-counts"] = test_data_og[11]
train_data["pants-fire-counts"] = train_data_og[12]
test_data["pants-fire-counts"] = test_data_og[12]

unigrams_list=set((word,1) for train_example in token_train_info for word in train_example)
bigrams_list=set((f"{w1} {w2}",1) for train_example in token_train_info for (w1,w2) in bigrams(train_example))

subjects_list=set(("sub"+subject,8) for example in train_data["subjects"] for subject in example) #weighted 
speakers_list=set(("s"+speaker,10) for speaker in train_data["speaker"]) #weighted 
speakers_hist_count={"s"+speaker:[0,0,0,0,0] for speaker in train_data["speaker"]}
speakers_curr_count={"s"+speaker:[0,0,0,0,0,0] for speaker in train_data["speaker"]}
job_list=set(("j"+job,1) for job in train_data["job"] if not pd.isna(job))
state_list=set(("st"+state,1) for state in train_data["state"] if not pd.isna(state))
party_list=set(("p"+party,1) for party in train_data["party"] if not pd.isna(party))

dictionary=unigrams_list
dictionary=dictionary.union(bigrams_list)
dictionary=dictionary.union(subjects_list)
dictionary=dictionary.union(speakers_list)
dictionary=dictionary.union(job_list)
dictionary=dictionary.union(state_list)
dictionary=dictionary.union(party_list)
dictionary={word: [index,f] for index,(word,f) in enumerate(dictionary)}
dictionary={word:[index,val[1]] for index,(word,val) in enumerate(dictionary.items())}
dic_size=len(dictionary)

def create_fvector(example,train=False): 
    token_info=example["token_info"]
    fvector=np.zeros(dic_size)
    for word in token_info:
        if word in dictionary:
            fvector[dictionary[word][0]]+=dictionary[word][1]
    for w1,w2 in bigrams(token_info):
        bigram=f"{w1} {w2}"
        if bigram in dictionary:
            fvector[dictionary[bigram][0]]+=dictionary[bigram][1]
    for subject in example["subjects"]:
        subject="sub"+subject
        if subject in dictionary:
            fvector[dictionary[subject][0]]+=dictionary[subject][1]
    speaker="s"+example["speaker"]
    if speaker in dictionary:
        fvector[dictionary[speaker][0]]+=dictionary[speaker][1]
    speakers_hist_count[speaker] = [
        example["barely-true-counts"],
        example["false-counts"],
        example["half-true-counts"],
        example["mostly-true-counts"],
        example["pants-fire-counts"],
    ]
    if speaker not in speakers_curr_count:
        speakers_curr_count[speaker]=[0,0,0,0,0,1]
    else:
        speakers_curr_count[speaker][5]+=1
    if train==True:
        if example["truth"] == "pants-fire":
            speakers_curr_count[speaker][4] += 1
        if example["truth"] == "false":
            speakers_curr_count[speaker][1] += 1
        if example["truth"] == "mostly-true":
            speakers_curr_count[speaker][3] += 1
        if example["truth"] == "barely-true":
            speakers_curr_count[speaker][0] += 1
        if example["truth"] == "half-true":
            speakers_curr_count[speaker][2] += 1
    if not pd.isna(example["job"]):
        job="j"+example["job"]
        if job in dictionary:
            fvector[dictionary[job][0]]+=dictionary[job][1]
    if not pd.isna(example["state"]):
        state="st"+example["state"]
        if state in dictionary:
            fvector[dictionary[state][0]]+=dictionary[state][1]
    if not pd.isna(example["party"]):
        party="p"+example["party"]
        if party in dictionary:
            fvector[dictionary[party][0]]+=dictionary[party][1]
    return fvector

x_train=np.array([create_fvector(example,train=True) for (_,example) in train_data.iterrows()])
y_train=train_data["truth"].values
x_test=np.array([create_fvector(example,train=False) for (_,example) in test_data.iterrows()])
y_test=test_data["truth"].values

truth_values=["pants-fire","false","barely-true","half-true","mostly-true","true"]
prior_log_prob={}
for t in truth_values:
    prior_log_prob[t]=np.log(float(np.sum(y_train==t))/len(y_train))

##################
alpha=25
#################

likelihood_word={}
laplace_total_words={}
for t in truth_values:
    x_t=x_train[y_train==t]
    #Laplace Smoothing
    word_count=np.sum(x_t,axis=0)+(1*alpha)
    laplace_total_words[t]=np.sum(x_t)+(dic_size*alpha)
    likelihood_word[t]=word_count/laplace_total_words[t]

def predict(sample,idx=None,test=False):
    prob_truth_values=[]
    for t in truth_values:
        log_like_occ=np.sum(sample*np.log(likelihood_word[t]))
        prob_truth_values.append(log_like_occ+prior_log_prob[t])
    prob_truth_values=np.array(prob_truth_values)
    if test==True:
        speaker="s"+test_data["speaker"][idx]
        speakers_hist=speakers_hist_count[speaker]
        speakers_curr=speakers_curr_count[speaker]
        total_count=speakers_curr[5]
        possible_barely_true=speakers_hist[0]-speakers_curr[0]
        possible_false=speakers_hist[1]-speakers_curr[1]
        possible_half_true=speakers_hist[2]-speakers_curr[2]
        possible_mostly_true=speakers_hist[3]-speakers_curr[3]
        possible_pants_fire=speakers_hist[4]-speakers_curr[4]
        possible_total=(total_count-speakers_curr[0]-speakers_curr[1]-speakers_curr[2]-speakers_curr[3]-speakers_curr[4])
        prob_truth_values=np.exp(prob_truth_values)
        if possible_false == 0:
            prob_truth_values[1] = 0
        if possible_mostly_true == 0:
            prob_truth_values[4] = 0
        if possible_half_true == 0:
            prob_truth_values[3] = 0
        if possible_pants_fire == 0:
            prob_truth_values[0] = 0
        if possible_barely_true == 0:
            prob_truth_values[2] = 0
    else:
        prob_truth_values=np.exp(prob_truth_values)
        
    return prob_truth_values

pred_train_prob=np.array([predict(example) for example in x_train])
pred_test_prob=np.array([predict(example,index,True) for (index,example) in enumerate(x_test)])
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
