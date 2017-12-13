# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:58:58 2017

@author: jpmaldonado
"""

import os
import numpy as np
from tqdm import tqdm

dir_name = "20_newsgroup"


## Create vectorizer
text = []
#categories = os.listdir(dir_name)

categories = ['soc.religion.christian','rec.sport.hockey']

for category in categories:
    sub_dir = os.path.join(dir_name,category)
    for fname in os.listdir(sub_dir):
        full_fname = os.path.join(sub_dir,fname)
        with open(full_fname, 'r') as f:
            for line in f:
                text.append(line)
                
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(stop_words='english',  # Remove common words
                              token_pattern=r'[a-zA-Z]{3,}',  # regex to choose the words
                              lowercase=True,
                              use_idf=True
                              ) 

print("Fitting the vectorizer")
vec.fit(text) # Fit the vectorizer, but not transform!

vocab_size = len(vec.vocabulary_)       
       


## Here we go again through the full corpus, this time to vectorize each line
X = np.empty((1,vocab_size)) #dummy vectorize: just take the line
y = np.empty((1,len(categories)))


print("Encoding categories")
idx_to_cat = {}
cat_to_idx = {}

for i, cat in enumerate(categories):
    idx_to_cat[i] = cat
    cat_to_idx[cat] = i    


## Generator for the state
def read():
    for category in categories:
        sub_dir = os.path.join(dir_name,category)
        for fname in os.listdir(sub_dir):
            full_fname = os.path.join(sub_dir,fname)
            with open(full_fname, 'r') as f:
                for line in f:
                    x = vec.transform([line])
                    y = cat_to_idx[category]
                    yield x,y

def policy(state,theta):
    action = 1 if np.dot(state.todense(),theta.T) > 0 else 0
    return action                        

print("Random search time!")
theta = np.random.normal(size=vocab_size)
rewards = []
best_reward = 0
best_theta = None
tot_reward = 0
for state, cat in tqdm(read()): # this is analog to the env.step() in OpenAI Gym, except that the state is simply moving to the next line.
    action = policy(state,theta)
    reward = 1 if cat==action else 0
    rewards.append(reward)
    if len(rewards)==100: # Check the theta vector in batches and update
        if sum(rewards)>best_reward:
            best_reward = sum(rewards)
            best_theta = theta
        else:
            theta = np.random.normal(size=vocab_size)
print("Best parameters found:")
print(best_theta)            
print("Best reward in 100 sentences")
print(best_reward)        
