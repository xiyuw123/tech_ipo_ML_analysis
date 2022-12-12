# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 11:42:54 2022

@author: w.xiyu
"""

workingdir = "C:/Users/w.xiyu/Box/ML Class Project/"

import pandas as pd
from striprtf.striprtf import rtf_to_text
from nltk.tokenize import word_tokenize
from nltk.stem import *
import os, glob
import re

import datetime
from datetime import datetime
import numpy as np

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from itertools import chain
from sklearn.model_selection import train_test_split

#run MOD_Load_MasterDictionary_v2022.py first

#process text: tokenize article text and stem tokens
def process(doc):
    #except for company names,
    #make all text lowercase, remove everything except alpha chars
    #sentences = doc.splitlines()
    stop_free = " ".join([i.lower() for i in doc.split() if i.lower() not in set(stopwords)])

    #remove all punctuation except for tickers
    punc_free = ''
    for word in stop_free.split():
        temp = word
        if len(word)!= 0:
            if word[0]!='#':
                temp = re.sub('\.', '', word)
                temp = re.sub('[^a-zA-Z#\s]', ' ', temp)
        temp = temp.strip()
        punc_free = punc_free + ' ' + temp        
    punc_free = re.sub('\ss\s', ' ', punc_free)
    punc_free = re.sub('\s\s+', ' ', punc_free)
    normalized = [lemma.lemmatize(word) for word in punc_free.split()]
    return normalized


#make sentiment dictionary from master dictionary
sentiment_dict = {}
for word in master_dictionary.keys():
    wd_dict = master_dictionary[word]
    if ( (wd_dict.positive>0) | (wd_dict.negative>0) | (wd_dict.uncertainty>0) | 
         (wd_dict.strong_modal>0) | (wd_dict.weak_modal>0) ):
        sentiment_dict[word.lower()] = [(wd_dict.positive > 0), 
                                        (wd_dict.negative > 0), 
                                        (wd_dict.uncertainty > 0), 
                                        (wd_dict.strong_modal > 0),
                                        (wd_dict.weak_modal > 0) ]
        
        
major_news_sources = ['The Wall Street Journal', 
    'The Wall Street Journal Online',
    'The Wall Street Journal (Europe Edition)'
    'The Wall Street Journal (Asia Edition)'
    'The Wall Street Journal Asia', 
    'The Wall Street Journal Europe',
    'International New York Times' 
    'Washington Post.com',
    'WSJ Pro Venture Capital',
    'WSJ Pro Private Equity',
    'WSJ Pro Cybersecurity',
    'WSJ Pro Financial Regulation',
    'WSJ Pro Central Banking',
    'WSJ Pro Artificial Intelligence',
    'NYT Blogs']
        
        
articles_df = pd.read_csv(workingdir+"articles_cleaned.csv")

docs = articles_df['cleaned_text']
docs = [x for x in docs if len(x)>0]

lemma = WordNetLemmatizer()

doc_cleaned = [process(x) for x in docs]
doc_cleaned = [x for x in doc_cleaned if len(x)>0]

n_words = []
pos = []
neg = []
unc = []
str_m = []
weak_m = []

for doc in doc_cleaned:
    n_words += [len(doc)]
    pos += [np.sum([sentiment_dict[word][0] for word in doc if word in sentiment_dict.keys()])]
    neg += [np.sum([sentiment_dict[word][1] for word in doc if word in sentiment_dict.keys()])]
    unc += [np.sum([sentiment_dict[word][2] for word in doc if word in sentiment_dict.keys()])]
    str_m += [np.sum([sentiment_dict[word][3] for word in doc if word in sentiment_dict.keys()])]
    weak_m += [np.sum([sentiment_dict[word][4] for word in doc if word in sentiment_dict.keys()])]
    
articles_df['n_words'] = n_words
articles_df['pos'] = pos
articles_df['neg'] = neg
articles_df['unc'] = unc
articles_df['str_m'] = str_m
articles_df['weak_m'] = weak_m


companies = articles_df['company'].unique()
final_df = pd.DataFrame()
n_words = []
n_articles_major = []
n_articles = []
pos = []
neg = []
unc = []
str_m = []
weak_m = []
for comp in companies:
    temp = articles_df.loc[articles_df['company']==comp]
    n_words += [np.average(temp['n_words'])]
    n_articles += [len(temp)]
    n_articles_major += [len(temp.loc[temp['source'].isin(major_news_sources)])]
    pos += [np.average(temp['pos'] / temp['n_words'])]
    neg += [np.average(temp['neg'] / temp['n_words'])]
    unc += [np.average(temp['unc'] / temp['n_words'])]
    str_m += [np.average(temp['str_m'] / temp['n_words'])]
    weak_m += [np.average(temp['weak_m'] / temp['n_words'])]
    
final_df['company'] = companies
final_df['n_articles'] = n_articles
final_df['n_articles_major'] = n_articles_major
final_df['n_words'] = n_words
final_df['pos'] = pos
final_df['neg'] = neg
final_df['unc'] = unc
final_df['str_m'] = str_m
final_df['weak_m'] = weak_m

final_df.to_csv(workingdir+'dict_sentiment_before_ipo.csv', index=False)

##########################################################################

temp1 = pd.read_csv(workingdir+"articles_cleaned_1wkafter_ipo.csv")
temp2 = pd.read_csv(workingdir+"articles_cleaned_3.csv")
articles_df = pd.concat([temp1, temp2], ignore_index=True)

docs = articles_df['cleaned_text']
docs = [x for x in docs if len(x)>0]

lemma = WordNetLemmatizer()

doc_cleaned = [process(x) for x in docs]
doc_cleaned = [x for x in doc_cleaned if len(x)>0]

n_words = []
pos = []
neg = []
unc = []
str_m = []
weak_m = []

for doc in doc_cleaned:
    n_words += [len(doc)]
    pos += [np.sum([sentiment_dict[word][0] for word in doc if word in sentiment_dict.keys()])]
    neg += [np.sum([sentiment_dict[word][1] for word in doc if word in sentiment_dict.keys()])]
    unc += [np.sum([sentiment_dict[word][2] for word in doc if word in sentiment_dict.keys()])]
    str_m += [np.sum([sentiment_dict[word][3] for word in doc if word in sentiment_dict.keys()])]
    weak_m += [np.sum([sentiment_dict[word][4] for word in doc if word in sentiment_dict.keys()])]
    
articles_df['n_words'] = n_words
articles_df['pos'] = pos
articles_df['neg'] = neg
articles_df['unc'] = unc
articles_df['str_m'] = str_m
articles_df['weak_m'] = weak_m


companies = articles_df['company'].unique()
final_df = pd.DataFrame()
n_words = []
n_articles_major = []
n_articles = []
pos = []
neg = []
unc = []
str_m = []
weak_m = []
for comp in companies:
    temp = articles_df.loc[articles_df['company']==comp]
    n_words += [np.average(temp['n_words'])]
    n_articles += [len(temp)]
    n_articles_major += [len(temp.loc[temp['source'].isin(major_news_sources)])]
    pos += [np.average(temp['pos'] / temp['n_words'])]
    neg += [np.average(temp['neg'] / temp['n_words'])]
    unc += [np.average(temp['unc'] / temp['n_words'])]
    str_m += [np.average(temp['str_m'] / temp['n_words'])]
    weak_m += [np.average(temp['weak_m'] / temp['n_words'])]
    
final_df['company'] = companies
final_df['n_articles'] = n_articles
final_df['n_articles_major'] = n_articles_major
final_df['n_words'] = n_words
final_df['pos'] = pos
final_df['neg'] = neg
final_df['unc'] = unc
final_df['str_m'] = str_m
final_df['weak_m'] = weak_m

final_df.to_csv(workingdir+'dict_sentiment_after_ipo.csv', index=False)