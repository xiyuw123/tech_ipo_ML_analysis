#!/usr/bin/env python
# coding: utf-8

# In[12]:


get_ipython().system('pip install gensim')
workingdir = "C:/Users/arjain/Box/ML Class Project/"
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import *
import os, glob
import re
from scipy.stats import ttest_ind

import datetime
from datetime import datetime
import numpy as np

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

import gensim
from gensim import corpora
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from itertools import chain
from sklearn.model_selection import train_test_split

from gensim.test.utils import datapath
from gensim.models.word2vec import Text8Corpus
from gensim.models.phrases import Phrases,ENGLISH_CONNECTOR_WORDS
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import TfidfModel
from gensim.parsing.preprocessing import strip_short
import matplotlib.pyplot as plt 


# In[2]:


before = pd.read_csv(workingdir+"final_data_before_ipo.csv")
after=pd.read_csv(workingdir+"final_data_after_ipo.csv")


# mean_before=before[['t1','t2','t3','t4']].mean()
# print (mean_before)

# In[9]:


mean_before=before[['t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18','t19','t20','t21','t22','t23','t24','t25','t26','t27','t28','t29','t30']].mean() 
print (mean_before)


# In[10]:


mean_after=after[['t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18','t19','t20','t21','t22','t23','t24','t25','t26','t27','t28','t29','t30']].mean() 
print (mean_after)


# In[21]:


ttest1=ttest_ind(before['t1'], after['t1'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t2'], after['t2'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t3'], after['t3'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t4'], after['t4'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t5'], after['t5'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t6'], after['t6'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t7'], after['t7'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t8'], after['t8'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t9'], after['t9'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t10'], after['t10'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t11'], after['t11'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t12'], after['t12'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t13'], after['t13'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t14'], after['t14'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t15'], after['t15'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t16'], after['t16'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t17'], after['t17'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t18'], after['t18'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t19'], after['t19'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t20'], after['t20'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t21'], after['t21'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t22'], after['t22'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t23'], after['t23'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t24'], after['t24'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t25'], after['t25'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t26'], after['t26'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t27'], after['t27'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t28'], after['t28'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t29'], after['t29'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['t30'], after['t30'],equal_var=False)
print (ttest1)


# In[24]:


before['Comp perform'] = before['t1'] + before['t19']
before['energy']=before['t2']
before['IPO offering']=before['t4']
before['firm product']=before['t6']+ before['t7']
before['IPO market']=before['t14']
before['corporate gove']=before['t25']
before['stock exchange']=before['t24']
before['underwriter']=before['t16']+before['t22']
before['buy']=before['t27']
before['investors']=before['t3']+before['t23']+before['t30']+before['t13']+before['t5']
before['SEC filling']=before['t9']+before['t10']+before['t21']+before['t17']
before['analyst opinion']=before['t29']+before['t12']


# In[27]:


after['Comp perform'] = after['t1'] + after['t19']
after['energy']=after['t2']
after['IPO offering']=after['t4']
after['firm product']=after['t6']+ after['t7']
after['IPO market']=after['t14']
after['corporate gove']=after['t25']
after['stock exchange']=after['t24']
after['underwriter']=after['t16']+after['t22']
after['buy']=after['t27']
after['investors']=after['t3']+after['t23']+after['t30']+after['t13']+after['t5']
after['SEC filling']=after['t9']+after['t10']+after['t21']+after['t17']
after['analyst opinion']=after['t29']+after['t12']


# In[28]:


ttest1=ttest_ind(before['Comp perform'], after['Comp perform'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['energy'], after['energy'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['IPO offering'], after['IPO offering'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['firm product'], after['firm product'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['IPO market'], after['IPO market'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['corporate gove'], after['corporate gove'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['stock exchange'], after['stock exchange'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['underwriter'], after['underwriter'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['buy'], after['buy'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['investors'], after['investors'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['SEC filling'], after['SEC filling'],equal_var=False)
print (ttest1)
ttest1=ttest_ind(before['analyst opinion'], after['analyst opinion'],equal_var=False)
print (ttest1)


# In[ ]:




