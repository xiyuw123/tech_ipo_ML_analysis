# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 08:50:26 2022

@author: w.xiyu
"""

#ISSUES
#retailmenot inc-20130619 to 20130818-2.txt is empty
#check cloudflare inc-20190814 to 20191013-1.txt

#factiva only lets you download 100 files at a time
#label each file with company ticker
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
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import TfidfModel
from gensim.models.ldamulticore import LdaMulticore

import matplotlib.pyplot as plt 

#convert txt files into dataframe
def clean_txt_file(filename):
    dates = []
    sources = []
    headlines = []
    texts = []
    doc_ids = []
    
    date = None
    source = None
    headline = None
    text = None
    doc_id = None
    
    extracting_text = False
    extracting_doc_id = False
    with open(workingdir+"csv/"+filename, encoding='utf-8') as infile:    
        for line in infile:
            if len(line) < 3:
                continue
            if extracting_text == True:
                #handle cases where we do not want to extract text anymore
                if " ".join(line.split()) == "TD":
                    continue
                elif line[0:3] == "AN ":
                    extracting_text = False
                    extracting_doc_id = True
                    continue
                elif (len(line)<=4) & (line.isupper()):
                    extracting_text = False
                    continue
                #otherwise extract text
                text = text + "\n" + line
            elif extracting_doc_id == True:
                if len(line)!= 0:
                    #document id has been extracted
                    doc_id = " ".join(line.split()[1:])
                    extracting_doc_id = False
                    #add current article to data, reset vars for next article
                    dates += [date]
                    sources += [source]
                    headlines += [headline]
                    texts += [text]
                    doc_ids += [doc_id] 
                    date = None
                    source = None
                    headline = None
                    text = None
                    doc_id = None
            #get headline        
            elif line[0:3] == "HD ":
                headline = " ".join(line.split()[1:])
            #get date
            elif line[0:3] == "PD ":
                date = pd.to_datetime(" ".join(line.split()[1:]))
            #get source
            elif line[0:3] == "SN ":
                source = " ".join(line.split()[1:])
            #start process of getting text
            elif (line[0:3] == "LP ") | (line[0:3] == "TD "):
                extracting_text = True
                if text==None:
                    if len(line) > 3:
                        text = " ".join(line.split()[1:])
                    else:
                        text = ""
            #start process of getting doc id
            elif line[0:3] == "AN ":
                extracting_text = False
                extracting_doc_id = True
    
    df = pd.DataFrame()
    df['date'] = dates
    df['source'] = sources
    df['headline'] = headlines
    df['text'] = texts
    df['doc_id'] = doc_ids
    df['file'] = filename
    return df

def remove_other_comps(matches, text, headline):
    for match in list(set(matches)):
        if [x.islower() for x in match.split()][0]==False:
            if match.lower() in headline.lower():
                #remove match from text
                text = re.sub(match, "", text)
    return text

def clean_other_comps(conml, text, headline):
    matches = re.findall("\w+\s+\w+\s+\w+\s" + conml, text)
    text = remove_other_comps(matches, text, headline)
    matches = re.findall("\w+\s+\w+\s" + conml, text)
    text = remove_other_comps(matches, text, headline)
    matches = re.findall("\w+\s" + conml, text)
    text = remove_other_comps(matches, text, headline)
    return text

def company_detection(articles_df, company_names):
    #find places where company is mentioned and replace all of those with special company name
    #get rid of all non alphanumeric characters, urls except in company names
    i = 0
    company_present = []
    cleaned_text = []
    companies = []
    excl_names = ['Communications', 'Frequency', "Integrated", 'National', 
                 'Technical', 'Universal', 'Western', 'Tech',  'Technical',
                 'Linear', 'Measurement', 'Fair',  'Applied', 'New', 'Code', 
                 "Wireless", "Bit", "Repay",  'Climb', 'Optical', 'Social',
                 'Top', 'Diamond', 'Official', 'Online', 'Move', 'Sustainable',
                 'Network',  'Semiconductor',  'Document', 'Smart', 'Core',
                 'Active', 'Advanced', 'Actions', 'Information', 'Software',
                 'Internet',  'Formula', 'United', 'On', 'My', 'Thunder',
                 'Link', 'Issuer', 'Net', 'Onto',  'Silicon', 'Park', 
                 'Digital', 'Scientific', 'Arm', 'PC', 'Power', 'Dot', 
                 'Check', 'Open', 'One', 'Speed', 'Sonic', 
                 'Key', 'Energy', 'Computer', 'Data', 'Energy', 'Key', 
                 'Research', 'Renaissance', 'Avenue',
                 'One', 'Two', 'Three', 'Q1', 'Q2', 'Q3', 'Q4', 'First', 
                 'Second', 'Third', 'Total',
                 'Vera', 'Smith', 'Henry', 'William', 'Jones', 'John',
                 'Ag', 'CA', 'CGI', 'Co',  'O.', 'U', 'the', 'St', 'Sc', 'P',
                 'On', '6M', 'CPS', 'The', 'Le', 'So', 'Rd', 'By', 'SIC', 'W', 
                 'Market','Preferred', 'Rally', 'Up', 'Capital', 'Financial',
                 'Floor', 'Investment', 'Equity', 'Exchange', 'Stock', 'Bank',
                 'Canada', 'Canadian', 'American', 'US', 'China', 'County',
                 'State', 'City', 'North', 'Northern', 'South', 'Southern', 'West',
                 'Western', 'East', 'Eastern', 'Road', 'Street', 'Avenue',
                 'World', 'Global', 'International','California', 'Boston', 
                 'Blue', 'Black', 'Red', 'Silver',]
    excl_names = [x.lower() for x in excl_names]
    excl_names += stopwords.words('english')
    excl_tics = ['IT', 'IM', 'IN', 'ST', 'III', 'ON', 'AM', 'PM', 'IQ', 'ADC',
                 'CPS', 'RD', 'ROAD', 'DATA', 'BY', 'SIC', 'W',
                 'CA', 'MA', 'IL', 'GA', 'MN', 'AL', 'DC', 'CO']
    while i < len(articles_df):
        filename = articles_df.iloc[i]['file']
        headline = articles_df.iloc[i]['headline']
        text = articles_df.iloc[i]['text']
        if type(text)!=str:
            text = " "+headline+'\n'
        else:
            text = " "+headline+'\n'+text
        conm = " ".join(filename.split('-')[:-2])
        tic = list(company_names.loc[company_names['conm']==conm, 'tic'])[0]
        conml = list(company_names.loc[company_names['conm']==conm, 'conml'])[0]
        companies += [tic]
        
        #remove urls and emails
        text = re.sub("http\S+", "", text)
        text = re.sub("www\S+", "", text)
        text = re.sub("\S+\@\S+\.com", "", text)
        text = re.sub("J[\.\s]+P[\.\s]+Morgan", "JP Morgan", text)
        text = re.sub("U[\.\s]+S[\.\s]+", "United States ", text)
        text = re.sub("\sUS\s", " United States ", text)
        text = re.sub("^US\s", " United States ", text)
        text = re.sub("E[\.\s]+U[\.\s]+", "European Union ", text)
        text = re.sub("\sEU\s", "European Union ", text)
        text = re.sub("^EU\s", "European Union ", text)
        text = re.sub("\se\-", " e", text)
        
        
        #try different variations of company name
        text_compcheck = clean_other_comps(conml, text, headline)
        text_compcheck = clean_other_comps(str.title(conml), text_compcheck, headline)
      
        if len(conml.split()) >= 2:
            conml_short = " ".join(conml.split()[:2])
            text_compcheck = clean_other_comps(conml_short, text_compcheck, headline)
            text_compcheck = clean_other_comps(str.title(conml_short), text_compcheck, headline)
        conml_shorter = " ".join(conml.split()[:1])
        if conml_shorter not in excl_names:
            text_compcheck = clean_other_comps(conml_shorter, text_compcheck, headline)
            text_compcheck = clean_other_comps(str.title(conml_shorter), text_compcheck, headline)
        
        #if company name in article replace it with ticker
        #flag articles for deletion if they do not contain info on the selected company
        #text = re.sub("\s"+tic, "#" + tic + '#', text)
        if ((conml.lower() in text_compcheck.lower()) | (conml_short.lower() in text_compcheck.lower()) | 
            (conml_short.replace(' ',', ') in text_compcheck) |
            (conml_short.replace(' ',', ').lower() in text_compcheck.lower()) |
            (conml_shorter in text_compcheck) |
            (str.title(conml_shorter) in text_compcheck) ):
            company_present += [True]   
        else:
            company_present += [False]
        
        text = re.sub("[^a-zA-Z]"+conml+"[^a-zA-Z]", " #" + tic + '# ', text, flags=re.IGNORECASE)
        text = re.sub("[^a-zA-Z]"+conml.replace(' ', ', ')+"[^a-zA-Z]", " #" + tic + '# ', text, flags=re.IGNORECASE)
        text = re.sub("[^a-zA-Z]"+conml_short+"[^a-zA-Z]", " #" + tic + '# ', text, flags=re.IGNORECASE)
        text = re.sub("[^a-zA-Z]"+conml_short.replace(' ', ', ')+"[^a-zA-Z]", " #" + tic + '# ', text, flags=re.IGNORECASE)
        text = re.sub("[^a-zA-Z]"+conml_shorter+"[^a-zA-Z]", " #" + tic + '# ', text, flags=re.IGNORECASE)
            
        #try to find other company names in text and correct their names
        for j in range(0, len(company_names)):
            conml = company_names.iloc[j]['conml']
            tic_j = company_names.iloc[j]['tic']
            
            text = re.sub("[^a-zA-Z\#]"+conml+"[^a-zA-Z\#]", " #" + tic_j + '# ', text, flags=re.IGNORECASE)

        for j in range(0, len(company_names)):
            conml = company_names.iloc[j]['conml']
            tic_j = company_names.iloc[j]['tic']
            if len(conml.split()) > 3:
                conml = " ".join(conml.split()[:3])
                text = re.sub("[^a-zA-Z\#]"+conml+"[^a-zA-Z\#]", " #" + tic_j + '# ', text, flags=re.IGNORECASE)
                text = re.sub("[^a-zA-Z\#]"+conml.replace(' ', ', ')+"[^a-zA-Z\#]", " #" + tic_j + '# ', text, flags=re.IGNORECASE)
                
                
        for j in range(0, len(company_names)):
            conml = company_names.iloc[j]['conml']
            tic_j = company_names.iloc[j]['tic']
            if len(conml.split()) > 2:
                conml = " ".join(conml.split()[:2])
                if ((conml.split()[0].lower() not in excl_names) & 
                    (conml.split()[1].lower() not in excl_names) ):
                    text = re.sub("[^a-zA-Z\#]"+conml+"[^a-zA-Z\#]", " #" + tic_j + '# ', text, flags=re.IGNORECASE)
                    text = re.sub("[^a-zA-Z\#]"+conml.replace(' ', ', ')+"[^a-zA-Z\#]", " #" + tic_j + '# ', text, flags=re.IGNORECASE)
        
        for j in range(0, len(company_names)):
            conml = company_names.iloc[j]['conml']
            tic_j = company_names.iloc[j]['tic']
            conml = " ".join(conml.split()[:1])
            if ((conml.lower() not in excl_names) & (not conml.isnumeric()) & (len(conml) > 1) ):
                text = re.sub("[^a-zA-Z\#]"+conml+"[^a-zA-Z\#]", " #" + tic_j + '# ', text)
                text = re.sub("[^a-zA-Z\#]"+str.title(conml)+"[^a-zA-Z\#]", " #" + tic_j + '# ', text)
        
        for j in range(0, len(company_names)):
            tic_j = company_names.iloc[j]['tic']
            if tic_j not in excl_tics:
                text = re.sub("[^a-zA-Z\#]"+tic_j+"[^a-zA-Z\#]", " #" + tic_j + '# ', text)         
        
                   
        cleaned_text += [text]
        i = i+1
        
    articles_df['company'] = companies
    articles_df['company_present'] = company_present
    articles_df['cleaned_text'] = cleaned_text
    return articles_df

#process text: tokenize article text and stem tokens
def process(doc):
    #except for company names,
    #make all text lowercase, remove everything except alpha chars
    #sentences = doc.splitlines()
    stop_free = " ".join([i.lower() for i in doc.split() if i.lower() not in set(stopwords.words('english'))])

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


def cross_val_k( test_data, train_data):
    chsize = 500
    train_data_matrix = [dictionary.doc2bow(doc) for doc in train_data]
    tfmodel = TfidfModel(train_data_matrix)
    train_data_matrix = tfmodel[train_data_matrix]
    test_data_matrix = [dictionary.doc2bow(doc) for doc in test_data]
    tfmodel = TfidfModel(train_data_matrix)
    test_data_matrix = tfmodel[test_data_matrix]
    
    ks = []
    scores = []
    k=5
    while k <= 40 :
        
        ldamodel = gensim.models.LdaMulticore(corpus=train_data_matrix,
                                              id2word=dictionary,
                                              num_topics=k,
                                              chunksize=chsize)
        ks += [k]
        scores += [2**ldamodel.log_perplexity(test_data_matrix)]

        k += 1

    plt.plot(ks, scores) 
    
    return None


os.chdir(workingdir+'csv')
filenames= glob.glob("*.txt")

articles_df = pd.DataFrame()
for filename in filenames:
    temp = clean_txt_file(filename)
    articles_df = pd.concat([articles_df, temp], ignore_index=True)
    
articles_df.to_csv(workingdir+"articles.csv",index=False)
articles_df = pd.read_csv(workingdir+"articles.csv")

df_filter1 = ~articles_df["headline"].str.contains("Highs And Lows", na=False)
df_filter2 = ~articles_df["headline"].str.contains("IPO Scorecard", na=False)
df_filter3 = ~articles_df["headline"].str.contains("Substantial Insider", na=False)    
articles_df = articles_df.loc[ df_filter1 & df_filter2 & df_filter3 ]
articles_df['ipo_date'] = [ pd.to_datetime(file.split('-')[-2].split()[0]) + pd.Timedelta(30, unit='d') for file in  articles_df['file']]
articles_df['date'] = pd.to_datetime(articles_df['date'], format='%Y-%m-%d')    
articles_df = articles_df.loc[ articles_df['date'] < articles_df['ipo_date'] ]
articles_df.to_csv(workingdir+"articles_before_ipo.csv",index=False)

articles_df['company'] = [" ".join(filename.split('-')[:-2]) for filename in articles_df['file']]
articles_df = articles_df.loc[articles_df['company'].isin(companies_sample)]

#pull up data to identify company names and tickers
filename = 'CRSP data/200912_201912_quarterly.csv'
crsp_data = pd.read_csv(workingdir+filename)
crsp_data = crsp_data.loc[crsp_data['gsector']==45]
company_names = crsp_data[['GVKEY','tic','conm','conml']].drop_duplicates(keep='first')
company_names['conm'] = company_names['conm'].str.lower()

filename = 'Tech compnaies IPO_2010_2019.xlsx'
ipo_data = pd.read_excel(workingdir+filename,sheet_name=1)
ipo_data = ipo_data[['Global Company Key','Ticker Symbol','Company Name']].drop_duplicates(keep='first')
ipo_data.columns = ['GVKEY','tic', 'conm']
ipo_data = ipo_data.dropna()
ipo_data['conm'] = [conm.lower().replace("-", ' ') for conm in ipo_data['conm']]
ipo_data['conml'] = [str.title(conm) for conm in ipo_data['conm']]

company_names = pd.concat([company_names, ipo_data], ignore_index=True)

#preprocess articles
articles_df = preprocess(articles_df, company_names)
articles_df.to_csv(workingdir+"articles_cleaned.csv",index=False)


articles = pd.read_csv(workingdir+"articles_cleaned.csv")
article_Week= pd.read_csv(workingdir+"articles_cleaned_1wkafter_ipo.csv")
articles_3=pd.read_csv(workingdir+"articles_cleaned_3.csv")
articles_df=pd.concat([articles, article_Week,articles_3])
stemmer = PorterStemmer()
lemma = WordNetLemmatizer()
docs = articles_df['cleaned_text']
docs = [doc.splitlines() for doc in docs]
docs = list(chain.from_iterable(docs))
docs = [x for x in docs if len(x)>0]

doc_cleaned = [process(x) for x in docs]
doc_cleaned = [x for x in doc_cleaned if len(x)>5]


phrase_model = Phrases(doc_cleaned, min_count=1, threshold=100, connector_words=ENGLISH_CONNECTOR_WORDS)
doc_cleaned = phrase_model[doc_cleaned]

docu_clean=[]
for x in doc_cleaned:
    x=[i for i in x if i != "inc"]
    x=[i for i in x if len(i)>2]
    x=[i for i in x if i != "million"]
    x=[i for i in x if i != "llc"]
    x=[i for i in x if i != "co"]
    x=[i for i in x if i != "billion"]
    x=[i for i in x if i != "united"]
    x=[i for i in x if i != "find_article"]
    x=[i for i in x if i != "useful_subscribe"]
    x=[i for i in x if i != "states"]
    x=[i for i in x if i != "m_shs"]
    x=[i for i in x if i != "keywords_news"]
    x=[i for i in x if i != "contact_information"]
    x=[i for i in x if i != "canadian"]
    x=[i for i in x if i != "usd"]
    x=[i for i in x if i != "journal_article"]
    x=[i for i in x if i != "bizjournalscom_subscribe"]
    x=[i for i in x if i != "ltd"]
    x=[i for i in x if i != "copyright_newsrx"]
    x=[i for i in x if i != "sic_code"]
    x=[i for i in x if i != "the"]
    x=[i for i in x if i != "that"]
    x=[i for i in x if i != "there"]
    x=[i for i in x if i != "said"]
   
   
   
   
   
    docu_clean.append(x)


dictionary = corpora.Dictionary(docu_clean)
dictionary.filter_extremes(no_below=50, no_above=0.5)

doc_term_matrix = [dictionary.doc2bow(doc) for doc in docu_clean]
tfmodel = TfidfModel(doc_term_matrix)
doc_term_matrix = tfmodel[doc_term_matrix]

ldamodel = LdaMulticore.load(workingdir+'ldamodel.model')

##########################################################


articles_df['sentences'] = [doc.splitlines() for doc in articles_df['cleaned_text']]
articles_df['sentences'] = [ [x for x in sentences if len(x) > 0] for sentences in articles_df['sentences']]
articles_df['sentences'] = [ [process(x) for x in sentences] for sentences in articles_df['sentences']]
articles_df['sentences'] = [ [x for x in sentences if len(x) > 2] for sentences in articles_df['sentences']]
articles_df['sentences_lda'] = [ [phrase_model[x] for x in sentences] for sentences in articles_df['sentences']]
articles_df['sentences_lda'] = [ [dictionary.doc2bow(x) for x in sentences] for sentences in articles_df['sentences_lda']]
#get most likely topic for sentence
articles_df['sentences_lda'] = [ [np.argmax([ x[1] for x in ldamodel[sen]]) for sen in sentences] for sentences in articles_df['sentences_lda']]

##########################################################

from transformers import BertTokenizer, BertForSequenceClassification

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

labels = {0:0, 1:1,2:-1}
articles_df['sentences_sent'] = [ [' '.join(x) for x in sentences] for sentences in articles_df['sentences']]


 

companies = articles_df['company'].unique()

filter1 = articles_df['date'] < articles_df['ipo_date']
filter2 = articles_df['date'] >= articles_df['ipo_date']
times = ['before_ipo', 'after_ipo']
n_topics = 30

i=0
while i < 2:
    
    final_df = pd.DataFrame()

    t1 = []
    t2 = []
    t3 = []
    t4 = []
    t5 = []
    t6 = []
    t7 = []
    t8 = []
    t9 = []
    t10 = []
    t11 = []
    t12 = []
    t13 = []
    t14 = []
    t15 = [] 
    t16 = [] 
    t17 = [] 
    t18 = [] 
    t19 = [] 
    t20 = [] 
    t21 = []
    t22 = []
    t23 = []
    t24 = []
    t25 = []
    t26 = []
    t27 = []
    t28 = []
    t29 = []
    t30 = []  
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    s5 = []
    s6 = []
    s7 = []
    s8 = []
    s9 = []
    s10 = []
    s11 = []
    s12 = []
    s13 = []
    s14 = []
    s15 = [] 
    s16 = [] 
    s17 = [] 
    s18 = [] 
    s19 = [] 
    s20 = [] 
    s21 = []
    s22 = []
    s23 = []
    s24 = []
    s25 = []
    s26 = []
    s27 = []
    s28 = []
    s29 = []
    s30 = [] 
    
    df_filter = [filter1, filter2][i]
    time = times[i]
    i += 1
    for comp in companies:        
        temp = articles_df.loc[(articles_df['company']==comp) & df_filter]
        #get topic distribution
        topics = temp['sentences_lda']
        topics = list((chain.from_iterable(topics)))
        try:
            topic_dist = { i: topics.count(i)/len(topics) for i in range(0, n_topics) }
        except:
            topic_dist = { i: np.nan for i in range(0, n_topics) }
        t1 += [topic_dist[0]]
        t2 += [topic_dist[1]]
        t3 += [topic_dist[2]]
        t4 += [topic_dist[3]]
        t5 += [topic_dist[4]]
        t6 += [topic_dist[5]]
        t7 += [topic_dist[6]]
        t8 += [topic_dist[7]]
        t9 += [topic_dist[8]]
        t10 += [topic_dist[9]]
        t11 += [topic_dist[10]]
        t12 += [topic_dist[11]]
        t13 += [topic_dist[12]]
        t14 += [topic_dist[13]]
        t15 += [topic_dist[14]]
        t16 += [topic_dist[15]]
        t17 += [topic_dist[16]]
        t18 += [topic_dist[17]]
        t19 += [topic_dist[18]]
        t20 += [topic_dist[19]]
        t21 += [topic_dist[20]]
        t22 += [topic_dist[21]]
        t23 += [topic_dist[22]]
        t24 += [topic_dist[23]]
        t25 += [topic_dist[24]]
        t26 += [topic_dist[25]]
        t27 += [topic_dist[26]]
        t28 += [topic_dist[27]]
        t29 += [topic_dist[28]]
        t30 += [topic_dist[29]]
        #get sentiment for topics
        sentences = temp['sentences_sent']
        if len(sentences) > 0:
            sentences = list((chain.from_iterable(sentences)))
            topic_sent = []
            for sentence in sentences:
                if len(sentence.split()) > 200:
                    sentence = " ".join(sentence.split()[:200])
                inputs = tokenizer(sentence, return_tensors="pt", padding=True)
                outputs = finbert(**inputs)[0]
                topic_sent += [ labels[np.argmax(x)] for x in outputs.detach().numpy()]
            s1 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==0 ])]
            s2 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==1 ])]
            s3 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==2 ])]
            s4 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==3 ])]
            s5 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==4 ])]
            s6 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==5 ])]
            s7 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==6 ])]
            s8 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==7 ])]
            s9 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==8 ])]
            s10 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==9 ])]
            s11 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==10])]
            s12 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==11])]
            s13 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==12])]
            s14 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==13])]
            s15 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==14])] 
            s16 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==15])]
            s17 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==16])]
            s18 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==17])]
            s19 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==18])]
            s20 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==19])]
            s21 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==20])]
            s22 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==21])]
            s23 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==22])]
            s24 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==23])]
            s25 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==24])]
            s26 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==25])]
            s27 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==26])]
            s28 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==27])]
            s29 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==28])]
            s30 += [np.mean( [0] + [ topic_sent[i] for i in range(0, len(topics)) if topics[i]==29])] 
        else:
            s1 += [0]
            s2 += [0]
            s3 += [0]
            s4 += [0]
            s5 += [0]
            s6 += [0]
            s7 += [0]
            s8 += [0]
            s9 += [0]
            s10 += [0]
            s11 += [0]
            s12 += [0]
            s13 += [0]
            s14 += [0]
            s15 += [0] 
            s16 += [0]
            s17 += [0]
            s18 += [0]
            s19 += [0]
            s20 += [0]
            s21 += [0]
            s22 += [0]
            s23 += [0]
            s24 += [0]
            s25 += [0]
            s26 += [0]
            s27 += [0]
            s28 += [0]
            s29 += [0]
            s30 += [0] 

        
    final_df['company'] = companies
    final_df['t1'] = t1
    final_df['t2'] = t2
    final_df['t3'] = t3
    final_df['t4'] = t4
    final_df['t5'] = t5
    final_df['t6'] = t6
    final_df['t7'] = t7
    final_df['t8'] = t8
    final_df['t9'] = t9
    final_df['t10'] = t10
    final_df['t11'] = t11
    final_df['t12'] = t12
    final_df['t13'] = t13
    final_df['t14'] = t14
    final_df['t15'] = t15
    final_df['t16'] = t16
    final_df['t17'] = t17
    final_df['t18'] = t18
    final_df['t19'] = t19
    final_df['t20'] = t20
    final_df['t21'] = t21
    final_df['t22'] = t22
    final_df['t23'] = t23
    final_df['t24'] = t24
    final_df['t25'] = t25
    final_df['t26'] = t26
    final_df['t27'] = t27
    final_df['t28'] = t28
    final_df['t29'] = t29
    final_df['t30'] = t30
    final_df['s1'] = s1
    final_df['s2'] = s2
    final_df['s3'] = s3
    final_df['s4'] = s4
    final_df['s5'] = s5
    final_df['s6'] = s6
    final_df['s7'] = s7
    final_df['s8'] = s8
    final_df['s9'] = s9
    final_df['s10'] = s10
    final_df['s11'] = s11
    final_df['s12'] = s12
    final_df['s13'] = s13
    final_df['s14'] = s14
    final_df['s15'] = s15
    final_df['s16'] = s16
    final_df['s17'] = s17
    final_df['s18'] = s18
    final_df['s19'] = s19
    final_df['s20'] = s20
    final_df['s21'] = s21
    final_df['s22'] = s22
    final_df['s23'] = s23
    final_df['s24'] = s24
    final_df['s25'] = s25
    final_df['s26'] = s26
    final_df['s27'] = s27
    final_df['s28'] = s28
    final_df['s29'] = s29
    final_df['s30'] = s30
    
    
    final_df.to_csv(workingdir+'topic_sentiment_'+time+'.csv', index=False)