# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 15:40:38 2022

@author: w.xiyu
"""
workingdir = "C:/Users/w.xiyu/Box/ML Class Project/"

from scipy.stats.mstats import winsorize

import matplotlib.pyplot as plt 

import pandas as pd
import datetime
from datetime import datetime
import numpy as np

from keras import models
from keras import layers
import keras
import tensorflow as tf


from xgboost import XGBRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler



def xgboost_model(x, y):
        
    xgb_model = XGBRegressor()
    clf = GridSearchCV(xgb_model, {'max_depth': [2, 4, 6, 8],
                                   'objective':['reg:squarederror'],
                                   'learning_rate': [0.05, .1, .3, .5, .7],
                                   'n_estimators': [50, 100, 150, 200, 250],
                                   'verbosity': [0]}, verbose=0,
                       n_jobs=2,
                       scoring="neg_mean_squared_error")
    clf.fit(x, y)
    #print(clf.best_score_)
    #print(clf.best_params_)
    
    xgb_model = XGBRegressor(**clf.best_params_)
    
    return xgb_model.fit(x, y, eval_set=[(x, y)], verbose=False)


def nn_model(x, y, model_width=1, num_epoch=100, batch_size=4, validation=0.2):
    
    model = tf.keras.Sequential([
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(rate=0.2, input_shape=(model_width, )),
      tf.keras.layers.Dense(model_width, activation='relu'),
      tf.keras.layers.Dense(model_width, activation='relu'),
      tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer="adam", loss="mean_squared_error")
    
    model.fit(x, y,
          batch_size=batch_size, 
          epochs=num_epoch, 
          validation_split = validation,
          verbose=0)
    
    return model


def evaluation(model, test_data, test_targets, nn_mod=False):
    mse = tf.keras.losses.MeanSquaredError()
    
    if nn_mod==True:
        y_pred = model.predict(test_data).flatten()
    else:
        y_pred = model.predict(test_data)

    out_of_sample_mse = mse(test_targets, y_pred).numpy()
    out_of_sample_mape = tf.keras.metrics.mean_absolute_percentage_error(test_targets, y_pred).numpy()

    return out_of_sample_mse, out_of_sample_mape


filename = 'final_data_before_ipo.csv'
#filename = 'final_data_after_ipo.csv'
labels = ['week_ret', 'week_std']
#labels=['month_ret', 'month_std']


for label in labels:

    final_df = pd.read_csv(workingdir+filename)
    quarter_dummies= pd.get_dummies(final_df['quarter'])
    rating_dummies = pd.get_dummies(final_df['rating'])
    
    
    predictors = ['n_articles', 'n_articles_major', 'n_words', 
                  'pos', 'neg', 'unc', 'str_m', 'weak_m']
    '''
    predictors = ['n_articles', 'n_articles_major', 'n_words', 
                  'pos', 'neg', 'unc', 'str_m', 'weak_m',
                  't1', 't2', 't3', 't4', 't5', 
                  't6', 't7', 't8', 't9', 't10', 
                  't11', 't12', 't13', 't14', 't15', 
                  't16', 't17', 't18', 't19', 't20', 
                  't21', 't22', 't23', 't24', 't25', 
                  't26', 't27', 't28', 't29', 't30', 
                  's1', 's2', 's3', 's4', 's5', 
                  's6', 's7', 's8', 's9', 's10', 
                  's11', 's12', 's13', 's14', 's15', 
                  's16', 's17', 's18', 's19', 's20', 
                  's21', 's22', 's23', 's24', 's25', 
                  's26', 's27', 's28', 's29', 's30',
                  's1*t1', 's2*t2', 's3*t3', 's4*t4', 's5*t5', 
                  's6*t6', 's7*t7', 's8*t8', 's9*t9', 's10*t10', 
                  's11*t11', 's12*t12', 's13*t13', 's14*t14', 's15*t15', 
                  's16*t16', 's17*t17', 's18*t18', 's19*t19', 's20*t20', 
                  's21*t21', 's22*t22', 's23*t23', 's24*t24', 's25*t25', 
                  's26*t26', 's27*t27', 's28*t28', 's29*t29', 's30*t30']
    
    predictors = ['n_articles', 'n_articles_major', 'n_words', 
                  'pos', 'neg', 'unc', 'str_m', 'weak_m',
                  'comp_perf', 'energy', 'ipo', 'product', 'corp_gov', 
                  'stock_ex', 'buy', 'underwriters', 'investors', 
                  'sec_filing', 'analyst', 
                  'sent_comp_perf', 'sent_energy', 'sent_ipo', 
                  'sent_product', 'sent_corp_gov', 'sent_stock_ex', 
                  'sent_buy', 'sent_underwriters', 'sent_investors', 
                  'sent_sec_filing', 'sent_analyst', 
                  'int_comp_perf', 'int_energy', 'int_ipo', 'int_product', 
                  'int_corp_gov', 'int_stock_ex', 'int_buy', 
                  'int_underwriters', 'int_investors', 'int_sec_filing', 
                  'int_analyst']
    '''
    predictors = ['n_articles', 'n_articles_major', 'n_words', 
                  'pos', 'neg', 'unc', 'str_m', 'weak_m',
                  'comp_perf', 'corp_gov', 'buy', 
                  'underwriters', 'analyst', 
                  'sent_comp_perf', 'sent_corp_gov', 'sent_buy', 
                  'sent_underwriters', 'sent_analyst', 
                  'int_comp_perf', 'int_corp_gov', 'int_buy', 
                  'int_underwriters', 'int_analyst']
    
    final_df = final_df.loc[final_df[label].notna()]
    y = final_df[label]*100
    y = winsorize(y, limits=[0, 0.01])
    x = final_df[predictors]
    x = x.join(quarter_dummies)
    x = x.join(rating_dummies)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 
          
    # linear regression
    reg = LinearRegression().fit(x_train, y_train)
    
    # Use LASSO to pick predictors for linear regression
    
    stand_scaler = StandardScaler()
    #lasso_x = x.copy()
    #lasso_x = stand_scaler.fit_transform(lasso_x)
    lasso_reg = LassoCV(cv=10, random_state=0, max_iter=10000, 
                        fit_intercept=False).fit(x_train, y_train)
    
    lasso_x = x.copy()
    lasso_x = lasso_x.iloc[:, lasso_reg.coef_!=0]
    #print(lasso_x.columns)
    
    
    # XGBoosted random forest
    xgb = xgboost_model(x_train, y_train)
    
    # Neural net with normalization layer, dense layer, dropout layer, dense layer
    nn = nn_model(x_train, y_train, model_width = len(predictors))
    
    print(filename)
    print(label)
    print(lasso_x.columns)
    
    #evaluate model on held out data
    print(evaluation(reg, x_test, y_test))
    print(evaluation(lasso_reg, x_test, y_test))
    print(evaluation(xgb, x_test, y_test))
    print(evaluation(nn, x_test, y_test, nn_mod=True))
    print('---------------------------------------------')
    
    
    
    
#filename = 'final_data_before_ipo.csv'
filename = 'final_data_after_ipo.csv'
#labels = ['week_ret', 'week_std']
labels=['month_ret', 'month_std']


for label in labels:

    final_df = pd.read_csv(workingdir+filename)
    quarter_dummies= pd.get_dummies(final_df['quarter'])
    rating_dummies = pd.get_dummies(final_df['rating'])
    
    
    predictors = ['n_articles', 'n_articles_major', 'n_words', 
                  'pos', 'neg', 'unc', 'str_m', 'weak_m']
    '''
    predictors = ['n_articles', 'n_articles_major', 'n_words', 
                  'pos', 'neg', 'unc', 'str_m', 'weak_m',
                  't1', 't2', 't3', 't4', 't5', 
                  't6', 't7', 't8', 't9', 't10', 
                  't11', 't12', 't13', 't14', 't15', 
                  't16', 't17', 't18', 't19', 't20', 
                  't21', 't22', 't23', 't24', 't25', 
                  't26', 't27', 't28', 't29', 't30', 
                  's1', 's2', 's3', 's4', 's5', 
                  's6', 's7', 's8', 's9', 's10', 
                  's11', 's12', 's13', 's14', 's15', 
                  's16', 's17', 's18', 's19', 's20', 
                  's21', 's22', 's23', 's24', 's25', 
                  's26', 's27', 's28', 's29', 's30',
                  's1*t1', 's2*t2', 's3*t3', 's4*t4', 's5*t5', 
                  's6*t6', 's7*t7', 's8*t8', 's9*t9', 's10*t10', 
                  's11*t11', 's12*t12', 's13*t13', 's14*t14', 's15*t15', 
                  's16*t16', 's17*t17', 's18*t18', 's19*t19', 's20*t20', 
                  's21*t21', 's22*t22', 's23*t23', 's24*t24', 's25*t25', 
                  's26*t26', 's27*t27', 's28*t28', 's29*t29', 's30*t30']
    
    predictors = ['n_articles', 'n_articles_major', 'n_words', 
                  'pos', 'neg', 'unc', 'str_m', 'weak_m',
                  'comp_perf', 'energy', 'ipo', 'product', 'corp_gov', 
                  'stock_ex', 'buy', 'underwriters', 'investors', 
                  'sec_filing', 'analyst', 
                  'sent_comp_perf', 'sent_energy', 'sent_ipo', 
                  'sent_product', 'sent_corp_gov', 'sent_stock_ex', 
                  'sent_buy', 'sent_underwriters', 'sent_investors', 
                  'sent_sec_filing', 'sent_analyst', 
                  'int_comp_perf', 'int_energy', 'int_ipo', 'int_product', 
                  'int_corp_gov', 'int_stock_ex', 'int_buy', 
                  'int_underwriters', 'int_investors', 'int_sec_filing', 
                  'int_analyst']
    '''
    predictors = ['n_articles', 'n_articles_major', 'n_words', 
                  'pos', 'neg', 'unc', 'str_m', 'weak_m',
                  'comp_perf', 'corp_gov', 'buy', 
                  'underwriters', 'analyst', 
                  'sent_comp_perf', 'sent_corp_gov', 'sent_buy', 
                  'sent_underwriters', 'sent_analyst', 
                  'int_comp_perf', 'int_corp_gov', 'int_buy', 
                  'int_underwriters', 'int_analyst']
    
    final_df = final_df.loc[final_df[label].notna()]
    y = final_df[label]*100
    y = winsorize(y, limits=[0, 0.01])
    x = final_df[predictors]
    x = x.join(quarter_dummies)
    x = x.join(rating_dummies)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 
          
    # linear regression
    reg = LinearRegression().fit(x_train, y_train)
    
    # Use LASSO to pick predictors for linear regression
    
    stand_scaler = StandardScaler()
    #lasso_x = x.copy()
    #lasso_x = stand_scaler.fit_transform(lasso_x)
    lasso_reg = LassoCV(cv=10, random_state=0, max_iter=10000, 
                        fit_intercept=False).fit(x_train, y_train)
    
    lasso_x = x.copy()
    lasso_x = lasso_x.iloc[:, lasso_reg.coef_!=0]
    #print(lasso_x.columns)
    
    
    # XGBoosted random forest
    xgb = xgboost_model(x_train, y_train)
    
    # Neural net with normalization layer, dense layer, dropout layer, dense layer
    nn = nn_model(x_train, y_train, model_width = len(predictors))
    
    print(filename)
    print(label)
    print(lasso_x.columns)
    
    #evaluate model on held out data
    print(evaluation(reg, x_test, y_test))
    print(evaluation(lasso_reg, x_test, y_test))
    print(evaluation(xgb, x_test, y_test))
    print(evaluation(nn, x_test, y_test, nn_mod=True))
    print('---------------------------------------------')
    

from scipy import stats
from xgboost import plot_importance






# t test to for MAPE significance
filename = 'final_data_before_ipo.csv'


label = 'week_ret'

predictors = ['n_articles', 'n_articles_major', 'n_words', 
              'pos', 'neg', 'unc', 'str_m', 'weak_m']

final_df = pd.read_csv(workingdir+filename)
quarter_dummies= pd.get_dummies(final_df['quarter'])
rating_dummies = pd.get_dummies(final_df['rating'])
final_df = final_df.loc[final_df[label].notna()]
y = final_df[label]*100
y = winsorize(y, limits=[0, 0.01])
x = final_df[predictors]
x = x.join(quarter_dummies)
x = x.join(rating_dummies)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

# XGBoosted random forest
xgb1 = xgboost_model(x_train, y_train)
y_pred = xgb1.predict(x_test)
err1 = 100*np.abs(y_pred - y_test) / np.abs(y_test)

predictors = ['n_articles', 'n_articles_major', 'n_words', 
              'pos', 'neg', 'unc', 'str_m', 'weak_m',
              'comp_perf', 'energy', 'ipo', 'product', 'corp_gov', 
              'stock_ex', 'buy', 'underwriters', 'investors', 
              'sec_filing', 'analyst', 
              'sent_comp_perf', 'sent_energy', 'sent_ipo', 
              'sent_product', 'sent_corp_gov', 'sent_stock_ex', 
              'sent_buy', 'sent_underwriters', 'sent_investors', 
              'sent_sec_filing', 'sent_analyst', 
              'int_comp_perf', 'int_energy', 'int_ipo', 'int_product', 
              'int_corp_gov', 'int_stock_ex', 'int_buy', 
              'int_underwriters', 'int_investors', 'int_sec_filing', 
              'int_analyst']

final_df = pd.read_csv(workingdir+filename)
quarter_dummies= pd.get_dummies(final_df['quarter'])
rating_dummies = pd.get_dummies(final_df['rating'])
final_df = final_df.loc[final_df[label].notna()]
y = final_df[label]*100
y = winsorize(y, limits=[0, 0.01])
x = final_df[predictors]
x = x.join(quarter_dummies)
x = x.join(rating_dummies)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

# XGBoosted random forest
xgb2 = xgboost_model(x_train, y_train)
y_pred = xgb2.predict(x_test)
err2 = 100*np.abs(y_pred - y_test) / np.abs(y_test)

stats.ttest_ind(err1, err2)

# plot feature importance
plot_importance(xgb2)
plt.show()

label = 'week_std'

predictors = ['n_articles', 'n_articles_major', 'n_words', 
              'pos', 'neg', 'unc', 'str_m', 'weak_m']

final_df = pd.read_csv(workingdir+filename)
quarter_dummies= pd.get_dummies(final_df['quarter'])
rating_dummies = pd.get_dummies(final_df['rating'])
final_df = final_df.loc[final_df[label].notna()]
y = final_df[label]*100
y = winsorize(y, limits=[0, 0.01])
x = final_df[predictors]
x = x.join(quarter_dummies)
x = x.join(rating_dummies)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

# XGBoosted random forest
xgb1 = xgboost_model(x_train, y_train)
y_pred = xgb1.predict(x_test)
err1 = 100*np.abs(y_pred - y_test) / np.abs(y_test)

predictors = ['n_articles', 'n_articles_major', 'n_words', 
              'pos', 'neg', 'unc', 'str_m', 'weak_m',
              'comp_perf', 'energy', 'ipo', 'product', 'corp_gov', 
              'stock_ex', 'buy', 'underwriters', 'investors', 
              'sec_filing', 'analyst', 
              'sent_comp_perf', 'sent_energy', 'sent_ipo', 
              'sent_product', 'sent_corp_gov', 'sent_stock_ex', 
              'sent_buy', 'sent_underwriters', 'sent_investors', 
              'sent_sec_filing', 'sent_analyst', 
              'int_comp_perf', 'int_energy', 'int_ipo', 'int_product', 
              'int_corp_gov', 'int_stock_ex', 'int_buy', 
              'int_underwriters', 'int_investors', 'int_sec_filing', 
              'int_analyst']

final_df = pd.read_csv(workingdir+filename)
quarter_dummies= pd.get_dummies(final_df['quarter'])
rating_dummies = pd.get_dummies(final_df['rating'])
final_df = final_df.loc[final_df[label].notna()]
y = final_df[label]*100
y = winsorize(y, limits=[0, 0.01])
x = final_df[predictors]
x = x.join(quarter_dummies)
x = x.join(rating_dummies)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

# XGBoosted random forest
xgb2 = xgboost_model(x_train, y_train)
y_pred = xgb2.predict(x_test)
err2 = np.abs(y_pred - y_test) / np.abs(y_test)

stats.ttest_ind(err1, err2)

# plot feature importance
plot_importance(xgb2)
plt.show()
    




# t test to for MAPE significance
filename = 'final_data_after_ipo.csv'

label = 'month_ret'

predictors = ['n_articles', 'n_articles_major', 'n_words', 
              'pos', 'neg', 'unc', 'str_m', 'weak_m']

final_df = final_df.loc[final_df[label].notna()]
y = final_df[label]*100
y = winsorize(y, limits=[0, 0.01])
x = final_df[predictors]
x = x.join(quarter_dummies)
x = x.join(rating_dummies)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

# XGBoosted random forest
xgb1 = xgboost_model(x_train, y_train)
y_pred = xgb1.predict(x_test)
err1 = 100*np.abs(y_pred - y_test) / np.abs(y_test)

predictors = ['n_articles', 'n_articles_major', 'n_words', 
              'pos', 'neg', 'unc', 'str_m', 'weak_m',
              'comp_perf', 'energy', 'ipo', 'product', 'corp_gov', 
              'stock_ex', 'buy', 'underwriters', 'investors', 
              'sec_filing', 'analyst', 
              'sent_comp_perf', 'sent_energy', 'sent_ipo', 
              'sent_product', 'sent_corp_gov', 'sent_stock_ex', 
              'sent_buy', 'sent_underwriters', 'sent_investors', 
              'sent_sec_filing', 'sent_analyst', 
              'int_comp_perf', 'int_energy', 'int_ipo', 'int_product', 
              'int_corp_gov', 'int_stock_ex', 'int_buy', 
              'int_underwriters', 'int_investors', 'int_sec_filing', 
              'int_analyst']

final_df = final_df.loc[final_df[label].notna()]
y = final_df[label]*100
y = winsorize(y, limits=[0, 0.01])
x = final_df[predictors]
x = x.join(quarter_dummies)
x = x.join(rating_dummies)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

# XGBoosted random forest
xgb2 = xgboost_model(x_train, y_train)
y_pred = xgb2.predict(x_test)
err2 = 100*np.abs(y_pred - y_test) / np.abs(y_test)

stats.ttest_ind(err1, err2)

# plot feature importance
plot_importance(xgb2)
plt.show()

label = 'month_std'

predictors = ['n_articles', 'n_articles_major', 'n_words', 
              'pos', 'neg', 'unc', 'str_m', 'weak_m']

final_df = final_df.loc[final_df[label].notna()]
y = final_df[label]*100
y = winsorize(y, limits=[0, 0.01])
x = final_df[predictors]
x = x.join(quarter_dummies)
x = x.join(rating_dummies)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

# XGBoosted random forest
xgb1 = xgboost_model(x_train, y_train)
y_pred = xgb1.predict(x_test)
err1 = 100*np.abs(y_pred - y_test) / np.abs(y_test)

predictors = ['n_articles', 'n_articles_major', 'n_words', 
              'pos', 'neg', 'unc', 'str_m', 'weak_m',
              'comp_perf', 'energy', 'ipo', 'product', 'corp_gov', 
              'stock_ex', 'buy', 'underwriters', 'investors', 
              'sec_filing', 'analyst', 
              'sent_comp_perf', 'sent_energy', 'sent_ipo', 
              'sent_product', 'sent_corp_gov', 'sent_stock_ex', 
              'sent_buy', 'sent_underwriters', 'sent_investors', 
              'sent_sec_filing', 'sent_analyst', 
              'int_comp_perf', 'int_energy', 'int_ipo', 'int_product', 
              'int_corp_gov', 'int_stock_ex', 'int_buy', 
              'int_underwriters', 'int_investors', 'int_sec_filing', 
              'int_analyst']

final_df = final_df.loc[final_df[label].notna()]
y = final_df[label]*100
y = winsorize(y, limits=[0, 0.01])
x = final_df[predictors]
x = x.join(quarter_dummies)
x = x.join(rating_dummies)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

# XGBoosted random forest
xgb2 = xgboost_model(x_train, y_train)
y_pred = xgb2.predict(x_test)
err2 = np.abs(y_pred - y_test) / np.abs(y_test)

stats.ttest_ind(err1, err2)

# plot feature importance
plot_importance(xgb2)
plt.show()
    