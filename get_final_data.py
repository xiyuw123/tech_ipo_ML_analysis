# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 13:40:52 2022

@author: xiyuw
"""
#workingdir = "C:/Users/w.xiyu/Box/ML Class Project/"
workingdir = "C:/Users/xiyuw/Downloads/"

import pandas as pd
import datetime
from datetime import datetime
import numpy as np

times = ['before_ipo', 'after_ipo']
for time in times:

    final_df = pd.read_csv(workingdir+'dict_sentiment_'+time+'.csv')
    temp = pd.read_csv(workingdir+'topic_sentiment_'+time+'.csv')   
    final_df = temp.merge(final_df, how='inner', on='company')
    
    for i in range(1,31):
        colname = 's' + str(i) + '*t' + str(i)
        t_col = 's' + str(i)
        s_col = 't' + str(i)
        final_df[colname] = final_df[s_col] * final_df[t_col]
    
    #topic groupings
    final_df['comp_perf'] = final_df.t1 + final_df.t19
    final_df['energy'] = final_df.t2
    final_df['ipo'] = final_df.t4 + final_df.t14
    final_df['product'] = final_df.t6 + final_df.t7
    final_df['corp_gov'] = final_df.t25
    final_df['stock_ex'] = final_df.t24
    final_df['buy'] = final_df.t27
    final_df['underwriters'] = final_df.t16 + final_df.t22
    final_df['investors'] = final_df.t3 + final_df.t23 + final_df.t30 + final_df.t13 + final_df.t5
    final_df['sec_filing'] = final_df.t9 + final_df.t10 + final_df.t21 + final_df.t17
    final_df['analyst'] = final_df.t29 + final_df.t1
    
    final_df['sent_comp_perf'] = (final_df.s1 + final_df.s19)/2
    final_df['sent_energy'] = final_df.s2
    final_df['sent_ipo'] = (final_df.s4 + final_df.s14)/2
    final_df['sent_product'] = (final_df.s6 + final_df.s7)/2
    final_df['sent_corp_gov'] = final_df.s25
    final_df['sent_stock_ex'] = final_df.s24
    final_df['sent_buy'] = final_df.s27
    final_df['sent_underwriters'] = (final_df.s16 + final_df.s22)/2
    final_df['sent_investors'] = (final_df.s3 + final_df.s23 + final_df.s30 + final_df.s13 + final_df.s5)/5
    final_df['sent_sec_filing'] = (final_df.s9 + final_df.s10 + final_df.s21 + final_df.s17)/4
    final_df['sent_analyst'] = (final_df.s29 + final_df.s1)/2
    
    final_df['int_comp_perf'] = final_df['comp_perf'] * final_df['sent_comp_perf'] 
    final_df['int_energy'] = final_df['energy'] * final_df['sent_energy'] 
    final_df['int_ipo'] = final_df['ipo'] * final_df['sent_ipo'] 
    final_df['int_product'] = final_df['product'] * final_df['sent_product'] 
    final_df['int_corp_gov'] = final_df['corp_gov'] * final_df['sent_corp_gov'] 
    final_df['int_stock_ex'] = final_df['stock_ex'] * final_df['sent_stock_ex'] 
    final_df['int_buy'] = final_df['buy'] * final_df['sent_buy'] 
    final_df['int_underwriters'] = final_df['underwriters'] * final_df['sent_underwriters'] 
    final_df['int_investors'] = final_df['investors'] * final_df['sent_investors'] 
    final_df['int_sec_filing'] = final_df['sec_filing'] * final_df['sent_sec_filing'] 
    final_df['int_analyst'] = final_df['analyst'] * final_df['sent_analyst'] 
    
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
    
    ipo_prices = pd.read_csv(workingdir+"ipo_price_history.csv")
    ipo_prices['price'] = ipo_prices['prccd'] / ipo_prices['ajexdi']
    ipo_prices['ipo_date'] = pd.to_datetime(ipo_prices['ipo_date'], format="%Y-%m-%d")
    ipo_prices['datadate'] = pd.to_datetime(ipo_prices['datadate'], format="%Y-%m-%d")
    ipo_prices['quarter'] = pd.PeriodIndex(ipo_prices['ipo_date'], freq='Q')
    ipo_prices['date_diff'] = ipo_prices['datadate'] - ipo_prices['ipo_date']
    ipo_prices['date_diff'] = ipo_prices['date_diff'] / np.timedelta64(1, 'D')
    ipo_prices['conm'] = [conm.lower().replace("-", ' ') for conm in ipo_prices['conm']]
    companies = ipo_prices['conm'].unique()
    ipo_final_df = pd.DataFrame()
    
    tickers = []
    week_ret = []
    week_std = []
    month_ret = []
    month_std = []
    ipo_quarters = []
    
    for comp in companies:
        tic = list(company_names.loc[company_names['conm']==comp, 'tic'])[0]
        tickers+=[tic]
        temp = ipo_prices.loc[ipo_prices['conm'] == comp].copy()
        ipo_quarters += [list(temp['quarter'])[0]]
        temp['return'] = temp['price'].pct_change()
        if temp['date_diff'].max() > 25:
            returns = list(temp['price'])
            month_ret += [ returns[-1] - returns[0]  ]
            month_std += [temp['return'].std()]
        else:
            month_ret += [None]
            month_std += [None]
        temp = temp.loc[temp['date_diff'] < 8]
        if len(temp) > 0:
            returns = list(temp['price'])
            week_ret += [ returns[-1] - returns[0]  ]
            week_std += [temp['return'].std()]
        else:
            week_ret += [None]
            week_std += [None]
            
    ipo_final_df['company'] = tickers
    ipo_final_df['quarter'] = ipo_quarters
    ipo_final_df['week_ret'] = week_ret
    ipo_final_df['week_std'] = week_std
    ipo_final_df['month_ret'] = month_ret
    ipo_final_df['month_std'] = month_std
    
    final_df = final_df.merge(ipo_final_df, how='inner', on='company')
    
    controls = pd.read_csv(workingdir+"ipo_price_history_controls.csv")
    controls = controls.drop_duplicates(subset = ['tic'], keep='first')
    controls = controls[['tic', 'Trading Volume - Monthly', 'Net Asset Value - Monthly', 
                         'Stock Exchange Code', 'S&P Rating']]
    controls.columns = ['company', 'vol_m', 'net_value_m', 'stock_exchange', 'rating']
    
    final_df = final_df.merge(controls, how='inner', on='company')
    final_df.loc[final_df['rating'].isnull(), 'rating' ] = 'unrated'
    final_df.to_csv(workingdir+'final_data_'+time+'.csv',index=False)