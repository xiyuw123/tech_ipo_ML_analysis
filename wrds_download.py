# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 17:10:16 2022

@author: xiyuw
"""

workingdir = "C:/Users/w.xiyu/Box/ML Class Project/"

import pandas as pd
from striprtf.striprtf import rtf_to_text

import wrds
import re

import datetime
from datetime import datetime
import numpy as np


filename = 'Tech compnaies IPO_2010_2019.csv'
ipos = pd.read_csv(workingdir+filename)
ipos['Company Initial Public Offering Date'] = pd.to_datetime(ipos['Company Initial Public Offering Date'])    
ipos = ipos.drop_duplicates(subset=['Company Initial Public Offering Date', 'Global Company Key'], keep='first')
ipos = ipos.sort_values(by=['Company Initial Public Offering Date'], ascending=False)



conn = wrds.Connection(wrds_username='xiyuw123')
#conn.create_pgpass_file()
#conn.list_libraries()
#conn.list_tables(library='comp')
#conn.get_table(library='comp', table='company', columns = ['conm', 'gvkey', 'cik'], obs=5)


ipos_prices = pd.DataFrame()

for i in ipos.index:
    comp = str(ipos.loc[i]['CUSIP'])
    start_date = pd.to_datetime(str(ipos.loc[i]['Company Initial Public Offering Date']), format="%Y-%m-%d")
    end_date = start_date + pd.Timedelta(31, "d")
    
    end = end_date.strftime("%Y-%m-%d")
    start = start_date.strftime("%Y-%m-%d")

    query = """select cusip, datadate, prccd, ajexdi
                    from comp.secd 
                    where cusip = '{comp}'
                    and datadate>='{start}' and datadate<='{end}' """.format(comp=comp, end=end, start=start)
    temp = conn.raw_sql(query, 
                        date_cols=['date'])
    if len(temp)==0:
        print(str(ipos.loc[i]['Company Name']))
        end = end_date.strftime("%m/%d/%Y")
        start = start_date.strftime("%m/%d/%Y")
        query = """select cusip, date, prc, cfacpr
                    from crsp.dsf 
                    where cusip = '{comp}'
                    and date>='{start}' and date<='{end}' """.format(comp=comp[:8], end=end, start=start)
        temp = conn.raw_sql(query, 
                            date_cols=['date'])
        if len(temp)>0:
            temp.columns = ['cusip', 'datadate', 'prccd', 'ajexdi']
            temp['cusip'] = comp
        
    temp["conm"] = str(ipos.loc[i]['Company Name'])
    temp['tic'] = str(ipos.loc[i]['Ticker Symbol'])
    temp['ipo_date'] = start_date
    
    ipos_prices = pd.concat([ipos_prices, temp], ignore_index=True)


conn.close()

ipos_prices.to_csv(workingdir+'ipo_price_history.csv', index=False)


'''
No compustat data for

SAFE T GROUP LTD
MOBILICOM LTD
JETPAY CORP
VRINGO INC  -OLD
NEXSAN CORP

No CRSP data for

WISEKEY INTERNATIONAL HOLD
DOCEBO INC
LIGHTSPEED COMMERCE INC
MMTEC INC
COOTEK (CAYMAN) INC
AMERICAN VIRTUAL CL TECH INC
MOBILICOM LTD
MONSTER DIGITAL INC -OLD
ATLASSIAN CORP
KINAXIS INC
CHEETAH MOBILE INC  -ADR
NET ELEMENT INC
AVIGILON CORP
NEXJ SYSTEMS INC
JETPAY CORP
VRINGO INC  -OLD
KINGTONE WIRELESS  -ADR -OLD
NEXSAN CORP
'''


###########################################################

