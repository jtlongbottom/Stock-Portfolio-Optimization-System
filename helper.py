#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 21:00:27 2025

@author: jakoblongbottom
"""

import yfinance as yf

import pandas as pd


df = pd.read_csv('top1000_assets_ranked_by_sharpe_ratio.csv')




tickers = df['asset'][:200]




def get_stock_names (tickers):
    names = []
    
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            if 'shortName' in list(info.keys()):
                name = info['shortName']
            #asset_type = info['typeDisp']
                
            elif 'longName' in list(info.keys()):
                name = info['longName']
            names.append(name)
        except Exception as e:
            print(f"Error fetching info for {ticker}: {e}")
            continue
    return names
        