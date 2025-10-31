
import finnhub
import yfinance as yf
import time
import pandas as pd
import sqlite3


#Connecting to FINNHUB to grab United States Stock Data
api_key = "d40ov09r01qqo3qj3lj0d40ov09r01qqo3qj3ljg"
finnhub_client = finnhub.Client(api_key=api_key)
us_stock_data = finnhub_client.stock_symbols('US')
print("There are a total of " + str(len(us_stock_data)) + " US Stocks")


us_stock_tickers = []
#This returns a lot of data for each stock, I am going to just create a list of Tickers (ex: Apples ticker is AAPL)
for stock in us_stock_data:
    us_stock_tickers.append(stock['displaySymbol'])
    
    
# #Going to store the data with SQLITE (Maybe add back later)
# conn = sqlite3.connect('us_stocks.db')
# print("Connected to SQLite DB" )

# I am going to use this list to download data from Yahoo Finance

#us_stock_tickers = us_stock_tickers[:25]



start_date = "2023-01-01"
end_date = "2025-10-28"
batch_size = 10 
all_data = []
batch_count = 1
total_batch = len(us_stock_tickers)/batch_size
for i in range(0, len(us_stock_tickers), batch_size):
    batch = us_stock_tickers[i:i+batch_size]
    df = yf.download(batch, start=start_date, end=end_date, auto_adjust=False)
    temp_df = df['Close'].copy()
    all_data.append(temp_df)
    print("Finished Batch: " + str(batch_count) +" of: " + str(total_batch))
    batch_count+=1
    time.sleep(1)  

main_df = pd.concat(all_data, axis=1)
main_df.to_csv("stock_data_v2.csv")
print("Done ")

