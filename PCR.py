import pandas as pd
import yfinance as yf
import numpy as np
import time
import os
import pickle

# 1.期間&標的函數
def get_rolling_base_data(tickers, start_date, end_date):
    
    pkl_file = f"quant_data_{start_date}_to_{end_date}.pkl"
    
    # 嘗試讀PKL存檔
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        return data['price'], data['volume'], data['high'], data['low'], data['fundamentals']

    
    # 時間處理&抓量價
    fetch_start_day = pd.to_datetime(start_date) - pd.DateOffset(years=1)
    fetch_start_day_str = fetch_start_day.strftime("%Y-%m-%d")
    raw_data = yf.download(tickers, start=fetch_start_day_str, end=end_date, auto_adjust=False)
    
    # 清空值+對齊
    price_df = raw_data['Adj Close'].dropna(axis=1, how='all')
    volume_df = raw_data['Volume'][price_df.columns]
    high_df = raw_data['High'][price_df.columns]
    low_df = raw_data['Low'][price_df.columns]
    valid_tickers = price_df.columns.tolist()

  
    # 抓取財報
    fundamentals = {}
    
    for i, ticker in enumerate(valid_tickers):
        try:
            stock = yf.Ticker(ticker)
            income = stock.quarterly_financials
            balance = stock.quarterly_balance_sheet
            cashflow = stock.quarterly_cashflow
            
            if not income.empty and not balance.empty:
                fs = pd.concat([income, balance, cashflow], axis=0)
                fs = fs[~fs.index.duplicated(keep='first')] # 刪除重複會計科目
                fundamentals[ticker] = fs
        except:
            pass
            
        if (i + 1) % 5 == 0:
            time.sleep(1)

    # 存檔
    # PKL
    with open(pkl_file, 'wb') as f:
        pickle.dump({
            'price': price_df,
            'volume': volume_df,
            'high_df': high_df,
            'low_df': low_df,
            'fundamentals': fundamentals
        }, f)
        
    # CSV
    price_df.to_csv(f"price_{start_date}_to_{end_date}.csv")
    volume_df.to_csv(f"volume_{start_date}_to_{end_date}.csv")
    high_df.to_csv(f"high_{start_date}_to_{end_date}.csv")
    low_df.to_csv(f"low_{start_date}_to_{end_date}.csv")
    if fundamentals:
        pd.concat(fundamentals).to_csv(f"fundamentals_{start_date}_to_{end_date}.csv")

    return price_df, volume_df, high_df, low_df, fundamentals

# 主程式
if __name__ == "__main__":
    test_tickers = ['AAPL', 'MSFT', 'NVDA']
    df_p, df_v, df_h, df_l, dict_f = get_rolling_base_data(test_tickers, "2024-01-01", "2026-03-28")