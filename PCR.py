import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import time
import os
import pickle

#=============抓資料=============
def get_rolling_base_data(tickers, start_date, end_date):

    pkl_file = f"quant_data_{start_date}_to_{end_date}.pkl"
    
    # 1. 初始化變數
    price_df = pd.DataFrame()
    volume_df = pd.DataFrame()
    fundamentals = {}

    #嘗試讀PKL檔
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, tuple):
            return data[0], data[1], data[2]
        else:
            return data['price'], data['volume'], data['fundamentals']

    #若沒存檔，開始下載
    fetch_start_day = pd.to_datetime(start_date) - pd.DateOffset(years=1)
    fetch_start_day_str = fetch_start_day.strftime("%Y-%m-%d")
    
    # 批次下載價格與成交量
    raw_data = yf.download(tickers, start=fetch_start_day_str, end=end_date, auto_adjust=False)
    
    if raw_data.empty:
        return price_df, volume_df, fundamentals

    price_df = raw_data['Adj Close'].dropna(axis=1, how='all')
    volume_df = raw_data['Volume'][price_df.columns]
    valid_tickers = price_df.columns.tolist()

    #抓財報
    for i, ticker in enumerate(valid_tickers):
        print(f"正在抓取 ({i+1}/{len(valid_tickers)}): {ticker}")
        try:
            stock = yf.Ticker(ticker)
            income = stock.quarterly_financials
            balance = stock.quarterly_balance_sheet
            cashflow = stock.quarterly_cashflow
            
            if not income.empty and not balance.empty:
                fs = pd.concat([income, balance, cashflow], axis=0)
                fs = fs[~fs.index.duplicated(keep='first')].T
                fs.index = pd.to_datetime(fs.index)
                fundamentals[ticker] = fs.sort_index()
        except Exception as e:
            print(f"   ! {ticker} 失敗: {e}")
            pass
            
        # 節流 防被Yahoo封鎖
        if (i + 1) % 5 == 0:
            time.sleep(1)

    #存檔
    with open(pkl_file, 'wb') as f:
        pickle.dump({'price': price_df, 'volume': volume_df, 'fundamentals': fundamentals}, f)
        
    price_df.to_csv(f"price_{start_date}_to_{end_date}.csv")
    volume_df.to_csv(f"volume_{start_date}_to_{end_date}.csv")

    return price_df, volume_df, fundamentals


#==================特徵工程====================

#算價量指標
def calculate_daily_log_factors(price_df, volume_df, market_price_df=None):
    """
    使用 ln(P_t / P_{t-1}) 計算日頻因子
    market_price_df: 用於計算 Beta 和殘差的大盤資料 (如 ^GSPC)
    """
    # 對數報酬率
    log_ret = np.log(price_df).diff()
    
    # 動能與反轉
    f_1w_rev  = log_ret.rolling(5).sum()
    f_1m_rev  = log_ret.rolling(21).sum()
    f_mom_3m  = log_ret.rolling(63).sum()
    f_mom_6m  = log_ret.rolling(126).sum()
    f_mom_12m = log_ret.rolling(252).sum() - f_1m_rev #12m skip 1m
    
    # 波動度
    f_vol_20d = log_ret.rolling(20).std() * np.sqrt(252)
    f_vol_60d = log_ret.rolling(60).std() * np.sqrt(252)
    
    # 下行波動
    f_vol_down = log_ret.rolling(60).apply(lambda x: x[x<0].std() * np.sqrt(252) if len(x[x<0]) > 2 else np.nan)

    # 流動性 
    f_turnover = volume_df.rolling(20).mean()
    f_turn_chg = f_turnover / f_turnover.shift(20) - 1
    
    # Amihud Illiquidity
    f_amihud = (log_ret.abs() / (price_df * volume_df)).rolling(20).mean()
    f_vol_z  = (volume_df - volume_df.rolling(60).mean()) / volume_df.rolling(60).std()
    f_zero_d = (volume_df == 0).rolling(21).sum()

    # 技術面
    f_ma20_bias = price_df / price_df.rolling(20).mean()
    f_ma60_bias = price_df / price_df.rolling(60).mean()
    
    # MACD (12, 26, 9)
    exp1 = price_df.ewm(span=12, adjust=False).mean()
    exp2 = price_df.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    f_macd_signal = macd - macd.ewm(span=9, adjust=False).mean()
    
    # Bollinger Band Position
    bb_mid = price_df.rolling(20).mean()
    bb_std = price_df.rolling(20).std()
    f_bb_pos = (price_df - (bb_mid - 2*bb_std)) / (4 * bb_std)

    # RSI (14D)
    delta = price_df.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    f_rsi = 100 - (100 / (1 + (gain/loss)))

    #打包
    daily_factors = {
        '1W_Rev': f_1w_rev, '1M_Rev': f_1m_rev, 'Mom_3M': f_mom_3m, 
        'Mom_6M': f_mom_6m, 'Mom_12M_1M': f_mom_12m, 'Vol_20D': f_vol_20d, 
        'Vol_60D': f_vol_60d, 'Vol_Down': f_vol_down, 'Turnover': f_turnover,
        'Turnover_Chg': f_turn_chg, 'Amihud': f_amihud, 'Vol_Z': f_vol_z,
        'Zero_Days': f_zero_d, 'MA20_Bias': f_ma20_bias, 'MA60_Bias': f_ma60_bias,
        'MACD': f_macd_signal, 'BB_Pos': f_bb_pos, 'RSI': f_rsi
    }
    return daily_factors

#抓財報指標&合併對齊
def finalize_quarterly_dataset(daily_factors, fund_dict, price_df, start_date, end_date):

    #季末日期列表
    all_q_dates = pd.date_range(start=start_date, end=end_date, freq='QE')
    final_panel = []

    for t in all_q_dates:
        # 找季末前最近的交易日
        if t not in price_df.index:
            valid_dates = price_df.index[price_df.index <= t]
            if valid_dates.empty: continue
            eval_date = valid_dates[-1]
        else:
            eval_date = t

        #抽取該日期的價量因子
        step_features = {}
        for factor_name, factor_df in daily_factors.items():
            if eval_date in factor_df.index:
                step_features[factor_name] = factor_df.loc[eval_date]
        
        step_df = pd.DataFrame(step_features)
        
        #抽取財報因子
        fund_features = []
        for ticker in step_df.index:
            df_fs = fund_dict.get(ticker)
            if df_fs is not None and not df_fs.empty:
                if not isinstance(df_fs.index, pd.DatetimeIndex):
                    try:
                        df_fs.index = pd.to_datetime(df_fs.index)
                    except:
                        continue
                        
                # 找至少 90 天前的財報
                past_fs = df_fs[df_fs.index <= (eval_date - pd.Timedelta(days=90))]
                
                if len(past_fs) >= 2:
                    latest_report = past_fs.iloc[-1]
                    prev_report = past_fs.iloc[-2]
                    
                    #抓會計科目
                    ni = latest_report.get('Net Income', np.nan)
                    equity = latest_report.get('Stockholders Equity', np.nan)
                    sales_t = latest_report.get('Total Revenue', np.nan)
                    sales_t1 = prev_report.get('Total Revenue', np.nan)
                    assets_t = latest_report.get('Total Assets', np.nan)
                    assets_t1 = prev_report.get('Total Assets', np.nan)
                    ocf = latest_report.get('Operating Cash Flow', np.nan)
                    
                    
                    #算財報因子
                    roe = ni / equity if equity != 0 else np.nan
                    sales_growth = (sales_t / sales_t1) - 1 if sales_t1 != 0 else np.nan
                    asset_growth = (assets_t / assets_t1) - 1 if assets_t1 != 0 else np.nan
                    accruals = ni - ocf if pd.notna(ni) and pd.notna(ocf) else np.nan
                    
                    #打包
                    fund_features.append({
                        'Ticker': ticker,
                        'ROE': roe,
                        'Sales_Growth': sales_growth,
                        'Asset_Growth': asset_growth,
                        'Accruals': accruals
                    })
        
        #合併基本面資料
        if fund_features:
            df_fund = pd.DataFrame(fund_features).set_index('Ticker')
            step_df = step_df.join(df_fund)

        #下一季對數報酬率(找下一季末最近的交易日)
        next_q_date = t + pd.DateOffset(months=3)
        next_valid_dates = price_df.index[price_df.index <= next_q_date]
        
        if len(next_valid_dates) > 0 and eval_date in price_df.index:
            actual_next_date = next_valid_dates[-1]
            step_df['Target_Y'] = np.log(price_df.loc[actual_next_date] / price_df.loc[eval_date])
        else:
            step_df['Target_Y'] = np.nan
            
        #截面資料清洗、縮尾與標準化
        step_df = step_df.dropna(subset=['Target_Y'])
        
        #處理自變數特徵
        features_cols = [c for c in step_df.columns if c != 'Target_Y']
        
        for col in features_cols:
            # 用當季截面中位數填補缺失值
            median_val = step_df[col].median()
            step_df[col] = step_df[col].fillna(median_val)
            
            #1%~99%縮尾
            lower = step_df[col].quantile(0.01)
            upper = step_df[col].quantile(0.99)
            step_df[col] = step_df[col].clip(lower, upper)
            
            #Z-Score標準化
            std_val = step_df[col].std()
            if std_val > 0:
                step_df[col] = (step_df[col] - step_df[col].mean()) / std_val
            else:
                step_df[col] = 0
                
        step_df['Date'] = eval_date
        final_panel.append(step_df)

    #合併所有季度的資料
    final_df = pd.concat(final_panel).reset_index()
    final_df = final_df.rename(columns={'index': 'Ticker'})
    
    return final_df

#======根據不同準則決定最佳的主成分數量 k======

def get_optimal_k(X_train, y_train, method='onatski', max_k=20):
    
    #配適整體PCA取得特徵值
    pca = PCA().fit(X_train)
    eigenvalues = pca.explained_variance_
    X_pca = pca.transform(X_train)
    
    n_samples = len(y_train)
    max_k = min(max_k, X_train.shape[1] - 2) #k不能超過特徵數或樣本數極限

    if method == 'onatski':
        diff1 = eigenvalues[:-1] - eigenvalues[1:]
        ratios = diff1[:-1] / diff1[1:]
        best_k = np.argmax(ratios[:max_k]) + 1
        return best_k

    elif method == 'aic':
        aic_values = []
        for k in range(1, max_k + 1):
            X_k = X_pca[:, :k]
            model = LinearRegression().fit(X_k, y_train)
            rss = np.sum((y_train - model.predict(X_k))**2)
            aic = n_samples * np.log(rss / n_samples) + 2 * k
            aic_values.append(aic)
        return np.argmin(aic_values) + 1
    
    elif method == 'cv':
        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        cv_errors = []
        
        for k in range(1, max_k + 1):
            k_fold_sse = 0
            
            for train_idx, val_idx in kf.split(X_train):
                
                X_cv_train, y_cv_train = X_train[train_idx], y_train[train_idx]
                X_cv_val, y_cv_val = X_train[val_idx], y_train[val_idx]
                
                
                pca_cv = PCA(n_components=k).fit(X_cv_train)
                X_cv_train_pca = pca_cv.transform(X_cv_train)
                
                
                model_cv = LinearRegression().fit(X_cv_train_pca, y_cv_train)
                
                
                X_cv_val_pca = pca_cv.transform(X_cv_val)
                
                preds = model_cv.predict(X_cv_val_pca)
                k_fold_sse += np.sum((y_cv_val - preds)**2)
                
            #累積的SSE
            cv_errors.append(k_fold_sse)
            
        return np.argmin(cv_errors) + 1
    
#=====執行三組PCR與一組PLS，回傳測試集的預測結果=====
    
def run_pcr_and_pls(X_train, y_train, X_test, k_dict):
    predictions = {}
    
    #做一次完整的PCA轉換
    pca_full = PCA().fit(X_train)
    X_train_pca = pca_full.transform(X_train)
    X_test_pca = pca_full.transform(X_test)
    
    for method, k in k_dict.items():
        #根據各準則選出的 k 截取主成分
        X_train_k = X_train_pca[:, :k]
        X_test_k = X_test_pca[:, :k]
        
        #OLS 模型
        model = LinearRegression().fit(X_train_k, y_train)
        predictions[f'PCR_{method}'] = model.predict(X_test_k)
        
    #PLS 預測
    pls_model = PLSRegression(n_components=1)
    pls_model.fit(X_train, y_train)
    
    #壓縮維度
    predictions['PLS_1_comp'] = pls_model.predict(X_test).ravel()
    
    return predictions

#======RMSE計算(有除以樣本數)======

def evaluate_RMSE(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred)**2)/len(y_true))

#======================
def walk_forward_backtest(fin_df, train_window_quarters=4):
    # 確保日期排序
    fin_df = fin_df.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
    quarters = np.sort(fin_df['Date'].unique())
    
    results_log = []
    
    for i in range(train_window_quarters, len(quarters)):
        train_dates = quarters[i - train_window_quarters : i]
        test_date = quarters[i]

        #定義與切分
        train_mask = fin_df['Date'].isin(train_dates)
        test_mask = fin_df['Date'] == test_date
        
        train_set = fin_df[train_mask].copy()
        test_set = fin_df[test_mask].copy()

        #剔除Target_Y
        train_set = train_set.dropna(subset=['Target_Y'])
        test_set = test_set.dropna(subset=['Target_Y'])
        
        if len(train_set) == 0 or len(test_set) == 0:
            print(f"跳過 {test_date}：樣本數不足")
            continue

        #處理X的缺失值與inf
        feature_cols = [c for c in fin_df.columns if c not in ['Date', 'Ticker', 'Target_Y']]
        
        #處理 X_train
        X_train_df = train_set[feature_cols].replace([np.inf, -np.inf], np.nan)
        X_train_df = X_train_df.fillna(X_train_df.median()).fillna(0)
        X_train = X_train_df.values  #Numpy Array
        y_train = train_set['Target_Y'].values

        # 處理 X_test
        X_test_df = test_set[feature_cols].replace([np.inf, -np.inf], np.nan)
        X_test_df = X_test_df.fillna(X_test_df.median()).fillna(0)
        X_test = X_test_df.values
        y_test = test_set['Target_Y'].values

        #找最佳k
        k_cv = get_optimal_k(X_train, y_train, method='cv')
        k_aic = get_optimal_k(X_train, y_train, method='aic')
        k_onatski = get_optimal_k(X_train, y_train, method='onatski')
        
        k_dict = {'CV': k_cv, 'AIC': k_aic, 'Onatski': k_onatski}
        
        #執行模型預測
        predictions = run_pcr_and_pls(X_train, y_train, X_test, k_dict)
        
        #計算 RMSE 並紀錄
        period_metrics = {'Test_Date': test_date, 'Sample_Size': len(y_test)}
        period_metrics.update({f'k_{k}': v for k, v in k_dict.items()})
        
        for model_name, y_pred in predictions.items():
            error_val = evaluate_RMSE(y_test, y_pred)
            period_metrics[f'Error_{model_name}'] = error_val
            
        results_log.append(period_metrics)
        print(f"完成季度: {test_date} | k_Onatski: {k_onatski}")
        
    results_df = pd.DataFrame(results_log)
    print("\n--- 測試集誤差評估結果 ---")
    print(results_df.to_string(index=False))
    
    return results_df


# =======主程式=======
if __name__ == "__main__":

    SP500 = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'GOOG', 'META', 'BRK-B', 'TSLA', 'AVGO', 
    'LLY', 'JPM', 'UNH', 'V', 'XOM', 'MA', 'HD', 'PG', 'COST', 'JNJ', 
    'ABBV', 'ORCL', 'NFLX', 'AMD', 'MRK', 'CVX', 'BAC', 'CRM', 'ADBE', 'PEP', 
    'TMO', 'WMT', 'WFC', 'KO', 'LIN', 'CSCO', 'ACN', 'INTC', 'DIS', 'ABT', 
    'MCD', 'QCOM', 'CAT', 'DHR', 'INTU', 'TXN', 'VZ', 'AMAT', 'MS', 'GE', 
    'PFE', 'IBM', 'AMGN', 'AXP', 'PM', 'UNP', 'NEE', 'LOW', 'SPGI', 'COP', 
    'RTX', 'HON', 'SYK', 'GS', 'BKNG', 'T', 'TJX', 'ETN', 'PLD', 'LMT', 
    'BLK', 'UPS', 'ELV', 'C', 'BA', 'REGN', 'MDLZ', 'VRTX', 'BMY', 'ADP', 
    'BSX', 'GILD', 'AMT', 'ADI', 'PANW', 'ISRG', 'LRCX', 'CB', 'MMC', 'MU', 
    'DE', 'CI', 'SBUX', 'TGT', 'FI', 'NOW', 'VRT', 'LITE', 'COHR', 
    'SATS', 'CIEN', 'PLTR', 'UBER', 'ABNB', 'BX', 'KVUE', 'LULU', 'CDNS', 'SNPS',
    'MAR', 'APH', 'ROP', 'KLAC', 'ADSK', 'MCO', 'EOG', 'PH', 'ITW', 'GD',
    'PGR', 'ECL', 'BDX', 'MCK', 'MPC', 'CTAS', 'EMR', 'NSC', 'TEL', 'TROW',
    'AIG', 'MET', 'D', 'SO', 'DUK', 'AEP', 'WM', 'O', 'KMB', 'STZ',
    'GIS', 'HSY', 'PAYX', 'AZO', 'ORLY', 'GPN', 'HUM', 'CNC', 'IDXX', 'IQV',
    'MDT', 'EW', 'SYK', 'DXCM', 'RMD', 'BIIB', 'VRSK', 'INFO', 'A', 'WELL',
    'PSA', 'DLR', 'VICI', 'CCI', 'AMT', 'EQIX', 'AVB', 'EQR', 'SPG', 'KIM',
    'TRV', 'ALL', 'CB', 'PGR', 'AFL', 'PRU', 'HIG', 'L', 'AJG', 'RE',
    'HLT', 'MAR', 'BKNG', 'YUM', 'DRI', 'CMG', 'NKE', 'TSCO', 'TJX', 'ROST',
    'LOW', 'HD', 'WMT', 'COST', 'TGT', 'DG', 'DLTR', 'EBAY', 'AMZN', 'BK',
    'STT', 'NTRS', 'USB', 'PNC', 'TFC', 'COF', 'DFS', 'SYF', 'AXP', 'MA',
    'V', 'PYPL', 'SQ', 'FI', 'FIS', 'JKHY', 'GPN', 'MCO', 'SPGI', 'NDAQ',
    'CME', 'ICE', 'MSCI', 'CBOE', 'FDS', 'MKTX', 'BEN', 'IVZ', 'BLK', 'AMP',
    'PRU', 'AFL', 'MET', 'AIG', 'L', 'TRV', 'ALL', 'HIG', 'PFG', 'UNM',
    'AON', 'MMC', 'AJG', 'WTW', 'BRO', 'PGR', 'WRB', 'RE', 'CB', 'CINF',
    'DOW', 'LYB', 'CTVA', 'FMC', 'NUE', 'STLD', 'FCX', 'NEM', 'APD', 'LIN',
    'VMC', 'MLM', 'SHW', 'PPG', 'ECL', 'ALB', 'MOS', 'CF', 'IFF', 'BALL',
    'AMCOR', 'PKG', 'IP', 'SEE', 'WRK', 'VNR', 'AVY', 'CE', 'EMN', 'MTD',
    'WAT', 'PKI', 'TMO', 'A', 'DHR', 'ILMN', 'ZBH', 'ABT', 'SYK', 'BSX',
    'EW', 'DXCM', 'RMD', 'MDT', 'ISRG', 'BDX', 'BAX', 'HOLX', 'STE', 'ALGN',
    'IDXX', 'VRTX', 'REGN', 'AMGN', 'GILD', 'BIIB', 'MRNA', 'PFE', 'MRK', 'JNJ',
    'ABBV', 'LLY', 'BMY', 'ELV', 'CI', 'HUM', 'UNH', 'CVS', 'MCK',
    'CAH', 'HSIC', 'XRAY', 'Dentsply', 'VTRS', 'MOH', 'CNC', 'HCA', 'UHS', 'DVA',
    'XOM', 'CVX', 'COP', 'EOG', 'PXD', 'SLB', 'HAL', 'DVN', 'HES', 'OXY',
    'WMB', 'OKE', 'KMI', 'PSX', 'MPC', 'VLO', 'FANG', 'APA', 'CTRA', 'MRO',
    'CHRW', 'EXPD', 'FDX', 'UPS', 'NSC', 'UNP', 'CSX', 'ODFL', 'JBHT', 'DAL',
    'UAL', 'LUV', 'AAL', 'ALK', 'SWK', 'SNA', 'PNR', 'LII', 'TT', 'CARR',
    'JCI', 'HON', 'GE', 'RTX', 'LMT', 'NOC', 'BA', 'GD', 'TDG', 'HEI',
    'HWM', 'AXON', 'TXT', 'CPRT', 'RSG', 'WM', 'SRCL', 'CTAS', 'PAYX', 'ADP',
    'INFO', 'EFX', 'VRSK', 'MCO', 'SPGI', 'FICO', 'TRU', 'NLSN', 'LDOS', 'LEIDOS',
    'CACI', 'SAIC', 'ST', 'BAH', 'KBR', 'J', 'ACM', 'EMR', 'ITW', 'PH',
    'AME', 'ROK', 'DOV', 'XYL', 'DE', 'CAT', 'PCAR', 'CMI', 'OSK', 'HUBB',
    'NDSN', 'OTIS', 'ALLE', 'FBHS', 'MAS', 'MHK', 'VMC', 'MLM', 'EXP', 'SUM',
    'VRT', 'LITE', 'COHR', 'SATS', 'CIEN', 'STX', 'WDC', 'HPQ', 'HPE', 'NTAP',
    'WDC', 'MU', 'INTC', 'NVDA', 'AMD', 'TXN', 'ADI', 'MCHP', 'NXPI', 'ON',
    'SWKS', 'QRVO', 'QCOM', 'AVGO', 'BRCM', 'MPWR', 'MRVL', 'TER', 'LRCX', 'AMAT',
    'KLAC', 'ASML', 'TEL', 'APH', 'MSI', 'CSCO', 'JNPR', 'ANET', 'FTNT', 'PANW',
    'CRWD', 'OKTA', 'ZS', 'DDOG', 'NET', 'TEAM', 'NOW', 'CRM', 'ORCL', 'ADBE',
    'INTU', 'MSFT', 'SAP', 'SNOW', 'PLTR', 'AI', 'PATH', 'UIPATH', 'U', 'UNITY',
    'ADSK', 'ANSY', 'PTC', 'CDNS', 'SNPS', 'TYL', 'PAYC', 'PCTY', 'WDAY', 'GDDY',
    'AKAM', 'VRSN', 'Zscaler', 'F5', 'GEN', 'CHKP', 'FTNT', 'PANW', 'IBM', 'ACN',
    'CDW', 'STN', 'WPP', 'OMC', 'IPG', 'DIS', 'NFLX', 'PARA', 'WBD', 'CHTR',
    'CMCSA', 'TMUS', 'VZ', 'T', 'META', 'GOOGL', 'GOOG', 'LYV', 'TTWO', 'EA',
    'ATVI', 'MTCH', 'IAC', 'DASH', 'UBER', 'LYFT', 'ABNB', 'BKNG', 'EXPE', 'TRIP'
]
    test_tickers = SP500
    df_p, df_v, dict_f = get_rolling_base_data(test_tickers, "2024-01-01", "2025-12-31")
    df_daily_fc = calculate_daily_log_factors(df_p, df_v, market_price_df=None)
    final_df = finalize_quarterly_dataset(df_daily_fc, dict_f, df_p, "2024-01-01", "2025-12-31")
    final_results = walk_forward_backtest(final_df, train_window_quarters=1)
