import pandas as pd
import numpy as np

def calculate_rsi(series, period=14):
    """
    Calculates the Relative Strength Index (RSI) using Wilder's Smoothing Method.
    Includes error handling for division by zero and NaN values.
    """
    if len(series) < period:
        return pd.Series([np.nan] * len(series))
    
    delta = series.diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))

    # Using Wilder's smoothing (EMA-based)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # Fill NaN for cases where avg_loss was 0
    return rsi.fillna(100)

def calculate_indicators(df):
    """
    Calculates technical indicators with robustness checks.
    """
    # Ensure dataframe is sorted by date
    df = df.sort_index()
    
    # Basic technicals
    df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['SMA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['SMA200'] = df['Close'].rolling(window=200, min_periods=1).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # RSI
    df['RSI'] = calculate_rsi(df['Close'])
    
    # NR7 detection
    df['Range'] = df['High'] - df['Low']
    df['NR7'] = df['Range'] == df['Range'].rolling(window=7).min()
    
    # Momentum Score (0-100)
    # Combined ROC (Rate of Change) and RSI for a more stable momentum reading
    roc = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12).replace(0, np.nan)) * 100
    df['Momentum_Score'] = (df['RSI'].fillna(50) * 0.6 + (roc.fillna(0) + 50) * 0.4).clip(0, 100)
    
    # VCP Pattern (Volatility Contraction Pattern)
    # Simplified: check if current 10-day volatility is lower than 20-day volatility
    df['Vol_20'] = df['Range'].rolling(window=20, min_periods=1).std()
    df['Vol_10'] = df['Range'].rolling(window=10, min_periods=1).std()
    df['VCP'] = df['Vol_10'] < df['Vol_20'] * 0.8
    
    # Pocket Pivot
    # Volume > max volume of last 10 down days
    down_days_vol = df['Volume'].where(df['Close'] < df['Close'].shift(1), 0)
    df['Max_Down_Vol_10'] = down_days_vol.rolling(window=10, min_periods=1).max()
    df['Pocket_Pivot'] = (df['Close'] > df['Close'].shift(1)) & (df['Volume'] > df['Max_Down_Vol_10'])
    
    return df

def calculate_rrg_metrics(df, benchmark_df):
    """
    Calculates RRG (Relative Rotation Graph) metrics: RS-Ratio and RS-Momentum.
    RS-Ratio: Relative strength of the asset vs benchmark.
    RS-Momentum: Momentum of the RS-Ratio.
    """
    # Align dataframes
    common_idx = df.index.intersection(benchmark_df.index)
    df = df.loc[common_idx]
    bench_df = benchmark_df.loc[common_idx]
    
    # 1. Calculate Relative Strength (RS)
    rs = (df['Close'] / bench_df['Close']) * 100
    
    # 2. Calculate RS-Ratio (using 14-period EMA of RS, normalized)
    # Professional RRG uses a double-smoothed EMA approach
    ema_rs = rs.ewm(span=14, adjust=False).mean()
    rs_ratio = (rs / ema_rs) * 100
    
    # 3. Calculate RS-Momentum (Rate of Change of RS-Ratio)
    rs_mom = (rs_ratio / rs_ratio.shift(10)) * 100
    
    return rs_ratio, rs_mom

def get_ai_grade(row):
    """
    Multi-factor grading system for stock quality.
    """
    # Handle NaN values in row
    if pd.isna(row['Close']) or pd.isna(row['SMA200']):
        return 'Grade C'
        
    score = 0
    if row['Close'] > row['SMA50']: score += 1
    if row['Close'] > row['SMA200']: score += 1
    if row['RSI'] > 60: score += 1
    if row['Momentum_Score'] > 70: score += 1
    if row['Pocket_Pivot']: score += 1
    
    if score >= 4:
        return 'Grade A'
    elif score >= 2:
        return 'Grade B'
    else:
        return 'Grade C'
