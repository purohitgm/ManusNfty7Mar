import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from indicators import calculate_indicators, get_ai_grade, calculate_rrg_metrics
from datetime import datetime
import time

# Page Config
st.set_page_config(layout="wide", page_title="Nifty Sectoral Analytics Dashboard", page_icon="📈")

# Constants
INDEX_TICKERS = {
    "Nifty 50": "^NSEI",
    "Nifty 100": "^CNX100",
    "Nifty 200": "^CNX200",
    "Nifty Bank": "^NSEBANK",
    "Nifty IT": "^CNXIT",
    "Nifty Auto": "^CNXAUTO",
    "Nifty FMCG": "^CNXFMCG",
    "Nifty Metal": "^CNXMETAL",
    "Nifty Pharma": "^CNXPHARMA",
    "Nifty Reality": "^CNXREALTY",
    "Nifty Energy": "^CNXENERGY",
    "Nifty Infra": "^CNXINFRA"
}

# Mapping Sectoral Indices to key stocks
SECTOR_STOCKS = {
    "Nifty Bank": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
    "Nifty IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
    "Nifty Auto": ["TATAMOTORS.NS", "M&M.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS"],
    "Nifty FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "VBL.NS"],
    "Nifty Metal": ["TATASTEEL.NS", "JINDALSTEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "VEDL.NS"],
    "Nifty Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"],
    "Nifty Reality": ["DLF.NS", "LODHA.NS", "OBEROIRLTY.NS", "PHOENIXLTD.NS", "PRESTIGE.NS"],
    "Nifty Energy": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "BPCL.NS"],
    "Nifty Infra": ["LT.NS", "ADANIPORTS.NS", "GRASIM.NS", "ULTRACEMCO.NS", "BHARTIARTL.NS"]
}

@st.cache_data(ttl=300)
def fetch_data(ticker, period="1y"):
    """
    Fetches data with caching and error handling.
    """
    try:
        # Using a timeout or error handling for yfinance
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if df is None or df.empty:
            return None
        # Handle multi-index columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None

def main():
    st.sidebar.title("Dashboard Settings")
    auto_refresh = st.sidebar.checkbox("Auto Refresh (60s)", value=False)
    
    st.title("📊 Indian Stock Market: Nifty Sectoral Analytics")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. LIVE INDEX SUMMARY
    st.subheader("Indices Performance")
    
    index_performance = []
    # Fetching indices in a loop (could be optimized with bulk download)
    for name, ticker in INDEX_TICKERS.items():
        df = fetch_data(ticker, period="5d")
        if df is not None and len(df) >= 2:
            last_close = float(df['Close'].iloc[-1])
            prev_close = float(df['Close'].iloc[-2])
            pct_change = ((last_close - prev_close) / prev_close) * 100
            
            index_performance.append({
                "Sector": name,
                "Change%": pct_change,
                "Price": last_close
            })
    
    if index_performance:
        # Display as a grid of metrics
        perf_df = pd.DataFrame(index_performance)
        cols = st.columns(4)
        for i, row in perf_df.iterrows():
            with cols[i % 4]:
                st.metric(row['Sector'], f"{row['Price']:,.2f}", f"{row['Change%']:+.2f}%")
        
        # 2. SECTOR HEATMAP & RRG
        st.divider()
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.subheader("Sectoral Momentum Ranking")
            perf_df_sorted = perf_df.sort_values(by="Change%", ascending=False)
            fig_bar = px.bar(perf_df_sorted, x="Sector", y="Change%", color="Change%",
                         color_continuous_scale='RdYlGn', title="Today's Sectoral Performance")
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_right:
            st.subheader("Relative Rotation Graph (RRG)")
            # Calculate RRG metrics for each sector vs Nifty 50 benchmark
            benchmark_ticker = INDEX_TICKERS["Nifty 50"]
            bench_df = fetch_data(benchmark_ticker, period="1y")
            
            rrg_data = []
            if bench_df is not None:
                for name, ticker in INDEX_TICKERS.items():
                    sec_df = fetch_data(ticker, period="1y")
                    if sec_df is not None:
                        rs_ratio, rs_mom = calculate_rrg_metrics(sec_df, bench_df)
                        rrg_data.append({
                            "Sector": name,
                            "RS-Ratio": rs_ratio.iloc[-1],
                            "RS-Momentum": rs_mom.iloc[-1]
                        })
            
            if rrg_data:
                rrg_df = pd.DataFrame(rrg_data)
                fig_rrg = px.scatter(rrg_df, x="RS-Ratio", y="RS-Momentum", text="Sector",
                                     color="Sector", size_max=60, title="RRG: Sector Rotation vs Nifty 50")
                
                # Add quadrant lines and labels
                fig_rrg.add_vline(x=100, line_dash="dash", line_color="gray")
                fig_rrg.add_hline(y=100, line_dash="dash", line_color="gray")
                
                # Quadrant Annotations
                fig_rrg.add_annotation(x=101, y=101, text="LEADING", showarrow=False, font=dict(color="green", size=12))
                fig_rrg.add_annotation(x=99, y=101, text="IMPROVING", showarrow=False, font=dict(color="blue", size=12))
                fig_rrg.add_annotation(x=99, y=99, text="LAGGING", showarrow=False, font=dict(color="red", size=12))
                fig_rrg.add_annotation(x=101, y=99, text="WEAKENING", showarrow=False, font=dict(color="orange", size=12))
                
                fig_rrg.update_layout(xaxis_title="RS-Ratio (Strength)", yaxis_title="RS-Momentum (Momentum)")
                st.plotly_chart(fig_rrg, use_container_width=True)

    # 3. STOCK LEVEL ANALYSIS
    st.divider()
    selected_sector = st.selectbox("Drill-down: Select Sector to Analyze Stocks", list(SECTOR_STOCKS.keys()))
    
    if selected_sector:
        tickers = SECTOR_STOCKS[selected_sector]
        stock_data = []
        
        with st.spinner(f"Analyzing stocks in {selected_sector}..."):
            for ticker in tickers:
                df = fetch_data(ticker, period="1y")
                if df is not None and len(df) > 20:
                    df = calculate_indicators(df)
                    last_row = df.iloc[-1]
                    prev_row = df.iloc[-2]
                    
                    stock_data.append({
                        "Ticker": ticker,
                        "Price": float(last_row['Close']),
                        "Change%": ((float(last_row['Close']) - float(prev_row['Close'])) / float(prev_row['Close'])) * 100,
                        "RSI(14)": float(last_row['RSI']),
                        "Momentum": float(last_row['Momentum_Score']),
                        "NR7": "Yes" if last_row['NR7'] else "No",
                        "VCP": "Detected" if last_row['VCP'] else "No",
                        "Pocket Pivot": "Yes" if last_row['Pocket_Pivot'] else "No",
                        "Above 200DMA": "Yes" if last_row['Close'] > last_row['SMA200'] else "No",
                        "Grade": get_ai_grade(last_row)
                    })
        
        if stock_data:
            df_stocks = pd.DataFrame(stock_data)
            
            # Visuals: Treemap
            st.subheader(f"Heatmap: {selected_sector} Constituents")
            fig_tree = px.treemap(df_stocks, path=['Ticker'], values='Price', color='Change%',
                                  color_continuous_scale='RdYlGn', hover_data=['RSI(14)', 'Grade'])
            st.plotly_chart(fig_tree, use_container_width=True)
            
            # Filters
            st.subheader("Interactive Filters")
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                rsi_range = st.slider("RSI Range", 0, 100, (0, 100))
            with f_col2:
                min_momentum = st.number_input("Min Momentum Score", 0, 100, 0)
            
            filtered_df = df_stocks[
                (df_stocks['RSI(14)'] >= rsi_range[0]) & 
                (df_stocks['RSI(14)'] <= rsi_range[1]) &
                (df_stocks['Momentum'] >= min_momentum)
            ]
            
            # Styling for the table
            def color_grade(val):
                color = 'white'
                if val == 'Grade A': color = '#90ee90'
                elif val == 'Grade C': color = '#ffcccb'
                elif val == 'Grade B': color = '#ffffba'
                return f'background-color: {color}'

            st.dataframe(filtered_df.style.applymap(color_grade, subset=['Grade']), use_container_width=True)

            # Advance-Decline Ratio
            advances = len(df_stocks[df_stocks['Change%'] > 0])
            declines = len(df_stocks[df_stocks['Change%'] < 0])
            st.info(f"**Market Breadth ({selected_sector}):** Advances: {advances} | Declines: {declines} | A/D Ratio: {advances/max(declines,1):.2f}")

    if auto_refresh:
        time.sleep(60)
        st.rerun()

if __name__ == "__main__":
    main()
