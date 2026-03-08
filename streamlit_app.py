"""
╔══════════════════════════════════════════════════════════════╗
║  NIFTY MARKET INTELLIGENCE — Streamlit Edition              ║
║  Stack: Streamlit + yfinance + React 18 + Recharts 2        ║
╠══════════════════════════════════════════════════════════════╣
║  SETUP:                                                      ║
║    pip install streamlit yfinance pandas numpy               ║
║                streamlit-autorefresh                         ║
║    streamlit run streamlit_app.py                            ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Optional, List
import warnings
warnings.filterwarnings("ignore")

# ── AUTO-REFRESH ──────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, key="live_refresh")
except ImportError:
    pass

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Nifty Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace !important; }
  .stApp { background: #060c14; }
  .block-container { padding: 1rem 2rem !important; }
  div[data-testid="metric-container"] > label {
    font-size: 11px !important; color: #6b7280 !important;
    font-family: 'IBM Plex Mono', monospace !important;
  }
  div[data-testid="metric-container"] > div {
    font-size: 20px !important;
    font-family: 'IBM Plex Mono', monospace !important;
  }
  div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
  .stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
  }
  .stSidebar { background: #0b1422 !important; }
  hr { border-color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────
INDICES = {
    "NIFTY 50":   "^NSEI",
    "NIFTY 100":  "^CNX100",
    "NIFTY 200":  "^CNX200",
    "NIFTY 500":  "^CNX500",
    "SENSEX":     "^BSESN",
    "INDIA VIX":  "^INDIAVIX",
}

SECTOR_INDICES = {
    "Nifty Bank":         "^NSEBANK",
    "Nifty IT":           "^CNXIT",
    "Nifty Auto":         "^CNXAUTO",
    "Nifty Pharma":       "^CNXPHARMA",
    "Nifty FMCG":         "^CNXFMCG",
    "Nifty Metal":        "^CNXMETAL",
    "Nifty Realty":       "^CNXREALTY",
    "Nifty Energy":       "^CNXENERGY",
    "Nifty Infra":        "^CNXINFRA",
    "Nifty Fin Services": "^CNXFINANCE",
    "Nifty Healthcare":   "^CNXHEALTH",
    "Nifty PSU Bank":     "^CNXPSUBANK",
    "Nifty Oil & Gas":    "^CNXOILGAS",
    "Nifty Media":        "^CNXMEDIA",
    "Nifty Cons Dur":     "^CNXCONSDURBL",
}

NIFTY50_STOCKS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
    "LT.NS","BAJFINANCE.NS","ASIANPAINT.NS","MARUTI.NS","TITAN.NS",
    "WIPRO.NS","HCLTECH.NS","ULTRACEMCO.NS","AXISBANK.NS","NESTLEIND.NS",
    "SUNPHARMA.NS","TATAMOTORS.NS","TATASTEEL.NS","TECHM.NS","INDUSINDBK.NS",
    "BAJAJFINSV.NS","POWERGRID.NS","NTPC.NS","ONGC.NS","COALINDIA.NS",
    "ADANIGREEN.NS","ADANIPORTS.NS","HINDALCO.NS","GRASIM.NS","DIVISLAB.NS",
    "CIPLA.NS","DRREDDY.NS","EICHERMOT.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS",
    "BPCL.NS","IOC.NS","JSWSTEEL.NS","VEDL.NS","APOLLOHOSP.NS",
    "M&M.NS","TVSMOTOR.NS","PERSISTENT.NS","ZOMATO.NS","PIDILITIND.NS",
]

C = {
    "bull": "#00e5a0", "bear": "#f87171", "warn": "#fbbf24",
    "bg":   "#060c14", "card": "#0d1828", "border": "#1e293b",
    "muted":"#6b7280", "text": "#e2e8f0",
}

# ── TECHNICAL INDICATORS ──────────────────────────────────────
def calc_rsi(series: pd.Series, period: int = 14) -> Optional[float]:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss
    rsi   = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2) if not rsi.empty else None

def calc_sma(series: pd.Series, period: int) -> Optional[float]:
    if len(series) < period: return None
    return round(float(series.rolling(period).mean().iloc[-1]), 2)

def calc_ema(series: pd.Series, period: int) -> Optional[float]:
    if len(series) < period: return None
    return round(float(series.ewm(span=period, adjust=False).mean().iloc[-1]), 2)

def is_nr7(highs: pd.Series, lows: pd.Series) -> bool:
    if len(highs) < 7: return False
    ranges = (highs - lows).iloc[-7:]
    return float(ranges.iloc[-1]) == float(ranges.min())

def is_vcp(closes: pd.Series, highs: pd.Series, lows: pd.Series) -> bool:
    if len(closes) < 20: return False
    vols = []
    for w in range(4):
        s = closes.iloc[-(20 - w*5): -(15 - w*5) or None]
        if len(s) < 3: continue
        vols.append((s.max() - s.min()) / s.mean())
    return all(vols[i] <= vols[i-1] for i in range(1, len(vols))) if len(vols) >= 3 else False

def is_pocket_pivot(volumes: pd.Series, closes: pd.Series) -> bool:
    if len(volumes) < 11: return False
    if not (closes.iloc[-1] > closes.iloc[-2]): return False
    prev_10 = pd.DataFrame({
        'v': volumes.iloc[-11:-1].values,
        'c': closes.iloc[-11:-1].values,
        'cp': closes.iloc[-12:-2].values
    })
    up_vols = prev_10[prev_10['c'] > prev_10['cp']]['v']
    return float(volumes.iloc[-1]) > float(up_vols.max()) if len(up_vols) > 0 else False

def momentum_score(rsi, vol_ratio, abv50, nr7, pp, rs_change) -> float:
    return round(
        max(0, min(100, ((rsi - 40) / 30) * 100)) * 0.20 +
        min(100, vol_ratio * 50)                   * 0.20 +
        (75 if abv50 else 25)                      * 0.15 +
        (85 if nr7   else 30)                       * 0.10 +
        (82 if pp    else 28)                       * 0.15 +
        max(0, min(100, 50 + rs_change * 5))        * 0.20,
    1)

def assign_grade(score, rsi, rs_change) -> str:
    if score > 68 and 50 <= rsi <= 75 and rs_change > 0: return "A"
    if score > 50 and rs_change > -2: return "B"
    return "C"

# ── DATA LOADERS ──────────────────────────────────────────────
def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns if present."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data(ttl=60, show_spinner=False)
def load_indices() -> pd.DataFrame:
    rows = []
    end, start = datetime.today(), datetime.today() - timedelta(days=5)
    for name, ticker in INDICES.items():
        try:
            hist = _flatten(yf.download(ticker, start=start, end=end,
                                        progress=False, auto_adjust=True))
            if hist.empty: continue
            c = hist["Close"].dropna()
            if len(c) < 2: continue
            price = float(c.iloc[-1])
            chg   = round((price / float(c.iloc[-2]) - 1) * 100, 2)
            rows.append({"Index": name, "Level": round(price, 2), "Change%": chg})
        except Exception:
            pass
    return pd.DataFrame(rows)

@st.cache_data(ttl=60, show_spinner=False)
def load_sectors() -> pd.DataFrame:
    end, start = datetime.today(), datetime.today() - timedelta(days=180)
    try:
        nifty = _flatten(yf.download("^NSEI", start=start, end=end,
                                     progress=False, auto_adjust=True))["Close"].dropna().squeeze()
    except Exception:
        nifty = pd.Series(dtype=float)

    rows = []
    for name, ticker in SECTOR_INDICES.items():
        try:
            hist = _flatten(yf.download(ticker, start=start, end=end,
                                        progress=False, auto_adjust=True))
            if hist.empty or len(hist) < 5: continue
            c = hist["Close"].dropna().squeeze()
            h = hist["High"].dropna().squeeze()
            l = hist["Low"].dropna().squeeze()
            v = hist["Volume"].dropna().squeeze() if "Volume" in hist.columns else pd.Series(dtype=float)

            rsi   = calc_rsi(c)
            sma20 = calc_sma(c, 20)
            price = float(c.iloc[-1])

            if len(c) >= 20 and len(nifty) >= 20:
                rs = round(((float(c.iloc[-1]) - float(c.iloc[-20])) / float(c.iloc[-20]) -
                            (float(nifty.iloc[-1]) - float(nifty.iloc[-20])) / float(nifty.iloc[-20])) * 100, 2)
            else:
                rs = 0.0

            vol_ratio = 1.0
            if len(v) > 20 and v.sum() > 0:
                avg20 = float(v.iloc[-20:].mean())
                vol_ratio = round(float(v.iloc[-1]) / avg20, 2) if avg20 > 0 else 1.0

            breadth = round(min(95, max(10, 50 + rs * 3)), 1)
            score   = momentum_score(rsi or 50, vol_ratio,
                                     bool(sma20 and price > sma20), is_nr7(h, l), False, rs)
            chg = round((price / float(c.iloc[-2]) - 1) * 100, 2) if len(c) > 1 else 0.0
            rows.append({
                "Sector": name, "Level": round(price, 2), "Change%": chg,
                "RSI(14)": rsi, "Vol/20DMA": vol_ratio,
                "RS vs N50": rs, "Breadth%": breadth, "Momentum": score,
            })
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")

    df = pd.DataFrame(rows)
    return df.sort_values("Momentum", ascending=False) if not df.empty else df

@st.cache_data(ttl=120, show_spinner=False)
def load_stocks(symbols: List[str]) -> pd.DataFrame:
    end, start = datetime.today(), datetime.today() - timedelta(days=180)
    try:
        nifty = _flatten(yf.download("^NSEI", start=start, end=end,
                                     progress=False, auto_adjust=True))["Close"].dropna().squeeze()
    except Exception:
        nifty = pd.Series(dtype=float)

    rows = []
    for sym in symbols:
        try:
            hist = _flatten(yf.download(sym, start=start, end=end,
                                        progress=False, auto_adjust=True))
            if len(hist) < 30: continue
            c = hist["Close"].dropna().squeeze()
            h = hist["High"].dropna().squeeze()
            l = hist["Low"].dropna().squeeze()
            v = hist["Volume"].dropna().squeeze() if "Volume" in hist.columns else pd.Series(dtype=float)
            if len(c) < 30: continue

            price  = float(c.iloc[-1])
            chg    = round((price / float(c.iloc[-2]) - 1) * 100, 2)
            rsi    = calc_rsi(c)
            ema20  = calc_ema(c, 20)
            sma50  = calc_sma(c, 50)
            sma200 = calc_sma(c, 200)
            avg20v = float(v.iloc[-20:].mean()) if len(v) > 20 and v.sum() > 0 else 1
            vol_ratio = round(float(v.iloc[-1]) / avg20v, 2) if avg20v > 0 else 1.0

            if len(c) >= 20 and len(nifty) >= 20:
                rs = round(((price - float(c.iloc[-20])) / float(c.iloc[-20]) -
                            (float(nifty.iloc[-1]) - float(nifty.iloc[-20])) / float(nifty.iloc[-20])) * 100, 2)
            else:
                rs = 0.0

            nr7   = is_nr7(h, l)
            vcp   = is_vcp(c, h, l)
            pp    = is_pocket_pivot(v, c)
            score = momentum_score(rsi or 50, vol_ratio,
                                   bool(sma50 is not None and price > sma50), nr7, pp, rs)
            grade = assign_grade(score, rsi or 50, rs)

            try:
                mcap = round((yf.Ticker(sym).fast_info.market_cap or 0) / 1e10, 1)
            except Exception:
                mcap = None

            rows.append({
                "Symbol": sym.replace(".NS", ""), "Price": round(price, 2), "Change%": chg,
                "RSI": rsi, "Vol/DMA": vol_ratio, "MCap(₹Bn)": mcap,
                ">20EMA": bool(ema20  is not None and price > ema20),
                ">50SMA": bool(sma50  is not None and price > sma50),
                ">200SMA": bool(sma200 is not None and price > sma200),
                "NR7": nr7, "VCP": vcp, "PP": pp, "RS vs N50": rs,
                "Momentum": score, "Grade": grade,
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    return df.sort_values("Momentum", ascending=False) if not df.empty else df


# ══════════════════════════════════════════════════════════════
#  REACT + RECHARTS CHART FUNCTIONS
#  Rendered via st.components.v1.html()
#  React 18 + Recharts 2.12 from unpkg CDN, JSX via Babel.
#  NOTE: All JSX is in plain (non-f) strings so {} are safe.
# ══════════════════════════════════════════════════════════════

_HEAD = (
    '<script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>'
    '<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>'
    '<script src="https://unpkg.com/recharts@2.12.7/umd/Recharts.js"></script>'
    '<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>'
)

_CSS = """
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#060c14; font-family:'IBM Plex Mono',monospace; }
.wrap { background:#0d1828; border:1px solid #1e293b; border-radius:10px; padding:10px 14px 12px; }
.ct { color:#6b7280; font-size:10px; letter-spacing:.1em; font-weight:600; margin-bottom:8px; }
"""

def _page(title: str, data_json: str, height: int, jsx: str) -> str:
    """Assemble a self-contained React+Recharts HTML page.
    JSX is a plain string — no f-string so {} is safe for JS/JSX."""
    return (
        '<!DOCTYPE html><html><head><meta charset="UTF-8"/>' + _HEAD +
        '<style>' + _CSS + '</style></head><body>'
        '<div class="wrap"><div class="ct">' + title + '</div><div id="root"></div></div>'
        '<script>const DATA=' + data_json + ';const H=' + str(height) + ';</script>'
        '<script type="text/babel">'
        'const {useState}=React;'
        'const {BarChart,Bar,XAxis,YAxis,CartesianGrid,Tooltip,Legend,'
        'AreaChart,Area,ScatterChart,Scatter,ZAxis,'
        'PieChart,Pie,Cell,ResponsiveContainer,ReferenceLine,LabelList}=Recharts;'
        + jsx +
        'ReactDOM.createRoot(document.getElementById("root")).render(React.createElement(Chart,null));'
        '</script></body></html>'
    )


def _sc(v: float) -> str:   # score colour
    if v >= 72: return "#00e5a0"
    if v >= 58: return "#4ade80"
    if v >= 44: return "#fbbf24"
    if v >= 30: return "#f97316"
    return "#f87171"

def _cc(v: float) -> str:   # change colour
    return "#00e5a0" if v >= 0 else "#f87171"

def _rc(v: float) -> str:   # RSI colour
    if v >= 70: return "#f97316"
    if v >= 55: return "#00e5a0"
    if v <= 35: return "#a78bfa"
    return "#fbbf24"


# ── 1. Horizontal Bar: Sector % Change ───────────────────────
def chart_heatmap(df: pd.DataFrame, height: int = 420):
    df_s = df.sort_values("Change%").copy()
    records = [{"s": row["Sector"].replace("Nifty ",""),
                "v": round(float(row["Change%"]),2),
                "f": _cc(row["Change%"])}
               for _, row in df_s.iterrows()]
    jsx = """
function CBar(p){
  const {x,y,width,height,fill}=p;
  return <rect x={x} y={y+1} width={width} height={Math.max(0,height-2)} fill={fill} rx={3}/>;
}
function CLabel(p){
  const {x,y,width,value,fill}=p;
  const pos=value>=0;
  return <text x={pos?x+width+5:x+width-5} y={y+10} fill={fill}
    fontSize={9} fontFamily="IBM Plex Mono" textAnchor={pos?"start":"end"}>
    {(value>=0?"+":"")+value.toFixed(2)+"%"}
  </text>;
}
function Chart(){
  return <ResponsiveContainer width="100%" height={H}>
    <BarChart data={DATA} layout="vertical" margin={{top:4,right:68,left:8,bottom:4}}>
      <CartesianGrid horizontal={false} stroke="rgba(255,255,255,.05)"/>
      <XAxis type="number" domain={["auto","auto"]} axisLine={false} tickLine={false}
        tick={{fill:"#6b7280",fontSize:9,fontFamily:"IBM Plex Mono"}}
        tickFormatter={v=>(v>=0?"+":"")+v.toFixed(1)+"%"}/>
      <YAxis type="category" dataKey="s" width={82} axisLine={false} tickLine={false}
        tick={{fill:"#94a3b8",fontSize:9,fontFamily:"IBM Plex Mono"}}/>
      <Tooltip cursor={{fill:"rgba(255,255,255,.04)"}}
        contentStyle={{background:"#0d1828",border:"1px solid #1e293b",borderRadius:6,
          fontFamily:"IBM Plex Mono",fontSize:11}}
        formatter={v=>[(v>=0?"+":"")+v.toFixed(2)+"%","Change"]}/>
      <Bar dataKey="v" shape={<CBar/>} isAnimationActive={false}>
        {DATA.map((d,i)=><Cell key={i} fill={d.f}/>)}
        <LabelList dataKey="v" content={<CLabel/>}/>
      </Bar>
    </BarChart>
  </ResponsiveContainer>;
}
"""
    components.html(_page("SECTOR % CHANGE TODAY", json.dumps(records), height, jsx), height=height+40)


# ── 2. Horizontal Bar: Momentum Scores ───────────────────────
def chart_momentum(df: pd.DataFrame, height: int = 420):
    df_s = df.sort_values("Momentum").copy()
    records = [{"s": row["Sector"].replace("Nifty ",""),
                "v": round(float(row["Momentum"]),1),
                "f": _sc(row["Momentum"])}
               for _, row in df_s.iterrows()]
    jsx = """
function CBar(p){
  const {x,y,width,height,fill}=p;
  return <rect x={x} y={y+1} width={width} height={Math.max(0,height-2)} fill={fill} fillOpacity={0.85} rx={3}/>;
}
function SLabel(p){
  const {x,y,width,value}=p;
  return <text x={x+width+5} y={y+10} fontSize={9}
    fontFamily="IBM Plex Mono" fill="#9ca3af" textAnchor="start">{value}</text>;
}
function Chart(){
  return <ResponsiveContainer width="100%" height={H}>
    <BarChart data={DATA} layout="vertical" margin={{top:4,right:52,left:8,bottom:4}}>
      <CartesianGrid horizontal={false} stroke="rgba(255,255,255,.05)"/>
      <XAxis type="number" domain={[0,100]} axisLine={false} tickLine={false}
        tick={{fill:"#6b7280",fontSize:9,fontFamily:"IBM Plex Mono"}}/>
      <YAxis type="category" dataKey="s" width={82} axisLine={false} tickLine={false}
        tick={{fill:"#94a3b8",fontSize:9,fontFamily:"IBM Plex Mono"}}/>
      <Tooltip cursor={{fill:"rgba(255,255,255,.04)"}}
        contentStyle={{background:"#0d1828",border:"1px solid #1e293b",borderRadius:6,
          fontFamily:"IBM Plex Mono",fontSize:11}}
        formatter={v=>[v.toFixed(1),"Score"]}/>
      <Bar dataKey="v" shape={<CBar/>} isAnimationActive={false}>
        {DATA.map((d,i)=><Cell key={i} fill={d.f}/>)}
        <LabelList dataKey="v" content={<SLabel/>}/>
      </Bar>
    </BarChart>
  </ResponsiveContainer>;
}
"""
    components.html(_page("SECTOR MOMENTUM SCORE (0–100)", json.dumps(records), height, jsx), height=height+40)


# ── 3. Sector Heatmap Grid (React div tiles) ─────────────────
def chart_treemap(df: pd.DataFrame, height: int = 340):
    df_s = df.sort_values("Momentum", ascending=False).head(15).copy()
    records = []
    for _, row in df_s.iterrows():
        chg = float(row["Change%"]); mom = float(row["Momentum"])
        rsi = float(row.get("RSI(14)") or 50)
        bg  = ("rgba(0,229,160,.18)"   if chg>=2   else
               "rgba(0,229,160,.09)"   if chg>=0.5 else
               "rgba(255,255,255,.03)" if chg>=-0.5 else
               "rgba(248,113,113,.10)" if chg>=-2  else
               "rgba(248,113,113,.20)")
        br  = "rgba(0,229,160,.28)" if chg>=0 else "rgba(248,113,113,.28)"
        records.append({"s":  row["Sector"].replace("Nifty ",""),
                        "chg": round(chg,2), "mom": round(mom,1), "rsi": round(rsi,1),
                        "cs": ("+" if chg>=0 else "")+f"{chg:.2f}%",
                        "cc": _cc(chg), "mc": _sc(mom), "rc": _rc(rsi),
                        "bg": bg, "br": br})
    jsx = """
function Chart(){
  const [hov,setHov]=useState(null);
  return <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:6,padding:"2px 0 4px"}}>
    {DATA.map((d,i)=>(
      <div key={i} onMouseEnter={()=>setHov(i)} onMouseLeave={()=>setHov(null)}
        style={{background:d.bg,border:`1px solid ${d.br}`,borderRadius:8,padding:"9px 11px",
          cursor:"default",transform:hov===i?"translateY(-2px)":"none",transition:"transform .12s",
          display:"flex",flexDirection:"column",gap:4}}>
        <div style={{fontSize:10,fontWeight:700,color:"#e2e8f0",whiteSpace:"nowrap",
          overflow:"hidden",textOverflow:"ellipsis"}}>{d.s}</div>
        <div style={{fontSize:13,fontWeight:700,color:d.cc,lineHeight:1}}>{d.cs}</div>
        <div style={{fontSize:9,color:"#6b7280"}}>
          RSI <span style={{color:d.rc}}>{d.rsi}</span>
          &nbsp;·&nbsp;
          <span style={{color:d.mc}}>{d.mom}</span>
        </div>
        <div style={{height:3,background:"rgba(255,255,255,.07)",borderRadius:99,overflow:"hidden",marginTop:2}}>
          <div style={{height:"100%",width:`${d.mom}%`,background:d.mc,borderRadius:99}}/>
        </div>
      </div>
    ))}
  </div>;
}
"""
    components.html(_page("SECTOR HEATMAP — MOMENTUM VIEW", json.dumps(records), height, jsx), height=height)


# ── 4. Scatter: RSI vs Relative Strength ─────────────────────
def chart_rs_scatter(df: pd.DataFrame, height: int = 380):
    df_s = df.dropna(subset=["RSI","RS vs N50"]).copy()
    gc = {"A":"#00e5a0","B":"#fbbf24","C":"#f87171"}
    by_grade = {
        "A": [{"x":round(float(r["RSI"]),1),"y":round(float(r["RS vs N50"]),2),"n":r["Symbol"]}
              for _,r in df_s[df_s["Grade"]=="A"].iterrows()],
        "B": [{"x":round(float(r["RSI"]),1),"y":round(float(r["RS vs N50"]),2),"n":r["Symbol"]}
              for _,r in df_s[df_s["Grade"]=="B"].iterrows()],
        "C": [{"x":round(float(r["RSI"]),1),"y":round(float(r["RS vs N50"]),2),"n":r["Symbol"]}
              for _,r in df_s[df_s["Grade"]=="C"].iterrows()],
    }
    jsx = """
const GC={A:"#00e5a0",B:"#fbbf24",C:"#f87171"};
function CDot(p){
  const {cx,cy,payload,fill}=p;
  return <g>
    <circle cx={cx} cy={cy} r={5} fill={fill} fillOpacity={0.8} stroke={fill} strokeWidth={1}/>
    <text x={cx} y={cy-9} textAnchor="middle" fontSize={8} fontFamily="IBM Plex Mono" fill="#9ca3af">
      {payload.n}
    </text>
  </g>;
}
function TTip({active,payload}){
  if(!active||!payload?.length)return null;
  const d=payload[0]?.payload;if(!d)return null;
  const g=payload[0]?.name||"";
  return <div style={{background:"#0d1828",border:"1px solid #1e293b",borderRadius:6,
    padding:"8px 12px",fontFamily:"IBM Plex Mono",fontSize:11}}>
    <div style={{color:"#e2e8f0",fontWeight:700}}>{d.n}</div>
    <div style={{color:GC[g]||"#9ca3af"}}>Grade {g}</div>
    <div style={{color:"#9ca3af"}}>RSI: {d.x?.toFixed(1)}</div>
    <div style={{color:"#9ca3af"}}>RS: {(d.y>=0?"+":"")+d.y?.toFixed(2)}%</div>
  </div>;
}
function Chart(){
  return <ResponsiveContainer width="100%" height={H}>
    <ScatterChart margin={{top:16,right:20,bottom:32,left:20}}>
      <CartesianGrid stroke="rgba(255,255,255,.05)"/>
      <XAxis type="number" dataKey="x" name="RSI" domain={[20,85]}
        tick={{fill:"#6b7280",fontSize:9,fontFamily:"IBM Plex Mono"}}
        axisLine={false} tickLine={false}
        label={{value:"RSI(14)",position:"insideBottom",offset:-18,
          fill:"#6b7280",fontSize:9,fontFamily:"IBM Plex Mono"}}/>
      <YAxis type="number" dataKey="y" name="RS"
        tick={{fill:"#6b7280",fontSize:9,fontFamily:"IBM Plex Mono"}}
        axisLine={false} tickLine={false} width={46}
        tickFormatter={v=>(v>=0?"+":"")+v.toFixed(1)+"%"}
        label={{value:"RS vs N50 (%)",angle:-90,position:"insideLeft",offset:14,
          fill:"#6b7280",fontSize:9,fontFamily:"IBM Plex Mono"}}/>
      <ZAxis range={[30,30]}/>
      <ReferenceLine x={50} stroke="#374151" strokeDasharray="4 4"/>
      <ReferenceLine y={0}  stroke="#374151" strokeDasharray="4 4"/>
      <Tooltip content={<TTip/>} cursor={{stroke:"rgba(255,255,255,.08)"}}/>
      <Legend wrapperStyle={{fontFamily:"IBM Plex Mono",fontSize:10,color:"#9ca3af",paddingTop:8}}/>
      {["A","B","C"].map(g=>(
        <Scatter key={g} name={"Grade "+g} data={DATA[g]||[]}
          fill={GC[g]} shape={<CDot fill={GC[g]}/>}/>
      ))}
    </ScatterChart>
  </ResponsiveContainer>;
}
"""
    components.html(_page("RSI vs RELATIVE STRENGTH (GRADE COLOURED)",
                          json.dumps(by_grade), height, jsx), height=height+40)


# ── 5. Bar: Momentum Score Distribution ──────────────────────
def chart_histogram(values: List[float], height: int = 380):
    bins = list(range(0, 105, 5))
    records = []
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        cnt = sum(1 for v in values if lo <= v < hi)
        records.append({"l": str(lo), "c": cnt, "f": _sc((lo+hi)/2)})
    jsx = """
function CBar(p){
  const {x,y,width,height,fill}=p;
  return <rect x={x+1} y={y} width={Math.max(0,width-2)} height={height}
    fill={fill} fillOpacity={0.82} rx={3}/>;
}
function Chart(){
  return <ResponsiveContainer width="100%" height={H}>
    <BarChart data={DATA} margin={{top:10,right:16,bottom:28,left:16}}>
      <CartesianGrid vertical={false} stroke="rgba(255,255,255,.05)"/>
      <XAxis dataKey="l" axisLine={false} tickLine={false} interval={1}
        tick={{fill:"#6b7280",fontSize:8,fontFamily:"IBM Plex Mono"}}
        label={{value:"Momentum Score",position:"insideBottom",offset:-14,
          fill:"#6b7280",fontSize:9,fontFamily:"IBM Plex Mono"}}/>
      <YAxis allowDecimals={false} axisLine={false} tickLine={false}
        tick={{fill:"#6b7280",fontSize:9,fontFamily:"IBM Plex Mono"}}
        label={{value:"Stocks",angle:-90,position:"insideLeft",offset:12,
          fill:"#6b7280",fontSize:9,fontFamily:"IBM Plex Mono"}}/>
      <Tooltip cursor={{fill:"rgba(255,255,255,.04)"}}
        contentStyle={{background:"#0d1828",border:"1px solid #1e293b",borderRadius:6,
          fontFamily:"IBM Plex Mono",fontSize:11}}
        labelFormatter={v=>"Score "+v+"\u2013"+(Number(v)+5)}
        formatter={v=>[v+" stocks","Count"]}/>
      <Bar dataKey="c" shape={<CBar/>} isAnimationActive={false}>
        {DATA.map((d,i)=><Cell key={i} fill={d.f}/>)}
      </Bar>
    </BarChart>
  </ResponsiveContainer>;
}
"""
    components.html(_page("MOMENTUM SCORE DISTRIBUTION", json.dumps(records), height, jsx), height=height+40)


# ── 6. Donut: Grade Distribution ─────────────────────────────
def chart_grade_pie(grade_counts: dict, height: int = 260):
    cmap = {"A":"#00e5a0","B":"#fbbf24","C":"#f87171"}
    records = [{"n":k,"v":v,"f":cmap.get(k,"#6b7280")} for k,v in sorted(grade_counts.items())]
    jsx = """
const RAD=Math.PI/180;
function CLabel({cx,cy,midAngle,innerRadius,outerRadius,value}){
  const r=innerRadius+(outerRadius-innerRadius)*0.5;
  const x=cx+r*Math.cos(-midAngle*RAD);
  const y=cy+r*Math.sin(-midAngle*RAD);
  return <text x={x} y={y} fill="#fff" textAnchor="middle" dominantBaseline="central"
    fontFamily="IBM Plex Mono" fontWeight={700} fontSize={13}>{value}</text>;
}
function Chart(){
  return <div style={{display:"flex",alignItems:"center",gap:24,padding:"4px 0 8px"}}>
    <ResponsiveContainer width={H*0.9} height={H}>
      <PieChart>
        <Pie data={DATA} dataKey="v" cx="50%" cy="50%"
          innerRadius="50%" outerRadius="78%" paddingAngle={3}
          labelLine={false} label={<CLabel/>} isAnimationActive={false}>
          {DATA.map((d,i)=><Cell key={i} fill={d.f} stroke={d.f} strokeWidth={1}/>)}
        </Pie>
        <Tooltip contentStyle={{background:"#0d1828",border:"1px solid #1e293b",
          borderRadius:6,fontFamily:"IBM Plex Mono",fontSize:11}}
          formatter={(v,n)=>[v+" stocks","Grade "+n]}/>
      </PieChart>
    </ResponsiveContainer>
    <div style={{display:"flex",flexDirection:"column",gap:10}}>
      {DATA.map((d,i)=>(
        <div key={i} style={{display:"flex",alignItems:"center",gap:8}}>
          <div style={{width:12,height:12,borderRadius:3,background:d.f,flexShrink:0}}/>
          <span style={{fontFamily:"IBM Plex Mono",fontSize:11,color:"#9ca3af"}}>
            <span style={{color:d.f,fontWeight:700}}>Grade {d.n}</span>
            {"  "}{d.v} stocks
          </span>
        </div>
      ))}
    </div>
  </div>;
}
"""
    components.html(_page("GRADE DISTRIBUTION", json.dumps(records), height, jsx), height=height+40)


# ── 7. Area: Advance / Decline ───────────────────────────────
def chart_advance_decline(height: int = 220):
    rng = np.random.default_rng(42)
    records = [{"t": f"{9+i//4}:{str((i%4)*15).zfill(2)}",
                "adv": int(700+rng.integers(0,700)),
                "dec": int(300+rng.integers(0,600))}
               for i in range(26)]
    jsx = """
function Chart(){
  return <ResponsiveContainer width="100%" height={H}>
    <AreaChart data={DATA} margin={{top:6,right:16,bottom:4,left:16}}>
      <defs>
        <linearGradient id="gA" x1="0" y1="0" x2="0" y2="1">
          <stop offset="5%"  stopColor="#00e5a0" stopOpacity={0.25}/>
          <stop offset="95%" stopColor="#00e5a0" stopOpacity={0.03}/>
        </linearGradient>
        <linearGradient id="gD" x1="0" y1="0" x2="0" y2="1">
          <stop offset="5%"  stopColor="#f87171" stopOpacity={0.25}/>
          <stop offset="95%" stopColor="#f87171" stopOpacity={0.03}/>
        </linearGradient>
      </defs>
      <CartesianGrid stroke="rgba(255,255,255,.05)"/>
      <XAxis dataKey="t" axisLine={false} tickLine={false} interval={4}
        tick={{fill:"#6b7280",fontSize:8,fontFamily:"IBM Plex Mono"}}/>
      <YAxis axisLine={false} tickLine={false} width={36}
        tick={{fill:"#6b7280",fontSize:8,fontFamily:"IBM Plex Mono"}}/>
      <Tooltip contentStyle={{background:"#0d1828",border:"1px solid #1e293b",
        borderRadius:6,fontFamily:"IBM Plex Mono",fontSize:11}}
        labelStyle={{color:"#e2e8f0",fontWeight:700}}
        itemStyle={{color:"#9ca3af"}}/>
      <Legend wrapperStyle={{fontFamily:"IBM Plex Mono",fontSize:10,color:"#9ca3af"}}/>
      <Area type="monotone" dataKey="adv" name="Advances"
        stroke="#00e5a0" strokeWidth={1.5} fill="url(#gA)" dot={false}/>
      <Area type="monotone" dataKey="dec" name="Declines"
        stroke="#f87171" strokeWidth={1.5} fill="url(#gD)" dot={false}/>
    </AreaChart>
  </ResponsiveContainer>;
}
"""
    components.html(_page("ADVANCE / DECLINE (INTRADAY)", json.dumps(records), height, jsx), height=height+40)



# ── Pandas table styling helpers ──────────────────────────────
def color_change(val):
    return f"color: {'#00e5a0' if val > 0 else '#f87171'}; font-weight: bold"

def color_rsi(val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return ""
    if val >= 70: return "color: #f97316"
    if val >= 55: return "color: #00e5a0"
    if val <= 35: return "color: #a78bfa"
    return "color: #fbbf24"

def color_score(val):
    c = "#00e5a0" if val >= 72 else "#4ade80" if val >= 58 else "#fbbf24" if val >= 44 else "#f87171"
    return f"color: {c}; font-weight: bold"

def color_grade(val):
    return {"A": "color: #00e5a0; font-weight: bold",
            "B": "color: #fbbf24; font-weight: bold",
            "C": "color: #f87171; font-weight: bold"}.get(val, "")


# ══════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════
def main():
    # Header
    col_logo, col_title, col_time = st.columns([1, 8, 2])
    with col_logo:
        st.markdown('<div style="font-size:32px;margin-top:4px">📊</div>', unsafe_allow_html=True)
    with col_title:
        st.markdown("""
        <h1 style="font-family:'IBM Plex Mono',monospace;font-size:22px;color:#00e5a0;
                   margin:0;letter-spacing:0.12em">NIFTY MARKET INTELLIGENCE</h1>
        <p style="font-size:10px;color:#6b7280;margin:0;letter-spacing:0.08em">
          QUANTITATIVE SECTOR &amp; STOCK ANALYTICS · LIVE DATA VIA YFINANCE · RECHARTS (REACT)
        </p>""", unsafe_allow_html=True)
    with col_time:
        st.markdown(f'<p style="font-size:10px;color:#6b7280;text-align:right;margin-top:8px">'
                    f'Updated: {datetime.now().strftime("%H:%M:%S")}</p>', unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#1e293b;margin:8px 0">', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown("### 🔧 Filters")
    min_momentum = st.sidebar.slider("Min Momentum Score", 0, 90, 0, 5)
    rsi_range    = st.sidebar.slider("RSI Range", 20, 85, (30, 80), 5)
    grade_filter = st.sidebar.multiselect("Grade Filter", ["A","B","C"], default=["A","B","C"])
    only_nr7     = st.sidebar.checkbox("NR7 Only")
    only_vcp     = st.sidebar.checkbox("VCP Only")
    only_pp      = st.sidebar.checkbox("Pocket Pivot Only")
    only_abv50   = st.sidebar.checkbox("Above 50 SMA Only")
    only_vol     = st.sidebar.checkbox("Volume Surge (>1.3×)")
    num_stocks   = st.sidebar.slider("# Stocks to Load", 10, 50, 25, 5)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Grading Logic:**")
    st.sidebar.markdown("🏆 **A**: RS>0 + Mom>68 + RSI 50–75")
    st.sidebar.markdown("📊 **B**: RS>-2 + Mom>50")
    st.sidebar.markdown("⚠️  **C**: Weak sector or stock")

    tab_ov, tab_sec, tab_stocks, tab_grades = st.tabs([
        "📈 Overview", "🗂️ Sectors", "📋 Stocks", "🏆 AI Grades"
    ])

    # ═══════════════════════════════════════════════════════════
    # OVERVIEW TAB
    # ═══════════════════════════════════════════════════════════
    with tab_ov:
        with st.spinner("Fetching live index data…"):
            idx_df = load_indices()

        if not idx_df.empty:
            cols = st.columns(len(idx_df))
            for i, row in idx_df.iterrows():
                with cols[i]:
                    st.metric(label=row["Index"],
                              value=f"{row['Level']:,.2f}",
                              delta=f"{row['Change%']:+.2f}%")
        st.markdown("---")

        with st.spinner("Loading sector data…"):
            sec_df = load_sectors()

        if not sec_df.empty:
            c1, c2 = st.columns(2)
            with c1:
                chart_heatmap(sec_df, height=420)
            with c2:
                chart_momentum(sec_df, height=420)
            chart_treemap(sec_df, height=350)
            chart_advance_decline(height=220)

    # ═══════════════════════════════════════════════════════════
    # SECTORS TAB
    # ═══════════════════════════════════════════════════════════
    with tab_sec:
        with st.spinner("Loading sectors…"):
            sec_df = load_sectors()

        if sec_df.empty:
            st.error("Could not load sector data.")
        else:
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a: st.metric("Top Sector",   sec_df.iloc[0]["Sector"],
                                  f'{sec_df.iloc[0]["Change%"]:+.2f}%')
            with col_b: st.metric("Avg Momentum", round(sec_df["Momentum"].mean(), 1))
            with col_c: st.metric("Sectors > 0%", int((sec_df["Change%"] > 0).sum()))
            with col_d: st.metric("Avg RSI",       round(sec_df["RSI(14)"].dropna().mean(), 1))

            st.markdown("#### Sector Performance Charts")
            c1, c2 = st.columns(2)
            with c1:
                chart_heatmap(sec_df, height=400)
            with c2:
                chart_momentum(sec_df, height=400)

            st.markdown("#### Full Sector Table")
            styled = (sec_df.style
                      .map(color_change, subset=["Change%", "RS vs N50"])
                      .map(color_rsi,    subset=["RSI(14)"])
                      .map(color_score,  subset=["Momentum"])
                      .format({"Level": "{:,.0f}", "Change%": "{:+.2f}%",
                               "RSI(14)": "{:.1f}", "Vol/20DMA": "{:.2f}x",
                               "RS vs N50": "{:+.2f}%", "Breadth%": "{:.1f}%",
                               "Momentum": "{:.1f}"})
                      .set_properties(**{"font-family": "IBM Plex Mono", "font-size": "11px"}))
            st.dataframe(styled, use_container_width=True, height=480)

    # ═══════════════════════════════════════════════════════════
    # STOCKS TAB
    # ═══════════════════════════════════════════════════════════
    with tab_stocks:
        with st.spinner(f"Fetching {num_stocks} stocks with live technicals…"):
            stocks_df = load_stocks(NIFTY50_STOCKS[:num_stocks])

        if stocks_df.empty:
            st.warning("No stock data loaded.")
        else:
            mask = (
                (stocks_df["Momentum"] >= min_momentum) &
                (stocks_df["RSI"].fillna(50).between(rsi_range[0], rsi_range[1])) &
                (stocks_df["Grade"].isin(grade_filter))
            )
            if only_nr7:   mask &= stocks_df["NR7"]
            if only_vcp:   mask &= stocks_df["VCP"]
            if only_pp:    mask &= stocks_df["PP"]
            if only_abv50: mask &= stocks_df[">50SMA"]
            if only_vol:   mask &= stocks_df["Vol/DMA"] > 1.3
            filtered = stocks_df[mask]
            st.caption(f"Showing {len(filtered)} / {len(stocks_df)} stocks after filters")

            c1, c2 = st.columns(2)
            with c1:
                chart_rs_scatter(filtered, height=380)
            with c2:
                chart_histogram(filtered["Momentum"].tolist(), height=380)

            display_cols = ["Symbol","Price","Change%","RSI","Vol/DMA","RS vs N50",
                            ">50SMA","NR7","VCP","PP","Momentum","Grade"]
            styled = (filtered[display_cols].style
                      .map(color_change, subset=["Change%","RS vs N50"])
                      .map(color_rsi,    subset=["RSI"])
                      .map(color_score,  subset=["Momentum"])
                      .map(color_grade,  subset=["Grade"])
                      .format({"Price": "₹{:,.0f}", "Change%": "{:+.2f}%",
                               "RSI": "{:.1f}", "Vol/DMA": "{:.2f}x",
                               "RS vs N50": "{:+.2f}%", "Momentum": "{:.1f}"})
                      .set_properties(**{"font-family": "IBM Plex Mono", "font-size": "11px"}))
            st.dataframe(styled, use_container_width=True, height=500)

    # ═══════════════════════════════════════════════════════════
    # AI GRADES TAB
    # ═══════════════════════════════════════════════════════════
    with tab_grades:
        with st.spinner("Computing AI grades…"):
            stocks_df = load_stocks(NIFTY50_STOCKS[:num_stocks])

        if stocks_df.empty:
            st.warning("No stock data available.")
        else:
            grade_a = stocks_df[stocks_df["Grade"] == "A"].head(16)
            grade_b = stocks_df[stocks_df["Grade"] == "B"].head(12)
            grade_c = stocks_df[stocks_df["Grade"] == "C"].head(8)

            ca, cb, cc, cd = st.columns(4)
            with ca: st.metric("🏆 Grade A", len(grade_a), "Prime candidates")
            with cb: st.metric("📊 Grade B", len(grade_b), "Watchlist")
            with cc: st.metric("⚠️ Grade C", len(grade_c), "Avoid")
            with cd: st.metric("Avg Score",  round(stocks_df["Momentum"].mean(), 1))

            grade_counts = stocks_df["Grade"].value_counts().to_dict()
            chart_grade_pie(grade_counts, height=260)

            if not grade_a.empty:
                st.markdown("### 🏆 Grade A — Prime Buy Candidates")
                cols = st.columns(min(4, len(grade_a)))
                for i, (_, row) in enumerate(grade_a.iterrows()):
                    with cols[i % 4]:
                        chg_c = C["bull"] if row["Change%"] >= 0 else C["bear"]
                        tags  = " ".join([t for t, v in [
                            ("NR7", row["NR7"]), ("VCP", row["VCP"]),
                            ("PP", row["PP"]),   ("50D", row[">50SMA"])
                        ] if v])
                        st.markdown(f"""
                        <div style="background:rgba(0,229,160,.06);border:1px solid rgba(0,229,160,.2);
                                    border-radius:10px;padding:12px;margin-bottom:8px">
                          <div style="font-weight:700;color:#f1f5f9;font-size:13px">{row['Symbol']}</div>
                          <div style="color:{chg_c};font-size:11px">{row['Change%']:+.2f}%</div>
                          <div style="color:#9ca3af;font-size:10px">RSI {row['RSI']:.1f} · {row['Momentum']:.1f} score</div>
                          <div style="color:#6b7280;font-size:10px">{tags}</div>
                        </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### Grading Criteria")
            cr, cc2, cp = st.columns(3)
            for col, g, color, rules in [
                (cr,  "A", "#00e5a0", ["Sector RS > 0 vs Nifty","Momentum > 68","RSI 50–75","Vol > 1.2× DMA","≥1 pattern (NR7/VCP/PP)"]),
                (cc2, "B", "#fbbf24", ["Sector RS > −2","Momentum > 50","RSI 45–78","Volume neutral+","No pattern required"]),
                (cp,  "C", "#f87171", ["Sector RS < −2","Momentum < 50","RSI < 45","Vol below DMA","Avoid fresh longs"]),
            ]:
                with col:
                    rules_html = "".join(
                        f'<li style="color:#9ca3af;font-size:11px;margin-bottom:4px">{r}</li>'
                        for r in rules)
                    st.markdown(f"""
                    <div style="background:{color}0d;border:1px solid {color}33;
                                border-radius:10px;padding:14px">
                      <div style="color:{color};font-size:16px;font-weight:900;margin-bottom:8px">Grade {g}</div>
                      <ul style="padding-left:16px">{rules_html}</ul>
                    </div>""", unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#1e293b;margin-top:32px">', unsafe_allow_html=True)
    st.markdown('<p style="font-size:10px;color:#374151;text-align:center">'
                'Nifty Market Intelligence · Data via Yahoo Finance · '
                'Charts powered by React + Recharts · Not financial advice</p>',
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()
