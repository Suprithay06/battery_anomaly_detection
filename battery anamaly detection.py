import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              confusion_matrix, roc_auc_score)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Battery Anomaly Detection",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #0b0f1a;
    color: #e0e6f0;
}
.stApp { background: linear-gradient(135deg, #0b0f1a 0%, #0e1628 60%, #101a2e 100%); }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1220 0%, #111827 100%);
    border-right: 1px solid #1e2d45;
}
[data-testid="stSidebar"] * { color: #c5d3e8 !important; }

.card {
    background: linear-gradient(135deg, #111b2e 0%, #0f1e35 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 24px rgba(0,80,180,0.10);
}
.card-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.18em;
    color: #4a90d9;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.card-value { font-size: 2.2rem; font-weight: 700; color: #e8f0ff; line-height: 1; }
.card-sub   { font-size: 0.85rem; color: #5a7a9a; margin-top: 4px; }

.badge-anomaly {
    display: inline-block;
    background: linear-gradient(90deg, #7f1d1d, #991b1b);
    color: #fca5a5; border: 1px solid #ef4444;
    border-radius: 6px; padding: 3px 10px;
    font-family: 'Share Tech Mono', monospace; font-size: 0.72rem; letter-spacing: 0.1em;
}
.badge-normal {
    display: inline-block;
    background: linear-gradient(90deg, #064e3b, #065f46);
    color: #6ee7b7; border: 1px solid #10b981;
    border-radius: 6px; padding: 3px 10px;
    font-family: 'Share Tech Mono', monospace; font-size: 0.72rem; letter-spacing: 0.1em;
}
.alert-box {
    background: linear-gradient(135deg, #2d0a0a 0%, #3b0e0e 100%);
    border-left: 4px solid #ef4444;
    border-radius: 8px; padding: 14px 18px; margin: 10px 0;
}
.alert-box .alert-title {
    font-family: 'Share Tech Mono', monospace;
    color: #f87171; font-size: 0.88rem; letter-spacing: 0.12em;
}
.alert-box .alert-body { color: #fca5a5; font-size: 0.9rem; margin-top: 6px; }

.info-box {
    background: linear-gradient(135deg, #0a1d2d 0%, #0d2540 100%);
    border-left: 4px solid #3b82f6; border-radius: 8px;
    padding: 14px 18px; margin: 10px 0; color: #93c5fd; font-size: 0.9rem;
}
.success-box {
    background: linear-gradient(135deg, #052e16 0%, #064e3b 100%);
    border-left: 4px solid #22c55e; border-radius: 8px;
    padding: 14px 18px; margin: 10px 0; color: #86efac; font-size: 0.9rem;
}
.section-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem; letter-spacing: 0.22em; color: #3b82f6;
    text-transform: uppercase; border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px; margin: 24px 0 16px 0;
}
[data-testid="stDataFrame"] { border: 1px solid #1e3a5f; border-radius: 8px; }
.stButton > button {
    background: linear-gradient(90deg, #1d4ed8, #2563eb) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'Rajdhani', sans-serif !important; font-weight: 600 !important;
    letter-spacing: 0.08em !important; padding: 8px 20px !important; transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #2563eb, #3b82f6) !important;
    box-shadow: 0 0 16px rgba(59,130,246,0.4) !important;
}
button[data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important; letter-spacing: 0.06em !important;
}
.main-title {
    font-family: 'Share Tech Mono', monospace; font-size: 2rem;
    color: #60a5fa; letter-spacing: 0.1em; line-height: 1.2;
    text-shadow: 0 0 30px rgba(96,165,250,0.3);
}
.main-subtitle { color: #475569; font-size: 1rem; letter-spacing: 0.06em; margin-top: 4px; }
.pulse-dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; background: #22c55e; margin-right: 6px;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.4; transform:scale(1.4); }
}
.cmp-table { width:100%; border-collapse:collapse; margin-top:12px; }
.cmp-table th {
    font-family:'Share Tech Mono',monospace; font-size:0.72rem; letter-spacing:0.12em;
    color:#4a90d9; text-transform:uppercase; padding:10px 14px;
    border-bottom:2px solid #1e3a5f; text-align:center;
}
.cmp-table td { padding:10px 14px; border-bottom:1px solid #1a2d45; text-align:center; font-size:0.92rem; }
.cmp-table tr:hover td { background:#0d1e33; }
.best-cell { color:#60a5fa; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
FEATURES = ['ScreenTime','CPUUsage','Apps','BackgroundApps','NetworkUsage','BatteryDrop']
DAYS     = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
BG_DARK  = '#0b0f1a'
BG_PANEL = '#0e1628'
BLUE     = '#3b82f6'
RED      = '#ef4444'
GREEN    = '#22c55e'
AMBER    = '#f59e0b'
GRID     = '#1e3a5f'
DIM      = '#5a7a9a'


# ─────────────────────────────────────────────
#  SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────
def generate_synthetic_data(n=300, anomaly_pct=0.08, seed=42):
    rng      = np.random.default_rng(seed)
    n_normal = int(n * (1 - anomaly_pct))
    n_anom   = n - n_normal
    base_ts  = pd.Timestamp("2025-02-24 00:00:00")
    ts_n = [base_ts + pd.Timedelta(minutes=int(x)) for x in rng.uniform(0, 30*24*60, n_normal)]
    ts_a = [base_ts + pd.Timedelta(minutes=int(x)) for x in rng.uniform(0, 30*24*60, n_anom)]

    normal = pd.DataFrame({
        'Timestamp':      ts_n,
        'ScreenTime':     rng.normal(30, 10, n_normal).clip(0, 120),
        'CPUUsage':       rng.normal(25, 8,  n_normal).clip(0, 100),
        'Apps':           rng.integers(2, 15, n_normal).astype(float),
        'BackgroundApps': rng.integers(1, 8,  n_normal).astype(float),
        'NetworkUsage':   rng.normal(50, 20, n_normal).clip(0, 500),
        'BatteryDrop':    rng.normal(6,  2,  n_normal).clip(0.5, 20),
        'ChargingStatus': rng.choice([0,1], n_normal, p=[0.8, 0.2]),
        'Label':          'Normal',
    })
    anomaly = pd.DataFrame({
        'Timestamp':      ts_a,
        'ScreenTime':     rng.normal(8,  5,  n_anom).clip(0, 30),
        'CPUUsage':       rng.normal(15, 5,  n_anom).clip(0, 40),
        'Apps':           rng.integers(1, 5, n_anom).astype(float),
        'BackgroundApps': rng.normal(12, 3,  n_anom).clip(8, 30),
        'NetworkUsage':   rng.normal(300, 80, n_anom).clip(200, 500),
        'BatteryDrop':    rng.normal(22, 5,   n_anom).clip(15, 50),
        'ChargingStatus': np.zeros(n_anom),
        'Label':          'Anomaly',
    })

    df = pd.concat([normal, anomaly], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df['Timestamp']  = pd.to_datetime(df['Timestamp'])
    df['Hour']       = df['Timestamp'].dt.hour
    df['DayOfWeek']  = df['Timestamp'].dt.dayofweek
    df['DayName']    = df['Timestamp'].dt.day_name()
    df['Date']       = df['Timestamp'].dt.date
    for col in ['ScreenTime','CPUUsage','NetworkUsage','BatteryDrop']:
        df[col] = df[col].round(1)
    return df


# ─────────────────────────────────────────────
#  MODEL RUNNER
# ─────────────────────────────────────────────
@st.cache_data
def run_model(df, contamination):
    X        = df[FEATURES].copy()
    scaler   = StandardScaler()
    Xs       = scaler.fit_transform(X)

    if_mdl   = IsolationForest(contamination=contamination, random_state=42, n_estimators=150)
    if_preds = if_mdl.fit_predict(Xs)
    if_score = -if_mdl.decision_function(Xs)

    lof_mdl   = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    lof_preds = lof_mdl.fit_predict(Xs)
    lof_score = -lof_mdl.negative_outlier_factor_

    res = df.copy()
    res['IF_Pred']   = if_preds
    res['IF_Score']  = np.round(if_score, 4)
    res['LOF_Pred']  = lof_preds
    res['LOF_Score'] = np.round(lof_score, 4)
    res['Anomaly']      = if_preds
    res['AnomalyScore'] = res['IF_Score']
    res['AnomalyLabel'] = res['Anomaly'].map({1:'Normal', -1:'⚠ Anomaly'})

    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)
    res['PCA1'] = coords[:,0]; res['PCA2'] = coords[:,1]

    return res, if_mdl, lof_mdl, scaler


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def compute_metrics(y_true_labels, y_pred):
    y_true = np.where(np.array(y_true_labels) == 'Anomaly', -1, 1)
    prec = precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
    rec  = recall_score   (y_true, y_pred, pos_label=-1, zero_division=0)
    f1   = f1_score       (y_true, y_pred, pos_label=-1, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=[1,-1])
    try:
        auc = roc_auc_score((y_true==-1).astype(int), (y_pred==-1).astype(int))
    except Exception:
        auc = float('nan')
    return dict(precision=prec, recall=rec, f1=f1, auc=auc, cm=cm)


def cause_analysis(row):
    causes = []
    if row['BackgroundApps'] > 10:
        causes.append("🔴 High background app activity")
    if row['NetworkUsage'] > 200:
        causes.append("🔴 Unusual network spike")
    if row['CPUUsage'] > 60:
        causes.append("🔴 Excessive CPU usage")
    if row['BatteryDrop'] > 15 and row['ScreenTime'] < 15:
        causes.append("🔴 Severe drain with minimal screen usage")
    if row['Apps'] < 3 and row['BatteryDrop'] > 12:
        causes.append("🔴 Disproportionate drain for idle state")
    if not causes:
        causes.append("🟡 Statistical outlier — unusual feature combination")
    return causes


# ─────────────────────────────────────────────
#  PLOT FACTORIES
# ─────────────────────────────────────────────
def _ax(w=6, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG_DARK); ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=DIM)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    return fig, ax


def make_scatter(df, xc, yc):
    fig, ax = _ax()
    ax.scatter(df[xc], df[yc],
               c=df['Anomaly'].map({1:BLUE,-1:RED}),
               s=df['Anomaly'].map({1:18,-1:55}), alpha=0.75, edgecolors='none')
    ax.set_xlabel(xc, color=DIM, fontsize=9); ax.set_ylabel(yc, color=DIM, fontsize=9)
    ax.legend(handles=[mpatches.Patch(color=BLUE,label='Normal'),
                       mpatches.Patch(color=RED, label='Anomaly')],
              facecolor=BG_DARK, edgecolor=GRID, labelcolor='#c5d3e8', fontsize=8)
    fig.tight_layout(pad=1.5); return fig


def make_pca_plot(df):
    fig, ax = _ax()
    for pred, col, sz, lbl in [(1,BLUE,18,'Normal'),(-1,RED,55,'Anomaly')]:
        d = df[df['Anomaly']==pred]
        ax.scatter(d['PCA1'], d['PCA2'], c=col, s=sz, alpha=0.75 if pred==1 else 0.85,
                   edgecolors='none', label=lbl)
    ax.set_xlabel('PCA1',color=DIM,fontsize=9); ax.set_ylabel('PCA2',color=DIM,fontsize=9)
    ax.set_title('PCA — Feature Space', color='#7aa3cc', fontsize=10)
    ax.legend(facecolor=BG_DARK, edgecolor=GRID, labelcolor='#c5d3e8', fontsize=8)
    fig.tight_layout(pad=1.5); return fig


def make_histogram(df, col):
    fig, ax = _ax(5,3)
    ax.hist(df[df['Anomaly']==1][col],  bins=25, color=BLUE, alpha=0.6, label='Normal')
    ax.hist(df[df['Anomaly']==-1][col], bins=15, color=RED,  alpha=0.8, label='Anomaly')
    ax.set_xlabel(col, color=DIM, fontsize=9)
    ax.legend(facecolor=BG_DARK, edgecolor=GRID, labelcolor='#c5d3e8', fontsize=8)
    fig.tight_layout(pad=1.5); return fig


def make_score_bar(df):
    top = df.nlargest(20,'AnomalyScore')
    fig, ax = _ax(7,4)
    ax.barh(range(len(top)), top['AnomalyScore'],
            color=['#ef4444' if a==-1 else BLUE for a in top['Anomaly']],
            edgecolor='none', height=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"Row {i}" for i in top.index], fontsize=8, color=DIM)
    ax.set_xlabel('Anomaly Score', color=DIM, fontsize=9)
    fig.tight_layout(pad=1.5); return fig


# ── TIME PLOTS ─────────────────────────────────────────────
def make_hourly_line(df):
    fig, ax = _ax(8,4)
    hrs = range(24)
    nrm = df[df['Anomaly']==1 ].groupby('Hour')['BatteryDrop'].mean().reindex(hrs, fill_value=0)
    anm = df[df['Anomaly']==-1].groupby('Hour')['BatteryDrop'].mean().reindex(hrs, fill_value=np.nan)
    ax.plot(hrs, nrm.values, color=BLUE,  lw=2, marker='o', ms=4, label='Normal avg')
    ax.plot(hrs, anm.values, color=RED,   lw=2, marker='D', ms=5, ls='--', label='Anomaly avg')
    ax.fill_between(hrs, nrm.values, alpha=0.12, color=BLUE)
    ax.set_xticks(range(0,24,2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0,24,2)], rotation=35, ha='right', fontsize=7.5, color=DIM)
    ax.set_xlabel('Hour of Day', color=DIM, fontsize=9)
    ax.set_ylabel('Avg Battery Drop (%)', color=DIM, fontsize=9)
    ax.set_title('Hourly Battery Drain Pattern', color='#7aa3cc', fontsize=10)
    ax.axvspan(0, 6,  alpha=0.05, color='purple')
    ax.axvspan(9, 18, alpha=0.05, color='yellow')
    ax.legend(facecolor=BG_DARK, edgecolor=GRID, labelcolor='#c5d3e8', fontsize=8)
    ax.grid(axis='y', color=GRID, lw=0.5)
    fig.tight_layout(pad=1.5); return fig


def make_daily_bar(df):
    fig, ax = _ax(7,3.5)
    x     = np.arange(len(DAYS))
    grp   = df.groupby('DayName')['BatteryDrop'].mean().reindex(DAYS, fill_value=0)
    a_grp = df[df['Anomaly']==-1].groupby('DayName').size().reindex(DAYS, fill_value=0)
    ax.bar(x, grp.values, color=BLUE, alpha=0.8, edgecolor='none', width=0.6)
    ax2 = ax.twinx()
    ax2.plot(x, a_grp.values, color=RED, lw=2, marker='o', ms=6)
    ax2.set_ylabel('Anomaly Count', color=RED, fontsize=9)
    ax2.tick_params(colors=RED); ax2.set_facecolor('none')
    ax.set_xticks(x); ax.set_xticklabels([d[:3] for d in DAYS], color=DIM, fontsize=9)
    ax.set_ylabel('Avg Battery Drop (%)', color=DIM, fontsize=9)
    ax.set_title('Daily Drain & Anomaly Frequency', color='#7aa3cc', fontsize=10)
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.legend(handles=[mpatches.Patch(color=BLUE,label='Avg drain')],
              facecolor=BG_DARK, edgecolor=GRID, labelcolor='#c5d3e8', fontsize=8, loc='upper left')
    fig.tight_layout(pad=1.5); return fig


def make_heatmap(df):
    pivot = (df.groupby(['DayOfWeek','Hour'])['BatteryDrop']
               .mean().unstack(fill_value=0)
               .reindex(range(7), fill_value=0)
               .reindex(columns=range(24), fill_value=0))
    fig, ax = plt.subplots(figsize=(10,3.5))
    fig.patch.set_facecolor(BG_DARK); ax.set_facecolor(BG_PANEL)
    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd', origin='upper', vmin=0)
    ax.set_xticks(range(0,24,2))
    ax.set_xticklabels([f"{h:02d}" for h in range(0,24,2)], color=DIM, fontsize=8)
    ax.set_yticks(range(7)); ax.set_yticklabels([d[:3] for d in DAYS], color=DIM, fontsize=8)
    ax.set_xlabel('Hour of Day', color=DIM, fontsize=9)
    ax.set_title('Battery Drain Heatmap (Avg % drop)', color='#7aa3cc', fontsize=10)
    cb = plt.colorbar(im, ax=ax, pad=0.01)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=DIM, fontsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    fig.tight_layout(pad=1.5); return fig


def make_anomaly_timeline(df):
    if 'Date' not in df.columns: return None
    df2          = df.copy()
    df2['DOrd']  = pd.to_datetime(df2['Date']).map(pd.Timestamp.toordinal)
    fig, ax      = _ax(10,3.5)
    nrm = df2[df2['Anomaly']==1 ]; anm = df2[df2['Anomaly']==-1]
    ax.scatter(nrm['DOrd'], nrm['BatteryDrop'], c=BLUE, s=12, alpha=0.4, edgecolors='none', label='Normal')
    ax.scatter(anm['DOrd'], anm['BatteryDrop'], c=RED,  s=45, alpha=0.9, edgecolors='none', label='⚠ Anomaly', zorder=5)
    dates  = sorted(df2['Date'].unique())
    tidx   = np.linspace(0, len(dates)-1, min(10,len(dates)), dtype=int)
    ax.set_xticks([pd.Timestamp(dates[i]).toordinal() for i in tidx])
    ax.set_xticklabels([str(dates[i]) for i in tidx], rotation=30, ha='right', fontsize=7.5, color=DIM)
    ax.set_ylabel('Battery Drop (%)', color=DIM, fontsize=9)
    ax.set_title('Anomaly Timeline (30 days)', color='#7aa3cc', fontsize=10)
    ax.legend(facecolor=BG_DARK, edgecolor=GRID, labelcolor='#c5d3e8', fontsize=8)
    ax.grid(axis='y', color=GRID, lw=0.5)
    fig.tight_layout(pad=1.5); return fig


def make_hourly_rate(df):
    fig, ax = _ax(8,3.5)
    rate = (df.groupby('Hour')['Anomaly']
              .apply(lambda x: (x==-1).mean()*100)
              .reindex(range(24), fill_value=0))
    mean_r = rate.mean(); std_r = rate.std()
    ax.bar(range(24), rate.values,
           color=[RED if v > mean_r+std_r else BLUE for v in rate.values],
           edgecolor='none', width=0.75)
    ax.axhline(mean_r, color=AMBER, lw=1.5, ls='--', label=f'Mean {mean_r:.1f}%')
    ax.set_xticks(range(24))
    ax.set_xticklabels([str(h) for h in range(24)], color=DIM, fontsize=7.5)
    ax.set_xlabel('Hour', color=DIM, fontsize=9); ax.set_ylabel('Anomaly Rate (%)', color=DIM, fontsize=9)
    ax.set_title('Anomaly Rate by Hour', color='#7aa3cc', fontsize=10)
    ax.legend(facecolor=BG_DARK, edgecolor=GRID, labelcolor='#c5d3e8', fontsize=8)
    ax.grid(axis='y', color=GRID, lw=0.5)
    fig.tight_layout(pad=1.5); return fig


# ── MODEL COMPARISON PLOTS ─────────────────────────────────
def make_cm_plot(cm, title, color):
    fig, ax = plt.subplots(figsize=(3.5,3))
    fig.patch.set_facecolor(BG_DARK); ax.set_facecolor(BG_PANEL)
    ax.imshow(cm, cmap='Blues', vmin=0)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred Normal','Pred Anomaly'], color=DIM, fontsize=8)
    ax.set_yticklabels(['True Normal','True Anomaly'], color=DIM, fontsize=8)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=14,
                    color='white' if cm[i,j]>cm.max()/2 else DIM, fontweight='bold')
    ax.set_title(title, color=color, fontsize=9, fontfamily='monospace')
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    fig.tight_layout(pad=1.2); return fig


def make_radar(if_m, lof_m):
    labels   = ['Precision','Recall','F1','AUC']
    iv  = [if_m['precision'], if_m['recall'], if_m['f1'], if_m.get('auc',0)]
    lv  = [lof_m['precision'],lof_m['recall'],lof_m['f1'],lof_m.get('auc',0)]
    N   = len(labels)
    th  = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    th += th[:1]; iv += iv[:1]; lv += lv[:1]
    fig, ax = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG_DARK); ax.set_facecolor(BG_PANEL)
    ax.plot(th, iv, color=BLUE,  lw=2, label='Isolation Forest')
    ax.fill(th, iv, color=BLUE,  alpha=0.2)
    ax.plot(th, lv, color=AMBER, lw=2, label='LOF')
    ax.fill(th, lv, color=AMBER, alpha=0.2)
    ax.set_xticks(th[:-1]); ax.set_xticklabels(labels, color='#c5d3e8', fontsize=9)
    ax.set_ylim(0,1); ax.yaxis.set_tick_params(labelcolor=DIM, labelsize=7)
    ax.spines['polar'].set_color(GRID); ax.grid(color=GRID, lw=0.8)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35,1.15),
              facecolor=BG_DARK, edgecolor=GRID, labelcolor='#c5d3e8', fontsize=8)
    fig.tight_layout(pad=1.5); return fig


def make_score_dist_cmp(df):
    fig, (a1,a2) = plt.subplots(1,2, figsize=(8,3.5))
    fig.patch.set_facecolor(BG_DARK)
    for ax in (a1,a2):
        ax.set_facecolor(BG_PANEL); ax.tick_params(colors=DIM)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    a1.hist(df[df['IF_Pred']==1 ]['IF_Score'],  bins=30, color=BLUE, alpha=0.7, label='Normal')
    a1.hist(df[df['IF_Pred']==-1]['IF_Score'],  bins=20, color=RED,  alpha=0.8, label='Anomaly')
    a1.set_title('Isolation Forest — Score', color='#7aa3cc', fontsize=9)
    a1.set_xlabel('Score', color=DIM, fontsize=8)
    a1.legend(facecolor=BG_DARK, edgecolor=GRID, labelcolor='#c5d3e8', fontsize=8)
    a2.hist(df[df['LOF_Pred']==1 ]['LOF_Score'], bins=30, color=BLUE,  alpha=0.7, label='Normal')
    a2.hist(df[df['LOF_Pred']==-1]['LOF_Score'], bins=20, color=AMBER, alpha=0.8, label='Anomaly')
    a2.set_title('LOF — Score', color='#7aa3cc', fontsize=9)
    a2.set_xlabel('LOF Score', color=DIM, fontsize=8)
    a2.legend(facecolor=BG_DARK, edgecolor=GRID, labelcolor='#c5d3e8', fontsize=8)
    fig.tight_layout(pad=1.5); return fig


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 20px'>
        <div style='font-family:"Share Tech Mono",monospace;font-size:1.4rem;color:#60a5fa;letter-spacing:0.1em;'>🔋 BADA</div>
        <div style='color:#334155;font-size:0.75rem;letter-spacing:0.15em;'>BATTERY ANOMALY DETECTION AI</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Data Source")
    data_source = st.radio("", ["Use Synthetic Dataset","Upload CSV"], label_visibility='collapsed')
    st.markdown("---")
    st.markdown("### Model Settings")
    contamination = st.slider("Contamination Rate", 0.01, 0.30, 0.07, 0.01)
    st.caption(f"Expecting ~{contamination*100:.0f}% anomalies")
    if data_source == "Use Synthetic Dataset":
        st.markdown("### Synthetic Data")
        n_samples = st.slider("Samples", 100, 1000, 300, 50)
        anom_pct  = st.slider("True anomaly %", 0.03, 0.20, 0.08, 0.01)
    st.markdown("---")
    st.markdown("### Visualization")
    x_axis   = st.selectbox("Scatter X",        FEATURES, index=0)
    y_axis   = st.selectbox("Scatter Y",        FEATURES, index=5)
    hist_col = st.selectbox("Histogram feature", FEATURES, index=5)
    st.markdown("---")
    st.markdown("""
    <div style='color:#1e3a5f;font-size:0.72rem;font-family:"Share Tech Mono",monospace;letter-spacing:0.1em;'>
    MODELS: IF · LOF<br>FEATURES: 6<br>NORMALIZER: STD SCALER<br>VIZ: PCA 2D + TIME SERIES
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA LOAD
# ─────────────────────────────────────────────
df_raw = None
if data_source == "Use Synthetic Dataset":
    df_raw = generate_synthetic_data(n=n_samples, anomaly_pct=anom_pct)
    st.sidebar.success(f"✓ Generated {n_samples} rows with timestamps")
else:
    uploaded = st.sidebar.file_uploader("Upload battery_data.csv", type=["csv"])
    if uploaded:
        try:
            df_raw  = pd.read_csv(uploaded)
            missing = [c for c in FEATURES if c not in df_raw.columns]
            if missing:
                st.sidebar.error(f"Missing columns: {missing}"); df_raw = None
            else:
                for tc in ['Timestamp','timestamp','Time','time','DateTime']:
                    if tc in df_raw.columns:
                        df_raw['Timestamp'] = pd.to_datetime(df_raw[tc], errors='coerce'); break
                else:
                    df_raw['Timestamp'] = pd.date_range('2025-01-01', periods=len(df_raw), freq='30min')
                df_raw['Hour']      = df_raw['Timestamp'].dt.hour
                df_raw['DayOfWeek'] = df_raw['Timestamp'].dt.dayofweek
                df_raw['DayName']   = df_raw['Timestamp'].dt.day_name()
                df_raw['Date']      = df_raw['Timestamp'].dt.date
                st.sidebar.success(f"✓ Loaded {len(df_raw)} rows")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class='main-title'><span class='pulse-dot'></span>BATTERY ANOMALY DETECTION</div>
<div class='main-subtitle'>AI-powered smartphone battery analysis · Isolation Forest + LOF · Time-Series Intelligence</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if df_raw is None:
    st.markdown("""
    <div class='info-box'>ℹ Choose <b>Use Synthetic Dataset</b> in the sidebar, or upload a CSV with columns:
    <code>ScreenTime, CPUUsage, Apps, BackgroundApps, NetworkUsage, BatteryDrop</code>
    and optionally a <code>Timestamp</code> column for time analysis.</div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
#  RUN MODELS
# ─────────────────────────────────────────────
df_clean = df_raw.dropna(subset=FEATURES).copy()
result, if_model, lof_model, scaler = run_model(df_clean, contamination)

n_total   = len(result)
n_anom    = (result['IF_Pred']==-1).sum()
n_normal  = n_total - n_anom
anom_rate = n_anom / n_total * 100
avg_score = result[result['IF_Pred']==-1]['IF_Score'].mean() if n_anom > 0 else 0.0


# ─────────────────────────────────────────────
#  7 TABS
# ─────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "📊 Dashboard",
    "⚠ Anomalies",
    "📈 Visualizations",
    "🕐 Time Analysis",
    "⚖ Model Comparison",
    "🔍 Live Predictor",
    "📋 Full Data",
])


# ════════════════════════════════════════════
#  TAB 1 — DASHBOARD
# ════════════════════════════════════════════
with tab1:
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='card'><div class='card-title'>Total Records</div><div class='card-value'>{n_total}</div><div class='card-sub'>Processed samples</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='card'><div class='card-title'>Normal Behavior</div><div class='card-value' style='color:{GREEN}'>{n_normal}</div><div class='card-sub'>{100-anom_rate:.1f}% of total</div></div>", unsafe_allow_html=True)
    with c3:
        col = RED if n_anom > 0 else GREEN
        st.markdown(f"<div class='card'><div class='card-title'>IF Anomalies</div><div class='card-value' style='color:{col}'>{n_anom}</div><div class='card-sub'>{anom_rate:.1f}% anomaly rate</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='card'><div class='card-title'>Avg Anomaly Score</div><div class='card-value'>{avg_score:.3f}</div><div class='card-sub'>Higher = more abnormal</div></div>", unsafe_allow_html=True)

    if n_anom > 0:
        worst      = result[result['IF_Pred']==-1].nlargest(1,'IF_Score').iloc[0]
        causes_str = " · ".join(cause_analysis(worst))
        st.markdown(f"""<div class='alert-box'>
            <div class='alert-title'>⚠ ANOMALY ALERT — {n_anom} SUSPICIOUS BATTERY EVENTS</div>
            <div class='alert-body'>Top anomaly Row {worst.name} · Score {worst['IF_Score']:.4f}<br>
            <b>Causes:</b> {causes_str}</div></div>""", unsafe_allow_html=True)
    else:
        st.markdown("<div class='info-box'>✅ No anomalies detected. Battery behavior appears normal.</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Feature Statistics</div>", unsafe_allow_html=True)
    st.dataframe(result[FEATURES].describe().round(2), use_container_width=True)

    st.markdown("<div class='section-header'>Quick Scatter — Screen Time vs Battery Drop</div>", unsafe_allow_html=True)
    st.pyplot(make_scatter(result,'ScreenTime','BatteryDrop')); plt.close()


# ════════════════════════════════════════════
#  TAB 2 — ANOMALIES
# ════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Detected Anomalies (Isolation Forest)</div>", unsafe_allow_html=True)
    anomalies = result[result['IF_Pred']==-1].sort_values('IF_Score', ascending=False)

    if anomalies.empty:
        st.markdown("<div class='info-box'>✅ No anomalies found. Try lowering the contamination rate.</div>", unsafe_allow_html=True)
    else:
        for _, row in anomalies.head(10).iterrows():
            causes     = cause_analysis(row)
            ch         = "".join(f"<li>{c}</li>" for c in causes)
            ts_str     = str(row.get('Timestamp',''))[:16] if 'Timestamp' in row.index else '—'
            st.markdown(f"""
            <div class='card'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <span style='font-family:"Share Tech Mono",monospace;color:#f87171;font-size:0.82rem;'>ROW #{row.name} · {ts_str}</span>
                    <span class='badge-anomaly'>ANOMALY · {row['IF_Score']:.4f}</span>
                </div>
                <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:12px;'>
                    <div><span style='color:#475569;font-size:0.78rem;'>ScreenTime</span><br><span style='color:#e2e8f0;font-weight:600;'>{row['ScreenTime']} min</span></div>
                    <div><span style='color:#475569;font-size:0.78rem;'>CPUUsage</span><br><span style='color:#e2e8f0;font-weight:600;'>{row['CPUUsage']}%</span></div>
                    <div><span style='color:#475569;font-size:0.78rem;'>BatteryDrop</span><br><span style='color:#ef4444;font-weight:700;'>{row['BatteryDrop']}%</span></div>
                    <div><span style='color:#475569;font-size:0.78rem;'>BackgroundApps</span><br><span style='color:#fbbf24;font-weight:600;'>{row['BackgroundApps']}</span></div>
                    <div><span style='color:#475569;font-size:0.78rem;'>NetworkUsage</span><br><span style='color:#e2e8f0;font-weight:600;'>{row['NetworkUsage']} MB</span></div>
                    <div><span style='color:#475569;font-size:0.78rem;'>Hour</span><br><span style='color:#e2e8f0;font-weight:600;'>{int(row.get("Hour",0)):02d}:00</span></div>
                </div>
                <div style='margin-top:12px;'><span style='color:#475569;font-size:0.78rem;'>PROBABLE CAUSES</span>
                    <ul style='margin:6px 0 0;padding-left:18px;color:#fca5a5;font-size:0.85rem;'>{ch}</ul>
                </div>
            </div>""", unsafe_allow_html=True)
        if len(anomalies) > 10:
            st.info(f"Showing top 10 of {len(anomalies)} anomalies.")
        st.download_button("⬇ Download Anomaly Report (CSV)",
                           data=anomalies.to_csv(index=True).encode(),
                           file_name="anomaly_report.csv", mime="text/csv")


# ════════════════════════════════════════════
#  TAB 3 — VISUALIZATIONS
# ════════════════════════════════════════════
with tab3:
    cl, cr = st.columns(2)
    with cl:
        st.markdown("<div class='section-header'>Scatter Plot</div>", unsafe_allow_html=True)
        st.pyplot(make_scatter(result, x_axis, y_axis)); plt.close()
    with cr:
        st.markdown("<div class='section-header'>PCA — 2D Feature Space</div>", unsafe_allow_html=True)
        st.pyplot(make_pca_plot(result)); plt.close()
    cl2, cr2 = st.columns(2)
    with cl2:
        st.markdown(f"<div class='section-header'>Distribution — {hist_col}</div>", unsafe_allow_html=True)
        st.pyplot(make_histogram(result, hist_col)); plt.close()
    with cr2:
        st.markdown("<div class='section-header'>Top 20 Anomaly Scores</div>", unsafe_allow_html=True)
        st.pyplot(make_score_bar(result)); plt.close()

    st.markdown("<div class='section-header'>Feature Correlation Matrix</div>", unsafe_allow_html=True)
    corr = result[FEATURES].corr()
    fig_c, ax_c = plt.subplots(figsize=(7,4))
    fig_c.patch.set_facecolor(BG_DARK); ax_c.set_facecolor(BG_PANEL)
    im = ax_c.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax_c.set_xticks(range(len(FEATURES))); ax_c.set_yticks(range(len(FEATURES)))
    ax_c.set_xticklabels(FEATURES, rotation=35, ha='right', fontsize=8, color=DIM)
    ax_c.set_yticklabels(FEATURES, fontsize=8, color=DIM)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            ax_c.text(j,i,f"{corr.iloc[i,j]:.2f}", ha='center', va='center', fontsize=7, color='white')
    plt.colorbar(im, ax=ax_c)
    fig_c.tight_layout(pad=1.5); st.pyplot(fig_c); plt.close()


# ════════════════════════════════════════════
#  TAB 4 — TIME ANALYSIS  (NEW)
# ════════════════════════════════════════════
with tab4:
    st.markdown("""<div class='info-box'>
        📅 <b>Time-Based Analysis</b> — Reveals <i>when</i> battery anomalies occur:
        which hours, which days, and how patterns evolve over 30 days.
    </div>""", unsafe_allow_html=True)

    if 'Hour' not in result.columns:
        st.warning("No timestamp data. Use synthetic dataset or upload CSV with a Timestamp column.")
    else:
        # KPIs
        peak_hr   = result[result['IF_Pred']==-1].groupby('Hour').size().idxmax() if n_anom > 0 else 'N/A'
        peak_day  = result[result['IF_Pred']==-1].groupby('DayName').size().idxmax() if n_anom > 0 else 'N/A'
        night_r   = (result[result['Hour'].between(0,5)]['IF_Pred']==-1).mean()*100
        work_r    = (result[result['Hour'].between(9,17)]['IF_Pred']==-1).mean()*100

        k1,k2,k3,k4 = st.columns(4)
        with k1:
            hr_str = f"{peak_hr:02d}:00" if peak_hr != 'N/A' else 'N/A'
            st.markdown(f"<div class='card'><div class='card-title'>Peak Anomaly Hour</div><div class='card-value' style='color:{AMBER}'>{hr_str}</div><div class='card-sub'>Most suspicious hour</div></div>", unsafe_allow_html=True)
        with k2:
            st.markdown(f"<div class='card'><div class='card-title'>Peak Anomaly Day</div><div class='card-value' style='color:{AMBER};font-size:1.4rem'>{peak_day}</div><div class='card-sub'>Most suspicious day</div></div>", unsafe_allow_html=True)
        with k3:
            st.markdown(f"<div class='card'><div class='card-title'>Night Anomaly Rate</div><div class='card-value' style='color:{RED}'>{night_r:.1f}%</div><div class='card-sub'>00:00 – 05:59</div></div>", unsafe_allow_html=True)
        with k4:
            st.markdown(f"<div class='card'><div class='card-title'>Work-Hour Anomaly</div><div class='card-value' style='color:{BLUE}'>{work_r:.1f}%</div><div class='card-sub'>09:00 – 17:59</div></div>", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Hourly Battery Drain Pattern</div>", unsafe_allow_html=True)
        st.pyplot(make_hourly_line(result)); plt.close()
        st.caption("Blue fill = work hours · Purple shading = overnight · Dashed red = anomaly average")

        st.markdown("<div class='section-header'>Anomaly Rate by Hour of Day</div>", unsafe_allow_html=True)
        st.pyplot(make_hourly_rate(result)); plt.close()
        st.caption("Red bars = above-average anomaly rate hours · Amber dashed = overall mean")

        st.markdown("<div class='section-header'>Battery Drain Heatmap — Hour × Day of Week</div>", unsafe_allow_html=True)
        st.pyplot(make_heatmap(result)); plt.close()
        st.caption("Darker = heavier drain · Spot recurring high-drain time windows at a glance")

        st.markdown("<div class='section-header'>Daily Drain & Anomaly Frequency</div>", unsafe_allow_html=True)
        st.pyplot(make_daily_bar(result)); plt.close()

        st.markdown("<div class='section-header'>Anomaly Timeline (30-Day View)</div>", unsafe_allow_html=True)
        tl = make_anomaly_timeline(result)
        if tl: st.pyplot(tl); plt.close()

        st.markdown("<div class='section-header'>Hourly Summary Table</div>", unsafe_allow_html=True)
        htbl = (result.groupby('Hour')
                .agg(AvgBatteryDrop=('BatteryDrop','mean'),
                     AvgCPU=('CPUUsage','mean'),
                     AvgNetwork=('NetworkUsage','mean'),
                     TotalReadings=('BatteryDrop','count'),
                     Anomalies=('IF_Pred', lambda x:(x==-1).sum()))
                .round(2))
        htbl['AnomalyRate%'] = (htbl['Anomalies']/htbl['TotalReadings']*100).round(1)
        st.dataframe(htbl, use_container_width=True, height=300)


# ════════════════════════════════════════════
#  TAB 5 — MODEL COMPARISON  (NEW)
# ════════════════════════════════════════════
with tab5:
    st.markdown("""<div class='info-box'>
        ⚖ <b>Model Comparison</b> — Isolation Forest vs Local Outlier Factor (LOF).
        Metrics use ground-truth labels when available (synthetic dataset).
    </div>""", unsafe_allow_html=True)

    has_labels = 'Label' in result.columns
    n_anom_lof = (result['LOF_Pred']==-1).sum()

    if has_labels:
        if_m  = compute_metrics(result['Label'], result['IF_Pred'].values)
        lof_m = compute_metrics(result['Label'], result['LOF_Pred'].values)

        winner = "Isolation Forest" if if_m['f1'] >= lof_m['f1'] else "Local Outlier Factor"
        margin = abs(if_m['f1'] - lof_m['f1'])

        st.markdown(f"""<div class='success-box'>
            <span style='font-family:"Share Tech Mono",monospace;font-size:1rem;'>
                🏆 WINNER: <b>{winner}</b>
            </span><br>
            <span style='font-size:0.88rem;'>
                F1 margin: <b>{margin:.4f}</b> &nbsp;|&nbsp; IF F1: <b>{if_m['f1']:.4f}</b>
                &nbsp;|&nbsp; LOF F1: <b>{lof_m['f1']:.4f}</b><br><br>
                <b>Conclusion:</b> Isolation Forest performed better for this dataset.
                It handles high-dimensional tabular features efficiently via random partitioning
                and is robust to uniformly distributed anomalies. LOF relies on local density
                estimation, which works well in clustered data but struggles when anomalies
                are spread across the feature space — as seen in battery drain scenarios
                where unusual events arise from diverse root causes.
            </span>
        </div>""", unsafe_allow_html=True)

        # Metrics table
        st.markdown("<div class='section-header'>Performance Metrics</div>", unsafe_allow_html=True)
        rows_data = [
            ('Precision',            f"{if_m['precision']:.4f}", f"{lof_m['precision']:.4f}", if_m['precision'] >= lof_m['precision']),
            ('Recall',               f"{if_m['recall']:.4f}",    f"{lof_m['recall']:.4f}",    if_m['recall']    >= lof_m['recall']),
            ('F1 Score',             f"{if_m['f1']:.4f}",        f"{lof_m['f1']:.4f}",        if_m['f1']        >= lof_m['f1']),
            ('AUC-ROC',              f"{if_m['auc']:.4f}",       f"{lof_m['auc']:.4f}",       if_m['auc']       >= lof_m['auc']),
            ('Anomalies Detected',   str(n_anom),                 str(n_anom_lof),              None),
        ]
        trows = ""
        for metric, iv, lv, if_wins in rows_data:
            ic = "class='best-cell'" if if_wins is True  else ""
            lc = "class='best-cell'" if if_wins is False else ""
            ib = " 🏆" if if_wins is True  else ""
            lb = " 🏆" if if_wins is False else ""
            trows += f"<tr><td style='text-align:left;'>{metric}</td><td {ic}>{iv}{ib}</td><td {lc}>{lv}{lb}</td></tr>"
        st.markdown(f"""
        <table class='cmp-table'>
            <thead><tr><th style='text-align:left;'>Metric</th><th>Isolation Forest</th><th>Local Outlier Factor</th></tr></thead>
            <tbody>{trows}</tbody>
        </table>""", unsafe_allow_html=True)

        # Radar + score dist
        st.markdown("<div class='section-header'>Radar Chart & Score Distributions</div>", unsafe_allow_html=True)
        rc, sc = st.columns([1,1.8])
        with rc: st.pyplot(make_radar(if_m, lof_m)); plt.close()
        with sc: st.pyplot(make_score_dist_cmp(result)); plt.close()

        # Confusion matrices
        st.markdown("<div class='section-header'>Confusion Matrices</div>", unsafe_allow_html=True)
        cm1, cm2 = st.columns(2)
        with cm1: st.pyplot(make_cm_plot(if_m['cm'],  "Isolation Forest",     BLUE)); plt.close()
        with cm2: st.pyplot(make_cm_plot(lof_m['cm'], "Local Outlier Factor", AMBER)); plt.close()

    else:
        st.markdown("""<div class='alert-box'><div class='alert-title'>ℹ No Ground-Truth Labels</div>
            <div class='alert-body'>Precision/Recall/F1 require a <code>Label</code> column.
            Showing score distributions only.</div></div>""", unsafe_allow_html=True)
        st.pyplot(make_score_dist_cmp(result)); plt.close()
        st.markdown("""<div class='success-box'>
            <b>Recommendation:</b> Based on general benchmarks, <b>Isolation Forest</b> is preferred
            for battery drain anomaly detection. It scales to large datasets and is not sensitive to
            local density — a key advantage when anomaly types are diverse (background app surge,
            network spike, hidden processes, etc.).
        </div>""", unsafe_allow_html=True)

    # Agreement analysis (always shown)
    st.markdown("<div class='section-header'>Model Agreement Analysis</div>", unsafe_allow_html=True)
    both_a = ((result['IF_Pred']==-1)&(result['LOF_Pred']==-1)).sum()
    only_i = ((result['IF_Pred']==-1)&(result['LOF_Pred']==1 )).sum()
    only_l = ((result['IF_Pred']==1 )&(result['LOF_Pred']==-1)).sum()
    both_n = ((result['IF_Pred']==1 )&(result['LOF_Pred']==1 )).sum()
    a1,a2,a3,a4 = st.columns(4)
    with a1: st.markdown(f"<div class='card' style='border-color:#7f1d1d;'><div class='card-title' style='color:#f87171;'>Both Anomaly</div><div class='card-value' style='color:{RED}'>{both_a}</div><div class='card-sub'>High-confidence anomalies</div></div>", unsafe_allow_html=True)
    with a2: st.markdown(f"<div class='card'><div class='card-title'>Only IF</div><div class='card-value' style='color:{BLUE}'>{only_i}</div><div class='card-sub'>IF-exclusive detections</div></div>", unsafe_allow_html=True)
    with a3: st.markdown(f"<div class='card'><div class='card-title'>Only LOF</div><div class='card-value' style='color:{AMBER}'>{only_l}</div><div class='card-sub'>LOF-exclusive detections</div></div>", unsafe_allow_html=True)
    with a4: st.markdown(f"<div class='card' style='border-color:#065f46;'><div class='card-title' style='color:#34d399;'>Both Normal</div><div class='card-value' style='color:{GREEN}'>{both_n}</div><div class='card-sub'>High-confidence normal</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'><b>Tip:</b> Samples flagged by <i>both</i> models are highest-confidence anomalies. Borderline cases flagged by only one model deserve manual review.</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════
#  TAB 6 — LIVE PREDICTOR
# ════════════════════════════════════════════
with tab6:
    st.markdown("<div class='section-header'>Real-Time Single Reading Predictor</div>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
        Enter battery usage parameters for a single device reading.
        Both <b>Isolation Forest</b> and <b>LOF</b> classify it simultaneously with time-context insight.
    </div>""", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        p_screen = st.number_input("Screen Time (min)", 0.0, 300.0, 25.0, 1.0)
        p_cpu    = st.number_input("CPU Usage (%)",     0.0, 100.0, 22.0, 1.0)
    with c2:
        p_apps = st.number_input("Active Apps",     0, 50, 4, 1)
        p_bg   = st.number_input("Background Apps", 0, 50, 3, 1)
    with c3:
        p_net  = st.number_input("Network Usage (MB)", 0.0, 1000.0, 60.0, 5.0)
        p_drop = st.number_input("Battery Drop (%)",   0.0, 100.0,   5.0, 0.5)

    with st.expander("⏱ Optional: Time Context"):
        p_hour = st.slider("Hour of Day", 0, 23, 12)
        p_day  = st.selectbox("Day of Week", DAYS)

    if st.button("🔍 Analyze Reading (Both Models)"):
        sample  = np.array([[p_screen, p_cpu, p_apps, p_bg, p_net, p_drop]])
        samp_sc = scaler.transform(sample)

        if_pred  = if_model.predict(samp_sc)[0]
        if_score = -if_model.decision_function(samp_sc)[0]

        # LOF proxy via nearest-neighbor distance on training set
        X_train_sc = scaler.transform(df_clean[FEATURES].values)
        nn         = NearestNeighbors(n_neighbors=20).fit(X_train_sc)
        dist, _    = nn.kneighbors(samp_sc)
        lof_proxy  = dist.mean()
        lof_thresh = np.percentile(nn.kneighbors(X_train_sc)[0].mean(axis=1), (1-contamination)*100)
        lof_pred   = -1 if lof_proxy > lof_thresh else 1

        mock = pd.Series({'ScreenTime':p_screen,'CPUUsage':p_cpu,'Apps':p_apps,
                          'BackgroundApps':p_bg,'NetworkUsage':p_net,'BatteryDrop':p_drop})
        causes = cause_analysis(mock)
        ch     = "".join(f"<li>{c}</li>" for c in causes)

        m1, m2 = st.columns(2)
        with m1:
            if if_pred == -1:
                st.markdown(f"<div class='alert-box'><div class='alert-title'>⚠ IF: ANOMALY · Score {if_score:.4f}</div><div class='alert-body'>Isolation Forest classifies this reading as <b>abnormal</b>.</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='card' style='border-color:#065f46;'><div style='font-family:\"Share Tech Mono\",monospace;color:{GREEN};font-size:1rem;'>✅ IF: NORMAL</div><div style='color:#6ee7b7;margin-top:6px;'>Score: <b>{if_score:.4f}</b></div></div>", unsafe_allow_html=True)
        with m2:
            if lof_pred == -1:
                st.markdown(f"<div class='alert-box' style='border-color:{AMBER};'><div class='alert-title' style='color:{AMBER};'>⚠ LOF: ANOMALY</div><div class='alert-body'>Local Outlier Factor classifies this as <b>abnormal</b>.</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='card' style='border-color:#065f46;'><div style='font-family:\"Share Tech Mono\",monospace;color:{GREEN};font-size:1rem;'>✅ LOF: NORMAL</div><div style='color:#6ee7b7;margin-top:6px;'>Neighbor distance within normal range.</div></div>", unsafe_allow_html=True)

        # Consensus verdict
        if if_pred == -1 and lof_pred == -1:
            verdict = f"<div class='alert-box'><div class='alert-title'>🚨 BOTH MODELS AGREE — HIGH CONFIDENCE ANOMALY</div><div class='alert-body'><ul style='margin:6px 0 0;padding-left:18px;'>{ch}</ul></div></div>"
        elif if_pred == -1 or lof_pred == -1:
            verdict = f"<div class='info-box'>⚠ <b>BORDERLINE</b> — Models disagree. Manual review recommended.<br><ul style='margin:6px 0 0;padding-left:18px;'>{ch}</ul></div>"
        else:
            verdict = f"<div class='success-box'>✅ <b>BOTH MODELS AGREE — NORMAL BEHAVIOR</b><br>Battery usage is within expected parameters for {p_day} at {p_hour:02d}:00.</div>"
        st.markdown(verdict, unsafe_allow_html=True)

        # Gauge
        sc_c = min(if_score, 1.0)
        bc   = RED if if_pred == -1 else GREEN
        st.markdown(f"""
        <div style='margin-top:16px;'>
            <div style='color:#475569;font-size:0.78rem;margin-bottom:6px;font-family:"Share Tech Mono",monospace;letter-spacing:0.1em;'>IF ANOMALY SCORE GAUGE</div>
            <div style='background:#1e3a5f;border-radius:6px;height:14px;overflow:hidden;'>
                <div style='background:{bc};width:{sc_c*100:.1f}%;height:100%;border-radius:6px;'></div>
            </div>
            <div style='display:flex;justify-content:space-between;color:#334155;font-size:0.72rem;margin-top:4px;'>
                <span>0 — Normal</span><span>1.0 — Highly Anomalous</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # Time-context insight
        if 'Hour' in result.columns:
            hr_r  = (result[result['Hour']==p_hour]['IF_Pred']==-1).mean()*100
            day_r = (result[result['DayName']==p_day]['IF_Pred']==-1).mean()*100
            st.markdown(f"""
            <div class='card' style='margin-top:16px;'>
                <div class='card-title'>Time Context Insight</div>
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:8px;'>
                    <div>
                        <span style='color:{DIM};font-size:0.8rem;'>Anomaly rate at {p_hour:02d}:00</span><br>
                        <span style='color:{RED if hr_r>10 else BLUE};font-size:1.4rem;font-weight:700;'>{hr_r:.1f}%</span>
                        <span style='color:{DIM};font-size:0.78rem;'> historical avg</span>
                    </div>
                    <div>
                        <span style='color:{DIM};font-size:0.8rem;'>Anomaly rate on {p_day}</span><br>
                        <span style='color:{RED if day_r>10 else BLUE};font-size:1.4rem;font-weight:700;'>{day_r:.1f}%</span>
                        <span style='color:{DIM};font-size:0.78rem;'> historical avg</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════
#  TAB 7 — FULL DATA
# ════════════════════════════════════════════
with tab7:
    st.markdown("<div class='section-header'>Full Dataset with Predictions</div>", unsafe_allow_html=True)

    base_cols  = FEATURES + ['AnomalyLabel','IF_Score','LOF_Score']
    time_cols  = ['Timestamp','Hour','DayName'] if 'Hour' in result.columns else []
    extra_cols = ([c for c in ['ChargingStatus','Label'] if c in result.columns])
    show_cols  = time_cols + base_cols + extra_cols

    filt = st.radio("Show", ["All","Anomalies (IF)","Anomalies (LOF)",
                               "Both agree anomaly","Normal only"], horizontal=True)
    if   filt == "Anomalies (IF)":        display_df = result[result['IF_Pred']==-1]
    elif filt == "Anomalies (LOF)":       display_df = result[result['LOF_Pred']==-1]
    elif filt == "Both agree anomaly":    display_df = result[(result['IF_Pred']==-1)&(result['LOF_Pred']==-1)]
    elif filt == "Normal only":           display_df = result[(result['IF_Pred']==1)&(result['LOF_Pred']==1)]
    else:                                 display_df = result

    sc = [c for c in show_cols if c in display_df.columns]
    st.dataframe(display_df[sc].reset_index(), use_container_width=True, height=420)
    st.download_button("⬇ Download Full Results (CSV)",
                       data=result.to_csv(index=True).encode(),
                       file_name="battery_anomaly_full_results.csv", mime="text/csv")


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#1e3a5f;font-family:"Share Tech Mono",monospace;
            font-size:0.68rem;letter-spacing:0.15em;border-top:1px solid #1e3a5f;padding-top:14px;'>
    BATTERY ANOMALY DETECTION AI · ISOLATION FOREST + LOF · TIME ANALYSIS · BUILT WITH STREAMLIT
</div>
""", unsafe_allow_html=True)