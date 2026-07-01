import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import esig
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn import hmm
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Path Signature Engine", layout="wide")
st.title("Path Signature Engine — Regime Detection")
st.caption(
    "Rough path theory · Terry Lyons iterated integrals · "
    "Signature featurization · Markov regime transitions"
)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.header("Parameters")
ticker     = st.sidebar.text_input("Ticker", "SPY")
period     = st.sidebar.selectbox("Period", ["3y","5y","10y"], index=1)
sig_depth  = st.sidebar.slider("Signature Depth", 1, 4, 3,
                                help="Higher depth captures more complex sequential geometry. "
                                     "Depth d gives 1+n+n²+...+nᵈ features for n-dim path.")
window     = st.sidebar.slider("Rolling Window (days)", 10, 60, 20)
n_regimes  = st.sidebar.slider("Number of Regimes", 2, 5, 3)
stride     = st.sidebar.slider("Stride (days)", 1, 10, 1,
                                help="How many days to advance the window each step.")
st.sidebar.divider()
st.sidebar.subheader("Path Channels")
use_price  = st.sidebar.checkbox("Log Price", value=True)
use_ret    = st.sidebar.checkbox("Log Return", value=True)
use_vol    = st.sidebar.checkbox("Rolling Volatility (20d)", value=True)

# ── DATA ───────────────────────────────────────────────────────────────────────
with st.spinner("Fetching data..."):
    raw    = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    prices = raw["Close"].dropna()
    volume = raw["Volume"].dropna()
    prices, volume = prices.align(volume, join="inner")

log_p   = np.log(prices.values)
log_ret = np.diff(log_p, prepend=log_p[0])
dates   = prices.index

# rolling vol
roll_vol = pd.Series(log_ret).rolling(20).std().fillna(0).values

channels = []
ch_names = []
if use_price:
    channels.append(log_p);   ch_names.append("log_price")
if use_ret:
    channels.append(log_ret); ch_names.append("log_return")
if use_vol:
    channels.append(roll_vol);ch_names.append("roll_vol")

if not channels:
    st.warning("Select at least one path channel.")
    st.stop()

# Stack into (T, d) path
path_data = np.column_stack(channels)
d = path_data.shape[1]

# ── COMPUTE SIGNATURES ─────────────────────────────────────────────────────────
# Expected signature size for depth k, dim d: sum_{i=0}^{k} d^i
sig_size = sum(d**i for i in range(sig_depth+1))
st.sidebar.caption(f"Signature features: {sig_size} per window (dim={d}, depth={sig_depth})")

with st.spinner(f"Computing path signatures (window={window}d, depth={sig_depth})..."):
    sigs  = []
    widxs = []
    i = 0
    while i + window <= len(path_data):
        segment = path_data[i:i+window].copy()
        # normalize each channel to [0,1] within window for numerical stability
        for j in range(d):
            lo, hi = segment[:,j].min(), segment[:,j].max()
            if hi > lo: segment[:,j] = (segment[:,j]-lo)/(hi-lo)
            else:       segment[:,j] = 0.0
        sig = esig.stream2sig(segment, sig_depth)
        sigs.append(sig)
        widxs.append(i + window - 1)
        i += stride

    sigs     = np.array(sigs)
    sig_dates = dates[widxs]

st.success(f"Computed {len(sigs)} signatures — {sig_size} features each.")

# ── CLUSTERING ─────────────────────────────────────────────────────────────────
scaler    = StandardScaler()
sigs_sc   = scaler.fit_transform(sigs)

km = KMeans(n_clusters=n_regimes, n_init=20, random_state=42)
labels    = km.fit_predict(sigs_sc)

# Name regimes by mean return in window
mean_rets = []
for c in range(n_regimes):
    idxs = np.where(labels==c)[0]
    ret  = []
    for idx in idxs:
        raw_i = widxs[idx]
        r_window = log_ret[max(0,raw_i-window):raw_i]
        ret.extend(r_window)
    mean_rets.append(np.mean(ret) if ret else 0)

sorted_by_ret = np.argsort(mean_rets)
regime_names  = {}
if n_regimes == 2:
    regime_names[sorted_by_ret[0]] = "Bear"
    regime_names[sorted_by_ret[-1]]= "Bull"
elif n_regimes == 3:
    regime_names[sorted_by_ret[0]] = "Bear"
    regime_names[sorted_by_ret[1]] = "Neutral"
    regime_names[sorted_by_ret[-1]]= "Bull"
else:
    for i,c in enumerate(sorted_by_ret):
        regime_names[c] = f"Regime {i+1}"

regime_colors = {
    "Bull":"#2ecc71","Bear":"#e74c3c","Neutral":"#f39c12",
    "Regime 1":"#3498db","Regime 2":"#9b59b6","Regime 3":"#e67e22",
    "Regime 4":"#1abc9c","Regime 5":"#e74c3c",
}

label_names  = [regime_names[l] for l in labels]
color_mapped = [regime_colors.get(n,"#aaa") for n in label_names]

# ── RETENTION ANALYSIS ─────────────────────────────────────────────────────────
def retention_stats(label_seq):
    runs = {}
    if not label_seq: return runs
    cur, count = label_seq[0], 1
    for l in label_seq[1:]:
        if l == cur: count += 1
        else:
            runs.setdefault(cur,[]).append(count)
            cur, count = l, 1
    runs.setdefault(cur,[]).append(count)
    return runs

ret_stats = retention_stats(label_names)

# ── TRANSITION MATRIX (MLE) ────────────────────────────────────────────────────
regime_list = list(regime_names.values())
n_r = len(regime_list)
trans_counts = np.zeros((n_r, n_r))
for i in range(len(label_names)-1):
    from_i = regime_list.index(label_names[i])
    to_i   = regime_list.index(label_names[i+1])
    trans_counts[from_i, to_i] += 1

row_sums = trans_counts.sum(axis=1, keepdims=True)
row_sums[row_sums==0] = 1
trans_mle = trans_counts / row_sums

# ── HMM COMPARISON ─────────────────────────────────────────────────────────────
sigs_hmm = sigs_sc.copy()
hmm_model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="diag",
                              n_iter=200, random_state=42)
hmm_model.fit(sigs_hmm)
hmm_labels_raw = hmm_model.predict(sigs_hmm)

# align HMM labels to same regime ordering
hmm_mean_rets = []
for c in range(n_regimes):
    idxs = np.where(hmm_labels_raw==c)[0]
    ret = []
    for idx in idxs:
        raw_i = widxs[idx]
        r_window = log_ret[max(0,raw_i-window):raw_i]
        ret.extend(r_window)
    hmm_mean_rets.append(np.mean(ret) if ret else 0)
hmm_sorted = np.argsort(hmm_mean_rets)
hmm_name_map = {hmm_sorted[i]: list(regime_names.values())[i] for i in range(n_regimes)}
hmm_label_names = [hmm_name_map[l] for l in hmm_labels_raw]

# ── LAYOUT ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Regime Timeline", "Retention & Transitions", "HMM Comparison"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — REGIME TIMELINE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Signature-Based Regime Detection")
    st.markdown(
        "Each window is featurized using a **path signature** — the collection of all "
        "iterated integrals of the path up to depth d. Signatures encode not just "
        "the endpoint but the *order*, *shape*, and *texture* of how prices moved. "
        "Windows are then clustered in signature space via K-Means."
    )

    # Price chart colored by regime
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=prices.values, mode="lines",
                              name=ticker, line=dict(color="rgba(255,255,255,0.3)", width=1)))

    for rname in regime_list:
        mask = [i for i,n in enumerate(label_names) if n==rname]
        if not mask: continue
        sdates = [sig_dates[m] for m in mask]
        sprices= [float(prices.loc[sig_dates[m]]) for m in mask]
        fig.add_trace(go.Scatter(x=sdates, y=sprices, mode="markers", name=rname,
                                  marker=dict(color=regime_colors.get(rname,"#aaa"),
                                              size=5, opacity=0.8)))
    fig.update_layout(title=f"{ticker} Price — Colored by Signature Regime",
                       yaxis_title="Price ($)", template="plotly_dark",
                       margin=dict(t=40,b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Regime label timeline
    fig2 = go.Figure()
    for rname in regime_list:
        mask   = [i for i,n in enumerate(label_names) if n==rname]
        sdates = [sig_dates[m] for m in mask]
        ynums  = [regime_list.index(rname)] * len(mask)
        fig2.add_trace(go.Scatter(x=sdates, y=ynums, mode="markers", name=rname,
                                   marker=dict(color=regime_colors.get(rname,"#aaa"),
                                               size=6, symbol="square")))
    fig2.update_layout(yaxis=dict(tickvals=list(range(n_r)), ticktext=regime_list),
                        title="Regime Classification Over Time",
                        template="plotly_dark", margin=dict(t=40,b=20))
    st.plotly_chart(fig2, use_container_width=True)

    # PCA visualization of signature space
    st.subheader("Signature Space — PCA Projection")
    st.caption("2D projection of the high-dimensional signature vectors. Clusters show regime separability.")
    pca  = PCA(n_components=2)
    sigs_pca = pca.fit_transform(sigs_sc)
    fig3 = go.Figure()
    for rname in regime_list:
        mask = [i for i,n in enumerate(label_names) if n==rname]
        fig3.add_trace(go.Scatter(x=sigs_pca[mask,0], y=sigs_pca[mask,1],
                                   mode="markers", name=rname,
                                   marker=dict(color=regime_colors.get(rname,"#aaa"),
                                               size=5, opacity=0.7)))
    fig3.update_layout(xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)",
                        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)",
                        template="plotly_dark", margin=dict(t=20,b=20))
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RETENTION & TRANSITIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Regime Retention Analysis")
    st.markdown(
        "**Retention** = how many consecutive classification steps a regime persists. "
        "High retention means the regime is stable; low retention means the market "
        "is toggling rapidly between states."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        rows = []
        for rname in regime_list:
            runs = ret_stats.get(rname,[])
            if not runs: continue
            rows.append({
                "Regime":rname,
                "Count":len(runs),
                "Mean Retention (windows)":f"{np.mean(runs):.1f}",
                "Max Retention":max(runs),
                "Approx. Calendar Days":f"~{np.mean(runs)*stride:.0f}d"
            })
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Regime"), use_container_width=True)

    with col_b:
        fig4 = go.Figure()
        for rname in regime_list:
            runs = ret_stats.get(rname,[])
            if not runs: continue
            fig4.add_trace(go.Box(y=runs, name=rname,
                                   marker_color=regime_colors.get(rname,"#aaa")))
        fig4.update_layout(title="Retention Distribution by Regime",
                            yaxis_title="Consecutive Windows in Regime",
                            template="plotly_dark", margin=dict(t=40,b=20))
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("MLE Transition Matrix")
    st.markdown(
        "Estimated from the sequence of regime classifications via Maximum Likelihood Estimation. "
        "Entry (i,j) = probability of transitioning from regime i to regime j in the next window."
    )
    fig5 = px.imshow(
        trans_mle, x=regime_list, y=regime_list,
        color_continuous_scale="Blues", text_auto=".2f",
        labels=dict(x="To", y="From", color="Probability")
    )
    fig5.update_layout(title="Regime Transition Matrix (MLE)", margin=dict(t=40,b=20))
    st.plotly_chart(fig5, use_container_width=True)

    # Multi-step transitions via matrix power (Chapman-Kolmogorov)
    st.subheader("Multi-Step Transition Probabilities — Chapman-Kolmogorov")
    st.caption("P(n) = P^n — the n-step transition matrix via matrix exponentiation.")
    steps = st.slider("Forecast Steps (n windows ahead)", 1, 30, 10)
    P_n   = np.linalg.matrix_power(trans_mle, steps)
    fig6  = px.imshow(
        P_n, x=regime_list, y=regime_list,
        color_continuous_scale="Purples", text_auto=".2f",
        labels=dict(x="To", y="From", color="Probability")
    )
    fig6.update_layout(title=f"P^{steps} — Regime Distribution {steps} Steps Ahead",
                        margin=dict(t=40,b=20))
    st.plotly_chart(fig6, use_container_width=True)

    # Current regime forecast
    if label_names:
        cur_regime = label_names[-1]
        if cur_regime in regime_list:
            cur_idx = regime_list.index(cur_regime)
            cur_dist= P_n[cur_idx]
            st.subheader(f"Starting from current regime: **{cur_regime}**")
            forecast_df = pd.DataFrame({"Regime":regime_list,
                                         f"P(regime | {steps} steps)":cur_dist})
            st.dataframe(forecast_df.set_index("Regime").style.format("{:.2%}"),
                          use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HMM COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Path Signatures vs. Hidden Markov Model — Baseline Comparison")
    st.markdown(
        "HMM assumes regimes have Gaussian emission distributions and estimates hidden states. "
        "Path signatures make no distributional assumption — they capture geometric structure. "
        "This comparison shows where the two methods agree and where they diverge."
    )

    # Agreement rate
    agreement = sum(a==b for a,b in zip(label_names, hmm_label_names)) / len(label_names)
    st.metric("Classification Agreement", f"{agreement:.1%}",
              help="Fraction of windows where signature clustering and HMM agree on regime label.")

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Signature Regimes")
        counts_sig = Counter(label_names)
        fig7 = px.pie(names=list(counts_sig.keys()), values=list(counts_sig.values()),
                      color=list(counts_sig.keys()),
                      color_discrete_map=regime_colors, title="Signature — Regime Share")
        st.plotly_chart(fig7, use_container_width=True)

    with col_d:
        st.subheader("HMM Regimes")
        counts_hmm = Counter(hmm_label_names)
        fig8 = px.pie(names=list(counts_hmm.keys()), values=list(counts_hmm.values()),
                      color=list(counts_hmm.keys()),
                      color_discrete_map=regime_colors, title="HMM — Regime Share")
        st.plotly_chart(fig8, use_container_width=True)

    # Side-by-side timeline
    fig9 = go.Figure()
    for rname in regime_list:
        mask_s = [i for i,n in enumerate(label_names)     if n==rname]
        mask_h = [i for i,n in enumerate(hmm_label_names) if n==rname]
        if mask_s:
            fig9.add_trace(go.Scatter(x=[sig_dates[m] for m in mask_s],
                                       y=[regime_list.index(rname)+0.1]*len(mask_s),
                                       mode="markers", name=f"Sig: {rname}",
                                       marker=dict(color=regime_colors.get(rname,"#aaa"),
                                                   size=5, symbol="square")))
        if mask_h:
            fig9.add_trace(go.Scatter(x=[sig_dates[m] for m in mask_h],
                                       y=[regime_list.index(rname)-0.1]*len(mask_h),
                                       mode="markers", name=f"HMM: {rname}",
                                       marker=dict(color=regime_colors.get(rname,"#aaa"),
                                                   size=5, symbol="circle", opacity=0.5),
                                       showlegend=True))
    fig9.update_layout(title="Regime Labels: Signature (square) vs HMM (circle)",
                        yaxis=dict(tickvals=list(range(n_r)), ticktext=regime_list),
                        template="plotly_dark", margin=dict(t=40,b=20))
    st.plotly_chart(fig9, use_container_width=True)

    st.info(
        "**Why signatures can outperform HMM:** HMM assumes each regime has a stationary "
        "Gaussian distribution and models only pairwise transitions. Path signatures capture "
        "*non-linear interactions between channels* and *the order in which events occurred*. "
        "A sharp intraday reversal and a gradual drift that both end at the same price look "
        "identical to HMM — signatures distinguish them. This is the core of Terry Lyons' "
        "rough path theory: the path, not just its endpoint, is what matters."
    )

st.caption("Built with Python · esig (rough path signatures) · hmmlearn · scikit-learn · Streamlit · Plotly")
