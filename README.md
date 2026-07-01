# Path Signature Engine
## Market regime detection using rough path theory and iterated integrals

Most regime detection approaches feed price levels or returns into a model. This one feeds the *geometry of the price path* — specifically, its path signature, a set of iterated integrals derived from rough path theory that capture the sequential shape of how a market moves, not just where it ends up. Rolling windows of log price, log return, and realized volatility are encoded into signatures, clustered into regimes, and connected by a Markov transition matrix estimated via MLE. A Hidden Markov Model runs in parallel as a baseline so you can see where the two approaches agree and where they don't — disagreements are the most interesting cases.

---

## Installation (run it yourself)

Requires Python 3.9+.

```bash
git clone https://github.com/CarleeSantel/path-signature-engine.git
cd path-signature-engine
pip install streamlit yfinance pandas numpy scipy scikit-learn hmmlearn plotly esig
streamlit run path_signatures.py
```

Note: `esig` can be finicky to install on some systems. If `pip install esig` fails, try `pip install esig --no-build-isolation`.

---

## Installation (contribute)

```bash
git clone https://github.com/CarleeSantel/path-signature-engine.git
cd path-signature-engine
pip install streamlit yfinance pandas numpy scipy scikit-learn hmmlearn plotly esig
```

The core pipeline is: raw OHLCV → rolling windows → normalize → `esig.stream2sig(path, depth)` → StandardScaler → KMeans → regime labels. Depth 3 on a 3-channel path produces 40 features per window. If you change depth or channels, feature count changes — update the scaler accordingly.

---

## Contributing

Open an issue before submitting a pull request. If you're proposing a change to the signature depth or channel construction, include a note on how it affects feature dimensionality and whether you've re-validated cluster stability.

---

## Known issues

- esig install fails on some ARM Macs — working on a fallback
- Regime labels (bull/bear/etc.) are assigned post-hoc by sorting clusters on mean return; this can flip between runs if cluster assignments shift
- HMM comparison tab assumes Gaussian emissions — works poorly in highly non-Gaussian regimes (exactly the case where signatures outperform)
