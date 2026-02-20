# ETF Regime Engine

### Regime Identification & Volatility Transition Framework


A quantitative framework that classifies market regimes using momentum and volatility, and (in V2) predicts where large moves are likely to occur next.

This project is designed to improve:

* Research prioritization
* Capital allocation
* Entry timing
* Risk management

---

# Dashboards

## V1 – Regime Classification






---

## V2 – Regime + Shock Score

*(Insert image below)*

```
[ PLACE V2 DASHBOARD IMAGE HERE ]
```

---

# What Problem Does This Solve?

Markets move in regimes:

* Trending
* Compressing
* Expanding
* Transitioning

Most strategies ignore regime structure.

This engine answers two simple but powerful questions:

1. What direction is the market structurally moving?
2. Where is volatility likely to expand soon?

---

# Full Pipeline Overview

Here is the complete workflow:

```
Market Data (Prices)
        ↓
Momentum Calculation
        ↓
Volatility Calculation
        ↓
Regime Classification
        ↓
Composite Score (Trend Strength)
        ↓
[V2 Only]
Shock Score (Expansion Probability)
        ↓
Sector Ranking & Dashboard Output
        ↓
Investment Decision Layer
```

Simple interpretation:

* Composite Score → Direction
* Shock Score → Movement Probability

---

# Version Breakdown

## V1 – Regime Engine

File: regimeid.py

Features:

* Momentum analysis
* Volatility analysis
* Regime classification
* Composite scoring
* Sector aggregation
* Visual dashboard

V1 tells you:

"What regime is the market in?"

---

## V2 – Institutional Version

File: regimeid2.py

Adds:

* Volatility acceleration detection
* Compression depth measurement
* Momentum instability analysis
* Shock score calculation

V2 tells you:

"What regime is the market in — and where will volatility expand next?"

---

# Core Concepts (Explained Simply)

## 1. Momentum

Measures price change over time.

If price is rising → bullish
If price is falling → bearish

---

## 2. Volatility

Measures how unstable price movement is.

Low volatility → compression
High volatility → expansion

---

## 3. Volatility Ratio

Compares short-term volatility to long-term volatility.

Short > Long → expansion
Short < Long → compression

---

# Regime Classification

| Trend | Volatility  | Regime           |
| ----- | ----------- | ---------------- |
| Up    | Expanding   | Bull Expansion   |
| Up    | Compressing | Bull Compression |
| Down  | Expanding   | Bear Expansion   |
| Down  | Compressing | Bear Compression |

---

# Composite Score

Each regime is converted into a number:

Bull Expansion = +2
Bull Compression = +1
Bear Compression = −1
Bear Expansion = −2

The final composite score blends short- and long-term regimes:

Composite Score → Measures structural trend strength

High positive → strong bullish
High negative → strong bearish

---

# V2 Shock Score (The Key Upgrade)

The Shock Score estimates how likely an asset is to experience a large move soon.

It measures:

* Rising volatility
* Deep compression
* Instability in trend

High shock score → Movement likely
Low shock score → Stable regime

Think of it as a pressure gauge.

---

# How to Interpret Results

Composite tells you direction.

Shock tells you timing.

Best setups:

High composite + high shock → strong long candidates
Low composite + high shock → strong short candidates
Neutral composite + high shock → transition setups

Low shock → low opportunity

---

# Configurable Parameters

Inside the config section:

```
SHORT_LOOKBACK = 20
LONG_LOOKBACK = 100
ACCEL_LOOKBACK = 5   # V2 only
TRADING_DAYS = 252
```

You can adjust sensitivity by changing these.

SHORT_LOOKBACK
Lower = faster signals
Higher = smoother signals

LONG_LOOKBACK
Higher = more structural stability
Lower = quicker regime shifts

ACCEL_LOOKBACK (V2)
Lower = faster transition detection
Higher = smoother shock signal

---

# Output Files

The engine automatically generates:

* etf_regime_output.csv
* category_composite.csv
* regime_dashboard.png
* institutional_dashboard.png (V2)

---

# Installation

```
pip install pandas numpy yfinance matplotlib seaborn
```

---

# Run

V1:

```
python regimeid.py
```

V2:

```
python regimeid2.py
```

---

# Example Applications

* Deep value research prioritization
* Sector rotation detection
* Volatility timing
* Options strategy alignment
* Portfolio allocation

---

# Key Insight

Markets do not move randomly.

They transition between compression and expansion regimes.

Volatility expansion creates opportunity.

This engine identifies those transitions before major moves occur.

---

# Future Possible Development

* Single-stock regime scoring
* Backtesting module
* Portfolio optimizer
* Real-time signal integration



