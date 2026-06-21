# Momentum-Anomaly-Seminar-
Python research project testing the momentum anomaly in U.S. and European large-cap stocks (2005–2025) using CAPM regressions, decile sorts, and a long-short portfolio strategy.

# Momentum Anomaly Analysis: USA vs. Europe (2005–2025)

Economics seminar research project, recognized for academic excellence, testing whether the momentum anomaly — the tendency of recent winning stocks to keep outperforming and recent losers to keep underperforming — holds in modern U.S. and European large-cap equity markets, and how the effect differs between the two regions.

## Data

- **612 U.S. tickers** (current and former S&P 500 constituents) and **500 European tickers** (current and former STOXX 600 constituents)
- Monthly stock returns, market returns, and risk-free rates collected via **Python (yfinance)**, covering **2005–2025** (~249,000 monthly observations)
- Some European series start later than 2005 due to data availability
- The post-2005 window was chosen deliberately: capital markets became significantly more retail-accessible and digitally tool-assisted from that point on, making it a more representative "modern market" sample

## Methodology

1. **Momentum signal (12-1 momentum):** geometric compounding of monthly returns from month *t-12* to *t-2*, skipping the most recent month to avoid short-term reversal effects (where the prior month's biggest losers often bounce back).
2. **Cross-sectional Z-score normalization:** each month, within each region, raw momentum is converted to a Z-score relative to that month's cross-section of stocks. This turns momentum into a relative ranking rather than a raw return — controlling for whether the overall market that month was up or down.
3. **CAPM + momentum regression:**

   `Exc_Stock = α + β₁·Exc_Mkt + β₂·Z_MOM`

   Run on the full 2005–2025 sample and separately by sub-period (2005–09, 2010–14, 2015–19, 2020–25) for both regions.
4. **Decile sorting:** each month, stocks are split into 10 deciles by raw 12-1 momentum (D1 = lowest, D10 = highest) to observe the return pattern directly, independent of the regression.
5. **Winner-minus-loser (W-L) strategy:** a long-short portfolio (long D10, short D1) to test whether momentum is tradeable, not just statistically present.
6. Survivorship bias (from using index-constituent ticker lists) is explicitly flagged as a limitation affecting the results, particularly for losing stocks.

## Key Findings

**United States — weak effect:** momentum among large caps is small and only marginally significant. The long-short strategy roughly broke even. Past "losers" delivered better-than-expected returns — likely a survivorship-bias artifact — which compressed the winner-loser spread.

**Europe — significant effect:** momentum is statistically significant with a clear, step-wise increase in returns from D1 to D10. The long-short strategy produced a modest but consistently positive cumulative return.

**Regime dependence:** momentum is strongest during optimistic, low-uncertainty markets and weakens or disappears during market stress (e.g., the 2008 crisis). The U.S. market's greater efficiency and the higher visibility of its large-cap names appear to compress the anomaly relative to Europe, where it has more room to persist.

## Limitations & Future Work

- Address survivorship bias more rigorously (delisted-stock data)
- Extend the universe to mid- and small-cap stocks in both regions to test whether momentum strengthens outside heavily-traded large caps
- Move beyond CAPM to multi-factor models (e.g., Fama-French) to control for additional risk factors

## Code Structure (`קוד_סופי.py`)

1. **Data download** — pulls daily prices for ~1,100 unique U.S. + European tickers via `yfinance` (2004–2025), resampled to month-end closes. Benchmarks: S&P 500 (`^GSPC`), Euro Stoxx (`^STOXX`), and the 13-week T-bill (`^IRX`) as the risk-free rate.
2. **Panel construction** — for every stock-month, computes the 12-1 momentum signal, the forward (next-month) return, and excess stock/market returns over the risk-free rate. Drops tickers with insufficient history, winsorizes returns at the 1st/99th percentiles to limit outlier distortion, then computes the cross-sectional, region-by-month Z-score. Exports three CSVs: the full panel, a regression-only panel, and a Fama-MacBeth panel.
3. **Regressions** — for each region and sub-period, runs three models: a month-by-month Fama-MacBeth regression of forward return on momentum Z-score (averaged across months with a t-stat), a baseline CAPM regression, and CAPM augmented with the momentum factor.
4. **Long/short portfolio** — each month, goes long the top 10% by raw momentum and short the bottom 10%, then reports annualized return, volatility, Sharpe ratio, and t-stat for the winners-minus-losers (WML) spread, plus a CAPM regression on the WML portfolio itself.
5. **Decile analysis** — sorts stocks into 10 monthly momentum deciles and reports annualized return/volatility/Sharpe/cumulative return per decile, to check whether the return pattern is monotonic.
6. **Visualization** — cumulative-return charts for the long/short portfolio vs. the market, a direct USA-vs-Europe comparison, and decile bar/line charts.

## Tools

Python · yfinance · pandas · NumPy · statsmodels · matplotlib · seaborn
