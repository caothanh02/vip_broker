# Recommendation research protocol

This protocol governs research-only BTC/USDT 1-hour recommendations. It does not enable live
trading, broker access, order submission, or API-key use.

## Frozen evaluation periods

The verified Binance Vision development dataset covers 2023-03-24T14:00:00Z through
2025-01-01T00:00:00Z. It is the only range allowed for rule, feature, threshold, or ML-filter
research. The first 200 closed candles are feature warm-up; they may be supplied to causal
features but are not evidence for a tuned claim.

Calendar year 2025 is a sealed holdout. It must not be used to select a feature, threshold,
confidence score, fee/slippage assumption, model, or recommendation policy. A strict OOS history
must continue to lock its UTC boundary, input SHA-256, and input range.

## Baseline observation

The untouched EMA/volume/ATR rule produced 53 applicable BUY_BIAS observations at 1h and 4h and
52 at 24h in the development range. Its after-cost directional accuracy was 24.5%, 35.8%, and
38.5% respectively. This is a negative baseline observation, not a performance claim and not a
reason to tune against the 2025 holdout.

The sparse candidate count means a supervised ML filter is not eligible until the development
sample is enlarged with additional verified pre-2025 history or a separately justified candidate
definition. Do not fit or present a model from the current 52--53 samples.

## Development workflow

1. Freeze the input checksum, feature schema, cost model, and candidate definition for one
   experiment.
2. Use expanding chronological folds within the development range. Every feature at decision
   candle T may use only closed candles at or before T; labels/outcomes may use future candles
   only after the decision is recorded.
3. Select at most one candidate policy from validation folds. Record every rejected alternative,
   its sample count, and the exact reason for rejection.
4. Run the chosen, frozen policy once against a new strict 2025 OOS history. Never retune after
   reading that report.

## Acceptance gates

Report 1h, 4h, and 24h independently. The primary statistic is after-cost directional accuracy
among non-NEUTRAL recommendations; it is not PnL, expected return, or investment advice.

A candidate is not eligible for an OOS claim unless each reported horizon has:

- at least 100 applicable, resolved recommendations;
- a two-sided 95% confidence interval whose lower bound is above 50%; and
- fixed fees/slippage and unchanged decision logic from the selected development experiment.

If these gates are not met, retain `NEUTRAL` as the safe default and label the output
`research-only` or `inconclusive`. `AVOID` remains an avoid-buy signal, never a short instruction.

## Audit requirements

Persist all research outputs outside Git. A report must identify the code commit, verified dataset
checksum/range, strict-OOS provenance, recommendation counts, coverage, applicable sample count,
confidence interval, and configured cost model. No raw market data, models, reports, caches, or
credentials may be committed.
