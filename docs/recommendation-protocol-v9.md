# Recommendation research protocol V9

**Status: `closed_input_unavailable`.** The single authorized paid CoinAPI availability audit ran
at `2026-08-24T03:39:47Z`; the provider rejected its request. No data was persisted and no strategy
result was evaluated.

V9 is now closed: `audit-protocol-v9-coinapi-historical-availability` fails before reading a local
credential or constructing a network client. It cannot retry, change the source/range, persist an
input, execute research, select a policy or authorize strict OOS.

The 2025 holdout remains sealed and the product default remains `NEUTRAL`; there is no broker,
order, live-trading or ML path. This is an input-availability finding, not a strategy result,
accuracy claim, or investment recommendation. Any future paid provider/account context requires a
new independent protocol rather than a V9 fallback or retry.

V8 remains closed and is never retried. A V9 audit result is an input-availability finding, never a
strategy result, accuracy claim, or investment recommendation.
