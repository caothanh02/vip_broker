# Recommendation research protocol V8

**Status: `closed_input_unavailable`.** The single authorized CoinAPI availability audit was run
at `2026-08-20T06:06:10Z` and the provider rejected its request. No data was persisted and no
strategy result was evaluated.

V8 is now closed: `audit-protocol-v8-coinapi-historical-availability` fails before reading local
credentials or constructing a network client. It cannot retry, change the range/source, persist an
input, execute research, select a policy or authorize strict OOS. The default remains `NEUTRAL`;
no broker, order, live-trading or ML path is loaded.

This is an input-availability finding, not a strategy or investment result. A future provider would
require a new independent protocol rather than a V8 fallback or retry.
