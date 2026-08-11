# Recommendation research protocol V4

**Status: `draft_availability_audit_required`.** V4 is not a candidate protocol. It has no
candidate, parameters, development range, dataset, input lock, selection artifact, performance
result, or OOS authorization.

## Permitted next step

Only a mechanical availability audit of public Binance Vision archives may propose a future V4
development range. The audit must require official checksum verification, BTC/USDT 1-hour UTC,
closed candles, and absolute continuity. It must not inspect or use a signal, return, accuracy,
backtest, or any performance metric to choose a range.

Any proposed range, candidate, parameters, costs, input lock, and selection rules require a new
reviewed V4 revision before implementation or execution. Until then, strict OOS 2025 is sealed and
the safe default remains `NEUTRAL`. This is research governance, not investment advice; no broker,
order, live-trading, ML, credential, or exchange-runtime path is authorized.

## Mechanical availability audit

`trading-bot audit-protocol-v4-availability --start <UTC-month-boundary> --end
<UTC-month-boundary>` is the only implemented V4 action. It reads public Binance Vision **monthly**
ZIP archives and their official checksum sidecars into memory, verifies archive/member identity,
closed 1-hour timestamps, and exact continuity, then prints a JSON result. It never writes a CSV,
cache, report, dataset, manifest, or selection artifact; it does not use REST, compute a feature,
signal, return, accuracy, backtest, or performance metric.

For raw timestamp identity it applies the repository's fixed, checksum-bound Binance Vision policy:
an early raw close of at most 60 seconds is canonicalized to its closed UTC hour and reported as an
accepted timestamp anomaly. No known market interruption is passed to this audit, so every actual
missing candle or larger/late timestamp deviation fails the absolute-continuity requirement.

The command accepts only complete UTC calendar months ending no later than the sealed 2025 OOS
boundary. A continuous result means `availability_verified_not_selected`: it may inform a later
governance review, but does not choose a range, candidate, parameter, policy, or OOS action. Any
request failure, checksum problem, malformed archive, open candle, duplicate, or gap returns
`availability_not_verified` and remains fail-closed.
