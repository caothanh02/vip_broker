# Recommendation research protocol V4

**Status: `closed_input_unavailable`.** V4 was not a candidate protocol and no strategy result,
selection, OOS evaluation, recommendation or trading activity occurred.

## Closure record

V4 mechanically audited public Binance Vision BTC/USDT 1-hour UTC archives from
`[2017-09-01T00:00:00Z, 2022-01-01T00:00:00Z)`. All 52 official archive checksums were verified,
but 24 monthly archives failed the fixed continuity or timestamp policy. The longest continuous
block was only `[2020-07-01T00:00:00Z, 2020-11-01T00:00:00Z)`, or four months.

This is an input-availability finding, not evidence about a strategy. V4 must not change its
continuity policy, retry with a fallback source, choose a range, freeze data, execute a candidate,
select a policy or authorize strict OOS after observing these facts. Its audit command now fails
closed before any network request.

The safe default remains `NEUTRAL`. No broker, order, live-trading, ML, credential, or OOS path is
authorized. A new source decision belongs only to [Protocol V5](recommendation-protocol-v5.md).
