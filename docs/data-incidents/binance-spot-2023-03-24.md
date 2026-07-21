# Binance Spot interruption — 2023-03-24

## Evidence and timeline

Binance's [official completion announcement](https://www.binance.com/en/support/announcement/detail/813a31506e9f478ea8c1058b425df87a)
states that Spot trading resumed at 14:00 UTC. Its [verified Binance Blog account explanation](https://www.binance.com/en/square/post/344026)
states that Spot trading was disabled at 11:27 UTC because of a matching-engine trailing-stop bug.

The event window used by the audited policy is therefore `[2023-03-24T11:27:00Z,
2023-03-24T14:00:00Z)`. It is a non-tradable interval, not a pricing or
interpolation rule.

## Independent raw-source audit

The official archive checksums observed during this audit were:

| Archive | SHA-256 | Inner CSV |
| --- | --- | --- |
| `BTCUSDT-1h-2023-03.zip` | `7f2afb8e0179a57ac31eab5205660298ba5eb77039ac2e21aef9b715ff3d06ce` | `BTCUSDT-1h-2023-03.csv` |
| `BTCUSDT-1h-2023-03-24.zip` | `ea9d94f28a39ad8029c9c2863cbb7769137188edd957fea35d0313ae4183561f` | `BTCUSDT-1h-2023-03-24.csv` |

Both verified archives and unauthenticated Binance REST `api/v3/klines` agree
on every row below. The daily archive has 23 rows, the monthly archive has 743
rows: both omit `2023-03-24T13:00:00Z`; neither contains a duplicate.

| Open UTC | Raw open | Raw close | Expected close | Duration ms | Early ms | OHLC | Volume | Trades | Monthly/daily/REST |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 09:00 | 1679648400000 | 1679651999999 | 1679651999999 | 3600000 | 0 | 28034.38 / 28190.35 / 28000 / 28041.11 | 4506.28931000 | 73822 | yes / yes / yes |
| 10:00 | 1679652000000 | 1679655599999 | 1679655599999 | 3600000 | 0 | 28041.11 / 28085.06 / 27941 / 28039.71 | 3639.24610000 | 65628 | yes / yes / yes |
| 11:00 | 1679655600000 | 1679659199999 | 1679659199999 | 3600000 | 0 | 28039.71 / 28091.03 / 27963.84 / 28080 | 1267.41714000 | 25083 | yes / yes / yes |
| 12:00 | 1679659200000 | 1679661581646 | 1679662799999 | 2381647 | 1218353 | 28080 / 28080 / 28080 / 28080 | 0 | 0 | yes / yes / yes |
| 13:00 | — | — | 1679666399999 | — | — | missing | — | — | missing / missing / missing |
| 14:00 | 1679666400000 | 1679669999999 | 1679669999999 | 3600000 | 0 | 28079.99 / 28253.01 / 27835 / 27989.06 | 8983.24018000 | 144497 | yes / yes / yes |
| 15:00 | 1679670000000 | 1679673599999 | 1679673599999 | 3600000 | 0 | 27989.07 / 28076.82 / 27843.41 / 28018.04 | 5198.28681000 | 92428 | yes / yes / yes |

## Decision

This is sufficient evidence for one exact allowlist event:
`binance-spot-2023-03-24-trailing-stop-maintenance`. It accepts only the exact
BTCUSDT Spot 1h raw 12:00 row above and only under either checksum listed
above. A changed archive checksum, archive name, raw timestamp, symbol, or
timeframe fails closed. The generic 60-second early-close policy is unchanged;
late closes remain rejected. The missing 13:00 interval is retained as a
sidecar-audited non-tradable gap and is never interpolated.
