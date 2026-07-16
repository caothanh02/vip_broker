# Strategy

`EmaVolumeAtrStrategy` buys only on a closed-candle EMA20 upward cross of EMA50, close above EMA200, and volume above 1.2×20-period volume SMA. Initial and trailing stops are 2×ATR; the trail never decreases. It holds no more than one spot BTC position, never shorts, pyramids, DCA's, or leverages.
