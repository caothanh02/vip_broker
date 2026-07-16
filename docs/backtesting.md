# Backtesting

The event-driven candle engine validates a closed candle, updates stops, processes conservative exits, generates entries and executes accepted entries on the next candle open with configurable fees/slippage. A stop reachable in an OHLC candle is selected before other exits because intrabar ordering is unknown. Reports must include fees/slippage, trades and equity curve alongside buy-and-hold comparison when real history is available.
