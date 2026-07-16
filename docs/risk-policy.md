# Risk policy

The risk engine is the final authority. Defaults: 0.5% equity risk/trade, 30% exposure, 3% daily loss, 10% drawdown breaker, and a 24-hour cooldown after three losses. It rejects invalid ATR/stops, insufficient balance, unhealthy state, ML rejection and a second position. Circuit state must be persisted by a runner before restart.
