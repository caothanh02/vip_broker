# Architecture

The core is dependency-directional: domain objects are pure dataclasses; data adapters normalize Binance `BTCUSDT` to `BTC/USDT`; features are shared by training and inference; strategy proposes only; risk authorizes; brokers execute simulations. Storage is SQLAlchemy/SQLite and can use a PostgreSQL URL without business-logic changes.
