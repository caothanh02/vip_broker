# Deployment

Copy `.env.example` locally, retain `BOT_MODE=backtest` or `dry_run`, and use SQLite for MVP. Docker Compose exposes an optional PostgreSQL profile. Never expose credentials, give withdrawal permission, or use production secrets. Live mode is intentionally non-functional in this release.
