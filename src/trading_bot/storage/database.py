from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Numeric, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class CircuitBreakerRecord(Base):
    __tablename__ = "circuit_breaker_state"
    id: Mapped[int] = mapped_column(primary_key=True)
    circuit_open: Mapped[bool] = mapped_column(default=False)
    cooldown_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class BotEventRecord(Base):
    __tablename__ = "bot_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    event_type: Mapped[str] = mapped_column(String(80))
    payload: Mapped[str] = mapped_column(String)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class PortfolioRecord(Base):
    __tablename__ = "portfolio_snapshots"
    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    cash: Mapped[Decimal] = mapped_column(Numeric(28, 8))
    equity: Mapped[Decimal] = mapped_column(Numeric(28, 8))


def database_session(url: str) -> sessionmaker[Session]:
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
