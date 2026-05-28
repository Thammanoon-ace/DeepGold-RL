"""
MetaTrader 5 connectivity bridge (requirement #9).

``MT5Bridge`` wraps the official ``MetaTrader5`` Python package, exposing the
small surface the live trader needs: connect, fetch recent bars, read account
state, query positions, and send/close market orders.

Architecture
------------
    Python RL agent  -->  MT5Bridge  -->  MetaTrader 5 terminal  -->  Broker

The ``MetaTrader5`` package is **Windows-only** and requires a running MT5
terminal.  The import is therefore done lazily so the rest of the framework
(training/backtesting) works on any OS without it installed.

Safety
------
This module is the *only* place that talks to the broker.  All order methods
honour a ``dry_run`` flag (default on) that logs the intended action without
sending it, so the architecture can be wired up and tested before any real
capital is at risk.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from config.config import LiveConfig

logger = logging.getLogger(__name__)


# MT5 timeframe constants are resolved lazily (only when the package is present).
_TIMEFRAME_NAMES = {"M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15", "H1": "TIMEFRAME_H1"}


class MT5Bridge:
    """Thin, defensively-coded wrapper around the MetaTrader5 API."""

    def __init__(self, config: LiveConfig) -> None:
        self.config = config
        self._mt5 = None  # the imported module, set in connect()
        self._connected = False

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #
    def connect(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None,
    ) -> bool:
        """Initialize the terminal connection.

        Credentials are optional: if the MT5 terminal is already logged in,
        ``initialize()`` alone is sufficient.  Never hard-code credentials —
        pass them from environment variables.
        """
        try:
            import MetaTrader5 as mt5  # noqa: N813 (vendor naming)
        except ImportError as exc:  # pragma: no cover - platform dependent
            raise RuntimeError(
                "The 'MetaTrader5' package is required for live trading and is "
                "only available on Windows. Install with `pip install MetaTrader5` "
                "and run an MT5 terminal."
            ) from exc

        self._mt5 = mt5
        kwargs = {}
        if path:
            kwargs["path"] = path
        if login is not None:
            kwargs.update(login=int(login), password=password, server=server)

        if not mt5.initialize(**kwargs):
            err = mt5.last_error()
            logger.error("MT5 initialize() failed: %s", err)
            return False

        # Make sure the symbol is visible in Market Watch.
        if not mt5.symbol_select(self.config.symbol, True):
            logger.warning("Could not select symbol %s in Market Watch.", self.config.symbol)

        self._connected = True
        logger.info("Connected to MT5; terminal=%s", mt5.terminal_info())
        return True

    def shutdown(self) -> None:
        """Cleanly disconnect from the terminal."""
        if self._mt5 is not None and self._connected:
            self._mt5.shutdown()
            self._connected = False
            logger.info("MT5 connection closed.")

    def _require(self):
        if not self._connected or self._mt5 is None:
            raise RuntimeError("MT5Bridge is not connected. Call connect() first.")
        return self._mt5

    # ------------------------------------------------------------------ #
    # Market data
    # ------------------------------------------------------------------ #
    def get_rates(self, count: int = 500, timeframe: Optional[str] = None) -> pd.DataFrame:
        """Fetch the most recent ``count`` bars as an OHLCV DataFrame.

        The returned frame uses the same canonical schema as the historical
        loader (``open/high/low/close/tick_volume`` indexed by ``time``), so the
        exact training-time feature pipeline can be reused unchanged.
        """
        mt5 = self._require()
        tf_name = _TIMEFRAME_NAMES[timeframe or self.config.timeframe]
        tf = getattr(mt5, tf_name)
        rates = mt5.copy_rates_from_pos(self.config.symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rates returned for {self.config.symbol}: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time")
        return df[["open", "high", "low", "close", "tick_volume"]]

    def get_account_info(self) -> dict:
        """Return key account fields (balance, equity, margin, free margin)."""
        mt5 = self._require()
        info = mt5.account_info()
        if info is None:
            raise RuntimeError(f"account_info() failed: {mt5.last_error()}")
        return {
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "margin_free": info.margin_free,
            "currency": info.currency,
            "leverage": info.leverage,
        }

    def get_positions(self) -> List[dict]:
        """Return open positions for the configured symbol + magic number."""
        mt5 = self._require()
        positions = mt5.positions_get(symbol=self.config.symbol)
        if positions is None:
            return []
        return [
            p._asdict() for p in positions if p.magic == self.config.magic_number
        ]

    # ------------------------------------------------------------------ #
    # Order execution
    # ------------------------------------------------------------------ #
    def _symbol_point(self) -> float:
        mt5 = self._require()
        info = mt5.symbol_info(self.config.symbol)
        return info.point if info else 0.01

    def market_order(
        self,
        direction: int,
        lots: float,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        comment: str = "deepgold-rl",
    ) -> Optional[dict]:
        """Send a market order (direction +1 buy / -1 sell).

        Honours ``config.dry_run``: when set, the intended request is logged and
        nothing is transmitted to the broker.
        """
        mt5 = self._require()
        lots = round(min(lots, self.config.max_lots), 2)
        if lots <= 0:
            logger.info("Skipping order: computed lots <= 0.")
            return None

        tick = mt5.symbol_info_tick(self.config.symbol)
        price = tick.ask if direction > 0 else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.symbol,
            "volume": float(lots),
            "type": order_type,
            "price": float(price),
            "deviation": 20,  # max slippage in points
            "magic": self.config.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if sl_price:
            request["sl"] = float(sl_price)
        if tp_price:
            request["tp"] = float(tp_price)

        if self.config.dry_run:
            logger.info("[DRY-RUN] Would send market order: %s", request)
            return {"dry_run": True, "request": request}

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("Order failed: %s | %s", getattr(result, "retcode", "?"), mt5.last_error())
            return None
        logger.info("Order executed: ticket=%s volume=%s", result.order, lots)
        return result._asdict()

    def close_position(self, ticket: int) -> Optional[dict]:
        """Close an open position identified by ``ticket``."""
        mt5 = self._require()
        positions = [p for p in (mt5.positions_get(ticket=ticket) or [])]
        if not positions:
            logger.warning("No open position with ticket %s.", ticket)
            return None
        pos = positions[0]

        # Closing a long means selling, and vice-versa.
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(self.config.symbol)
        price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": float(price),
            "deviation": 20,
            "magic": self.config.magic_number,
            "comment": "deepgold-rl-close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if self.config.dry_run:
            logger.info("[DRY-RUN] Would close position: %s", request)
            return {"dry_run": True, "request": request}

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("Close failed: %s | %s", getattr(result, "retcode", "?"), mt5.last_error())
            return None
        logger.info("Closed position ticket=%s", ticket)
        return result._asdict()
