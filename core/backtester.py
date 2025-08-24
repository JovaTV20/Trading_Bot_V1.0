"""
DATETIME FIX für core/backtester.py
Problem: 'int' object has no attribute 'isoformat'
Lösung: Sichere Timestamp-Konvertierung + SHARPE RATIO Fix
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from core.base_strategy import StrategyBase

def safe_datetime_convert(timestamp) -> datetime:
    """Sichere Konvertierung zu datetime"""
    if timestamp is None:
        return datetime.now()
    
    # Bereits datetime
    if isinstance(timestamp, datetime):
        return timestamp
    
    # Pandas Timestamp
    if hasattr(timestamp, 'to_pydatetime'):
        return timestamp.to_pydatetime()
    
    # String
    if isinstance(timestamp, str):
        try:
            return pd.to_datetime(timestamp).to_pydatetime()
        except:
            return datetime.now()
    
    # Integer (Unix timestamp)
    if isinstance(timestamp, (int, float)):
        try:
            return datetime.fromtimestamp(timestamp)
        except:
            # Fallback: Verwende als Tage seit Epoch
            return datetime(1970, 1, 1) + timedelta(days=int(timestamp))
    
    # Numpy datetime64
    if isinstance(timestamp, np.datetime64):
        return pd.to_datetime(timestamp).to_pydatetime()
    
    # Fallback
    return datetime.now()

@dataclass
class Trade:
    """Trade mit sicherer DateTime-Behandlung"""
    
    def __init__(self, symbol: str, entry_time, entry_price: float,
                 quantity: float, side: str, strategy_signal: Dict[str, Any] = None):
        self.symbol = symbol
        self.entry_time = safe_datetime_convert(entry_time)  # SICHERE KONVERTIERUNG
        self.entry_price = entry_price
        self.quantity = abs(quantity)
        self.side = side.lower()
        self.exit_time: Optional[datetime] = None
        self.exit_price: Optional[float] = None
        self.pnl: float = 0.0
        self.commission: float = 0.0
        self.is_closed = False
        self.return_pct: float = 0.0
        self.duration_days: int = 0
        
        # Strategie-Informationen
        self.strategy_signal = strategy_signal or {}
        self.confidence = strategy_signal.get('confidence', 0.0) if strategy_signal else 0.0
        
        # Exit-Parameter
        self.max_holding_days = 15
        self.stop_loss_pct = 0.03
        self.take_profit_pct = 0.05
        
        if strategy_signal:
            self.stop_loss_pct = strategy_signal.get('stop_loss_pct', 0.03)
            self.take_profit_pct = strategy_signal.get('take_profit_pct', 0.05)
        
        # Berechne Stop/Take-Profit Preise
        if self.side == 'buy':
            self.stop_loss_price = entry_price * (1 - self.stop_loss_pct)
            self.take_profit_price = entry_price * (1 + self.take_profit_pct)
        else:
            self.stop_loss_price = entry_price * (1 + self.stop_loss_pct)
            self.take_profit_price = entry_price * (1 - self.take_profit_pct)
        
        self.days_held = 0
        
    def update_trade(self, current_price: float, current_time) -> str:
        """Update mit sicherer DateTime-Behandlung"""
        if self.is_closed:
            return 'hold'
        
        # SICHERE DateTime-Konvertierung
        current_dt = safe_datetime_convert(current_time)
        
        # Berechne Tage seit Entry
        try:
            self.days_held = (current_dt - self.entry_time).days
        except:
            self.days_held += 1  # Fallback: Inkrementiere
        
        # Timeout nach max_holding_days
        if self.days_held >= self.max_holding_days:
            return 'timeout'
        
        # Take-Profit
        if self.side == 'buy' and current_price >= self.take_profit_price:
            return 'take_profit'
        elif self.side == 'sell' and current_price <= self.take_profit_price:
            return 'take_profit'
        
        # Stop-Loss
        if self.side == 'buy' and current_price <= self.stop_loss_price:
            return 'stop_loss'
        elif self.side == 'sell' and current_price >= self.stop_loss_price:
            return 'stop_loss'
        
        # Trailing Stop nach 5+ Tagen im Gewinn
        if self.days_held >= 5:
            unrealized_pnl = self._calculate_unrealized_pnl(current_price)
            if unrealized_pnl > 0:
                if self.side == 'buy':
                    if not hasattr(self, 'highest_price'):
                        self.highest_price = current_price
                    elif current_price > self.highest_price:
                        self.highest_price = current_price
                    
                    trailing_stop = self.highest_price * 0.98
                    if current_price <= trailing_stop:
                        return 'trailing_stop'
        
        return 'hold'
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Unrealisierten P&L berechnen"""
        if self.side == 'buy':
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity
        
    def close_trade(self, exit_time, exit_price: float, 
                   commission: float = 0.0, reason: str = 'manual'):
        """Trade schließen mit sicherer DateTime-Behandlung"""
        if self.is_closed:
            return
            
        self.exit_time = safe_datetime_convert(exit_time)  # SICHERE KONVERTIERUNG
        self.exit_price = exit_price
        self.commission = commission
        self.is_closed = True
        self.exit_reason = reason
        
        # P&L berechnen
        if self.side == 'buy':
            self.pnl = (exit_price - self.entry_price) * self.quantity - commission
        else:
            self.pnl = (self.entry_price - exit_price) * self.quantity - commission
        
        # Return Percentage
        if self.entry_price > 0:
            investment = self.entry_price * self.quantity
            self.return_pct = self.pnl / investment
        
        # Duration
        try:
            self.duration_days = (self.exit_time - self.entry_time).days
            if self.duration_days == 0:
                self.duration_days = 1
        except:
            self.duration_days = max(1, self.days_held)
    
    def to_dict(self) -> Dict[str, Any]:
        """Zu Dictionary mit sicherer Timestamp-Serialisierung"""
        def safe_timestamp(dt):
            """Sichere Timestamp-Serialisierung"""
            if dt is None:
                return None
            if isinstance(dt, datetime):
                return dt.isoformat()
            # Fallback für andere Typen
            try:
                return safe_datetime_convert(dt).isoformat()
            except:
                return str(dt)
        
        return {
            'symbol': self.symbol,
            'entry_time': safe_timestamp(self.entry_time),
            'entry_price': self.entry_price,
            'exit_time': safe_timestamp(self.exit_time),
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'side': self.side,
            'pnl': round(self.pnl, 2),
            'return_pct': round(self.return_pct, 4),
            'commission': self.commission,
            'duration_days': self.duration_days,
            'is_closed': self.is_closed,
            'confidence': self.confidence,
            'exit_reason': getattr(self, 'exit_reason', 'unknown'),
            'stop_loss_pct': round(self.stop_loss_pct, 3),
            'take_profit_pct': round(self.take_profit_pct, 3)
        }

class RealisticPortfolio:
    """Portfolio mit sicherer DateTime-Behandlung"""
    
    def __init__(self, initial_capital: float, commission: float = 0.0, slippage: float = 0.001):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission
        self.slippage_rate = slippage
        
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_history: List[Dict[str, Any]] = []
        
        # Limits
        self.max_concurrent_trades = 3
        self.min_trade_value = 500
        self.max_position_pct = 0.15
        
        self.total_commission_paid = 0.0
        self.trades_opened_today = 0
        self.last_trade_date = None
    
    def can_open_new_trade(self, current_date, trade_value: float) -> bool:
        """Prüfung für neuen Trade"""
        current_dt = safe_datetime_convert(current_date)
        
        if len(self.open_trades) >= self.max_concurrent_trades:
            return False
        
        if trade_value < self.min_trade_value:
            return False
        
        if self.cash < trade_value * 1.1:
            return False
        
        # Rate Limiting
        if (self.last_trade_date and 
            self.last_trade_date.date() == current_dt.date() and 
            self.trades_opened_today >= 2):
            return False
        
        return True
    
    def open_position(self, symbol: str, current_price: float, 
                     timestamp, side: str, 
                     strategy_signal: Dict[str, Any] = None) -> Optional[Trade]:
        """Position öffnen"""
        timestamp_dt = safe_datetime_convert(timestamp)
        
        # Trade-Größe berechnen
        current_equity = self.get_current_equity(current_price)
        position_pct = min(strategy_signal.get('position_size', 0.1), self.max_position_pct)
        trade_value = current_equity * position_pct
        quantity = trade_value / current_price
        
        if not self.can_open_new_trade(timestamp_dt, trade_value):
            return None
        
        # Slippage
        execution_price = current_price * (1 + self.slippage_rate if side == 'buy' else 1 - self.slippage_rate)
        
        # Kosten
        actual_trade_value = quantity * execution_price
        commission_cost = actual_trade_value * self.commission_rate
        total_cost = actual_trade_value + commission_cost
        
        # Cash Update
        if side == 'buy':
            self.cash -= total_cost
        else:
            self.cash += actual_trade_value - commission_cost
        
        # Trade erstellen
        trade = Trade(symbol, timestamp_dt, execution_price, quantity, side, strategy_signal)
        
        self.total_commission_paid += commission_cost
        self.open_trades.append(trade)
        
        # Statistiken
        if not self.last_trade_date or self.last_trade_date.date() != timestamp_dt.date():
            self.trades_opened_today = 1
        else:
            self.trades_opened_today += 1
        self.last_trade_date = timestamp_dt
        
        return trade
    
    def update_and_close_trades(self, current_price: float, current_time):
        """Trades aktualisieren und schließen"""
        current_dt = safe_datetime_convert(current_time)
        
        for trade in self.open_trades.copy():
            exit_reason = trade.update_trade(current_price, current_dt)
            
            if exit_reason != 'hold':
                self.close_position(trade, current_price, current_dt, exit_reason)
    
    def close_position(self, trade: Trade, current_price: float, 
                      timestamp, reason: str = 'manual') -> bool:
        """Position schließen"""
        timestamp_dt = safe_datetime_convert(timestamp)
        
        if trade.is_closed or trade not in self.open_trades:
            return False
        
        # Slippage beim Schließen
        exit_side = 'sell' if trade.side == 'buy' else 'buy'
        execution_price = current_price * (1 - self.slippage_rate if exit_side == 'sell' else 1 + self.slippage_rate)
        
        # Kosten
        trade_value = trade.quantity * execution_price
        commission_cost = trade_value * self.commission_rate
        
        # Cash Update
        if trade.side == 'buy':
            self.cash += trade_value - commission_cost
        else:
            self.cash -= trade_value + commission_cost
        
        self.total_commission_paid += commission_cost
        
        # Trade schließen
        trade.close_trade(timestamp_dt, execution_price, commission_cost, reason)
        
        # Move zu closed
        self.open_trades.remove(trade)
        self.closed_trades.append(trade)
        
        return True
    
    def get_current_equity(self, current_price: float) -> float:
        """Aktuelles Equity berechnen"""
        total_position_value = 0.0
        
        for trade in self.open_trades:
            if trade.side == 'buy':
                position_value = trade.quantity * current_price
            else:
                position_value = trade.quantity * (2 * trade.entry_price - current_price)
            total_position_value += position_value
        
        return self.cash + total_position_value
    
    def update_equity_history(self, timestamp, current_price: float):
        """Equity-Historie aktualisieren"""
        timestamp_dt = safe_datetime_convert(timestamp)
        current_equity = self.get_current_equity(current_price)
        total_return = (current_equity - self.initial_capital) / self.initial_capital
        
        unrealized_pnl = sum(trade._calculate_unrealized_pnl(current_price) for trade in self.open_trades)
        
        self.equity_history.append({
            'timestamp': timestamp_dt,  # Echtes datetime-Objekt
            'equity': current_equity,
            'cash': self.cash,
            'unrealized_pnl': unrealized_pnl,
            'open_positions': len(self.open_trades),
            'total_return': total_return,
            'price': current_price
        })

class Backtester:
    """Backtester mit sicherer DateTime-Behandlung"""
    
    def __init__(self, strategy: StrategyBase, initial_capital: float = 10000.0,
                 commission: float = 0.0, slippage: float = 0.001):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def run(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Backtest mit sicherer DateTime-Behandlung"""
        
        self.logger.info(f"Starte DATETIME-SICHEREN Backtest für {symbol}")
        
        # Validierung
        if not self.strategy.validate_data(data) or len(data) < 50:
            raise ValueError("Ungültige Daten")
        
        # WICHTIG: Stelle sicher, dass Index datetime ist
        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("Index ist kein DatetimeIndex - konvertiere")
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                self.logger.error(f"Index-Konvertierung fehlgeschlagen: {e}")
                # Erstelle künstlichen datetime-Index
                start_date = datetime(2023, 1, 1)
                date_range = pd.date_range(start=start_date, periods=len(data), freq='D')
                data.index = date_range
                self.logger.info("Künstlicher DateTime-Index erstellt")
        
        # Trainiere Strategie
        try:
            self.strategy.fit(data)
        except Exception as e:
            self.logger.error(f"Strategie-Training fehlgeschlagen: {e}")
            raise
        
        # Portfolio initialisieren
        portfolio = RealisticPortfolio(self.initial_capital, self.commission, self.slippage)
        
        signals_generated = 0
        
        # HAUPT-LOOP mit sicherer DateTime-Behandlung
        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_price = row['close']
            
            # Sichere Timestamp-Konvertierung
            try:
                timestamp_dt = safe_datetime_convert(timestamp)
            except Exception as e:
                self.logger.warning(f"Timestamp-Konvertierung fehlgeschlagen für Index {i}: {e}")
                timestamp_dt = datetime(2023, 1, 1) + timedelta(days=i)
            
            # Portfolio-Historie aktualisieren
            portfolio.update_equity_history(timestamp_dt, current_price)
            
            # Überspringe frühe Daten
            if i < self.strategy.lookback_period:
                continue
            
            # Bestehende Trades aktualisieren
            portfolio.update_and_close_trades(current_price, timestamp_dt)
            
            # Neues Signal alle 3 Tage
            if i % 3 == 0:
                try:
                    signal = self.strategy.generate_signal(row)
                    signal['symbol'] = symbol
                    
                    if signal['action'] != 'hold':
                        signals_generated += 1
                        
                        new_trade = portfolio.open_position(
                            symbol=symbol,
                            current_price=current_price,
                            timestamp=timestamp_dt,
                            side=signal['action'],
                            strategy_signal=signal
                        )
                        
                        if new_trade:
                            self.logger.debug(f"Trade eröffnet: {signal['action']} @ ${current_price:.2f}")
                            
                except Exception as e:
                    self.logger.warning(f"Signal-Fehler bei Index {i}: {e}")
                    continue
        
        # Alle offenen Trades schließen
        if len(data) > 0:
            final_price = data.iloc[-1]['close']
            final_timestamp = safe_datetime_convert(data.index[-1])
            
            for trade in portfolio.open_trades.copy():
                portfolio.close_position(trade, final_price, final_timestamp, 'backtest_end')
            
            # Final update
            portfolio.update_equity_history(final_timestamp, final_price)
        
        # Ergebnisse berechnen
        return self._calculate_results(portfolio, symbol)
    
    def _calculate_results(self, portfolio: RealisticPortfolio, symbol: str) -> Dict[str, Any]:
        """Ergebnisse mit sicherer Serialisierung und SHARPE RATIO"""
        
        # Metriken berechnen
        final_equity = portfolio.equity_history[-1]['equity'] if portfolio.equity_history else portfolio.initial_capital
        total_return = (final_equity - portfolio.initial_capital) / portfolio.initial_capital
        
        all_trades = portfolio.closed_trades
        winning_trades = [t for t in all_trades if t.pnl > 0.50]
        losing_trades = [t for t in all_trades if t.pnl < -0.50]
        
        total_trades = len(all_trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        avg_duration = np.mean([t.duration_days for t in all_trades]) if all_trades else 0
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else 99.9
        
        # REPARIERT: SHARPE RATIO BERECHNUNG
        sharpe_ratio = 0.0
        if portfolio.equity_history and len(portfolio.equity_history) > 1:
            try:
                # Berechne täglich Returns aus Equity-Historie
                equity_values = [entry['equity'] for entry in portfolio.equity_history]
                daily_returns = []
                
                for i in range(1, len(equity_values)):
                    if equity_values[i-1] > 0:
                        daily_return = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
                        daily_returns.append(daily_return)
                
                if len(daily_returns) > 1:
                    mean_return = np.mean(daily_returns)
                    std_return = np.std(daily_returns)
                    
                    if std_return > 0:
                        # Annualisierte Sharpe Ratio (252 Trading-Tage)
                        sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
                    else:
                        sharpe_ratio = 0.0
                
            except Exception as e:
                self.logger.warning(f"Sharpe-Ratio Berechnung fehlgeschlagen: {e}")
                sharpe_ratio = 0.0
        
        def safe_float(value):
            if value is None or np.isnan(value) or np.isinf(value):
                return 0.0
            return float(value)
        
        # SICHERE Equity-Curve Serialisierung
        equity_df = pd.DataFrame(portfolio.equity_history) if portfolio.equity_history else pd.DataFrame()
        if not equity_df.empty:
            try:
                equity_df.set_index('timestamp', inplace=True)
                # Drawdown berechnen
                equity_df['peak'] = equity_df['equity'].cummax()
                equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
            except Exception as e:
                self.logger.warning(f"Equity-DataFrame Verarbeitung fehlgeschlagen: {e}")
        
        max_drawdown = 0
        if not equity_df.empty and 'drawdown' in equity_df.columns:
            max_drawdown = abs(equity_df['drawdown'].min())
        
        # Ergebnisse - REPARIERT mit allen erforderlichen Metriken
        results = {
            'initial_capital': safe_float(portfolio.initial_capital),
            'final_capital': safe_float(final_equity),
            'total_return': safe_float(total_return),
            'total_return_pct': safe_float(total_return * 100),
            'max_drawdown': safe_float(max_drawdown),
            'sharpe_ratio': safe_float(sharpe_ratio),  # REPARIERT: Jetzt immer vorhanden
            
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'neutral_trades': total_trades - win_count - loss_count,
            'win_rate': safe_float(win_rate),
            
            'total_pnl': safe_float(sum(t.pnl for t in all_trades)),
            'avg_win': safe_float(avg_win),
            'avg_loss': safe_float(avg_loss),
            'profit_factor': safe_float(profit_factor),
            'max_win': safe_float(max([t.pnl for t in all_trades], default=0)),
            'max_loss': safe_float(min([t.pnl for t in all_trades], default=0)),
            
            'avg_trade_duration': safe_float(avg_duration),
            'total_commission': safe_float(portfolio.total_commission_paid),
            
            # SICHERE Serialisierung
            'trades': [t.to_dict() for t in all_trades],
            'equity_curve': equity_df,
            
            # REPARIERT: Cash-Werte hinzugefügt
            'final_cash': safe_float(portfolio.cash),
            
            'backtest_info': {
                'datetime_fix_applied': 'TRUE',
                'strategy_name': self.strategy.__class__.__name__,
                'avg_duration_days': round(avg_duration, 1),
                'sharpe_calculation': 'included',  # REPARIERT: Bestätigung
                'total_equity_points': len(portfolio.equity_history)
            }
        }
        
        # Logging
        self.logger.info(f"=== REPARIERTE ERGEBNISSE ===")
        self.logger.info(f"Total Return: {results['total_return']:.2%}")
        self.logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")  # REPARIERT: Logging
        self.logger.info(f"Trades: {total_trades} (Avg Duration: {avg_duration:.1f} Tage)")
        self.logger.info(f"Win Rate: {win_rate:.2%}")
        
        return results