"""
Backtester für Trading-Strategien - KOMPLETT NEU GESCHRIEBEN
Simuliert historisches Trading und berechnet Performance-Metriken
ERSETZE: core/backtester.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from core.base_strategy import StrategyBase

@dataclass
class Trade:
    """Repräsentiert einen einzelnen Trade - Vereinfacht"""
    
    def __init__(self, symbol: str, entry_time: datetime, entry_price: float,
                 quantity: float, side: str):
        self.symbol = symbol
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.quantity = abs(quantity)  # Immer positive Menge
        self.side = side.lower()  # 'buy' oder 'sell'
        self.exit_time: Optional[datetime] = None
        self.exit_price: Optional[float] = None
        self.pnl: float = 0.0
        self.commission: float = 0.0
        self.is_closed = False
        self.return_pct: float = 0.0
        self.duration_days: int = 0
        
    def close_trade(self, exit_time: datetime, exit_price: float, commission: float = 0.0):
        """Schließt den Trade und berechnet P&L"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.commission = commission
        self.is_closed = True
        
        # Berechne P&L basierend auf Trade-Richtung
        if self.side == 'buy':
            # Long Position: Gewinn wenn Preis steigt
            self.pnl = (exit_price - self.entry_price) * self.quantity - commission
        else:  # sell/short
            # Short Position: Gewinn wenn Preis fällt
            self.pnl = (self.entry_price - exit_price) * self.quantity - commission
        
        # Return Percentage
        if self.entry_price > 0:
            self.return_pct = self.pnl / (self.entry_price * self.quantity)
        
        # Duration
        if self.exit_time and self.entry_time:
            self.duration_days = (self.exit_time - self.entry_time).days
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert Trade zu Dictionary"""
        return {
            'symbol': self.symbol,
            'entry_time': self.entry_time,
            'entry_price': self.entry_price,
            'exit_time': self.exit_time,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'side': self.side,
            'pnl': self.pnl,
            'return_pct': self.return_pct,
            'commission': self.commission,
            'duration_days': self.duration_days,
            'is_closed': self.is_closed
        }

class Portfolio:
    """Verwaltet Portfolio-Status während Backtest - Vereinfacht"""
    
    def __init__(self, initial_capital: float, commission: float = 0.0, slippage: float = 0.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission
        self.slippage_rate = slippage
        
        # Position-Tracking
        self.current_position: float = 0.0  # Aktuelle Position (positiv=long, negativ=short)
        self.position_value: float = 0.0
        self.last_price: float = 0.0
        
        # Historie
        self.trades: List[Trade] = []
        self.equity_history: List[Dict[str, Any]] = []
        
        # Statistiken
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission_paid = 0.0
        
    def get_current_equity(self, current_price: float) -> float:
        """Berechnet aktuelles Gesamtkapital"""
        position_value = self.current_position * current_price if self.current_position != 0 else 0
        return self.cash + position_value
    
    def can_execute_trade(self, quantity: float, price: float, side: str) -> bool:
        """Prüft ob Trade ausführbar ist"""
        trade_value = abs(quantity) * price
        total_cost = trade_value * (1 + self.commission_rate + self.slippage_rate)
        
        if side == 'buy':
            return self.cash >= total_cost
        else:  # sell
            # Für Short-Verkauf oder Position schließen
            return True  # Vereinfacht - keine Margin-Prüfungen
    
    def execute_trade(self, symbol: str, quantity: float, price: float, 
                     timestamp: datetime, side: str) -> Optional[Trade]:
        """Führt Trade aus und updated Portfolio"""
        
        if not self.can_execute_trade(quantity, price, side):
            return None
        
        # Berücksichtige Slippage
        execution_price = price * (1 + self.slippage_rate if side == 'buy' else 1 - self.slippage_rate)
        
        # Berechne Kosten
        trade_value = abs(quantity) * execution_price
        commission_cost = trade_value * self.commission_rate
        
        # Erstelle Trade
        trade = Trade(symbol, timestamp, execution_price, abs(quantity), side)
        
        # Update Portfolio basierend auf Aktion
        if side == 'buy':
            # Kaufe Position
            if self.current_position < 0:
                # Schließe Short-Position
                close_quantity = min(abs(quantity), abs(self.current_position))
                self.current_position += close_quantity
                if quantity > close_quantity:
                    # Überschüssige Menge = neue Long-Position
                    self.current_position += (quantity - close_quantity)
            else:
                # Erweitere Long-Position
                self.current_position += quantity
            
            self.cash -= (trade_value + commission_cost)
            
        else:  # sell
            # Verkaufe Position
            if self.current_position > 0:
                # Schließe Long-Position
                close_quantity = min(abs(quantity), abs(self.current_position))
                self.current_position -= close_quantity
                if quantity > close_quantity:
                    # Überschüssige Menge = neue Short-Position
                    self.current_position -= (quantity - close_quantity)
            else:
                # Erweitere Short-Position
                self.current_position -= quantity
            
            self.cash += (trade_value - commission_cost)
        
        # Schließe Trade sofort für Backtest
        trade.close_trade(timestamp, execution_price, commission_cost)
        
        # Update Statistiken
        self.total_trades += 1
        self.total_commission_paid += commission_cost
        
        if trade.pnl > 0:
            self.winning_trades += 1
        elif trade.pnl < 0:
            self.losing_trades += 1
        
        # Speichere Trade
        self.trades.append(trade)
        self.last_price = execution_price
        
        return trade
    
    def update_equity_history(self, timestamp: datetime, current_price: float):
        """Aktualisiert Equity-Historie"""
        current_equity = self.get_current_equity(current_price)
        total_return = (current_equity - self.initial_capital) / self.initial_capital
        
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': current_equity,
            'cash': self.cash,
            'position_value': self.current_position * current_price,
            'position_size': self.current_position,
            'total_return': total_return,
            'price': current_price
        })

class Backtester:
    """
    Backtesting-Engine für Trading-Strategien - NEU GESCHRIEBEN
    """
    
    def __init__(self, strategy: StrategyBase, initial_capital: float = 10000.0,
                 commission: float = 0.0, slippage: float = 0.001):
        """
        Initialisiert Backtester
        
        Args:
            strategy: Trading-Strategie
            initial_capital: Startkapital
            commission: Kommissionsrate (z.B. 0.001 = 0.1%)
            slippage: Slippage-Rate (z.B. 0.001 = 0.1%)
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def run(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Führt Backtest aus - VEREINFACHT UND ROBUST
        
        Args:
            data: Historische OHLCV-Daten mit DateTime-Index
            symbol: Trading-Symbol
            
        Returns:
            Dictionary mit Backtest-Ergebnissen
        """
        self.logger.info(f"Starte Backtest für {symbol} mit {len(data)} Datenpunkten")
        
        # Validiere Eingangsdaten
        if not self.strategy.validate_data(data):
            raise ValueError("Ungültige Eingangsdaten für Backtest")
            
        if len(data) < 20:
            raise ValueError("Nicht genügend Daten für Backtest (mindestens 20 Datenpunkte)")
        
        # Trainiere Strategie
        self.logger.info("Trainiere Strategie...")
        try:
            self.strategy.fit(data)
        except Exception as e:
            self.logger.error(f"Strategie-Training fehlgeschlagen: {e}")
            raise
        
        # Initialisiere Portfolio
        portfolio = Portfolio(self.initial_capital, self.commission, self.slippage)
        
        # Simulation-Parameter
        signals_generated = 0
        trades_executed = 0
        
        self.logger.info("Starte Trading-Simulation...")
        
        # Durchlaufe alle Datenpunkte
        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_price = row['close']
            
            # Update Portfolio-Historie
            portfolio.update_equity_history(timestamp, current_price)
            
            # Überspringe erste Punkte (nicht genug Historie für Strategie)
            if i < self.strategy.lookback_period:
                continue
            
            # Generiere Trading-Signal
            try:
                signal = self.strategy.generate_signal(row)
                
                if signal['action'] != 'hold':
                    signals_generated += 1
                    
                    # Berechne Trade-Größe
                    current_equity = portfolio.get_current_equity(current_price)
                    position_size_pct = signal.get('position_size', 0.1)  # Default 10%
                    max_trade_value = current_equity * position_size_pct
                    trade_quantity = max_trade_value / current_price
                    
                    # Mindest-Trade-Größe
                    if trade_quantity * current_price < 100:  # Mindestens $100 Trade
                        continue
                    
                    # Führe Trade aus
                    trade = portfolio.execute_trade(
                        symbol=symbol,
                        quantity=trade_quantity,
                        price=current_price,
                        timestamp=timestamp,
                        side=signal['action']
                    )
                    
                    if trade:
                        trades_executed += 1
                        self.logger.debug(f"Trade {trades_executed}: {signal['action']} {trade_quantity:.2f} @ ${current_price:.2f}")
                        
                        # Risk Management: Stop Loss / Take Profit
                        if hasattr(signal, 'stop_loss') and signal.get('stop_loss'):
                            # TODO: Stop-Loss Implementation
                            pass
                            
            except Exception as e:
                self.logger.warning(f"Fehler bei Signal-Generierung für Index {i}: {e}")
                continue
        
        # Final update
        if len(data) > 0:
            final_price = data.iloc[-1]['close']
            portfolio.update_equity_history(data.index[-1], final_price)
        
        self.logger.info(f"Backtest abgeschlossen: {signals_generated} Signale, {trades_executed} Trades")
        
        # Berechne und gib Ergebnisse zurück
        return self._calculate_results(portfolio, symbol)
    
    def _calculate_results(self, portfolio: Portfolio, symbol: str) -> Dict[str, Any]:
        """Berechnet finale Backtest-Ergebnisse"""
        
        # Finale Metriken
        final_equity = portfolio.equity_history[-1]['equity'] if portfolio.equity_history else portfolio.initial_capital
        total_return = (final_equity - portfolio.initial_capital) / portfolio.initial_capital
        
        # Trade-Analysen
        all_trades = portfolio.trades
        winning_trades = [t for t in all_trades if t.pnl > 0.01]  # Mindestens 1 Cent Gewinn
        losing_trades = [t for t in all_trades if t.pnl < -0.01]   # Mindestens 1 Cent Verlust
        neutral_trades = [t for t in all_trades if -0.01 <= t.pnl <= 0.01]
        
        # Statistiken
        total_trades = len(all_trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        neutral_count = len(neutral_trades)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        loss_rate = loss_count / total_trades if total_trades > 0 else 0
        
        # Durchschnittliche Gewinne/Verluste
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Profit Factor
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        
        if total_losses > 0:
            profit_factor = total_wins / total_losses
        elif total_wins > 0:
            profit_factor = 99.9  # Sehr hoch aber nicht unrealistisch
        else:
            profit_factor = 1.0
        
        # Begrenze auf realistische Werte
        profit_factor = min(profit_factor, 99.9)
        profit_factor = max(profit_factor, 0.01)
        
        # Drawdown-Analyse
        equity_df = pd.DataFrame(portfolio.equity_history)
        max_drawdown = 0
        sharpe_ratio = 0
        
        if not equity_df.empty and len(equity_df) > 1:
            equity_df.set_index('timestamp', inplace=True)
            
            # Drawdown
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
            max_drawdown = abs(equity_df['drawdown'].min())
            
            # Sharpe Ratio
            daily_returns = equity_df['total_return'].diff().dropna()
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        
        # Weitere Metriken
        longest_winning_streak = self._calculate_streak(all_trades, 'win')
        longest_losing_streak = self._calculate_streak(all_trades, 'loss')
        
        # Beste und schlechteste Trades
        best_trade = max(all_trades, key=lambda t: t.pnl) if all_trades else None
        worst_trade = min(all_trades, key=lambda t: t.pnl) if all_trades else None
        
        # Sanitize alle numerischen Werte für JSON
        def safe_float(value):
            """Konvertiert zu float und behandelt inf/nan"""
            if value is None:
                return 0.0
            if np.isinf(value) or np.isnan(value):
                return 0.0
            return float(value)
        
        # Erstelle Ergebnis-Dictionary
        results = {
            # Haupt-Performance-Metriken
            'initial_capital': safe_float(portfolio.initial_capital),
            'final_capital': safe_float(final_equity),
            'total_return': safe_float(total_return),
            'total_return_pct': safe_float(total_return * 100),
            
            # Risk-Metriken
            'max_drawdown': safe_float(max_drawdown),
            'sharpe_ratio': safe_float(sharpe_ratio),
            
            # Trade-Statistiken
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'neutral_trades': neutral_count,
            'win_rate': safe_float(win_rate),
            'loss_rate': safe_float(loss_rate),
            
            # P&L Statistiken
            'total_pnl': safe_float(final_equity - portfolio.initial_capital),
            'total_wins_amount': safe_float(total_wins),
            'total_losses_amount': safe_float(total_losses),
            'avg_win': safe_float(avg_win),
            'avg_loss': safe_float(avg_loss),
            'profit_factor': safe_float(profit_factor),
            
            # Weitere Statistiken
            'longest_winning_streak': longest_winning_streak,
            'longest_losing_streak': longest_losing_streak,
            'avg_trade_duration': safe_float(np.mean([t.duration_days for t in all_trades])) if all_trades else 0,
            'total_commission': safe_float(portfolio.total_commission_paid),
            
            # Portfolio-Details
            'final_cash': safe_float(portfolio.cash),
            'final_position': safe_float(portfolio.current_position),
            'final_position_value': safe_float(portfolio.position_value),
            
            # Beste/Schlechteste Trades
            'best_trade_pnl': safe_float(best_trade.pnl if best_trade else 0),
            'worst_trade_pnl': safe_float(worst_trade.pnl if worst_trade else 0),
            
            # Equity Curve für Charts
            'equity_curve': equity_df if not equity_df.empty else pd.DataFrame(),
            
            # Alle Trades
            'trades': [t.to_dict() for t in all_trades],
            
            # Debug/Info
            'backtest_info': {
                'strategy_name': self.strategy.__class__.__name__,
                'data_points': len(equity_df) if not equity_df.empty else 0,
                'commission_rate': self.commission,
                'slippage_rate': self.slippage,
                'lookback_period': getattr(self.strategy, 'lookback_period', 20)
            }
        }
        
        return results
    
    def _calculate_streak(self, trades: List[Trade], streak_type: str) -> int:
        """Berechnet längste Gewinn- oder Verlust-Serie"""
        if not trades:
            return 0
            
        current_streak = 0
        max_streak = 0
        
        for trade in trades:
            if streak_type == 'win' and trade.pnl > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            elif streak_type == 'loss' and trade.pnl < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
                
        return max_streak
    
    def optimize_parameters(self, data: pd.DataFrame, symbol: str, 
                          param_ranges: Dict[str, List], metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Parameter-Optimierung - VEREINFACHT
        
        Args:
            data: Historische Daten
            symbol: Trading-Symbol
            param_ranges: Parameter-Bereiche zum Testen
            metric: Zu optimierende Metrik
            
        Returns:
            Beste Parameter und Ergebnisse
        """
        self.logger.info(f"Starte Parameter-Optimierung für {len(param_ranges)} Parameter")
        
        import itertools
        
        best_score = float('-inf') if metric != 'max_drawdown' else float('inf')
        best_params = {}
        best_results = {}
        
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        total_combinations = len(list(itertools.product(*param_values)))
        self.logger.info(f"Teste {total_combinations} Parameter-Kombinationen")
        
        for i, combination in enumerate(itertools.product(*param_values)):
            current_params = dict(zip(param_names, combination))
            
            # Update Strategie-Parameter
            for param, value in current_params.items():
                if hasattr(self.strategy, param):
                    setattr(self.strategy, param, value)
                elif hasattr(self.strategy, 'parameters') and param in self.strategy.parameters:
                    self.strategy.parameters[param] = value
            
            try:
                # Führe Backtest aus
                results = self.run(data.copy(), symbol)
                current_score = results.get(metric, 0)
                
                # Prüfe ob besser
                is_better = (current_score > best_score) if metric != 'max_drawdown' else (current_score < best_score)
                
                if is_better:
                    best_score = current_score
                    best_params = current_params.copy()
                    best_results = results.copy()
                    # Entferne große DataFrames aus gespeicherten Ergebnissen
                    if 'equity_curve' in best_results:
                        del best_results['equity_curve']
                    if 'trades' in best_results:
                        best_results['trades'] = best_results['trades'][:10]  # Nur erste 10 Trades
                    
                if (i + 1) % max(1, total_combinations // 10) == 0:
                    progress = (i + 1) / total_combinations * 100
                    self.logger.info(f"Fortschritt: {progress:.1f}% - Bester {metric}: {best_score:.4f}")
                    
            except Exception as e:
                self.logger.warning(f"Fehler bei Parameter-Kombination {current_params}: {e}")
                continue
        
        self.logger.info(f"Optimierung abgeschlossen. Beste Parameter: {best_params}")
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'best_results': best_results,
            'optimization_metric': metric,
            'total_combinations_tested': total_combinations
        }