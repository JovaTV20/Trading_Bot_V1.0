"""
Backtester für Trading-Strategien
Simuliert historisches Trading und berechnet Performance-Metriken
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from core.base_strategy import StrategyBase

class Trade:
    """Repräsentiert einen einzelnen Trade"""
    
    def __init__(self, symbol: str, entry_time: datetime, entry_price: float,
                 quantity: float, side: str):
        self.symbol = symbol
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.quantity = quantity
        self.side = side  # 'buy' oder 'sell'
        self.exit_time: Optional[datetime] = None
        self.exit_price: Optional[float] = None
        self.pnl: float = 0.0
        self.commission: float = 0.0
        self.is_closed = False
        
    def close_trade(self, exit_time: datetime, exit_price: float, commission: float = 0.0):
        """Schließt den Trade"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.commission = commission
        self.is_closed = True
        
        # Berechne P&L
        if self.side == 'buy':
            self.pnl = (exit_price - self.entry_price) * self.quantity - commission
        else:  # sell (short)
            self.pnl = (self.entry_price - exit_price) * self.quantity - commission
    
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
            'commission': self.commission,
            'return_pct': (self.pnl / (self.entry_price * self.quantity)) if self.entry_price * self.quantity > 0 else 0,
            'duration_days': (self.exit_time - self.entry_time).days if self.exit_time else None,
            'is_closed': self.is_closed
        }

class Portfolio:
    """Verwaltet Portfolio-Status während Backtest"""
    
    def __init__(self, initial_capital: float, commission: float = 0.0, slippage: float = 0.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.trades: List[Trade] = []
        self.equity_history: List[Dict[str, Any]] = []
        
    def get_position(self, symbol: str) -> float:
        """Gibt aktuelle Position für Symbol zurück"""
        return self.positions.get(symbol, 0.0)
    
    def get_total_equity(self, prices: Dict[str, float]) -> float:
        """Berechnet Gesamtequity"""
        equity = self.cash
        
        for symbol, quantity in self.positions.items():
            if symbol in prices and quantity != 0:
                equity += quantity * prices[symbol]
                
        return equity
    
    def can_trade(self, symbol: str, quantity: float, price: float) -> bool:
        """Prüft ob Trade ausführbar ist"""
        if quantity > 0:  # Buy
            required_cash = quantity * price * (1 + self.commission + self.slippage)
            return self.cash >= required_cash
        else:  # Sell
            current_position = self.get_position(symbol)
            return current_position >= abs(quantity)
    
    def execute_trade(self, symbol: str, quantity: float, price: float, timestamp: datetime) -> Optional[Trade]:
        """Führt Trade aus"""
        if not self.can_trade(symbol, quantity, price):
            return None
            
        # Berücksichtige Slippage
        execution_price = price * (1 + self.slippage if quantity > 0 else 1 - self.slippage)
        
        # Berechne Kosten
        trade_value = abs(quantity) * execution_price
        commission_cost = trade_value * self.commission
        
        # Erstelle Trade-Objekt
        side = 'buy' if quantity > 0 else 'sell'
        trade = Trade(symbol, timestamp, execution_price, abs(quantity), side)
        
        # Update Portfolio
        if quantity > 0:  # Buy
            self.cash -= (trade_value + commission_cost)
            self.positions[symbol] = self.get_position(symbol) + quantity
        else:  # Sell
            self.cash += (trade_value - commission_cost)
            self.positions[symbol] = self.get_position(symbol) + quantity  # quantity ist negativ
            
        # Schließe Trade sofort (für Backtesting)
        trade.close_trade(timestamp, execution_price, commission_cost)
        self.trades.append(trade)
        
        return trade
    
    def update_equity_history(self, timestamp: datetime, prices: Dict[str, float]):
        """Aktualisiert Equity-Historie"""
        equity = self.get_total_equity(prices)
        
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': equity,
            'cash': self.cash,
            'positions_value': equity - self.cash,
            'total_return': (equity - self.initial_capital) / self.initial_capital
        })

class Backtester:
    """
    Backtesting-Engine für Trading-Strategien
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
        Führt Backtest aus
        
        Args:
            data: Historische OHLCV-Daten
            symbol: Trading-Symbol
            
        Returns:
            Dictionary mit Backtest-Ergebnissen
        """
        self.logger.info(f"Starte Backtest für {symbol} mit {len(data)} Datenpunkten")
        
        # Validiere Daten
        if not self.strategy.validate_data(data):
            raise ValueError("Ungültige Eingangsdaten")
            
        # Trainiere Strategie
        self.logger.info("Trainiere Strategie...")
        self.strategy.fit(data)
        
        # Initialisiere Portfolio
        portfolio = Portfolio(self.initial_capital, self.commission, self.slippage)
        
        # Simuliere Trading
        current_position = 0.0
        signals_generated = 0
        trades_executed = 0
        
        self.logger.info("Simuliere Trading...")
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Generiere Signal
            signal = self.strategy.generate_signal(row)
            current_price = row['close']
            
            if signal['action'] != 'hold':
                signals_generated += 1
                
                # Berechne Positionsgröße
                equity = portfolio.get_total_equity({symbol: current_price})
                max_position_value = equity * signal.get('position_size', 0.1)
                target_quantity = max_position_value / current_price
                
                # Bestimme tatsächliche Order-Menge
                current_position = portfolio.get_position(symbol)
                
                if signal['action'] == 'buy' and current_position <= 0:
                    # Kaufsignal - schließe Short und öffne Long
                    quantity = target_quantity + abs(current_position)
                elif signal['action'] == 'sell' and current_position >= 0:
                    # Verkaufssignal - schließe Long und öffne Short
                    quantity = -(target_quantity + current_position)
                else:
                    quantity = 0  # Kein Trade notwendig
                    
                # Führe Trade aus
                if quantity != 0:
                    trade = portfolio.execute_trade(symbol, quantity, current_price, timestamp)
                    if trade:
                        trades_executed += 1
                        self.logger.debug(f"Trade {trades_executed}: {signal['action']} {abs(quantity):.2f} @ {current_price:.2f}")
            
            # Update Equity-Historie
            portfolio.update_equity_history(timestamp, {symbol: current_price})
        
        self.logger.info(f"Backtest abgeschlossen. Signale: {signals_generated}, Trades: {trades_executed}")
        
        # Berechne Ergebnisse
        return self._calculate_results(portfolio, symbol)
    
    def _calculate_results(self, portfolio: Portfolio, symbol: str) -> Dict[str, Any]:
        """Berechnet Backtest-Ergebnisse"""
        
        # Basis-Metriken
        final_equity = portfolio.equity_history[-1]['equity'] if portfolio.equity_history else portfolio.initial_capital
        total_return = (final_equity - portfolio.initial_capital) / portfolio.initial_capital
        
        # Trade-Statistiken
        closed_trades = [t for t in portfolio.trades if t.is_closed]
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Equity Curve DataFrame
        equity_df = pd.DataFrame(portfolio.equity_history)
        if not equity_df.empty:
            equity_df.set_index('timestamp', inplace=True)
        
        # Drawdown-Analyse
        if not equity_df.empty:
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
            max_drawdown = equity_df['drawdown'].min()
            
            # Sharpe Ratio (vereinfacht)
            daily_returns = equity_df['total_return'].diff().fillna(0)
            sharpe_ratio = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        else:
            max_drawdown = 0
            sharpe_ratio = 0
        
        results = {
            # Performance-Metriken
            'initial_capital': portfolio.initial_capital,
            'final_capital': final_equity,
            'total_return': total_return,
            'max_drawdown': abs(max_drawdown),
            'sharpe_ratio': sharpe_ratio,
            
            # Trade-Statistiken
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            
            # Portfolio-Details
            'final_cash': portfolio.cash,
            'final_positions': portfolio.positions,
            'total_commission': sum(t.commission for t in closed_trades),
            
            # Equity Curve
            'equity_curve': equity_df,
            'trades': [t.to_dict() for t in closed_trades]
        }
        
        return results
    
    def optimize_parameters(self, data: pd.DataFrame, symbol: str, 
                          param_ranges: Dict[str, List], metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Optimiert Strategie-Parameter
        
        Args:
            data: Historische Daten
            symbol: Trading-Symbol
            param_ranges: Dictionary mit Parameter-Bereichen
            metric: Zu optimierende Metrik
            
        Returns:
            Dict mit besten Parametern und Ergebnissen
        """
        self.logger.info(f"Starte Parameter-Optimierung für {len(param_ranges)} Parameter")
        
        best_score = float('-inf') if metric != 'max_drawdown' else float('inf')
        best_params = {}
        best_results = {}
        
        # Einfache Grid-Search (kann erweitert werden)
        import itertools
        
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        total_combinations = len(list(itertools.product(*param_values)))
        self.logger.info(f"Teste {total_combinations} Parameter-Kombinationen")
        
        for i, combination in enumerate(itertools.product(*param_values)):
            # Update Strategie-Parameter
            current_params = dict(zip(param_names, combination))
            
            # Aktualisiere Strategie-Konfiguration
            for param, value in current_params.items():
                if param in self.strategy.parameters:
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
                    best_results = results
                    
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Fortschritt: {i + 1}/{total_combinations}, Bester {metric}: {best_score:.4f}")
                    
            except Exception as e:
                self.logger.warning(f"Fehler bei Parameter-Kombination {current_params}: {e}")
                continue
        
        self.logger.info(f"Optimierung abgeschlossen. Beste Parameter: {best_params}")
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'best_results': best_results,
            'optimization_metric': metric
        }