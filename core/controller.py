"""
Trading Controller - Zentrale Steuerung für Live-Trading und Backtesting
"""

import time
import schedule
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging
import json
import traceback
from typing import List


from core.base_strategy import StrategyBase
from core.base_data import DataProviderBase  
from core.base_execution import ExecutionBase
from core.backtester import Backtester
from data_providers.alpaca_data import AlpacaDataProvider
from execution.alpaca_exec import AlpacaExecution
from strategies.ml_strategy import MLStrategy
from utils.email_alerts import EmailAlerter

class TradingController:
    """
    Zentrale Controller-Klasse für das Trading-System
    
    Koordiniert Datenquellen, Strategien und Execution
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialisiert den Controller
        
        Args:
            config: Konfigurationsdictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_running = False
        self.last_signal_time = None
        
        # Initialisiere Komponenten
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialisiert alle System-Komponenten"""
        
        try:
            # Data Provider initialisieren
            self.data_provider = AlpacaDataProvider(self.config.get('data', {}))
            self.logger.info("Alpaca Data Provider initialisiert")
            
            # Execution Provider initialisieren
            self.execution = AlpacaExecution(self.config.get('execution', {}))
            self.logger.info("Alpaca Execution Provider initialisiert")
            
            # Strategie initialisieren
            strategy_name = self.config.get('strategy', {}).get('name', 'ml_strategy')
            if strategy_name == 'ml_strategy':
                self.strategy = MLStrategy(self.config.get('strategy', {}))
            else:
                raise ValueError(f"Unbekannte Strategie: {strategy_name}")
            self.logger.info(f"Strategie {strategy_name} initialisiert")
            
            # Email Alerter initialisieren
            if self.config.get('alerts', {}).get('email_on_error', False):
                self.email_alerter = EmailAlerter()
                self.logger.info("Email-Alerts aktiviert")
            else:
                self.email_alerter = None
                
        except Exception as e:
            self.logger.error(f"Fehler bei Initialisierung: {e}", exc_info=True)
            if self.email_alerter:
                self.email_alerter.send_error_alert("Controller Initialisierung", str(e))
            raise
    
    def run_backtest(self, symbol: str, start_date: str, end_date: str, 
                    initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Führt Backtest aus
        
        Args:
            symbol: Trading-Symbol
            start_date: Startdatum (YYYY-MM-DD)
            end_date: Enddatum (YYYY-MM-DD)  
            initial_capital: Startkapital
            
        Returns:
            Backtest-Ergebnisse
        """
        self.logger.info(f"Starte Backtest: {symbol} ({start_date} - {end_date})")
        
        try:
            # Lade historische Daten
            self.logger.info("Lade historische Daten...")
            data = self.data_provider.get_historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=self.config.get('data', {}).get('timeframe', '1Day')
            )
            
            if data.empty:
                raise ValueError("Keine Daten erhalten")
                
            self.logger.info(f"Daten geladen: {len(data)} Datenpunkte")
            
            # Konfiguriere Backtester
            backtest_config = self.config.get('backtest', {})
            backtester = Backtester(
                strategy=self.strategy,
                initial_capital=initial_capital,
                commission=backtest_config.get('commission', 0.0),
                slippage=backtest_config.get('slippage', 0.001)
            )
            
            # Führe Backtest aus
            results = backtester.run(data, symbol)
            
            # Sende Email-Summary (optional)
            if self.email_alerter and self.config.get('alerts', {}).get('email_daily_summary', False):
                self._send_backtest_summary(symbol, results)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest-Fehler: {e}", exc_info=True)
            if self.email_alerter:
                self.email_alerter.send_error_alert("Backtest", str(e))
            raise
    
    def run_live(self, symbol: str):
        """
        Startet Live-Trading
        
        Args:
            symbol: Trading-Symbol
        """
        self.logger.info(f"Starte Live-Trading für {symbol}")
        
        # Prüfe Marktzeiten
        live_config = self.config.get('live_trading', {})
        market_hours_only = live_config.get('market_hours_only', True)
        
        if market_hours_only and not self.data_provider.is_market_open():
            self.logger.warning("Markt ist geschlossen - warte auf Öffnung")
            
        self.is_running = True
        
        try:
            # Lade initiale Daten und trainiere Strategie
            self._initialize_strategy_for_live(symbol)
            
            # Konfiguriere Scheduler
            update_interval = live_config.get('update_interval', 60)  # Sekunden
            
            def trading_job():
                if not market_hours_only or self.data_provider.is_market_open():
                    self._execute_trading_cycle(symbol)
                else:
                    self.logger.debug("Markt geschlossen - überspringe Trading-Zyklus")
            
            # Schedule Jobs
            if update_interval < 60:
                # Für Intervalle < 1 Minute verwende threading
                self._run_continuous_trading(symbol, update_interval)
            else:
                # Für längere Intervalle verwende schedule
                schedule.every(update_interval).seconds.do(trading_job)
                self._run_scheduled_trading()
                
        except KeyboardInterrupt:
            self.logger.info("Live-Trading durch Benutzer gestoppt")
            self.is_running = False
        except Exception as e:
            self.logger.error(f"Live-Trading Fehler: {e}", exc_info=True)
            if self.email_alerter:
                self.email_alerter.send_error_alert("Live-Trading", str(e))
            self.is_running = False
            raise
    
    def _initialize_strategy_for_live(self, symbol: str):
        """Initialisiert Strategie für Live-Trading"""
        
        self.logger.info("Initialisiere Strategie für Live-Trading...")
        
        # Lade historische Daten für Training
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        training_data = self.data_provider.get_historical(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=self.config.get('data', {}).get('timeframe', '1Day')
        )
        
        if training_data.empty:
            raise ValueError("Keine Trainingsdaten verfügbar")
            
        # Trainiere Strategie
        self.strategy.fit(training_data)
        self.logger.info(f"Strategie trainiert mit {len(training_data)} Datenpunkten")
    
    def _run_scheduled_trading(self):
        """Führt geplantes Trading aus"""
        
        self.logger.info("Starte geplantes Trading...")
        
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Scheduler-Fehler: {e}", exc_info=True)
                if self.email_alerter:
                    self.email_alerter.send_error_alert("Trading Scheduler", str(e))
                time.sleep(10)  # Kurze Pause vor Neustart
    
    def _run_continuous_trading(self, symbol: str, interval: int):
        """Führt kontinuierliches Trading aus"""
        
        self.logger.info(f"Starte kontinuierliches Trading (Intervall: {interval}s)")
        
        while self.is_running:
            try:
                live_config = self.config.get('live_trading', {})
                market_hours_only = live_config.get('market_hours_only', True)
                
                if not market_hours_only or self.data_provider.is_market_open():
                    self._execute_trading_cycle(symbol)
                else:
                    self.logger.debug("Markt geschlossen")
                    
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Trading-Zyklus Fehler: {e}", exc_info=True)
                if self.email_alerter:
                    self.email_alerter.send_error_alert("Trading Cycle", str(e))
                time.sleep(interval * 2)  # Längere Pause bei Fehlern
    
    def _execute_trading_cycle(self, symbol: str):
        """Führt einen einzelnen Trading-Zyklus aus"""
        
        try:
            # Lade aktuelle Daten
            current_data = self.data_provider.get_latest(
                symbol=symbol,
                timeframe=self.config.get('data', {}).get('timeframe', '1Day'),
                limit=1
            )
            
            if current_data.empty:
                self.logger.warning("Keine aktuellen Daten erhalten")
                return
                
            # Generiere Signal
            latest_row = current_data.iloc[-1]
            signal = self.strategy.generate_signal(latest_row)
            
            current_time = datetime.now()
            self.logger.debug(f"Signal generiert: {signal['action']} (Confidence: {signal.get('confidence', 0):.2f})")
            
            # Prüfe ob Trading notwendig
            if signal['action'] == 'hold':
                return
                
            # Prüfe Rate Limiting
            if self._should_skip_trade(signal, current_time):
                return
                
            # Führe Trade aus
            self._execute_trade(symbol, signal, latest_row['close'], current_time)
            
        except Exception as e:
            self.logger.error(f"Trading-Zyklus Fehler: {e}", exc_info=True)
            raise
    
    def _should_skip_trade(self, signal: Dict[str, Any], current_time: datetime) -> bool:
        """Prüft ob Trade übersprungen werden sollte"""
        
        # Rate Limiting
        max_daily_trades = self.config.get('risk_management', {}).get('max_daily_trades', 5)
        if max_daily_trades > 0:
            # Zähle heutige Trades (vereinfacht)
            # In einer echten Implementation würde man eine Datenbank verwenden
            pass
            
        # Mindest-Confidence
        min_confidence = self.config.get('strategy', {}).get('parameters', {}).get('prediction_threshold', 0.6)
        if signal.get('confidence', 0) < min_confidence:
            self.logger.debug(f"Signal-Confidence zu niedrig: {signal.get('confidence', 0):.2f} < {min_confidence}")
            return True
            
        # Zeitlicher Abstand zum letzten Signal
        if self.last_signal_time:
            time_diff = (current_time - self.last_signal_time).total_seconds()
            if time_diff < 300:  # 5 Minuten Mindestabstand
                self.logger.debug("Zu kurzer Abstand zum letzten Signal")
                return True
                
        return False
    
    def _execute_trade(self, symbol: str, signal: Dict[str, Any], current_price: float, timestamp: datetime):
        """Führt einen Trade aus"""
        
        try:
            # Hole Account-Info
            account_info = self.execution.get_account_info()
            buying_power = float(account_info.get('buying_power', 0))
            
            # Berechne Positionsgröße
            position_size = signal.get('position_size', 0.1)
            trade_value = buying_power * position_size
            quantity = self.execution.calculate_order_quantity(symbol, buying_power, position_size, current_price)
            
            if quantity <= 0:
                self.logger.warning("Berechnete Quantity <= 0")
                return
                
            # Hole aktuelle Position
            current_position = self.execution.get_position_for_symbol(symbol)
            current_qty = float(current_position.get('qty', 0)) if current_position else 0
            
            # Bestimme Order-Details
            side = signal['action']  # 'buy' or 'sell'
            
            # Für einfache Implementierung: schließe alte Position und öffne neue
            orders_placed = []
            
            # Schließe bestehende Position falls vorhanden
            if current_qty != 0:
                close_side = 'sell' if current_qty > 0 else 'buy'
                close_order = self.execution.place_order(
                    symbol=symbol,
                    side=close_side,
                    quantity=abs(current_qty),
                    order_type='market'
                )
                orders_placed.append(close_order)
                self.logger.info(f"Position geschlossen: {close_side} {abs(current_qty)} {symbol}")
            
            # Öffne neue Position
            if side in ['buy', 'sell']:
                # Berechne Stop-Loss und Take-Profit
                stop_loss = signal.get('stop_loss') or self.strategy.get_stop_loss(current_price, side)
                take_profit = signal.get('take_profit') or self.strategy.get_take_profit(current_price, side)
                
                # Platziere Hauptorder
                main_order = self.execution.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type='market'
                )
                orders_placed.append(main_order)
                
                self.logger.info(f"Order platziert: {side} {quantity} {symbol} @ {current_price:.2f}")
                
                # Platziere Stop-Loss/Take-Profit (falls unterstützt)
                if stop_loss or take_profit:
                    bracket_orders = self.execution.create_bracket_order(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    orders_placed.extend(bracket_orders[1:])  # Erste Order ist Hauptorder
                
            # Update letzter Signal-Zeitpunkt
            self.last_signal_time = timestamp
            
            # Sende Email-Alert (optional)
            if self.email_alerter and self.config.get('alerts', {}).get('email_on_trade', False):
                self._send_trade_alert(symbol, signal, orders_placed, current_price)
                
        except Exception as e:
            self.logger.error(f"Trade-Ausführung Fehler: {e}", exc_info=True)
            if self.email_alerter:
                self.email_alerter.send_error_alert("Trade Execution", str(e))
            raise
    
    def _send_trade_alert(self, symbol: str, signal: Dict[str, Any], orders: List[Dict], price: float):
        """Sendet Trade-Alert per Email"""
        
        try:
            subject = f"Trade Alert: {signal['action'].upper()} {symbol}"
            
            message = f"""
Trading Alert - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Symbol: {symbol}
Action: {signal['action'].upper()}
Price: ${price:.2f}
Confidence: {signal.get('confidence', 0):.2%}
Position Size: {signal.get('position_size', 0):.2%}

Orders placed: {len(orders)}
            """
            
            for i, order in enumerate(orders, 1):
                message += f"\nOrder {i}: {order.get('side', 'N/A')} {order.get('quantity', 'N/A')} @ {order.get('status', 'N/A')}"
            
            self.email_alerter.send_alert(subject, message)
            
        except Exception as e:
            self.logger.warning(f"Email-Alert Fehler: {e}")
    
    def _send_backtest_summary(self, symbol: str, results: Dict[str, Any]):
        """Sendet Backtest-Summary per Email"""
        
        try:
            subject = f"Backtest Summary: {symbol}"
            
            message = f"""
Backtest Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Symbol: {symbol}
Initial Capital: ${results['initial_capital']:,.2f}
Final Capital: ${results['final_capital']:,.2f}
Total Return: {results['total_return']:.2%}

Performance Metrics:
- Total Trades: {results['total_trades']}
- Win Rate: {results['win_rate']:.2%}
- Max Drawdown: {results['max_drawdown']:.2%}
- Sharpe Ratio: {results['sharpe_ratio']:.2f}
- Profit Factor: {results['profit_factor']:.2f}

Trade Statistics:
- Winning Trades: {results['winning_trades']}
- Losing Trades: {results['losing_trades']}
- Average Win: ${results['avg_win']:.2f}
- Average Loss: ${results['avg_loss']:.2f}
            """
            
            self.email_alerter.send_alert(subject, message)
            
        except Exception as e:
            self.logger.warning(f"Email-Summary Fehler: {e}")
    
    def stop(self):
        """Stoppt den Controller"""
        self.logger.info("Stoppe Trading Controller...")
        self.is_running = False
        
        # Schließe alle offenen Positionen (optional)
        try:
            positions = self.execution.get_positions()
            for position in positions:
                if float(position.get('qty', 0)) != 0:
                    symbol = position['symbol']
                    self.logger.info(f"Schließe Position: {symbol}")
                    self.execution.close_position(symbol)
        except Exception as e:
            self.logger.error(f"Fehler beim Schließen der Positionen: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Gibt aktuellen Status zurück"""
        
        try:
            account_info = self.execution.get_account_info()
            positions = self.execution.get_positions()
            
            return {
                'is_running': self.is_running,
                'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
                'account_value': account_info.get('equity', 0),
                'buying_power': account_info.get('buying_power', 0),
                'positions_count': len(positions),
                'market_open': self.data_provider.is_market_open(),
                'strategy_fitted': self.strategy.is_fitted if hasattr(self.strategy, 'is_fitted') else False
            }
            
        except Exception as e:
            self.logger.error(f"Status-Abfrage Fehler: {e}")
            return {
                'is_running': self.is_running,
                'error': str(e)
            }