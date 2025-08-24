"""
Trading Controller - ERWEITERT für IntelligentMLStrategy
Zentrale Steuerung für Live-Trading und Backtesting mit intelligenter ML-Strategie
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

# ERWEITERT: Import der intelligenten ML-Strategie
try:
    from strategies.ml_strategy import IntelligentMLStrategy
    INTELLIGENT_ML_AVAILABLE = True
except ImportError:
    try:
        # Fallback für alte Strategie
        from strategies.ml_strategy import MLStrategy as IntelligentMLStrategy
        INTELLIGENT_ML_AVAILABLE = True
    except ImportError:
        INTELLIGENT_ML_AVAILABLE = False
        logging.error("Keine ML-Strategie verfügbar!")

from utils.email_alerts import EmailAlerter

class TradingController:
    """
    ERWEITERTE Trading Controller-Klasse
    
    Koordiniert Datenquellen, Strategien und Execution mit:
    - Intelligenter ML-Strategie Integration
    - Online Learning Support
    - Regime-Erkennung
    - Adaptive Parameter-Anpassung
    - Erweiterte Performance-Metriken
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialisiert den erweiterten Controller
        
        Args:
            config: Konfigurationsdictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_running = False
        self.last_signal_time = None
        
        # ERWEITERT: Intelligent Strategy State
        self.strategy_performance = {}
        self.regime_history = []
        self.learning_metrics = {}
        self.adaptive_params = {}
        
        # Initialisiere Komponenten
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialisiert alle System-Komponenten mit intelligenter Strategie-Unterstützung"""
        
        try:
            # Data Provider initialisieren
            self.data_provider = AlpacaDataProvider(self.config.get('data', {}))
            self.logger.info("Alpaca Data Provider initialisiert")
            
            # Execution Provider initialisieren
            self.execution = AlpacaExecution(self.config.get('execution', {}))
            self.logger.info("Alpaca Execution Provider initialisiert")
            
            # ERWEITERT: Intelligente Strategie initialisieren
            self._initialize_intelligent_strategy()
            
            # Email Alerter initialisieren
            if self.config.get('alerts', {}).get('email_on_error', False):
                self.email_alerter = EmailAlerter()
                self.logger.info("Email-Alerts aktiviert")
            else:
                self.email_alerter = None
                
        except Exception as e:
            self.logger.error(f"Fehler bei Initialisierung: {e}", exc_info=True)
            if hasattr(self, 'email_alerter') and self.email_alerter:
                self.email_alerter.send_error_alert("Controller Initialisierung", str(e))
            raise
    
    def _initialize_intelligent_strategy(self):
        """ERWEITERT: Initialisiert intelligente ML-Strategie"""
        
        strategy_config = self.config.get('strategy', {})
        strategy_name = strategy_config.get('name', 'ml_strategy')
        
        self.logger.info(f"Initialisiere Strategie: {strategy_name}")
        
        try:
            if not INTELLIGENT_ML_AVAILABLE:
                raise ImportError("IntelligentMLStrategy nicht verfügbar")
            
            # Erstelle intelligente Strategie
            if strategy_name in ['intelligent_ml_strategy', 'ml_strategy']:
                self.strategy = IntelligentMLStrategy(strategy_config)
                self.logger.info("✅ IntelligentMLStrategy erfolgreich initialisiert")
                
                # ERWEITERT: Strategie-spezifische Einstellungen
                self._setup_intelligent_strategy_monitoring()
                
            else:
                # Fallback
                self.strategy = IntelligentMLStrategy(strategy_config)
                self.logger.warning(f"⚠️ Unbekannte Strategie '{strategy_name}', verwende IntelligentMLStrategy")
            
            # Validiere Strategie-Interface
            required_methods = ['fit', 'generate_signal', 'get_info']
            for method in required_methods:
                if not hasattr(self.strategy, method):
                    raise ValueError(f"Strategie implementiert nicht die erforderliche Methode: {method}")
            
            self.logger.info("✅ Strategie-Interface validiert")
            
        except Exception as e:
            self.logger.error(f"❌ Strategie-Initialisierung fehlgeschlagen: {e}")
            raise
    
    def _setup_intelligent_strategy_monitoring(self):
        """ERWEITERT: Setup für intelligente Strategie-Überwachung"""
        
        try:
            # Performance Tracking initialisieren
            self.strategy_performance = {
                'signals_generated': 0,
                'signals_executed': 0,
                'confidence_history': [],
                'regime_changes': 0,
                'online_learning_updates': 0,
                'model_retrains': 0
            }
            
            # Regime-Historie initialisieren
            self.regime_history = []
            
            # Learning-Metriken initialisieren
            self.learning_metrics = {
                'prediction_accuracy': 0.0,
                'confidence_calibration': 0.0,
                'regime_detection_accuracy': 0.0,
                'adaptation_speed': 0.0
            }
            
            # Adaptive Parameter initialisieren
            self.adaptive_params = {
                'confidence_threshold': self.strategy.prediction_threshold,
                'position_sizing_multiplier': 1.0,
                'regime_sensitivity': 1.0,
                'learning_rate_adjustment': 1.0
            }
            
            self.logger.info("✅ Intelligente Strategie-Überwachung eingerichtet")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Intelligent Live-Trading Cleanup fehlgeschlagen: {e}")
    
    # ERWEITERTE HILFSMETHODEN - FORTSETZUNG
    
    def _load_enhanced_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Lädt erweiterte historische Daten mit Qualitätsprüfung"""
        try:
            data = self.data_provider.get_historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=self.config.get('data', {}).get('timeframe', '1Day')
            )
            
            if data.empty:
                # Paper Account Fallback-Strategie
                self.logger.warning("⚠️ Keine Daten für gewünschten Zeitraum, versuche älteren Zeitraum...")
                
                end_date_corrected = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                start_date_corrected = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                
                data = self.data_provider.get_historical(
                    symbol=symbol,
                    start_date=start_date_corrected,
                    end_date=end_date_corrected,
                    timeframe='1Day'
                )
                
                if not data.empty:
                    self.logger.info(f"✅ Fallback-Daten geladen: {start_date_corrected} bis {end_date_corrected}")
            
            # Erweiterte Datenqualitätsprüfung
            if not data.empty:
                self._validate_and_enhance_data(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Fehler beim Laden erweiterte historischer Daten: {e}")
            return pd.DataFrame()
    
    def _validate_and_enhance_data(self, data: pd.DataFrame):
        """Validiert und erweitert Datenqualität"""
        try:
            # Prüfe auf fehlende Daten
            missing_data = data.isnull().sum()
            if missing_data.any():
                self.logger.warning(f"⚠️ Fehlende Daten gefunden: {missing_data[missing_data > 0].to_dict()}")
            
            # Prüfe auf unrealistische Preise
            for col in ['open', 'high', 'low', 'close']:
                if col in data.columns:
                    if (data[col] <= 0).any():
                        self.logger.warning(f"⚠️ Negative oder null Preise in {col} gefunden")
                    
                    # Prüfe auf extreme Sprünge
                    price_changes = data[col].pct_change().abs()
                    extreme_changes = price_changes > 0.5  # > 50% Änderung
                    if extreme_changes.any():
                        self.logger.warning(f"⚠️ Extreme Preissprünge in {col}: {extreme_changes.sum()} Fälle")
            
            # Füge erweiterte technische Indikatoren hinzu (falls nicht vorhanden)
            if hasattr(self.data_provider, 'calculate_technical_indicators'):
                enhanced_data = self.data_provider.calculate_technical_indicators(data)
                return enhanced_data
                
        except Exception as e:
            self.logger.warning(f"⚠️ Datenvalidierung/Enhancement fehlgeschlagen: {e}")
    
    def _validate_backtest_inputs(self, symbol: str, start_date: str, end_date: str, initial_capital: float):
        """Erweiterte Backtest-Eingaben-Validierung"""
        # Symbol Validation
        if not symbol or len(symbol) < 1 or len(symbol) > 10:
            raise ValueError(f"Ungültiges Symbol: {symbol}")
        
        # Datum Validation mit Paper Account Berücksichtigung
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"Ungültiges Datumsformat: {e}")
        
        if start_dt >= end_dt:
            raise ValueError("Startdatum muss vor Enddatum liegen")
        
        # Paper Account spezifische Warnung
        max_end_date = datetime.now() - timedelta(days=15)
        if end_dt > max_end_date:
            self.logger.warning(f"⚠️ Enddatum möglicherweise zu aktuell für Paper Account ({end_date})")
        
        # Capital Validation
        if initial_capital <= 0 or initial_capital > 10_000_000:
            raise ValueError(f"Ungültiges Startkapital: ${initial_capital:,.2f}")
    
    def _get_enhanced_intelligent_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Lädt erweiterte intelligente Marktdaten"""
        try:
            # Bestimme optimale Datenmenge basierend auf Strategie
            if hasattr(self.strategy, 'lookback_period'):
                limit = max(10, self.strategy.lookback_period // 5)
            else:
                limit = 20
            
            current_data = self.data_provider.get_latest(
                symbol=symbol,
                timeframe=self.config.get('data', {}).get('timeframe', '1Day'),
                limit=limit
            )
            
            if current_data.empty:
                return None
            
            # Erweitere um technische Indikatoren falls verfügbar
            if hasattr(self.data_provider, 'calculate_technical_indicators'):
                current_data = self.data_provider.calculate_technical_indicators(current_data)
            
            # Zusätzliche Echtzeit-Indikatoren
            current_data = self._add_realtime_intelligence_indicators(current_data)
            
            return current_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Fehler beim Laden intelligenter Marktdaten: {e}")
            return None
    
    def _add_realtime_intelligence_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fügt Echtzeit-Intelligence-Indikatoren hinzu"""
        try:
            # Momentum-basierte Indikatoren
            data['price_acceleration'] = data['close'].pct_change().diff()
            data['volume_price_ratio'] = data.get('volume', 1) / data['close']
            
            # Volatilitäts-Indikatoren
            returns = data['close'].pct_change()
            data['realized_volatility'] = returns.rolling(min(10, len(returns))).std()
            data['volatility_persistence'] = data['realized_volatility'].rolling(min(5, len(data))).corr(
                data['realized_volatility'].shift(1)
            )
            
            # Trend-Stärke-Indikatoren
            if len(data) >= 10:
                short_ma = data['close'].rolling(5).mean()
                long_ma = data['close'].rolling(10).mean()
                data['trend_strength'] = (short_ma - long_ma) / long_ma
                data['trend_acceleration'] = data['trend_strength'].diff()
            
            # Fülle NaN-Werte
            data = data.fillna(method='ffill').fillna(0)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Echtzeit-Intelligence-Indikatoren fehlgeschlagen: {e}")
        
        return data
    
    def _check_portfolio_constraints(self) -> Dict[str, Any]:
        """Prüft Portfolio-Constraints für Trading-Entscheidungen"""
        result = {
            'can_trade': True,
            'reason': 'all_constraints_satisfied'
        }
        
        try:
            # Hole Account-Informationen
            account_info = self.execution.get_account_info()
            
            # Buying Power Check
            buying_power = float(account_info.get('buying_power', 0))
            min_buying_power = 1000  # Minimum $1000
            
            if buying_power < min_buying_power:
                result['can_trade'] = False
                result['reason'] = f'insufficient_buying_power: ${buying_power:.2f}'
                return result
            
            # Daily Trade Limit Check
            max_daily_trades = self.config.get('risk_management', {}).get('max_daily_trades', 5)
            if max_daily_trades > 0:
                # Vereinfachte Tages-Trade-Zählung (in Realität würde man Datenbank verwenden)
                daily_trades_count = getattr(self, 'daily_trades_count', 0)
                if daily_trades_count >= max_daily_trades:
                    result['can_trade'] = False
                    result['reason'] = f'daily_trade_limit_reached: {daily_trades_count}/{max_daily_trades}'
                    return result
            
            # Portfolio Value vs Max Drawdown Check
            current_equity = float(account_info.get('equity', 0))
            initial_equity = getattr(self, 'initial_equity', current_equity)
            
            if initial_equity > 0:
                current_drawdown = (initial_equity - current_equity) / initial_equity
                max_allowed_drawdown = self.config.get('risk_management', {}).get('max_drawdown_pct', 0.15)
                
                if current_drawdown > max_allowed_drawdown:
                    result['can_trade'] = False
                    result['reason'] = f'max_drawdown_exceeded: {current_drawdown:.2%} > {max_allowed_drawdown:.2%}'
                    return result
            
            # Position Concentration Check
            positions = self.execution.get_positions()
            if positions:
                total_position_value = sum(float(pos.get('market_value', 0)) for pos in positions)
                portfolio_value = float(account_info.get('portfolio_value', 1))
                
                position_concentration = total_position_value / portfolio_value if portfolio_value > 0 else 0
                max_concentration = 0.95  # Max 95% invested
                
                if position_concentration > max_concentration:
                    result['can_trade'] = False
                    result['reason'] = f'high_concentration: {position_concentration:.2%}'
                    return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ Portfolio Constraint Check fehlgeschlagen: {e}")
            result['can_trade'] = False
            result['reason'] = f'constraint_check_error: {e}'
        
        return result
    
    def _run_intelligent_continuous_trading(self, symbol: str, interval: int):
        """Führt intelligentes kontinuierliches Trading aus"""
        
        self.logger.info(f"⚡ Starte intelligentes kontinuierliches Trading (Intervall: {interval}s)")
        
        def intelligent_trading_worker():
            while not getattr(self, 'stop_event', threading.Event()).is_set() and self.is_running:
                try:
                    live_config = self.config.get('live_trading', {})
                    market_hours_only = live_config.get('market_hours_only', True)
                    
                    if not market_hours_only or self.data_provider.is_market_open():
                        self._execute_intelligent_trading_cycle(symbol)
                        self._update_live_monitoring()
                        
                        # Erweiterte Monitoring-Aufgaben alle N Zyklen
                        if hasattr(self, 'monitoring_metrics'):
                            cycles = self.monitoring_metrics.get('cycles_completed', 0)
                            
                            if cycles % 10 == 0:  # Alle 10 Zyklen
                                self._monitor_strategy_health()
                            
                            if cycles % 30 == 0:  # Alle 30 Zyklen
                                self._adaptive_parameter_update()
                    else:
                        self.logger.debug("Markt geschlossen - intelligenter Modus wartend")
                    
                    getattr(self, 'stop_event', threading.Event()).wait(interval)
                    
                except Exception as e:
                    self.logger.error(f"❌ Fehler im intelligenten Trading-Worker: {e}", exc_info=True)
                    getattr(self, 'stop_event', threading.Event()).wait(interval * 2)
        
        # Starte intelligenten Trading in separatem Thread
        self.trading_thread = threading.Thread(target=intelligent_trading_worker, daemon=True)
        self.stop_event = threading.Event()
        self.trading_thread.start()
        
        # Warte auf Thread oder Stop-Signal
        try:
            while self.trading_thread.is_alive() and self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("⏹️ KeyboardInterrupt für intelligentes Trading empfangen")
            self.stop_event.set()
            
        if self.trading_thread.is_alive():
            self.trading_thread.join(timeout=5)
    
    def _run_scheduled_intelligent_trading(self):
        """Führt geplantes intelligentes Trading aus"""
        
        self.logger.info("📅 Starte geplantes intelligentes Trading...")
        
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(1)
                
                # Erweiterte System-Health-Checks
                if hasattr(self, 'monitoring_metrics'):
                    last_check = self.monitoring_metrics.get('last_health_check', datetime.now())
                    if (datetime.now() - last_check).total_seconds() > 3600:  # 1 Stunde
                        self._comprehensive_health_check()
                        
            except Exception as e:
                self.logger.error(f"❌ Intelligenter Scheduler-Fehler: {e}", exc_info=True)
                if self.email_alerter:
                    self.email_alerter.send_error_alert("Intelligent Trading Scheduler", str(e))
                time.sleep(10)
    
    def _comprehensive_health_check(self):
        """Umfassende System-Gesundheitsprüfung"""
        try:
            health_report = {
                'timestamp': datetime.now(),
                'overall_status': 'healthy',
                'components': {},
                'recommendations': []
            }
            
            # Data Provider Health
            try:
                test_data = self.data_provider.get_latest('AAPL', limit=1)
                health_report['components']['data_provider'] = 'healthy' if not test_data.empty else 'degraded'
            except Exception as e:
                health_report['components']['data_provider'] = 'unhealthy'
                health_report['overall_status'] = 'degraded'
            
            # Execution Provider Health  
            try:
                account_info = self.execution.get_account_info()
                health_report['components']['execution_provider'] = 'healthy' if 'error' not in account_info else 'unhealthy'
            except Exception as e:
                health_report['components']['execution_provider'] = 'unhealthy'
                health_report['overall_status'] = 'degraded'
            
            # Strategy Health
            if hasattr(self, 'confidence_performance_tracking') and self.confidence_performance_tracking:
                recent_performance = self.confidence_performance_tracking[-20:]
                success_rate = sum(1 for t in recent_performance if t['success']) / len(recent_performance)
                
                if success_rate > 0.6:
                    health_report['components']['strategy'] = 'healthy'
                elif success_rate > 0.4:
                    health_report['components']['strategy'] = 'degraded'
                    health_report['recommendations'].append('monitor_strategy_performance')
                else:
                    health_report['components']['strategy'] = 'unhealthy'
                    health_report['overall_status'] = 'unhealthy'
                    health_report['recommendations'].append('urgent_strategy_review_needed')
            else:
                health_report['components']['strategy'] = 'unknown'
            
            # Memory Usage Check
            try:
                import psutil
                memory_usage = psutil.virtual_memory().percent
                if memory_usage > 90:
                    health_report['components']['system_resources'] = 'critical'
                    health_report['overall_status'] = 'critical'
                    health_report['recommendations'].append('high_memory_usage_detected')
                elif memory_usage > 80:
                    health_report['components']['system_resources'] = 'warning'
                    health_report['recommendations'].append('monitor_memory_usage')
                else:
                    health_report['components']['system_resources'] = 'healthy'
            except ImportError:
                health_report['components']['system_resources'] = 'unknown'
            
            # Log Health Report
            status = health_report['overall_status']
            if status == 'healthy':
                self.logger.info("💚 Comprehensive Health Check: System gesund")
            elif status == 'degraded':
                self.logger.warning(f"🟡 Comprehensive Health Check: System beeinträchtigt - {health_report['recommendations']}")
            else:
                self.logger.error(f"🔴 Comprehensive Health Check: System ungesund - {health_report['recommendations']}")
                
                # Sende Critical Health Alert
                if self.email_alerter:
                    self.email_alerter.send_error_alert(
                        "Critical System Health Alert",
                        f"System Status: {status}\nIssues: {health_report['components']}\nRecommendations: {health_report['recommendations']}"
                    )
            
            # Update Monitoring
            if hasattr(self, 'monitoring_metrics'):
                self.monitoring_metrics['last_health_check'] = datetime.now()
                self.monitoring_metrics['last_health_status'] = status
                
        except Exception as e:
            self.logger.error(f"❌ Comprehensive Health Check fehlgeschlagen: {e}")
    
    # BENACHRICHTIGUNGS- UND LOGGING-METHODEN
    
    def _send_intelligent_trade_alert(self, symbol: str, signal: Dict[str, Any], 
                                    orders: List[Dict], price: float,
                                    execution_decision: Dict[str, Any],
                                    market_context: Dict[str, Any]):
        """Sendet erweiterte intelligente Trade-Benachrichtigung"""
        
        try:
            # Erweiterte Subject-Line
            confidence = signal.get('confidence', 0)
            decision_conf = execution_decision.get('confidence_in_decision', 0)
            regime = market_context.get('regime', {}).get('current', 'unknown')
            
            subject = f"🧠 Intelligent Trade: {signal['action'].upper()} {symbol} (Conf: {confidence:.1%}, Regime: {regime})"
            
            # Erweiterte Nachricht
            message = f"""
🤖 INTELLIGENT TRADING ALERT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

TRADE DETAILS:
• Symbol: {symbol}
• Action: {signal['action'].upper()}
• Price: ${price:.2f}
• Position Size: {signal.get('position_size', 0):.2%}

INTELLIGENCE METRICS:
• Signal Confidence: {confidence:.2%}
• Decision Confidence: {decision_conf:.2%}
• Market Regime: {regime}
• Signal Quality: {execution_decision.get('signal_analysis', {}).get('signal_quality', 'unknown')}

MARKET CONTEXT:
• Volatility Regime: {market_context.get('volatility', {}).get('regime', 'unknown')}
• Momentum Direction: {market_context.get('momentum', {}).get('direction', 'unknown')}
• Momentum Strength: {market_context.get('momentum', {}).get('strength', 0):.2%}

EXECUTION DETAILS:
• Decision Reason: {execution_decision.get('reason', 'unknown')}
• Adjustments Applied: {', '.join(execution_decision.get('adjustments', {}).keys()) or 'None'}
• Orders Placed: {len(orders)}

STRATEGY STATUS:
• Model Performance: {getattr(self.strategy, 'model_performance', {}).get('main', {}).get('cv_mean', 'N/A')}
• Online Learning: {'Active' if hasattr(self.strategy, 'data_buffer') else 'Inactive'}
• Last Retrain: {getattr(self.strategy, 'last_retrain_date', 'Never')}

This is an automated intelligent trading alert from your TradingBot.
            """
            
            self.email_alerter.send_alert(subject, message.strip())
            
        except Exception as e:
            self.logger.warning(f"⚠️ Intelligente Trade-Benachrichtigung fehlgeschlagen: {e}")
    
    def _send_regime_change_alert(self, old_regime: str, new_regime: str, symbol: str):
        """Sendet Regime-Wechsel-Benachrichtigung"""
        
        try:
            subject = f"📊 Market Regime Change: {old_regime.upper()} → {new_regime.upper()}"
            
            message = f"""
🌊 MARKET REGIME CHANGE DETECTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

REGIME TRANSITION:
• From: {old_regime.upper()}
• To: {new_regime.upper()}
• Symbol: {symbol}
• Detection Time: {datetime.now().strftime('%H:%M:%S')}

STRATEGY IMPLICATIONS:
• Position Sizing: {'More Conservative' if new_regime == 'bear' else 'Standard' if new_regime == 'sideways' else 'More Aggressive'}
• Risk Management: {'Tighter' if new_regime in ['bear', 'high_vol'] else 'Standard'}
• Signal Threshold: {'Higher' if new_regime == 'bear' else 'Standard'}

ADAPTIVE ACTIONS TAKEN:
• Strategy parameters automatically adjusted
• Risk management rules updated
• Position sizing algorithms adapted

RECENT REGIME HISTORY:
            """
            
            # Füge letzte 5 Regime-Wechsel hinzu
            for entry in self.regime_history[-5:]:
                timestamp = entry['timestamp'].strftime('%Y-%m-%d %H:%M')
                if 'old_regime' in entry:
                    message += f"• {timestamp}: {entry['old_regime']} → {entry['new_regime']}\n"
                else:
                    message += f"• {timestamp}: Detected {entry['regime']}\n"
            
            message += "\nThis is an automated regime change notification from your Intelligent TradingBot."
            
            self.email_alerter.send_alert(subject, message.strip())
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime-Change-Benachrichtigung fehlgeschlagen: {e}")
    
    def _log_intelligent_backtest_summary(self, results: Dict[str, Any]):
        """Loggt erweiterte intelligente Backtest-Zusammenfassung"""
        
        try:
            self.logger.info("="*60)
            self.logger.info("🧠 INTELLIGENT BACKTEST RESULTS")
            self.logger.info("="*60)
            
            # Standard-Metriken
            self.logger.info(f"📊 Standard Metrics:")
            self.logger.info(f"   • Total Return: {results.get('total_return', 0):.2%}")
            self.logger.info(f"   • Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
            self.logger.info(f"   • Max Drawdown: {results.get('max_drawdown', 0):.2%}")
            self.logger.info(f"   • Win Rate: {results.get('win_rate', 0):.2%}")
            self.logger.info(f"   • Total Trades: {results.get('total_trades', 0)}")
            
            # Intelligente Metriken
            if 'ml_performance' in results:
                ml_perf = results['ml_performance']
                self.logger.info(f"🤖 ML Performance:")
                model_scores = ml_perf.get('model_scores', {})
                for model_name, scores in model_scores.items():
                    cv_score = scores.get('cv_mean', scores.get('train_score', 0))
                    self.logger.info(f"   • {model_name} Model: {cv_score:.3f}")
            
            # Regime-Analyse
            if 'regime_analysis' in results:
                regime_perf = results['regime_analysis']
                self.logger.info(f"🌊 Regime Performance:")
                for regime, perf in regime_perf.items():
                    if isinstance(perf, dict) and 'return' in perf:
                        self.logger.info(f"   • {regime}: {perf['return']:.2%} ({perf.get('trades', 0)} trades)")
            
            # Online Learning
            if 'online_learning' in results:
                ol = results['online_learning']
                self.logger.info(f"🔄 Online Learning:")
                self.logger.info(f"   • Adaptation Events: {ol.get('adaptation_events', 0)}")
                self.logger.info(f"   • Concept Drift Detected: {ol.get('concept_drift_detection', False)}")
            
            # Confidence-Analyse
            if 'confidence_analysis' in results:
                conf = results['confidence_analysis']
                optimal_threshold = conf.get('optimal_confidence_threshold', 0)
                self.logger.info(f"🎯 Confidence Analysis:")
                self.logger.info(f"   • Optimal Threshold: {optimal_threshold:.3f}")
                self.logger.info(f"   • Confidence-Accuracy Correlation: {conf.get('confidence_accuracy_correlation', 0):.3f}
• Distribution Quality: {conf.get('confidence_distribution', 'balanced')}
                """
            
            if 'forward_predictions' in results:
                forward = results['forward_predictions']
                message += f"""

🔮 FORWARD PREDICTIONS:
• Next Regime Prediction: {forward.get('next_regime_prediction', 'Unknown')}
• Performance Forecast: {forward.get('performance_forecast', 'Stable')}
• Strategy Recommendations: {', '.join(forward.get('recommended_adjustments', ['None']))}
                """
            
            if 'intelligent_risk_metrics' in results:
                risk = results['intelligent_risk_metrics']
                message += f"""

🛡️ INTELLIGENT RISK ANALYSIS:
• Regime-Adjusted Sharpe: {risk.get('regime_adjusted_sharpe', 0):.3f}
• Adaptive Max Drawdown: {risk.get('adaptive_max_drawdown', 0):.2%}
• Dynamic VaR (95%): {risk.get('dynamic_var', 0):.2%}
• Cross-Regime Stability: {risk.get('cross_regime_stability', 'Stable')}
                """
            
            message += f"""

📋 STRATEGY DETAILS:
• Lookback Period: {getattr(self.strategy, 'lookback_period', 'N/A')}
• Prediction Threshold: {getattr(self.strategy, 'prediction_threshold', 'N/A')}
• Market Regime: {getattr(self.strategy, 'market_regime', 'Unknown')}
• Online Learning: {'Active' if hasattr(self.strategy, 'data_buffer') else 'Inactive'}

This is your intelligent backtest summary with advanced ML analytics.
            """
            
            self.email_alerter.send_alert(subject, message.strip())
            
        except Exception as e:
            self.logger.warning(f"⚠️ Intelligent Backtest Email-Summary fehlgeschlagen: {e}")
    
    def _log_health_summary(self):
        """Loggt System-Gesundheits-Zusammenfassung"""
        
        try:
            if hasattr(self, 'monitoring_metrics'):
                metrics = self.monitoring_metrics
                
                uptime = datetime.now() - getattr(self, 'monitoring_start_time', datetime.now())
                
                self.logger.info("💡 SYSTEM HEALTH SUMMARY:")
                self.logger.info(f"   • Uptime: {uptime}")
                self.logger.info(f"   • Cycles Completed: {metrics.get('cycles_completed', 0)}")
                self.logger.info(f"   • Signals Generated: {self.strategy_performance.get('signals_generated', 0)}")
                self.logger.info(f"   • Trades Executed: {self.strategy_performance.get('signals_executed', 0)}")
                
                if self.strategy_performance.get('signals_generated', 0) > 0:
                    execution_rate = (self.strategy_performance.get('signals_executed', 0) / 
                                    self.strategy_performance.get('signals_generated', 1))
                    self.logger.info(f"   • Execution Rate: {execution_rate:.1%}")
                
                # Confidence-Performance
                if hasattr(self, 'confidence_performance_tracking') and self.confidence_performance_tracking:
                    recent_performance = self.confidence_performance_tracking[-20:]
                    success_rate = sum(1 for t in recent_performance if t['success']) / len(recent_performance)
                    avg_confidence = sum(t['signal_confidence'] for t in recent_performance) / len(recent_performance)
                    
                    self.logger.info(f"   • Recent Success Rate: {success_rate:.1%}")
                    self.logger.info(f"   • Average Confidence: {avg_confidence:.2%}")
                
                # Regime-Aktivität
                regime_changes = self.strategy_performance.get('regime_changes', 0)
                self.logger.info(f"   • Regime Changes: {regime_changes}")
                
                # Online Learning-Aktivität
                ol_updates = self.strategy_performance.get('online_learning_updates', 0)
                if ol_updates > 0:
                    self.logger.info(f"   • Online Learning Updates: {ol_updates}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Health Summary Logging fehlgeschlagen: {e}")
    
    # ANALYSE-HILFSMETHODEN (für Backtest-Enhancement)
    
    def _check_data_consistency(self, data: pd.DataFrame) -> float:
        """Prüft Datenkonsistenz"""
        try:
            consistency_score = 1.0
            
            # OHLC-Konsistenz
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                # High sollte >= Open, Close, Low sein
                high_violations = (
                    (data['high'] < data['open']) | 
                    (data['high'] < data['close']) | 
                    (data['high'] < data['low'])
                ).sum()
                
                # Low sollte <= Open, Close, High sein  
                low_violations = (
                    (data['low'] > data['open']) |
                    (data['low'] > data['close']) |
                    (data['low'] > data['high'])
                ).sum()
                
                total_violations = high_violations + low_violations
                consistency_score *= max(0, 1 - total_violations / len(data))
            
            return consistency_score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Data Consistency Check fehlgeschlagen: {e}")
            return 0.5
    
    def _detect_price_outliers(self, data: pd.DataFrame) -> int:
        """Erkennt Preis-Outliers"""
        try:
            outliers = 0
            
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                
                if len(returns) > 10:
                    # Z-Score basierte Outlier-Erkennung
                    z_scores = np.abs((returns - returns.mean()) / returns.std())
                    outliers = (z_scores > 3).sum()  # 3-Sigma-Regel
            
            return outliers
            
        except Exception as e:
            self.logger.warning(f"⚠️ Price Outlier Detection fehlgeschlagen: {e}")
            return 0
    
    def _detect_data_gaps(self, data: pd.DataFrame) -> int:
        """Erkennt Datenlücken"""
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                return 0
            
            # Erwartete Frequenz basierend auf ersten paar Einträgen
            if len(data) < 3:
                return 0
            
            time_diffs = data.index[1:3] - data.index[0:2]
            expected_freq = time_diffs[0]
            
            # Zähle größere Lücken
            actual_diffs = data.index[1:] - data.index[:-1]
            gaps = (actual_diffs > expected_freq * 1.5).sum()
            
            return gaps
            
        except Exception as e:
            self.logger.warning(f"⚠️ Data Gap Detection fehlgeschlagen: {e}")
            return 0
    
    def _analyze_regime_in_period(self, data: pd.DataFrame) -> str:
        """Analysiert Markt-Regime in einem Zeitraum"""
        try:
            if len(data) < 10:
                return 'unknown'
            
            # Trend-Analyse
            prices = data['close']
            start_price = prices.iloc[0]
            end_price = prices.iloc[-1]
            total_return = (end_price - start_price) / start_price
            
            # Volatilitäts-Analyse
            returns = prices.pct_change().dropna()
            volatility = returns.std()
            
            # Regime-Klassifikation
            if total_return > 0.05 and volatility < 0.02:
                return 'bull_low_vol'
            elif total_return > 0.02:
                return 'bull'
            elif total_return < -0.05:
                return 'bear'
            elif volatility > 0.03:
                return 'high_volatility'
            else:
                return 'sideways'
                
        except Exception as e:
            self.logger.warning(f"⚠️ Regime Analysis fehlgeschlagen: {e}")
            return 'unknown'
    
    def _detect_volatility_clusters(self, returns: pd.Series) -> bool:
        """Erkennt Volatilitäts-Cluster"""
        try:
            if len(returns) < 20:
                return False
            
            # Rolling Volatility
            vol = returns.rolling(5).std()
            
            # Autokorrelation der Volatilität
            autocorr = vol.autocorr(lag=1)
            
            # Cluster wenn Autokorrelation > 0.3
            return autocorr > 0.3
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volatility Cluster Detection fehlgeschlagen: {e}")
            return False
    
    def _classify_volatility_regime(self, returns: pd.Series) -> str:
        """Klassifiziert Volatilitäts-Regime"""
        try:
            if len(returns) < 10:
                return 'unknown'
            
            current_vol = returns.std()
            
            # Annualisierte Volatilität
            annual_vol = current_vol * np.sqrt(252)
            
            if annual_vol > 0.4:
                return 'very_high'
            elif annual_vol > 0.25:
                return 'high'
            elif annual_vol > 0.15:
                return 'normal'
            elif annual_vol > 0.08:
                return 'low'
            else:
                return 'very_low'
                
        except Exception as e:
            self.logger.warning(f"⚠️ Volatility Classification fehlgeschlagen: {e}")
            return 'unknown'
    
    def _analyze_overall_trend(self, data: pd.DataFrame) -> str:
        """Analysiert Gesamt-Trend"""
        try:
            if 'close' not in data.columns or len(data) < 10:
                return 'unknown'
            
            prices = data['close']
            
            # Linear Regression für Trend
            x = np.arange(len(prices))
            coeffs = np.polyfit(x, prices, 1)
            slope = coeffs[0]
            
            # Normiere Slope durch Durchschnittspreis
            normalized_slope = slope / prices.mean() * len(prices)
            
            if normalized_slope > 0.1:
                return 'strong_uptrend'
            elif normalized_slope > 0.02:
                return 'uptrend'
            elif normalized_slope > -0.02:
                return 'sideways'
            elif normalized_slope > -0.1:
                return 'downtrend'
            else:
                return 'strong_downtrend'
                
        except Exception as e:
            self.logger.warning(f"⚠️ Overall Trend Analysis fehlgeschlagen: {e}")
            return 'unknown'
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Berechnet Trend-Stärke"""
        try:
            if 'close' not in data.columns or len(data) < 20:
                return 0.0
            
            prices = data['close']
            
            # R-squared von linearer Regression
            x = np.arange(len(prices))
            coeffs = np.polyfit(x, prices, 1)
            trend_line = coeffs[0] * x + coeffs[1]
            
            ss_res = np.sum((prices - trend_line) ** 2)
            ss_tot = np.sum((prices - prices.mean()) ** 2)
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return max(0, min(1, r_squared))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trend Strength Calculation fehlgeschlagen: {e}")
            return 0.0
    
    def _analyze_trend_consistency(self, data: pd.DataFrame) -> float:
        """Analysiert Trend-Konsistenz"""
        try:
            if 'close' not in data.columns or len(data) < 30:
                return 0.5
            
            prices = data['close']
            
            # Short-term vs Long-term Trend Konsistenz
            short_returns = prices.pct_change(5).dropna()
            long_returns = prices.pct_change(20).dropna()
            
            if len(short_returns) > 0 and len(long_returns) > 0:
                # Korrelation zwischen short- und long-term returns
                min_len = min(len(short_returns), len(long_returns))
                correlation = np.corrcoef(
                    short_returns.iloc[-min_len:],
                    long_returns.iloc[-min_len:]
                )[0, 1]
                
                return max(0, min(1, (correlation + 1) / 2))  # Normiere auf 0-1
            
            return 0.5
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trend Consistency Analysis fehlgeschlagen: {e}")
            return 0.5
    
    def _generate_pre_backtest_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generiert Pre-Backtest-Empfehlungen"""
        recommendations = []
        
        try:
            # Datenqualitäts-Empfehlungen
            data_quality = analysis.get('data_quality', {})
            if data_quality.get('completeness', 1) < 0.95:
                recommendations.append("improve_data_completeness")
            if data_quality.get('outliers_detected', 0) > 10:
                recommendations.append("review_outlier_handling")
            
            # Volatilitäts-Empfehlungen
            vol_analysis = analysis.get('volatility_analysis', {})
            vol_regime = vol_analysis.get('volatility_regime', 'normal')
            if vol_regime in ['very_high', 'high']:
                recommendations.append("use_conservative_position_sizing")
                recommendations.append("implement_dynamic_stop_losses")
            elif vol_regime in ['very_low', 'low']:
                recommendations.append("consider_increased_position_sizes")
            
            # Trend-Empfehlungen
            trend_analysis = analysis.get('trend_analysis', {})
            trend_strength = trend_analysis.get('trend_strength', 0)
            if trend_strength > 0.8:
                recommendations.append("favor_trend_following_signals")
            elif trend_strength < 0.3:
                recommendations.append("use_mean_reversion_bias")
            
            # Regime-Empfehlungen
            regimes = analysis.get('market_regimes', {})
            if regimes:
                dominant_regime = max(regimes.values(), key=lambda x: x if isinstance(x, (int, float)) else 0)
                if isinstance(dominant_regime, str) and 'bear' in dominant_regime:
                    recommendations.append("increase_confidence_threshold")
                    recommendations.append("reduce_position_sizes")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Pre-Backtest Recommendations fehlgeschlagen: {e}")
        
        return recommendations
    
    # ERWEITERTE ANALYSE-METHODEN (für Backtest-Enhancement)
    
    def _calculate_ensemble_weights(self) -> Dict[str, float]:
        """Berechnet Ensemble-Gewichte basierend auf Performance"""
        try:
            if not hasattr(self.strategy, 'model_performance'):
                return {}
            
            weights = {}
            total_performance = 0
            
            for model_name, perf in self.strategy.model_performance.items():
                score = perf.get('cv_mean', perf.get('train_score', 0))
                weights[model_name] = max(0.1, score)  # Minimum 10% Gewicht
                total_performance += weights[model_name]
            
            # Normalisiere Gewichte
            if total_performance > 0:
                for model_name in weights:
                    weights[model_name] /= total_performance
            
            return weights
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ensemble Weights Calculation fehlgeschlagen: {e}")
            return {}
    
    def _extract_feature_importance(self) -> Dict[str, float]:
        """Extrahiert Feature-Wichtigkeit aus Modellen"""
        try:
            if not hasattr(self.strategy, 'models'):
                return {}
            
            importance_dict = {}
            
            # Hauptmodell (Random Forest)
            main_model = self.strategy.models.get('main')
            if hasattr(main_model, 'feature_importances_') and hasattr(self.strategy, 'feature_names'):
                importances = main_model.feature_importances_
                feature_names = self.strategy.feature_names
                
                if len(importances) == len(feature_names):
                    for i, importance in enumerate(importances):
                        importance_dict[feature_names[i]] = float(importance)
            
            # Sortiere nach Wichtigkeit
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature Importance Extraction fehlgeschlagen: {e}")
            return {}
    
    def _analyze_prediction_calibration(self, results: Dict[str, Any]) -> str:
        """Analysiert Prediction-Kalibrierung"""
        try:
            trades = results.get('trades', [])
            if not trades:
                return 'no_data'
            
            # Sammle Confidence vs Success
            confidences = []
            successes = []
            
            for trade in trades:
                if 'confidence' in trade and 'pnl' in trade:
                    confidences.append(trade['confidence'])
                    successes.append(1 if trade['pnl'] > 0 else 0)
            
            if len(confidences) < 10:
                return 'insufficient_data'
            
            # Berechne Kalibrierung in Bins
            bins = np.linspace(0, 1, 6)  # 5 Bins
            bin_accuracies = []
            
            for i in range(len(bins)-1):
                mask = (np.array(confidences) >= bins[i]) & (np.array(confidences) < bins[i+1])
                if mask.sum() > 0:
                    accuracy = np.array(successes)[mask].mean()
                    expected = (bins[i] + bins[i+1]) / 2
                    bin_accuracies.append(abs(accuracy - expected))
            
            if bin_accuracies:
                avg_calibration_error = np.mean(bin_accuracies)
                if avg_calibration_error < 0.1:
                    return 'well_calibrated'
                elif avg_calibration_error < 0.2:
                    return 'moderately_calibrated'
                else:
                    return 'poorly_calibrated'
            
            return 'unknown'
            
        except Exception as e:
            self.logger.warning(f"⚠️ Prediction Calibration Analysis fehlgeschlagen: {e}")
            return 'error'
    
    def stop(self):
        """ERWEITERT: Stoppt den intelligenten Controller"""
        self.logger.info("⏹️ Stoppe intelligenten Trading Controller...")
        self.is_running = False
        
        # Stoppe Threading
        if hasattr(self, 'stop_event'):
            self.stop_event.set()
        
        # Warte auf Trading-Thread
        if hasattr(self, 'trading_thread') and self.trading_thread and self.trading_thread.is_alive():
            self.logger.info("⏳ Warte auf Trading-Thread...")
            self.trading_thread.join(timeout=10)
        
        # ERWEITERT: Intelligentes Shutdown
        try:
            # Finale Metriken loggen
            if hasattr(self, 'strategy_performance'):
                self.logger.info("📊 Finale Performance-Metriken:")
                for key, value in self.strategy_performance.items():
                    self.logger.info(f"   • {key}: {value}")
            
            # Schließe alle offenen Positionen (optional)
            try:
                positions = self.execution.get_positions()
                for position in positions:
                    if float(position.get('qty', 0)) != 0:
                        symbol = position['symbol']
                        self.logger.info(f"📤 Schließe finale Position: {symbol}")
                        self.execution.close_position(symbol)
            except Exception as e:
                self.logger.error(f"❌ Fehler beim Schließen finaler Positionen: {e}")
            
            # ERWEITERT: Finale Strategie-Persistierung
            if hasattr(self.strategy, '_save_training_state'):
                try:
                    final_data = pd.DataFrame([{
                        'timestamp': datetime.now(),
                        'shutdown': True,
                        'final_regime': getattr(self.strategy, 'market_regime', 'unknown')
                    }])
                    self.strategy._save_training_state(final_data)
                    self.logger.info("💾 Finale Strategie-Zustand gespeichert")
                except Exception as e:
                    self.logger.warning(f"⚠️ Finale Strategie-Speicherung fehlgeschlagen: {e}")
            
            # ERWEITERT: Finale System-Benachrichtigung
            if self.email_alerter:
                try:
                    runtime = datetime.now() - getattr(self, 'monitoring_start_time', datetime.now())
                    final_stats = {
                        'runtime': str(runtime),
                        'signals_generated': self.strategy_performance.get('signals_generated', 0),
                        'trades_executed': self.strategy_performance.get('signals_executed', 0),
                        'regime_changes': self.strategy_performance.get('regime_changes', 0),
                        'online_learning_updates': self.strategy_performance.get('online_learning_updates', 0)
                    }
                    
                    self.email_alerter.send_system_status(
                        status="stopped",
                        uptime=str(runtime),
                        last_trade=f"{final_stats['trades_executed']} trades executed"
                    )
                except Exception as e:
                    self.logger.warning(f"⚠️ Finale Benachrichtigung fehlgeschlagen: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ Fehler beim intelligenten Shutdown: {e}")
        
        self.logger.info("✅ Intelligenter Trading Controller gestoppt")
    
    def get_status(self) -> Dict[str, Any]:
        """ERWEITERT: Gibt erweiterten aktuellen Status zurück"""
        
        try:
            base_status = {
                'is_running': self.is_running,
                'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
                'strategy_fitted': getattr(self.strategy, 'is_fitted', False) if hasattr(self.strategy, 'is_fitted') else False
            }
            
            # Account-Informationen
            try:
                account_info = self.execution.get_account_info()
                base_status.update({
                    'account_value': account_info.get('equity', 0),
                    'buying_power': account_info.get('buying_power', 0),
                    'cash': account_info.get('cash', 0)
                })
            except Exception as e:
                base_status['account_error'] = str(e)
            
            # Positionen
            try:
                positions = self.execution.get_positions()
                base_status.update({
                    'positions_count': len(positions),
                    'positions': positions[:5]  # Nur erste 5 für Status
                })
            except Exception as e:
                base_status['positions_error'] = str(e)
            
            # Markt-Status
            try:
                base_status['market_open'] = self.data_provider.is_market_open()
            except Exception as e:
                base_status['market_status_error'] = str(e)
            
            # ERWEITERT: Intelligente Status-Informationen
            if hasattr(self, 'strategy_performance'):
                base_status['intelligent_metrics'] = {
                    'signals_generated': self.strategy_performance.get('signals_generated', 0),
                    'signals_executed': self.strategy_performance.get('signals_executed', 0),
                    'regime_changes': self.strategy_performance.get('regime_changes', 0),
                    'online_learning_updates': self.strategy_performance.get('online_learning_updates', 0)
                }
                
                # Execution Rate
                if self.strategy_performance.get('signals_generated', 0) > 0:
                    base_status['intelligent_metrics']['execution_rate'] = (
                        self.strategy_performance.get('signals_executed', 0) / 
                        self.strategy_performance.get('signals_generated', 1)
                    )
            
            # Strategie-spezifische Informationen
            if hasattr(self.strategy, 'market_regime'):
                base_status['current_regime'] = self.strategy.market_regime
                
            if hasattr(self.strategy, 'model_performance'):
                base_status['model_performance'] = self.strategy.model_performance
            
            # Monitoring-Informationen
            if hasattr(self, 'monitoring_start_time'):
                base_status['uptime'] = str(datetime.now() - self.monitoring_start_time)
                
            if hasattr(self, 'monitoring_metrics'):
                base_status['monitoring'] = self.monitoring_metrics
            
            # Confidence-Performance (letzte 10)
            if hasattr(self, 'confidence_performance_tracking') and self.confidence_performance_tracking:
                recent_performance = self.confidence_performance_tracking[-10:]
                base_status['recent_performance'] = {
                    'trades_count': len(recent_performance),
                    'success_rate': sum(1 for t in recent_performance if t['success']) / len(recent_performance),
                    'avg_confidence': sum(t['signal_confidence'] for t in recent_performance) / len(recent_performance)
                }
            
            # Regime-Historie (letzte 3)
            if hasattr(self, 'regime_history') and self.regime_history:
                base_status['recent_regimes'] = [
                    {
                        'timestamp': entry['timestamp'].isoformat(),
                        'regime': entry.get('new_regime', entry.get('regime', 'unknown'))
                    }
                    for entry in self.regime_history[-3:]
                ]
            
            return base_status
            
        except Exception as e:
            self.logger.error(f"❌ Status-Abfrage fehlgeschlagen: {e}")
            return {
                'is_running': self.is_running,
                'error': str(e),
                'status_timestamp': datetime.now().isoformat()
            }('confidence_accuracy_correlation', 0):.3f}")
            
            # Zukunfts-Prognosen
            if 'forward_predictions' in results:
                forward = results['forward_predictions']
                self.logger.info(f"🔮 Forward Predictions:")
                self.logger.info(f"   • Next Regime: {forward.get('next_regime_prediction', 'unknown')}")
                recommendations = forward.get('recommended_adjustments', [])
                if recommendations:
                    self.logger.info(f"   • Recommendations: {', '.join(recommendations)}")
            
            self.logger.info("="*60)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Intelligent Backtest Summary Logging fehlgeschlagen: {e}")
    
    def _send_intelligent_backtest_summary(self, symbol: str, results: Dict[str, Any]):
        """Sendet erweiterte intelligente Backtest-Zusammenfassung"""
        
        try:
            # Erweiterte Subject-Line mit KI-Metriken
            total_return = results.get('total_return', 0)
            sharpe = results.get('sharpe_ratio', 0)
            
            ml_indicator = ""
            if 'ml_performance' in results:
                main_score = results['ml_performance'].get('model_scores', {}).get('main', {}).get('cv_mean', 0)
                ml_indicator = f" | ML: {main_score:.2f}"
            
            subject = f"🧠 Intelligent Backtest: {symbol} | Return: {total_return:.1%} | Sharpe: {sharpe:.2f}{ml_indicator}"
            
            # Basis-Nachricht (behält bestehende Struktur)
            message = f"""
🤖 INTELLIGENT BACKTEST SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SYMBOL: {symbol}
STRATEGY: {getattr(self.strategy, '__class__', type(self.strategy)).__name__}

📊 STANDARD PERFORMANCE METRICS:
• Initial Capital: ${results['initial_capital']:,.2f}
• Final Capital: ${results['final_capital']:,.2f}  
• Total Return: {results['total_return']:.2%}
• Sharpe Ratio: {results['sharpe_ratio']:.3f}
• Max Drawdown: {results['max_drawdown']:.2%}

📈 TRADING STATISTICS:
• Total Trades: {results['total_trades']}
• Win Rate: {results['win_rate']:.2%}
• Winning Trades: {results['winning_trades']}
• Losing Trades: {results['losing_trades']}
• Profit Factor: {results['profit_factor']:.2f}
            """
            
            # Erweiterte intelligente Metriken
            if 'ml_performance' in results:
                ml_perf = results['ml_performance']
                message += f"""

🤖 MACHINE LEARNING PERFORMANCE:
• Model Ensemble: {len(ml_perf.get('model_scores', {}))} models
• Cross-Validation Score: {ml_perf.get('model_scores', {}).get('main', {}).get('cv_mean', 0):.3f}
• Feature Importance: {len(ml_perf.get('feature_importance', {}))} features analyzed
• Prediction Calibration: {ml_perf.get('prediction_calibration', 'N/A')}
                """
            
            if 'regime_analysis' in results:
                message += f"""

🌊 MARKET REGIME ANALYSIS:
                """
                for regime, perf in results['regime_analysis'].items():
                    if isinstance(perf, dict):
                        regime_return = perf.get('return', 0)
                        regime_trades = perf.get('trades', 0)
                        message += f"• {regime.title()}: {regime_return:.1%} ({regime_trades} trades)\n"
            
            if 'online_learning' in results:
                ol = results['online_learning']
                message += f"""

🔄 ADAPTIVE LEARNING:
• Adaptation Events: {ol.get('adaptation_events', 0)}
• Learning Curve Trend: {ol.get('learning_curve', 'stable')}
• Concept Drift: {'Detected' if ol.get('concept_drift_detection', False) else 'None'}
• Retrain Frequency: {ol.get('retrain_frequency_optimal', 'Standard')}
                """
            
            if 'confidence_analysis' in results:
                conf = results['confidence_analysis']
                message += f"""

🎯 CONFIDENCE CALIBRATION:
• Optimal Threshold: {conf.get('optimal_confidence_threshold', 0):.3f}
• Confidence-Accuracy Correlation: {conf.get"⚠️ Strategie-Monitoring Setup fehlgeschlagen: {e}")
    
    def run_backtest(self, symbol: str, start_date: str, end_date: str, 
                    initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        ERWEITERT: Führt Backtest mit intelligenter Analyse aus
        """
        self.logger.info(f"🔄 Starte erweiterten Backtest: {symbol} ({start_date} - {end_date})")
        
        try:
            # Validiere Eingaben
            self._validate_backtest_inputs(symbol, start_date, end_date, initial_capital)
            
            # Lade historische Daten
            self.logger.info("📊 Lade historische Daten...")
            data = self._load_enhanced_historical_data(symbol, start_date, end_date)
            
            if data.empty:
                raise ValueError("Keine historischen Daten erhalten - prüfe Symbol und Zeitraum")
                
            self.logger.info(f"✅ Daten geladen: {len(data)} Datenpunkte von {data.index[0]} bis {data.index[-1]}")
            
            # ERWEITERT: Pre-Backtest Analyse
            pre_analysis = self._perform_pre_backtest_analysis(data, symbol)
            
            # Konfiguriere erweiterten Backtester
            backtest_config = self.config.get('backtest', {})
            backtester = Backtester(
                strategy=self.strategy,
                initial_capital=initial_capital,
                commission=backtest_config.get('commission', 0.0),
                slippage=backtest_config.get('slippage', 0.001)
            )
            
            # Führe Backtest mit Monitoring aus
            self.logger.info("⚡ Führe erweiterten Backtest aus...")
            results = backtester.run(data, symbol)
            
            # ERWEITERT: Post-Backtest Intelligent Analysis
            enhanced_results = self._enhance_backtest_results_intelligent(
                results, data, symbol, pre_analysis
            )
            
            # ERWEITERT: Adaptive Learning von Backtest-Ergebnissen
            self._learn_from_backtest_results(enhanced_results, data)
            
            # Performance Summary mit intelligenten Metriken
            self._log_intelligent_backtest_summary(enhanced_results)
            
            # ERWEITERT: Email-Summary mit Regime-Analyse
            if self.email_alerter and self.config.get('alerts', {}).get('email_daily_summary', False):
                self._send_intelligent_backtest_summary(symbol, enhanced_results)
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"❌ Erweiterter Backtest fehlgeschlagen: {e}", exc_info=True)
            if self.email_alerter:
                self.email_alerter.send_error_alert("Intelligent Backtest Error", str(e))
            raise
    
    def _perform_pre_backtest_analysis(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """ERWEITERT: Führt Pre-Backtest Analyse für intelligente Strategie durch"""
        
        analysis = {
            'data_quality': {},
            'market_regimes': {},
            'volatility_analysis': {},
            'trend_analysis': {},
            'recommendations': []
        }
        
        try:
            # Datenqualitäts-Analyse
            analysis['data_quality'] = {
                'completeness': (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))),
                'consistency': self._check_data_consistency(data),
                'outliers_detected': self._detect_price_outliers(data),
                'gaps_detected': self._detect_data_gaps(data)
            }
            
            # Markt-Regime Vorab-Erkennung
            if hasattr(self.strategy, '_detect_market_regime'):
                regime_windows = [30, 60, 90]
                regimes = {}
                
                for window in regime_windows:
                    if len(data) >= window:
                        sample_data = data.tail(window)
                        # Simuliere Regime-Erkennung
                        regimes[f'{window}d'] = self._analyze_regime_in_period(sample_data)
                
                analysis['market_regimes'] = regimes
            
            # Volatilitäts-Analyse
            returns = data['close'].pct_change().dropna()
            analysis['volatility_analysis'] = {
                'mean_volatility': returns.std() * (252**0.5),  # Annualisiert
                'volatility_clusters': self._detect_volatility_clusters(returns),
                'volatility_regime': self._classify_volatility_regime(returns)
            }
            
            # Trend-Analyse
            analysis['trend_analysis'] = {
                'overall_trend': self._analyze_overall_trend(data),
                'trend_strength': self._calculate_trend_strength(data),
                'trend_consistency': self._analyze_trend_consistency(data)
            }
            
            # Strategische Empfehlungen
            analysis['recommendations'] = self._generate_pre_backtest_recommendations(analysis)
            
            self.logger.info(f"✅ Pre-Backtest Analyse abgeschlossen für {symbol}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Pre-Backtest Analyse teilweise fehlgeschlagen: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _enhance_backtest_results_intelligent(self, results: Dict[str, Any], 
                                            data: pd.DataFrame, symbol: str,
                                            pre_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """ERWEITERT: Erweitert Backtest-Ergebnisse um intelligente Analyse"""
        
        enhanced = results.copy()
        
        try:
            # Original-Erweiterungen (bestehend)
            if hasattr(self.strategy, 'market_regime'):
                regime_performance = self._analyze_regime_performance(results.get('trades', []))
                enhanced['regime_analysis'] = regime_performance
            
            # NEUE INTELLIGENTE ERWEITERUNGEN
            
            # 1. Adaptive Learning Performance
            if hasattr(self.strategy, 'model_performance'):
                enhanced['ml_performance'] = {
                    'model_scores': getattr(self.strategy, 'model_performance', {}),
                    'ensemble_weights': self._calculate_ensemble_weights(),
                    'feature_importance': self._extract_feature_importance(),
                    'prediction_calibration': self._analyze_prediction_calibration(results)
                }
            
            # 2. Online Learning Effectiveness
            enhanced['online_learning'] = {
                'adaptation_events': self._count_adaptation_events(),
                'learning_curve': self._analyze_learning_curve(),
                'concept_drift_detection': self._detect_concept_drift(),
                'retrain_frequency_optimal': self._analyze_retrain_frequency()
            }
            
            # 3. Regime-Adaptive Performance
            enhanced['regime_adaptation'] = {
                'regime_transitions': self._analyze_regime_transitions(),
                'adaptation_lag': self._calculate_adaptation_lag(),
                'regime_prediction_accuracy': self._evaluate_regime_predictions(),
                'cross_regime_stability': self._analyze_cross_regime_stability(results)
            }
            
            # 4. Confidence-Performance Korrelation
            enhanced['confidence_analysis'] = {
                'confidence_distribution': self._analyze_confidence_distribution(results),
                'confidence_accuracy_correlation': self._correlate_confidence_accuracy(results),
                'optimal_confidence_threshold': self._find_optimal_confidence_threshold(results),
                'confidence_calibration_curve': self._generate_calibration_curve(results)
            }
            
            # 5. Advanced Risk Metrics
            enhanced['intelligent_risk_metrics'] = {
                'regime_adjusted_sharpe': self._calculate_regime_adjusted_sharpe(results),
                'adaptive_max_drawdown': self._calculate_adaptive_drawdown(results),
                'tail_risk_by_regime': self._analyze_tail_risk_by_regime(results),
                'dynamic_var': self._calculate_dynamic_var(results)
            }
            
            # 6. Forward-Looking Predictions
            enhanced['forward_predictions'] = {
                'next_regime_prediction': self._predict_next_regime(),
                'performance_forecast': self._forecast_performance_metrics(),
                'recommended_adjustments': self._recommend_strategy_adjustments(enhanced)
            }
            
            # 7. Integration mit Pre-Analysis
            enhanced['pre_post_analysis'] = {
                'predictions_vs_reality': self._compare_predictions_reality(pre_analysis, results),
                'regime_prediction_accuracy': self._evaluate_regime_prediction_accuracy(pre_analysis, results),
                'volatility_forecast_accuracy': self._evaluate_volatility_forecasts(pre_analysis, data)
            }
            
            self.logger.info("✅ Intelligente Backtest-Erweiterung abgeschlossen")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Intelligente Erweiterung teilweise fehlgeschlagen: {e}")
            enhanced['enhancement_error'] = str(e)
        
        return enhanced
    
    def _learn_from_backtest_results(self, results: Dict[str, Any], data: pd.DataFrame):
        """ERWEITERT: Lernt aus Backtest-Ergebnissen für zukünftige Optimierung"""
        
        try:
            # Update Strategie-Performance Metriken
            self.strategy_performance.update({
                'last_backtest_return': results.get('total_return', 0),
                'last_backtest_sharpe': results.get('sharpe_ratio', 0),
                'last_backtest_win_rate': results.get('win_rate', 0),
                'last_backtest_trades': results.get('total_trades', 0)
            })
            
            # Adaptive Parameter-Anpassung basierend auf Ergebnissen
            self._adapt_parameters_from_results(results)
            
            # Regime-spezifische Learnings
            if 'regime_analysis' in results:
                self._learn_regime_patterns(results['regime_analysis'])
            
            # Confidence-Threshold Optimierung
            if 'confidence_analysis' in results:
                optimal_threshold = results['confidence_analysis'].get('optimal_confidence_threshold')
                if optimal_threshold and hasattr(self.strategy, 'prediction_threshold'):
                    old_threshold = self.strategy.prediction_threshold
                    # Sanfte Anpassung (10% in Richtung Optimum)
                    new_threshold = old_threshold + 0.1 * (optimal_threshold - old_threshold)
                    self.strategy.prediction_threshold = new_threshold
                    
                    self.logger.info(f"🎯 Confidence-Threshold adaptiert: {old_threshold:.3f} → {new_threshold:.3f}")
            
            self.logger.info("✅ Backtest-Learning abgeschlossen")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Backtest-Learning fehlgeschlagen: {e}")
    
    def run_live(self, symbol: str):
        """
        ERWEITERT: Startet intelligentes Live-Trading mit adaptiver Steuerung
        """
        self.logger.info(f"🚀 Starte intelligentes Live-Trading für {symbol}")
        
        # Prüfe Marktzeiten
        live_config = self.config.get('live_trading', {})
        market_hours_only = live_config.get('market_hours_only', True)
        
        if market_hours_only and not self.data_provider.is_market_open():
            self.logger.warning("Markt ist geschlossen - warte auf Öffnung")
            
        self.is_running = True
        
        try:
            # ERWEITERT: Lade initiale Daten und trainiere intelligente Strategie
            self._initialize_intelligent_strategy_for_live(symbol)
            
            # ERWEITERT: Setup für Online Learning und Monitoring
            self._setup_live_monitoring()
            
            # Konfiguriere erweiterten Scheduler
            update_interval = live_config.get('update_interval', 60)
            
            def intelligent_trading_job():
                if not market_hours_only or self.data_provider.is_market_open():
                    self._execute_intelligent_trading_cycle(symbol)
                    self._update_live_monitoring()
                else:
                    self.logger.debug("Markt geschlossen - überspringe intelligenten Trading-Zyklus")
            
            # Schedule Jobs mit intelligenter Überwachung
            if update_interval < 60:
                self._run_intelligent_continuous_trading(symbol, update_interval)
            else:
                # Für längere Intervalle verwende schedule mit Monitoring
                schedule.every(update_interval).seconds.do(intelligent_trading_job)
                
                # ERWEITERT: Zusätzliche Monitoring-Jobs
                schedule.every(5).minutes.do(lambda: self._monitor_strategy_health())
                schedule.every(30).minutes.do(lambda: self._adaptive_parameter_update())
                schedule.every(1).hour.do(lambda: self._regime_check_and_adapt(symbol))
                
                self._run_scheduled_intelligent_trading()
                
        except KeyboardInterrupt:
            self.logger.info("⏹️ Intelligentes Live-Trading durch Benutzer gestoppt")
            self.is_running = False
        except Exception as e:
            self.logger.error(f"❌ Intelligentes Live-Trading Fehler: {e}", exc_info=True)
            if self.email_alerter:
                self.email_alerter.send_error_alert("Intelligent Live Trading Error", str(e))
            self.is_running = False
            raise
        finally:
            self._cleanup_intelligent_live_trading()
    
    def _initialize_intelligent_strategy_for_live(self, symbol: str):
        """ERWEITERT: Initialisiert intelligente Strategie für Live-Trading"""
        
        self.logger.info("🎓 Initialisiere intelligente Strategie für Live-Trading...")
        
        try:
            # Erweiterte Trainingsdaten-Bestimmung
            if hasattr(self.strategy, 'lookback_period'):
                lookback = max(365, self.strategy.lookback_period * 10)
            else:
                lookback = 365
            
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=lookback)).strftime('%Y-%m-%d')
            
            # Lade erweiterte Trainingsdaten
            training_data = self._load_enhanced_historical_data(symbol, start_date, end_date)
            
            if training_data.empty:
                raise ValueError("Keine Trainingsdaten für intelligentes Live-Trading verfügbar")
            
            # ERWEITERT: Pre-Training Analysis
            pre_training_analysis = self._analyze_training_data_quality(training_data)
            
            # Trainiere intelligente Strategie
            self.logger.info("🧠 Trainiere intelligente Strategie...")
            self.strategy.fit(training_data)
            
            # ERWEITERT: Post-Training Validation
            self._validate_intelligent_strategy_training()
            
            # ERWEITERT: Initiale Regime-Erkennung
            if hasattr(self.strategy, 'market_regime'):
                initial_regime = self.strategy.market_regime
                self.regime_history.append({
                    'timestamp': datetime.now(),
                    'regime': initial_regime,
                    'confidence': getattr(self.strategy, 'regime_confidence', 0.5)
                })
                self.logger.info(f"📊 Initiales Markt-Regime erkannt: {initial_regime}")
            
            # ERWEITERT: Setup für Online Learning
            if hasattr(self.strategy, 'data_buffer'):
                self.logger.info("🔄 Online Learning aktiviert")
            
            self.logger.info(f"✅ Intelligente Strategie trainiert mit {len(training_data)} Datenpunkten")
            
        except Exception as e:
            self.logger.error(f"❌ Intelligente Strategie-Initialisierung für Live-Trading fehlgeschlagen: {e}")
            raise
    
    def _execute_intelligent_trading_cycle(self, symbol: str):
        """ERWEITERT: Führt intelligenten Trading-Zyklus aus"""
        
        try:
            cycle_start = time.time()
            
            # 1. Erweiterte Marktdaten mit Qualitätsprüfung
            current_data = self._get_enhanced_intelligent_market_data(symbol)
            if current_data is None or current_data.empty:
                self.logger.debug("⏭️ Keine aktuellen Daten - überspringe intelligenten Zyklus")
                return
            
            # 2. ERWEITERT: Pre-Signal Intelligence Gathering
            market_context = self._gather_market_intelligence(current_data, symbol)
            
            # 3. Generiere intelligentes Signal mit Kontext
            latest_row = current_data.iloc[-1]
            signal = self._generate_enhanced_intelligent_signal(latest_row, symbol, market_context)
            
            # 4. ERWEITERT: Signal-Intelligenz-Analyse
            signal_analysis = self._analyze_signal_intelligence(signal, latest_row, market_context)
            
            # 5. Update Metriken
            self.strategy_performance['signals_generated'] += 1
            if signal.get('confidence'):
                self.strategy_performance['confidence_history'].append(signal['confidence'])
            
            # 6. ERWEITERT: Intelligente Entscheidungslogik
            execution_decision = self._make_intelligent_execution_decision(
                signal, signal_analysis, market_context
            )
            
            if execution_decision['action'] == 'skip':
                self.strategy_performance['signals_executed'] += 0  # Skip zählt nicht
                self.logger.debug(f"🤔 Intelligente Entscheidung: SKIP - {execution_decision['reason']}")
                return
            elif execution_decision['action'] == 'hold':
                self.logger.debug("📊 Intelligente Entscheidung: HOLD")
                return
            
            # 7. ERWEITERT: Führe intelligenten Trade aus
            success = self._execute_enhanced_intelligent_trade(
                symbol, signal, latest_row['close'], execution_decision, market_context
            )
            
            if success:
                self.strategy_performance['signals_executed'] += 1
            
            # 8. ERWEITERT: Post-Trade Learning Update
            self._post_trade_learning_update(symbol, signal, execution_decision, success)
            
            # 9. Performance Tracking
            cycle_time = time.time() - cycle_start
            self.logger.debug(f"⚡ Intelligenter Trading-Zyklus completed in {cycle_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"❌ Intelligenter Trading-Zyklus Fehler: {e}", exc_info=True)
            if self.email_alerter:
                self.email_alerter.send_error_alert("Intelligent Trading Cycle Error", str(e))
    
    # ERWEITERTE HILFSMETHODEN (Neue intelligente Funktionen)
    
    def _gather_market_intelligence(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Sammelt umfassende Marktintelligenz"""
        
        intelligence = {}
        
        try:
            # Aktuelle Marktbedingungen
            latest_price = data.iloc[-1]['close']
            price_change = (latest_price / data.iloc[-2]['close'] - 1) if len(data) > 1 else 0
            
            intelligence['current_conditions'] = {
                'price': latest_price,
                'price_change': price_change,
                'volume': data.iloc[-1].get('volume', 0),
                'timestamp': datetime.now()
            }
            
            # Volatilitäts-Intelligence
            returns = data['close'].pct_change().dropna()
            if len(returns) >= 10:
                recent_vol = returns.tail(10).std()
                historical_vol = returns.std()
                
                intelligence['volatility'] = {
                    'current': recent_vol,
                    'historical': historical_vol,
                    'ratio': recent_vol / historical_vol if historical_vol > 0 else 1,
                    'regime': 'high' if recent_vol > historical_vol * 1.5 else 'low' if recent_vol < historical_vol * 0.5 else 'normal'
                }
            
            # Momentum-Intelligence
            if len(data) >= 20:
                momentum_5 = (latest_price / data.iloc[-6]['close'] - 1) if len(data) > 5 else 0
                momentum_20 = (latest_price / data.iloc[-21]['close'] - 1) if len(data) > 20 else 0
                
                intelligence['momentum'] = {
                    'short_term': momentum_5,
                    'medium_term': momentum_20,
                    'strength': abs(momentum_5),
                    'direction': 'up' if momentum_5 > 0 else 'down'
                }
            
            # Regime-Intelligence (falls verfügbar)
            if hasattr(self.strategy, 'market_regime'):
                intelligence['regime'] = {
                    'current': self.strategy.market_regime,
                    'confidence': getattr(self.strategy, 'regime_confidence', 0.5),
                    'duration': self._calculate_regime_duration(),
                    'stability': self._calculate_regime_stability()
                }
        
        except Exception as e:
            self.logger.warning(f"⚠️ Market Intelligence Gathering teilweise fehlgeschlagen: {e}")
            intelligence['error'] = str(e)
        
        return intelligence
    
    def _generate_enhanced_intelligent_signal(self, row: pd.Series, symbol: str, 
                                            market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generiert verstärktes intelligentes Signal mit Marktkontext"""
        
        try:
            # Basis-Signal von der Strategie
            base_signal = self.strategy.generate_signal(row)
            
            # ERWEITERT: Intelligente Signal-Enhancement
            enhanced_signal = base_signal.copy()
            
            # Kontext-basierte Adjustierungen
            if 'volatility' in market_context:
                vol_regime = market_context['volatility'].get('regime', 'normal')
                if vol_regime == 'high':
                    # Reduziere Position Size bei hoher Volatilität
                    enhanced_signal['position_size'] = enhanced_signal.get('position_size', 0.1) * 0.8
                    enhanced_signal['confidence'] = enhanced_signal.get('confidence', 0.5) * 0.9
                elif vol_regime == 'low':
                    # Erhöhe Position Size bei niedriger Volatilität
                    enhanced_signal['position_size'] = min(
                        enhanced_signal.get('position_size', 0.1) * 1.2, 
                        self.config.get('risk_management', {}).get('max_position_size', 0.15)
                    )
            
            # Momentum-basierte Adjustierungen
            if 'momentum' in market_context:
                momentum_strength = market_context['momentum'].get('strength', 0)
                if momentum_strength > 0.05:  # Starker Momentum
                    if (enhanced_signal['action'] == 'buy' and 
                        market_context['momentum'].get('direction') == 'up'):
                        enhanced_signal['confidence'] = min(1.0, enhanced_signal.get('confidence', 0.5) * 1.1)
            
            # Regime-basierte Adjustierungen
            if 'regime' in market_context and hasattr(self.strategy, '_apply_regime_adjustments'):
                enhanced_signal = self.strategy._apply_regime_adjustments(enhanced_signal, row)
            
            # Meta-Informationen hinzufügen
            enhanced_signal['market_context'] = market_context
            enhanced_signal['enhancement_applied'] = True
            enhanced_signal['enhancement_timestamp'] = datetime.now()
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.warning(f"⚠️ Signal-Enhancement fehlgeschlagen: {e}")
            # Fallback zum Basis-Signal
            return self.strategy.generate_signal(row)
    
    def _analyze_signal_intelligence(self, signal: Dict[str, Any], 
                                   current_row: pd.Series,
                                   market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analysiert die Intelligenz und Qualität des Signals"""
        
        analysis = {
            'signal_quality': 'unknown',
            'confidence_assessment': 'unknown',
            'risk_assessment': 'unknown',
            'timing_assessment': 'unknown',
            'recommendations': []
        }
        
        try:
            # Signal-Qualitäts-Bewertung
            confidence = signal.get('confidence', 0.5)
            if confidence > 0.8:
                analysis['signal_quality'] = 'excellent'
            elif confidence > 0.6:
                analysis['signal_quality'] = 'good'
            elif confidence > 0.4:
                analysis['signal_quality'] = 'fair'
            else:
                analysis['signal_quality'] = 'poor'
            
            # Confidence-Assessment
            if hasattr(self.strategy, 'prediction_threshold'):
                threshold = self.strategy.prediction_threshold
                if confidence > threshold * 1.2:
                    analysis['confidence_assessment'] = 'high_confidence'
                elif confidence > threshold:
                    analysis['confidence_assessment'] = 'adequate_confidence'
                else:
                    analysis['confidence_assessment'] = 'low_confidence'
            
            # Risk-Assessment basierend auf Marktkontext
            risk_factors = []
            
            if 'volatility' in market_context:
                vol_regime = market_context['volatility'].get('regime')
                if vol_regime == 'high':
                    risk_factors.append('high_volatility')
                elif vol_regime == 'low':
                    risk_factors.append('low_volatility')
            
            if 'momentum' in market_context:
                momentum_strength = market_context['momentum'].get('strength', 0)
                if momentum_strength > 0.1:
                    risk_factors.append('high_momentum')
            
            if len(risk_factors) >= 2:
                analysis['risk_assessment'] = 'high_risk'
            elif len(risk_factors) == 1:
                analysis['risk_assessment'] = 'medium_risk'
            else:
                analysis['risk_assessment'] = 'low_risk'
            
            # Timing-Assessment
            current_time = datetime.now()
            if hasattr(self, 'last_signal_time') and self.last_signal_time:
                time_since_last = (current_time - self.last_signal_time).total_seconds() / 60
                if time_since_last < 5:
                    analysis['timing_assessment'] = 'too_frequent'
                elif time_since_last > 60:
                    analysis['timing_assessment'] = 'good_spacing'
                else:
                    analysis['timing_assessment'] = 'adequate_spacing'
            else:
                analysis['timing_assessment'] = 'first_signal'
            
            # Empfehlungen generieren
            if analysis['signal_quality'] == 'poor':
                analysis['recommendations'].append('consider_skipping_low_quality_signal')
            
            if analysis['confidence_assessment'] == 'low_confidence':
                analysis['recommendations'].append('reduce_position_size')
            
            if analysis['risk_assessment'] == 'high_risk':
                analysis['recommendations'].append('apply_conservative_position_sizing')
            
            if analysis['timing_assessment'] == 'too_frequent':
                analysis['recommendations'].append('apply_rate_limiting')
        
        except Exception as e:
            self.logger.warning(f"⚠️ Signal-Intelligence-Analyse fehlgeschlagen: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _make_intelligent_execution_decision(self, signal: Dict[str, Any],
                                           signal_analysis: Dict[str, Any],
                                           market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Trifft intelligente Entscheidung über Signal-Ausführung"""
        
        decision = {
            'action': 'execute',  # execute, skip, hold
            'reason': 'signal_approved',
            'adjustments': {},
            'confidence_in_decision': 0.5
        }
        
        try:
            # Basis-Entscheidungslogik
            if signal.get('action') == 'hold':
                decision['action'] = 'hold'
                decision['reason'] = 'strategy_recommends_hold'
                return decision
            
            # Qualitäts-basierte Entscheidungen
            if signal_analysis.get('signal_quality') == 'poor':
                decision['action'] = 'skip'
                decision['reason'] = 'poor_signal_quality'
                return decision
            
            # Timing-basierte Entscheidungen
            if signal_analysis.get('timing_assessment') == 'too_frequent':
                decision['action'] = 'skip'
                decision['reason'] = 'too_frequent_signals'
                return decision
            
            # Confidence-basierte Entscheidungen
            if signal_analysis.get('confidence_assessment') == 'low_confidence':
                if signal.get('confidence', 0) < 0.4:
                    decision['action'] = 'skip'
                    decision['reason'] = 'insufficient_confidence'
                    return decision
                else:
                    # Reduziere Position Size
                    decision['adjustments']['position_size_multiplier'] = 0.5
                    decision['reason'] = 'reduced_position_due_to_low_confidence'
            
            # Risk-basierte Adjustierungen
            if signal_analysis.get('risk_assessment') == 'high_risk':
                decision['adjustments']['position_size_multiplier'] = 0.7
                decision['adjustments']['tighter_stops'] = True
                decision['reason'] = 'conservative_due_to_high_risk'
            
            # Portfolio-basierte Entscheidungen
            portfolio_check = self._check_portfolio_constraints()
            if not portfolio_check['can_trade']:
                decision['action'] = 'skip'
                decision['reason'] = f"portfolio_constraint: {portfolio_check['reason']}"
                return decision
            
            # Regime-basierte Adjustierungen
            if 'regime' in market_context:
                regime = market_context['regime'].get('current')
                if regime == 'bear' and signal.get('action') == 'buy':
                    # Sehr konservativ in Bear Markets
                    if signal.get('confidence', 0) < 0.75:
                        decision['action'] = 'skip'
                        decision['reason'] = 'bear_market_high_confidence_required'
                        return decision
                    else:
                        decision['adjustments']['position_size_multiplier'] = 0.6
            
            # Final Decision Confidence
            factors_positive = []
            factors_negative = []
            
            if signal_analysis.get('signal_quality') in ['excellent', 'good']:
                factors_positive.append('good_signal_quality')
            if signal_analysis.get('confidence_assessment') == 'high_confidence':
                factors_positive.append('high_confidence')
            if signal_analysis.get('timing_assessment') in ['good_spacing', 'adequate_spacing']:
                factors_positive.append('good_timing')
            
            if signal_analysis.get('risk_assessment') == 'high_risk':
                factors_negative.append('high_risk')
            if 'position_size_multiplier' in decision.get('adjustments', {}):
                factors_negative.append('position_reduced')
            
            # Berechne Decision Confidence
            positive_weight = len(factors_positive) * 0.3
            negative_weight = len(factors_negative) * 0.2
            base_confidence = signal.get('confidence', 0.5)
            
            decision['confidence_in_decision'] = max(0.1, 
                min(1.0, base_confidence + positive_weight - negative_weight))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Intelligente Execution Decision fehlgeschlagen: {e}")
            decision['error'] = str(e)
        
        return decision
    
    def _execute_enhanced_intelligent_trade(self, symbol: str, signal: Dict[str, Any], 
                                          current_price: float, execution_decision: Dict[str, Any],
                                          market_context: Dict[str, Any]) -> bool:
        """Führt erweiterten intelligenten Trade aus"""
        
        try:
            # Hole Account-Info
            account_info = self.execution.get_account_info()
            buying_power = float(account_info.get('buying_power', 0))
            
            # Berechne intelligente Positionsgröße
            base_position_size = signal.get('position_size', 0.1)
            
            # Wende Adjustierungen an
            adjustments = execution_decision.get('adjustments', {})
            final_position_size = base_position_size
            
            if 'position_size_multiplier' in adjustments:
                final_position_size *= adjustments['position_size_multiplier']
            
            # Zusätzliche intelligente Adjustierungen
            if 'volatility' in market_context:
                vol_ratio = market_context['volatility'].get('ratio', 1.0)
                if vol_ratio > 2.0:  # Sehr hohe Volatilität
                    final_position_size *= 0.5
                elif vol_ratio > 1.5:  # Hohe Volatilität
                    final_position_size *= 0.75
            
            # Berechne finale Quantity
            trade_value = buying_power * final_position_size
            quantity = self.execution.calculate_order_quantity(
                symbol, buying_power, final_position_size, current_price
            )
            
            if quantity <= 0:
                self.logger.warning("❌ Berechnete Quantity <= 0")
                return False
            
            # Hole aktuelle Position
            current_position = self.execution.get_position_for_symbol(symbol)
            current_qty = float(current_position.get('qty', 0)) if current_position else 0
            
            side = signal['action']
            orders_placed = []
            
            # Intelligente Order-Platzierung
            
            # 1. Schließe bestehende Position falls notwendig
            if current_qty != 0:
                close_side = 'sell' if current_qty > 0 else 'buy'
                close_order = self.execution.place_order(
                    symbol=symbol,
                    side=close_side,
                    quantity=abs(current_qty),
                    order_type='market'
                )
                orders_placed.append(close_order)
                self.logger.info(f"📤 Position geschlossen: {close_side} {abs(current_qty)} {symbol}")
            
            # 2. Öffne neue Position mit intelligenten Parametern
            if side in ['buy', 'sell']:
                # Erweiterte Stop-Loss und Take-Profit Berechnung
                stop_loss = signal.get('stop_loss')
                take_profit = signal.get('take_profit')
                
                # Intelligente Stop/TP Adjustierung basierend auf Volatilität
                if 'volatility' in market_context:
                    vol_adjustment = market_context['volatility'].get('ratio', 1.0)
                    if stop_loss and vol_adjustment > 1.5:
                        # Erweitere Stops bei hoher Volatilität
                        if side == 'buy':
                            stop_loss = current_price * (1 - (current_price - stop_loss) / current_price * vol_adjustment)
                        else:
                            stop_loss = current_price * (1 + (stop_loss - current_price) / current_price * vol_adjustment)
                
                # Fallback auf Strategie-Methoden
                if not stop_loss:
                    stop_loss = self.strategy.get_stop_loss(current_price, side)
                if not take_profit:
                    take_profit = self.strategy.get_take_profit(current_price, side)
                
                # Platziere Hauptorder
                main_order = self.execution.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type='market'
                )
                orders_placed.append(main_order)
                
                self.logger.info(f"🎯 Intelligente Order platziert: {side} {quantity} {symbol} @ ${current_price:.2f} "
                               f"(Confidence: {signal.get('confidence', 0):.2%}, "
                               f"Decision Confidence: {execution_decision.get('confidence_in_decision', 0):.2%})")
                
                # Erweiterte Risk Management Orders
                if stop_loss or take_profit:
                    try:
                        bracket_orders = self.execution.create_bracket_order(
                            symbol=symbol,
                            side=side,
                            quantity=quantity,
                            stop_loss=stop_loss,
                            take_profit=take_profit
                        )
                        orders_placed.extend(bracket_orders[1:])
                        
                        self.logger.info(f"🛡️ Risk Management Orders platziert - SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}")
                        
                    except Exception as e:
                        self.logger.warning(f"⚠️ Bracket Orders fehlgeschlagen: {e}")
            
            # Update Zeitstempel
            self.last_signal_time = datetime.now()
            
            # Erweiterte Benachrichtigung
            if self.email_alerter and self.config.get('alerts', {}).get('email_on_trade', False):
                self._send_intelligent_trade_alert(
                    symbol, signal, orders_placed, current_price, 
                    execution_decision, market_context
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erweiterte intelligente Trade-Ausführung fehlgeschlagen: {e}", exc_info=True)
            if self.email_alerter:
                self.email_alerter.send_error_alert("Enhanced Intelligent Trade Execution", str(e))
            return False
    
    def _post_trade_learning_update(self, symbol: str, signal: Dict[str, Any], 
                                  execution_decision: Dict[str, Any], success: bool):
        """Post-Trade Learning Update für kontinuierliche Verbesserung"""
        
        try:
            # Update Learning-Metriken
            self.learning_metrics['last_trade_timestamp'] = datetime.now()
            
            if success:
                self.learning_metrics['successful_trades'] = self.learning_metrics.get('successful_trades', 0) + 1
            else:
                self.learning_metrics['failed_trades'] = self.learning_metrics.get('failed_trades', 0) + 1
            
            # Confidence-Performance Tracking
            confidence = signal.get('confidence', 0.5)
            decision_confidence = execution_decision.get('confidence_in_decision', 0.5)
            
            if not hasattr(self, 'confidence_performance_tracking'):
                self.confidence_performance_tracking = []
            
            self.confidence_performance_tracking.append({
                'timestamp': datetime.now(),
                'signal_confidence': confidence,
                'decision_confidence': decision_confidence,
                'success': success,
                'symbol': symbol
            })
            
            # Halte nur die letzten 100 Einträge
            if len(self.confidence_performance_tracking) > 100:
                self.confidence_performance_tracking = self.confidence_performance_tracking[-100:]
            
            # Update der Strategie mit Learning-Daten (falls verfügbar)
            if hasattr(self.strategy, '_update_online_learning') and success:
                try:
                    current_row = pd.Series({
                        'close': signal.get('current_price', 100),
                        'timestamp': datetime.now()
                    })
                    features = signal.get('features', [])
                    if features:
                        self.strategy._update_online_learning(current_row, features)
                        self.strategy_performance['online_learning_updates'] = \
                            self.strategy_performance.get('online_learning_updates', 0) + 1
                except Exception as e:
                    self.logger.warning(f"⚠️ Online Learning Update fehlgeschlagen: {e}")
            
            # Adaptive Parameter Learning
            if success and execution_decision.get('adjustments'):
                # Merke erfolgreiche Adjustierungen für zukünftige Verwendung
                adjustment_type = execution_decision.get('reason', 'unknown')
                if adjustment_type not in self.adaptive_params:
                    self.adaptive_params[adjustment_type] = {'success_count': 0, 'total_count': 0}
                
                self.adaptive_params[adjustment_type]['success_count'] += 1
                self.adaptive_params[adjustment_type]['total_count'] += 1
            elif not success:
                adjustment_type = execution_decision.get('reason', 'unknown')
                if adjustment_type not in self.adaptive_params:
                    self.adaptive_params[adjustment_type] = {'success_count': 0, 'total_count': 0}
                
                self.adaptive_params[adjustment_type]['total_count'] += 1
            
        except Exception as e:
            self.logger.warning(f"⚠️ Post-Trade Learning Update fehlgeschlagen: {e}")
    
    # MONITORING UND MAINTENANCE METHODEN
    
    def _setup_live_monitoring(self):
        """Setup für Live-Trading Monitoring"""
        try:
            self.monitoring_start_time = datetime.now()
            self.monitoring_metrics = {
                'cycles_completed': 0,
                'signals_generated': 0,
                'trades_executed': 0,
                'errors_encountered': 0,
                'last_health_check': datetime.now()
            }
            
            self.logger.info("✅ Live-Monitoring eingerichtet")
        except Exception as e:
            self.logger.warning(f"⚠️ Live-Monitoring Setup fehlgeschlagen: {e}")
    
    def _update_live_monitoring(self):
        """Update Live-Monitoring Metriken"""
        try:
            self.monitoring_metrics['cycles_completed'] += 1
            self.monitoring_metrics['last_health_check'] = datetime.now()
            
            # Alle 50 Zyklen: Health Summary
            if self.monitoring_metrics['cycles_completed'] % 50 == 0:
                self._log_health_summary()
                
        except Exception as e:
            self.logger.warning(f"⚠️ Live-Monitoring Update fehlgeschlagen: {e}")
    
    def _monitor_strategy_health(self):
        """Überwacht Strategie-Gesundheit"""
        try:
            health_status = {
                'overall': 'healthy',
                'issues': [],
                'recommendations': []
            }
            
            # Prüfe Strategie-Performance
            if hasattr(self, 'confidence_performance_tracking') and self.confidence_performance_tracking:
                recent_trades = self.confidence_performance_tracking[-10:]
                success_rate = sum(1 for t in recent_trades if t['success']) / len(recent_trades)
                
                if success_rate < 0.3:
                    health_status['overall'] = 'unhealthy'
                    health_status['issues'].append(f'low_success_rate: {success_rate:.2%}')
                    health_status['recommendations'].append('consider_strategy_retraining')
                elif success_rate < 0.5:
                    health_status['overall'] = 'degraded'
                    health_status['issues'].append(f'declining_performance: {success_rate:.2%}')
            
            # Prüfe Online Learning Aktivität
            if hasattr(self.strategy, 'last_retrain_date'):
                if self.strategy.last_retrain_date:
                    days_since_retrain = (datetime.now() - self.strategy.last_retrain_date).days
                    if days_since_retrain > 7:
                        health_status['issues'].append(f'no_recent_retraining: {days_since_retrain} days')
                        health_status['recommendations'].append('schedule_manual_retrain')
            
            # Prüfe Regime-Erkennungs-Aktivität
            if len(self.regime_history) > 1:
                last_regime_change = max((entry['timestamp'] for entry in self.regime_history))
                days_since_regime_change = (datetime.now() - last_regime_change).days
                if days_since_regime_change > 30:
                    health_status['issues'].append('static_regime_detection')
            
            # Log Health Status
            if health_status['overall'] != 'healthy':
                self.logger.warning(f"🏥 Strategie-Gesundheit: {health_status['overall']}")
                for issue in health_status['issues']:
                    self.logger.warning(f"  - Issue: {issue}")
                for rec in health_status['recommendations']:
                    self.logger.info(f"  - Empfehlung: {rec}")
            else:
                self.logger.debug("💚 Strategie-Gesundheit: gesund")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Strategy Health Monitoring fehlgeschlagen: {e}")
    
    def _adaptive_parameter_update(self):
        """Adaptive Parameter-Updates basierend auf Performance"""
        try:
            updates_made = []
            
            # Confidence Threshold Optimization
            if hasattr(self, 'confidence_performance_tracking') and len(self.confidence_performance_tracking) >= 20:
                recent_data = self.confidence_performance_tracking[-20:]
                
                # Analysiere Confidence vs Success Rate
                high_conf_trades = [t for t in recent_data if t['signal_confidence'] > 0.7]
                low_conf_trades = [t for t in recent_data if t['signal_confidence'] < 0.5]
                
                if high_conf_trades and low_conf_trades:
                    high_conf_success = sum(1 for t in high_conf_trades if t['success']) / len(high_conf_trades)
                    low_conf_success = sum(1 for t in low_conf_trades if t['success']) / len(low_conf_trades)
                    
                    if high_conf_success > low_conf_success + 0.2:  # Signifikanter Unterschied
                        # Erhöhe Threshold leicht
                        old_threshold = getattr(self.strategy, 'prediction_threshold', 0.6)
                        new_threshold = min(0.9, old_threshold + 0.02)
                        if hasattr(self.strategy, 'prediction_threshold'):
                            self.strategy.prediction_threshold = new_threshold
                            updates_made.append(f'confidence_threshold: {old_threshold:.3f} → {new_threshold:.3f}')
            
            # Position Size Multiplier Updates
            for param_key, param_data in self.adaptive_params.items():
                if param_data['total_count'] >= 5:
                    success_rate = param_data['success_count'] / param_data['total_count']
                    
                    # Update Multiplier basierend auf Success Rate
                    current_multiplier = param_data.get('multiplier', 1.0)
                    
                    if success_rate > 0.7:
                        new_multiplier = min(1.5, current_multiplier * 1.05)
                    elif success_rate < 0.3:
                        new_multiplier = max(0.5, current_multiplier * 0.95)
                    else:
                        new_multiplier = current_multiplier
                    
                    if abs(new_multiplier - current_multiplier) > 0.01:
                        param_data['multiplier'] = new_multiplier
                        updates_made.append(f'{param_key}_multiplier: {current_multiplier:.3f} → {new_multiplier:.3f}')
            
            if updates_made:
                self.logger.info(f"🎯 Adaptive Parameter Updates: {', '.join(updates_made)}")
            else:
                self.logger.debug("📊 Adaptive Parameters: Keine Updates erforderlich")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Adaptive Parameter Update fehlgeschlagen: {e}")
    
    def _regime_check_and_adapt(self, symbol: str):
        """Stündliche Regime-Prüfung und Anpassung"""
        try:
            if not hasattr(self.strategy, 'market_regime'):
                return
            
            current_regime = self.strategy.market_regime
            
            # Lade aktuelle Daten für Regime-Check
            recent_data = self.data_provider.get_latest(symbol, limit=100)
            if not recent_data.empty and hasattr(self.strategy, '_detect_market_regime'):
                self.strategy._detect_market_regime(recent_data)
                
                new_regime = self.strategy.market_regime
                
                if new_regime != current_regime:
                    self.regime_history.append({
                        'timestamp': datetime.now(),
                        'old_regime': current_regime,
                        'new_regime': new_regime,
                        'confidence': getattr(self.strategy, 'regime_confidence', 0.5)
                    })
                    
                    self.strategy_performance['regime_changes'] = \
                        self.strategy_performance.get('regime_changes', 0) + 1
                    
                    self.logger.info(f"📊 Regime-Wechsel erkannt: {current_regime} → {new_regime}")
                    
                    # Regime-Change Alert
                    if self.email_alerter and self.config.get('alerts', {}).get('email_regime_change', False):
                        self._send_regime_change_alert(current_regime, new_regime, symbol)
                        
        except Exception as e:
            self.logger.warning(f"⚠️ Regime Check fehlgeschlagen: {e}")
    
    def _cleanup_intelligent_live_trading(self):
        """Cleanup nach intelligentem Live-Trading"""
        try:
            self.logger.info("🧹 Cleanup intelligentes Live-Trading...")
            
            # Finale Performance-Zusammenfassung
            if hasattr(self, 'monitoring_start_time'):
                runtime = datetime.now() - self.monitoring_start_time
                self.logger.info(f"📊 Live-Trading Laufzeit: {runtime}")
                
                if hasattr(self, 'monitoring_metrics'):
                    metrics = self.monitoring_metrics
                    self.logger.info(f"📈 Trading Statistiken:")
                    self.logger.info(f"  - Zyklen: {metrics.get('cycles_completed', 0)}")
                    self.logger.info(f"  - Signale: {self.strategy_performance.get('signals_generated', 0)}")
                    self.logger.info(f"  - Trades: {self.strategy_performance.get('signals_executed', 0)}")
                    self.logger.info(f"  - Regime-Wechsel: {self.strategy_performance.get('regime_changes', 0)}")
                    self.logger.info(f"  - Online Learning Updates: {self.strategy_performance.get('online_learning_updates', 0)}")
            
            # Optional: Speichere finale Strategie-Zustand
            if hasattr(self.strategy, '_save_training_state'):
                try:
                    recent_data = pd.DataFrame([{
                        'timestamp': datetime.now(),
                        'final_cleanup': True
                    }])
                    self.strategy._save_training_state(recent_data)
                    self.logger.info("💾 Finaler Strategie-Zustand gespeichert")
                except Exception as e:
                    self.logger.warning(f"⚠️ Finaler Strategie-Save fehlgeschlagen: {e}")
            
        except Exception as e:
            self.logger.warning(f