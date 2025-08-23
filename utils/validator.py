"""
Validation Utilities für TradingBot
Validiert Eingabedaten, Konfigurationen und Parameter
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

class ConfigValidator:
    """
    Validiert Konfigurationsdateien und Parameter
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Definiere gültige Werte
        self.valid_timeframes = ['1Min', '5Min', '15Min', '30Min', '1Hour', '1Day', '1Week', '1Month']
        self.valid_order_types = ['market', 'limit', 'stop', 'stop_limit']
        self.valid_time_in_force = ['day', 'gtc', 'ioc', 'fok']
        self.valid_actions = ['buy', 'sell', 'hold']
        self.valid_model_types = ['random_forest', 'logistic_regression', 'svm']
        
    def validate_config_file(self, config_path: str) -> Dict[str, Any]:
        """
        Validiert komplette Konfigurationsdatei
        
        Args:
            config_path: Pfad zur Config-Datei
            
        Returns:
            Dict mit Validierungsergebnissen
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'config': None
        }
        
        try:
            # Prüfe ob Datei existiert
            if not Path(config_path).exists():
                result['valid'] = False
                result['errors'].append(f"Config-Datei nicht gefunden: {config_path}")
                return result
            
            # Lade und parse JSON
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                result['config'] = config
            except json.JSONDecodeError as e:
                result['valid'] = False
                result['errors'].append(f"Ungültiges JSON-Format: {e}")
                return result
            
            # Validiere Hauptsektionen
            required_sections = ['strategy', 'risk_management', 'execution', 'backtest']
            for section in required_sections:
                if section not in config:
                    result['errors'].append(f"Fehlende Sektion: {section}")
                    result['valid'] = False
            
            # Validiere Strategy-Sektion
            if 'strategy' in config:
                self._validate_strategy_config(config['strategy'], result)
            
            # Validiere Risk-Management
            if 'risk_management' in config:
                self._validate_risk_config(config['risk_management'], result)
            
            # Validiere Execution
            if 'execution' in config:
                self._validate_execution_config(config['execution'], result)
            
            # Validiere Backtest
            if 'backtest' in config:
                self._validate_backtest_config(config['backtest'], result)
            
            # Validiere Data-Sektion
            if 'data' in config:
                self._validate_data_config(config['data'], result)
            
            self.logger.info(f"Config-Validierung: {'✅ Gültig' if result['valid'] else '❌ Ungültig'}")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Unerwarteter Fehler: {e}")
            self.logger.error(f"Config-Validierung fehlgeschlagen: {e}")
        
        return result
    
    def _validate_strategy_config(self, strategy_config: Dict[str, Any], result: Dict[str, Any]):
        """Validiert Strategy-Konfiguration"""
        
        # Strategy Name
        if 'name' not in strategy_config:
            result['errors'].append("Strategy: 'name' fehlt")
            result['valid'] = False
        elif strategy_config['name'] not in ['ml_strategy']:
            result['warnings'].append(f"Unbekannte Strategie: {strategy_config['name']}")
        
        # Parameter
        if 'parameters' in strategy_config:
            params = strategy_config['parameters']
            
            # Lookback Period
            if 'lookback_period' in params:
                if not isinstance(params['lookback_period'], int) or params['lookback_period'] < 1:
                    result['errors'].append("Strategy: lookback_period muss positive Ganzzahl sein")
                    result['valid'] = False
            
            # Prediction Threshold
            if 'prediction_threshold' in params:
                if not isinstance(params['prediction_threshold'], (int, float)) or not 0 <= params['prediction_threshold'] <= 1:
                    result['errors'].append("Strategy: prediction_threshold muss zwischen 0 und 1 liegen")
                    result['valid'] = False
            
            # Train-Test Split
            if 'train_test_split' in params:
                if not isinstance(params['train_test_split'], (int, float)) or not 0 < params['train_test_split'] < 1:
                    result['errors'].append("Strategy: train_test_split muss zwischen 0 und 1 liegen")
                    result['valid'] = False
            
            # Model Type
            if 'model_type' in params:
                if params['model_type'] not in self.valid_model_types:
                    result['warnings'].append(f"Unbekannter Model-Type: {params['model_type']}")
    
    def _validate_risk_config(self, risk_config: Dict[str, Any], result: Dict[str, Any]):
        """Validiert Risk-Management-Konfiguration"""
        
        # Max Position Size
        if 'max_position_size' in risk_config:
            if not isinstance(risk_config['max_position_size'], (int, float)) or not 0 < risk_config['max_position_size'] <= 1:
                result['errors'].append("Risk: max_position_size muss zwischen 0 und 1 liegen")
                result['valid'] = False
        
        # Stop Loss
        if 'stop_loss_pct' in risk_config:
            if not isinstance(risk_config['stop_loss_pct'], (int, float)) or risk_config['stop_loss_pct'] < 0:
                result['errors'].append("Risk: stop_loss_pct muss >= 0 sein")
                result['valid'] = False
        
        # Take Profit
        if 'take_profit_pct' in risk_config:
            if not isinstance(risk_config['take_profit_pct'], (int, float)) or risk_config['take_profit_pct'] < 0:
                result['errors'].append("Risk: take_profit_pct muss >= 0 sein")
                result['valid'] = False
        
        # Max Daily Trades
        if 'max_daily_trades' in risk_config:
            if not isinstance(risk_config['max_daily_trades'], int) or risk_config['max_daily_trades'] < 0:
                result['errors'].append("Risk: max_daily_trades muss >= 0 sein")
                result['valid'] = False
    
    def _validate_execution_config(self, exec_config: Dict[str, Any], result: Dict[str, Any]):
        """Validiert Execution-Konfiguration"""
        
        # Order Type
        if 'order_type' in exec_config:
            if exec_config['order_type'] not in self.valid_order_types:
                result['errors'].append(f"Execution: Ungültiger order_type: {exec_config['order_type']}")
                result['valid'] = False
        
        # Time in Force
        if 'time_in_force' in exec_config:
            if exec_config['time_in_force'] not in self.valid_time_in_force:
                result['errors'].append(f"Execution: Ungültiger time_in_force: {exec_config['time_in_force']}")
                result['valid'] = False
    
    def _validate_backtest_config(self, backtest_config: Dict[str, Any], result: Dict[str, Any]):
        """Validiert Backtest-Konfiguration"""
        
        # Commission
        if 'commission' in backtest_config:
            if not isinstance(backtest_config['commission'], (int, float)) or backtest_config['commission'] < 0:
                result['errors'].append("Backtest: commission muss >= 0 sein")
                result['valid'] = False
        
        # Slippage
        if 'slippage' in backtest_config:
            if not isinstance(backtest_config['slippage'], (int, float)) or backtest_config['slippage'] < 0:
                result['errors'].append("Backtest: slippage muss >= 0 sein")
                result['valid'] = False
        
        # Initial Capital
        if 'initial_capital' in backtest_config:
            if not isinstance(backtest_config['initial_capital'], (int, float)) or backtest_config['initial_capital'] <= 0:
                result['errors'].append("Backtest: initial_capital muss > 0 sein")
                result['valid'] = False
    
    def _validate_data_config(self, data_config: Dict[str, Any], result: Dict[str, Any]):
        """Validiert Data-Konfiguration"""
        
        # Timeframe
        if 'timeframe' in data_config:
            if data_config['timeframe'] not in self.valid_timeframes:
                result['errors'].append(f"Data: Ungültiger timeframe: {data_config['timeframe']}")
                result['valid'] = False

class DataValidator:
    """
    Validiert Trading-Daten und DataFrames
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Erwartete OHLCV-Spalten
        self.required_ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        
    def validate_ohlcv_data(self, df: pd.DataFrame, symbol: str = None) -> Dict[str, Any]:
        """
        Validiert OHLCV-DataFrame
        
        Args:
            df: OHLCV-DataFrame
            symbol: Symbol-Name für Logging
            
        Returns:
            Dict mit Validierungsergebnissen
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            symbol_str = f" für {symbol}" if symbol else ""
            self.logger.debug(f"Validiere OHLCV-Daten{symbol_str}")
            
            # Basis-Validierung
            if df.empty:
                result['valid'] = False
                result['errors'].append("DataFrame ist leer")
                return result
            
            # Spalten prüfen
            missing_columns = [col for col in self.required_ohlcv_columns if col not in df.columns]
            if missing_columns:
                result['valid'] = False
                result['errors'].append(f"Fehlende Spalten: {missing_columns}")
            
            # Index prüfen (sollte DateTime sein)
            if not isinstance(df.index, pd.DatetimeIndex):
                result['warnings'].append("Index ist kein DatetimeIndex")
            
            # Datentypen prüfen
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    result['warnings'].append(f"Spalte {col} ist nicht numerisch")
            
            # OHLC-Logik prüfen
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # High >= Low
                invalid_high_low = df['high'] < df['low']
                if invalid_high_low.any():
                    count = invalid_high_low.sum()
                    result['errors'].append(f"{count} Zeilen wo High < Low")
                    result['valid'] = False
                
                # High >= Open, Close
                invalid_high_open = df['high'] < df['open']
                invalid_high_close = df['high'] < df['close']
                if invalid_high_open.any():
                    count = invalid_high_open.sum()
                    result['warnings'].append(f"{count} Zeilen wo High < Open")
                if invalid_high_close.any():
                    count = invalid_high_close.sum()
                    result['warnings'].append(f"{count} Zeilen wo High < Close")
                
                # Low <= Open, Close
                invalid_low_open = df['low'] > df['open']
                invalid_low_close = df['low'] > df['close']
                if invalid_low_open.any():
                    count = invalid_low_open.sum()
                    result['warnings'].append(f"{count} Zeilen wo Low > Open")
                if invalid_low_close.any():
                    count = invalid_low_close.sum()
                    result['warnings'].append(f"{count} Zeilen wo Low > Close")
            
            # Negative Preise
            price_columns = [col for col in ['open', 'high', 'low', 'close'] if col in df.columns]
            for col in price_columns:
                negative_prices = df[col] <= 0
                if negative_prices.any():
                    count = negative_prices.sum()
                    result['errors'].append(f"{count} negative/null Preise in {col}")
                    result['valid'] = False
            
            # Negative Volume
            if 'volume' in df.columns:
                negative_volume = df['volume'] < 0
                if negative_volume.any():
                    count = negative_volume.sum()
                    result['warnings'].append(f"{count} negative Volume-Werte")
            
            # NaN-Werte
            nan_counts = df.isnull().sum()
            if nan_counts.any():
                for col, count in nan_counts[nan_counts > 0].items():
                    result['warnings'].append(f"{count} NaN-Werte in {col}")
            
            # Duplikate im Index
            if df.index.duplicated().any():
                count = df.index.duplicated().sum()
                result['warnings'].append(f"{count} doppelte Timestamps")
            
            # Statistiken
            result['stats'] = {
                'total_rows': len(df),
                'date_range': {
                    'start': df.index.min() if isinstance(df.index, pd.DatetimeIndex) else None,
                    'end': df.index.max() if isinstance(df.index, pd.DatetimeIndex) else None
                },
                'columns': list(df.columns),
                'nan_counts': nan_counts.to_dict()
            }
            
            # Preisstatistiken
            if 'close' in df.columns:
                result['stats']['price_stats'] = {
                    'min': df['close'].min(),
                    'max': df['close'].max(),
                    'mean': df['close'].mean(),
                    'std': df['close'].std()
                }
            
            # Volume-Statistiken
            if 'volume' in df.columns:
                result['stats']['volume_stats'] = {
                    'min': df['volume'].min(),
                    'max': df['volume'].max(),
                    'mean': df['volume'].mean(),
                    'zero_volume_days': (df['volume'] == 0).sum()
                }
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validierungsfehler: {e}")
            self.logger.error(f"OHLCV-Validierung fehlgeschlagen: {e}")
        
        return result
    
    def validate_features_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validiert Feature-DataFrame für ML
        
        Args:
            df: Features-DataFrame
            
        Returns:
            Dict mit Validierungsergebnissen
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            if df.empty:
                result['valid'] = False
                result['errors'].append("Features-DataFrame ist leer")
                return result
            
            # Unendliche Werte
            inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
            if inf_counts.any():
                for col, count in inf_counts[inf_counts > 0].items():
                    result['warnings'].append(f"{count} unendliche Werte in {col}")
            
            # Konstante Spalten
            constant_columns = []
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].nunique() <= 1:
                    constant_columns.append(col)
            
            if constant_columns:
                result['warnings'].append(f"Konstante Spalten: {constant_columns}")
            
            # Hohe Korrelationen
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr().abs()
                high_corr_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if corr_matrix.iloc[i, j] > 0.95:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                
                if high_corr_pairs:
                    result['warnings'].append(f"Hohe Korrelationen: {high_corr_pairs[:5]}")  # Nur erste 5
            
            # Statistiken
            result['stats'] = {
                'total_rows': len(df),
                'total_features': len(df.columns),
                'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(df.select_dtypes(include=['object', 'category']).columns),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                'constant_columns': constant_columns
            }
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Feature-Validierungsfehler: {e}")
            self.logger.error(f"Feature-Validierung fehlgeschlagen: {e}")
        
        return result

class TradingValidator:
    """
    Validiert Trading-spezifische Parameter und Signale
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_trading_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validiert Trading-Signal
        
        Args:
            signal: Signal-Dictionary
            
        Returns:
            Dict mit Validierungsergebnissen
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Action
            if 'action' not in signal:
                result['valid'] = False
                result['errors'].append("Signal: 'action' fehlt")
            elif signal['action'] not in ['buy', 'sell', 'hold']:
                result['valid'] = False
                result['errors'].append(f"Signal: Ungültige action: {signal['action']}")
            
            # Confidence
            if 'confidence' in signal:
                confidence = signal['confidence']
                if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                    result['errors'].append("Signal: confidence muss zwischen 0 und 1 liegen")
                    result['valid'] = False
            
            # Position Size
            if 'position_size' in signal:
                pos_size = signal['position_size']
                if not isinstance(pos_size, (int, float)) or pos_size < 0:
                    result['errors'].append("Signal: position_size muss >= 0 sein")
                    result['valid'] = False
                elif pos_size > 1:
                    result['warnings'].append("Signal: position_size > 100% des Kapitals")
            
            # Stop Loss
            if 'stop_loss' in signal and signal['stop_loss'] is not None:
                if not isinstance(signal['stop_loss'], (int, float)) or signal['stop_loss'] <= 0:
                    result['errors'].append("Signal: stop_loss muss > 0 sein")
                    result['valid'] = False
            
            # Take Profit
            if 'take_profit' in signal and signal['take_profit'] is not None:
                if not isinstance(signal['take_profit'], (int, float)) or signal['take_profit'] <= 0:
                    result['errors'].append("Signal: take_profit muss > 0 sein")
                    result['valid'] = False
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Signal-Validierungsfehler: {e}")
            self.logger.error(f"Signal-Validierung fehlgeschlagen: {e}")
        
        return result
    
    def validate_order_parameters(self, symbol: str, side: str, quantity: float,
                                order_type: str = "market", price: float = None) -> Dict[str, Any]:
        """
        Validiert Order-Parameter
        
        Args:
            symbol: Trading-Symbol
            side: buy/sell
            quantity: Menge
            order_type: Order-Typ
            price: Preis (optional)
            
        Returns:
            Dict mit Validierungsergebnissen
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Symbol
            if not self.validate_symbol(symbol):
                result['valid'] = False
                result['errors'].append(f"Ungültiges Symbol: {symbol}")
            
            # Side
            if side not in ['buy', 'sell']:
                result['valid'] = False
                result['errors'].append(f"Ungültige Side: {side}")
            
            # Quantity
            if not isinstance(quantity, (int, float)) or quantity <= 0:
                result['valid'] = False
                result['errors'].append("Quantity muss > 0 sein")
            
            # Order Type
            valid_types = ['market', 'limit', 'stop', 'stop_limit']
            if order_type not in valid_types:
                result['valid'] = False
                result['errors'].append(f"Ungültiger Order-Type: {order_type}")
            
            # Preis bei Limit-Orders
            if order_type in ['limit', 'stop_limit']:
                if price is None or not isinstance(price, (int, float)) or price <= 0:
                    result['valid'] = False
                    result['errors'].append("Limit-Orders benötigen gültigen Preis")
            
            # Warn bei sehr großen Quantities
            if quantity > 10000:
                result['warnings'].append("Sehr große Quantity - prüfen Sie die Order")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Order-Validierungsfehler: {e}")
            self.logger.error(f"Order-Validierung fehlgeschlagen: {e}")
        
        return result
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validiert Trading-Symbol
        
        Args:
            symbol: Trading-Symbol
            
        Returns:
            True wenn gültig
        """
        if not symbol or not isinstance(symbol, str):
            return False
        
        # Basis-Validierung
        if len(symbol) < 1 or len(symbol) > 10:
            return False
        
        # Nur Buchstaben, Zahlen, Punkt und Bindestrich erlaubt
        if not re.match(r'^[A-Za-z0-9.-]+$', symbol):
            return False
        
        return True
    
    def validate_date_range(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Validiert Datumsbereich
        
        Args:
            start_date: Startdatum (YYYY-MM-DD)
            end_date: Enddatum (YYYY-MM-DD)
            
        Returns:
            Dict mit Validierungsergebnissen
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Datumsformat prüfen
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError as e:
                result['valid'] = False
                result['errors'].append(f"Ungültiges Datumsformat: {e}")
                return result
            
            # Start < End
            if start_dt >= end_dt:
                result['valid'] = False
                result['errors'].append("Startdatum muss vor Enddatum liegen")
            
            # Zeitraum nicht zu lang
            time_diff = end_dt - start_dt
            if time_diff.days > 365 * 5:  # 5 Jahre
                result['warnings'].append("Sehr langer Zeitraum (>5 Jahre) - Performance könnte leiden")
            
            # Nicht in der Zukunft
            today = datetime.now()
            if end_dt > today:
                result['warnings'].append("Enddatum liegt in der Zukunft")
            
            # Nicht zu alt
            if start_dt < datetime.now() - timedelta(days=365 * 20):  # 20 Jahre
                result['warnings'].append("Startdatum sehr alt (>20 Jahre)")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Datums-Validierungsfehler: {e}")
            self.logger.error(f"Datums-Validierung fehlgeschlagen: {e}")
        
        return result

class EnvironmentValidator:
    """
    Validiert Environment-Variablen und System-Setup
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_environment(self) -> Dict[str, Any]:
        """
        Validiert komplettes Environment-Setup
        
        Returns:
            Dict mit Validierungsergebnissen
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'checks': {}
        }
        
        try:
            # Alpaca API Keys
            alpaca_check = self.validate_alpaca_credentials()
            result['checks']['alpaca'] = alpaca_check
            if not alpaca_check['valid']:
                result['valid'] = False
                result['errors'].extend(alpaca_check['errors'])
            
            # Email-Konfiguration
            email_check = self.validate_email_config()
            result['checks']['email'] = email_check
            if not email_check['valid']:
                result['warnings'].extend(email_check['errors'])  # Email-Fehler sind nur Warnings
            
            # Python-Pakete
            packages_check = self.validate_required_packages()
            result['checks']['packages'] = packages_check
            if not packages_check['valid']:
                result['valid'] = False
                result['errors'].extend(packages_check['errors'])
            
            # Verzeichnisse
            dirs_check = self.validate_directories()
            result['checks']['directories'] = dirs_check
            if not dirs_check['valid']:
                result['warnings'].extend(dirs_check['errors'])
            
            # Config-Datei
            config_check = self.validate_config_exists()
            result['checks']['config'] = config_check
            if not config_check['valid']:
                result['valid'] = False
                result['errors'].extend(config_check['errors'])
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Environment-Validierungsfehler: {e}")
            self.logger.error(f"Environment-Validierung fehlgeschlagen: {e}")
        
        return result
    
    def validate_alpaca_credentials(self) -> Dict[str, Any]:
        """Validiert Alpaca API Credentials"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        import os
        
        # API Key
        api_key = os.getenv('ALPACA_API_KEY')
        if not api_key:
            result['valid'] = False
            result['errors'].append("ALPACA_API_KEY nicht gesetzt")
        elif len(api_key) < 20:  # Alpaca Keys sind länger
            result['warnings'].append("ALPACA_API_KEY scheint sehr kurz")
        
        # Secret Key
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        if not secret_key:
            result['valid'] = False
            result['errors'].append("ALPACA_SECRET_KEY nicht gesetzt")
        elif len(secret_key) < 40:  # Alpaca Secret Keys sind länger
            result['warnings'].append("ALPACA_SECRET_KEY scheint sehr kurz")
        
        # Base URL
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        if 'paper-api' in base_url:
            result['warnings'].append("Paper Trading aktiviert (ALPACA_BASE_URL)")
        
        return result
    
    def validate_email_config(self) -> Dict[str, Any]:
        """Validiert Email-Konfiguration"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        import os
        
        # Email-Adresse
        email_address = os.getenv('EMAIL_ADDRESS')
        if not email_address:
            result['valid'] = False
            result['errors'].append("EMAIL_ADDRESS nicht gesetzt")
        elif '@' not in email_address:
            result['valid'] = False
            result['errors'].append("EMAIL_ADDRESS ungültiges Format")
        
        # Email-Passwort
        email_password = os.getenv('EMAIL_PASSWORD')
        if not email_password:
            result['valid'] = False
            result['errors'].append("EMAIL_PASSWORD nicht gesetzt")
        
        # SMTP-Server
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        if not smtp_server:
            result['warnings'].append("SMTP_SERVER nicht gesetzt, verwende Standard")
        
        # Empfänger
        recipients = os.getenv('ALERT_RECIPIENTS', '')
        if not recipients:
            result['valid'] = False
            result['errors'].append("ALERT_RECIPIENTS nicht gesetzt")
        
        return result
    
    def validate_required_packages(self) -> Dict[str, Any]:
        """Validiert erforderliche Python-Pakete"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'installed': {},
            'missing': []
        }
        
        required_packages = {
            'pandas': '2.0.0',
            'numpy': '1.24.0',
            'scikit-learn': '1.3.0',
            'alpaca-trade-api': '3.0.0',
            'python-dotenv': '1.0.0',
            'matplotlib': '3.7.0',
            'ta': '0.10.0'
        }
        
        for package, min_version in required_packages.items():
            try:
                if package == 'alpaca-trade-api':
                    import alpaca_trade_api
                    version = alpaca_trade_api.__version__
                    result['installed'][package] = version
                elif package == 'scikit-learn':
                    import sklearn
                    version = sklearn.__version__
                    result['installed'][package] = version
                elif package == 'python-dotenv':
                    import dotenv
                    version = getattr(dotenv, '__version__', 'unknown')
                    result['installed'][package] = version
                else:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'unknown')
                    result['installed'][package] = version
                    
            except ImportError:
                result['missing'].append(package)
                result['valid'] = False
                result['errors'].append(f"Paket nicht installiert: {package}")
        
        return result
    
    def validate_directories(self) -> Dict[str, Any]:
        """Validiert erforderliche Verzeichnisse"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        required_dirs = ['logs', 'data', 'models']
        
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    result['warnings'].append(f"Verzeichnis erstellt: {dir_name}")
                except Exception as e:
                    result['valid'] = False
                    result['errors'].append(f"Konnte Verzeichnis nicht erstellen: {dir_name} - {e}")
            elif not dir_path.is_dir():
                result['valid'] = False
                result['errors'].append(f"{dir_name} existiert, ist aber kein Verzeichnis")
        
        return result
    
    def validate_config_exists(self) -> Dict[str, Any]:
        """Validiert ob config.json existiert"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        config_path = Path('config.json')
        if not config_path.exists():
            result['valid'] = False
            result['errors'].append("config.json nicht gefunden")
        elif not config_path.is_file():
            result['valid'] = False
            result['errors'].append("config.json ist keine Datei")
        
        return result

def validate_trading_setup(config_path: str = 'config.json') -> Dict[str, Any]:
    """
    Komplette Validierung des Trading-Setups
    
    Args:
        config_path: Pfad zur Config-Datei
        
    Returns:
        Dict mit allen Validierungsergebnissen
    """
    logger = logging.getLogger('TradingSetupValidator')
    
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'validations': {}
    }
    
    try:
        logger.info("Starte komplette Setup-Validierung...")
        
        # Environment validieren
        env_validator = EnvironmentValidator()
        env_result = env_validator.validate_environment()
        result['validations']['environment'] = env_result
        
        if not env_result['valid']:
            result['valid'] = False
            result['errors'].extend(env_result['errors'])
        result['warnings'].extend(env_result['warnings'])
        
        # Config validieren
        config_validator = ConfigValidator()
        config_result = config_validator.validate_config_file(config_path)
        result['validations']['config'] = config_result
        
        if not config_result['valid']:
            result['valid'] = False
            result['errors'].extend(config_result['errors'])
        result['warnings'].extend(config_result['warnings'])
        
        # Zusammenfassung
        total_errors = len(result['errors'])
        total_warnings = len(result['warnings'])
        
        if result['valid']:
            logger.info(f"✅ Setup-Validierung erfolgreich! ({total_warnings} Warnings)")
        else:
            logger.error(f"❌ Setup-Validierung fehlgeschlagen! ({total_errors} Errors, {total_warnings} Warnings)")
        
        result['summary'] = {
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'status': '✅ Gültig' if result['valid'] else '❌ Ungültig'
        }
        
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Setup-Validierungsfehler: {e}")
        logger.error(f"Setup-Validierung fehlgeschlagen: {e}")
    
    return result

def print_validation_report(validation_result: Dict[str, Any]):
    """
    Druckt formatierten Validierungsbericht
    
    Args:
        validation_result: Validierungsergebnis von validate_trading_setup()
    """
    print("\n" + "="*60)
    print("TRADING SETUP VALIDATION REPORT")
    print("="*60)
    
    # Status
    status = validation_result.get('summary', {}).get('status', 'Unbekannt')
    print(f"\nStatus: {status}")
    
    # Zusammenfassung
    summary = validation_result.get('summary', {})
    print(f"Errors: {summary.get('total_errors', 0)}")
    print(f"Warnings: {summary.get('total_warnings', 0)}")
    
    # Detaillierte Ergebnisse
    validations = validation_result.get('validations', {})
    
    for section, section_result in validations.items():
        print(f"\n{section.upper()}:")
        print("-" * 40)
        
        if section_result.get('valid', False):
            print("✅ Gültig")
        else:
            print("❌ Ungültig")
        
        # Errors
        errors = section_result.get('errors', [])
        if errors:
            print("\nErrors:")
            for error in errors:
                print(f"  • {error}")
        
        # Warnings
        warnings = section_result.get('warnings', [])
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  • {warning}")
        
        # Spezielle Informationen
        if section == 'environment':
            checks = section_result.get('checks', {})
            if 'packages' in checks:
                installed = checks['packages'].get('installed', {})
                if installed:
                    print("\nInstallierte Pakete:")
                    for pkg, version in installed.items():
                        print(f"  • {pkg}: {version}")
    
    print("\n" + "="*60)

# Convenience-Funktionen für häufige Validierungen

def quick_validate_symbol(symbol: str) -> bool:
    """Schnelle Symbol-Validierung"""
    validator = TradingValidator()
    return validator.validate_symbol(symbol)

def quick_validate_ohlcv(df: pd.DataFrame) -> bool:
    """Schnelle OHLCV-Validierung"""
    validator = DataValidator()
    result = validator.validate_ohlcv_data(df)
    return result['valid']

def quick_validate_signal(signal: Dict[str, Any]) -> bool:
    """Schnelle Signal-Validierung"""
    validator = TradingValidator()
    result = validator.validate_trading_signal(signal)
    return result['valid']

def quick_validate_config(config_path: str = 'config.json') -> bool:
    """Schnelle Config-Validierung"""
    validator = ConfigValidator()
    result = validator.validate_config_file(config_path)
    return result['valid']

# Decorator für automatische Parameter-Validierung

def validate_parameters(**param_validators):
    """
    Decorator für automatische Parameter-Validierung
    
    Usage:
        @validate_parameters(symbol=quick_validate_symbol, quantity=lambda x: x > 0)
        def place_order(symbol, quantity):
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get function parameter names
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate parameters
            for param_name, validator in param_validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Validierung fehlgeschlagen für Parameter '{param_name}': {value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    # Test-Validierung beim direkten Ausführen
    result = validate_trading_setup()
    print_validation_report(result)