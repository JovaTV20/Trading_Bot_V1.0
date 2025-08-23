"""
Machine Learning Trading Strategy
Implementiert eine ML-basierte Trading-Strategie mit Scikit-Learn
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import logging

# Machine Learning Imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, accuracy_score
    import joblib
except ImportError as e:
    logging.error("Scikit-learn nicht installiert. Installiere mit: pip install scikit-learn")
    raise ImportError("Scikit-learn Paket erforderlich") from e

# Technical Analysis
try:
    import ta
except ImportError:
    ta = None
    logging.warning("TA-Lib nicht verfügbar. Verwende einfache technische Indikatoren.")

from core.base_strategy import StrategyBase

class MLStrategy(StrategyBase):
    """
    Machine Learning Trading Strategy
    
    Verwendet verschiedene ML-Modelle für Trading-Signale:
    - Random Forest Classifier
    - Logistic Regression
    - Support Vector Machine (optional)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialisiert ML-Strategie
        
        Args:
            config: Strategie-Konfiguration
        """
        super().__init__(config)
        
        # ML-Parameter
        self.lookback_period = self.parameters.get('lookback_period', 20)
        self.prediction_threshold = self.parameters.get('prediction_threshold', 0.6)
        self.train_test_split_ratio = self.parameters.get('train_test_split', 0.8)
        self.model_type = self.parameters.get('model_type', 'random_forest')
        
        # Feature-Konfiguration
        self.feature_config = self.parameters.get('features', {
            'returns': True,
            'moving_averages': [5, 10, 20],
            'rsi': True,
            'bollinger_bands': True,
            'momentum': True,
            'volume': True
        })
        
        # Modell und Scaler
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.training_score = 0.0
        
        # Initialisiere Modell
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialisiert das ML-Modell"""
        
        model_type = self.model_type.lower()
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver='liblinear'
            )
        else:
            self.logger.warning(f"Unbekannter Modell-Typ: {model_type}. Verwende Random Forest.")
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            
        self.logger.info(f"ML-Modell initialisiert: {type(self.model).__name__}")
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Trainiert das ML-Modell mit historischen Daten
        
        Args:
            data: DataFrame mit OHLCV-Daten
        """
        if not self.validate_data(data):
            raise ValueError("Ungültige Trainingsdaten")
            
        self.logger.info(f"Trainiere ML-Modell mit {len(data)} Datenpunkten")
        
        try:
            # Bereite Features vor
            features_df = self._prepare_features(data)
            
            if features_df.empty:
                raise ValueError("Keine Features generiert")
            
            # Erstelle Targets (1 = Buy, 0 = Hold, -1 = Sell -> 2, 1, 0 für Classifier)
            targets = self._create_targets(data)
            
            # Entferne NaN-Werte
            combined_df = features_df.join(targets, how='inner')
            combined_df = combined_df.dropna()
            
            if len(combined_df) < self.lookback_period * 2:
                raise ValueError("Nicht genügend gültige Daten nach Feature-Engineering")
            
            X = combined_df.drop('target', axis=1)
            y = combined_df['target']
            
            # Speichere Feature-Namen
            self.feature_names = list(X.columns)
            
            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=1 - self.train_test_split_ratio,
                random_state=42,
                stratify=y
            )
            
            # Skaliere Features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Trainiere Modell
            self.logger.info("Trainiere Modell...")
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluiere Modell
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            self.training_score = test_score
            
            self.logger.info(f"Modell trainiert - Train Score: {train_score:.3f}, Test Score: {test_score:.3f}")
            
            # Detaillierte Evaluation
            y_pred = self.model.predict(X_test_scaled)
            self.logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
            
            # Feature Importance (falls verfügbar)
            if hasattr(self.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                self.logger.info("Top 5 wichtigste Features:")
                for _, row in importance_df.head().iterrows():
                    self.logger.info(f"  {row['feature']}: {row['importance']:.3f}")
            
            self.is_fitted = True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Training: {e}", exc_info=True)
            raise
    
    def generate_signal(self, row: pd.Series) -> Dict[str, Any]:
        """
        Generiert Trading-Signal für eine Datenzeile
        
        Args:
            row: Series mit OHLCV-Daten
            
        Returns:
            Dict mit Signal-Informationen
        """
        if not self.is_fitted:
            return {
                'action': 'hold',
                'confidence': 0.0,
                'position_size': 0.0
            }
        
        try:
            # Konvertiere Series zu DataFrame für Feature-Erstellung
            df = pd.DataFrame([row])
            
            # Erstelle Features
            features = self._prepare_features_single(df)
            
            if features is None or len(features) == 0:
                return {
                    'action': 'hold',
                    'confidence': 0.0,
                    'position_size': 0.0
                }
            
            # Skaliere Features
            features_scaled = self.scaler.transform([features])
            
            # Vorhersage
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Konvertiere Prediction zurück zu Action
            if prediction == 2:  # Buy
                action = 'buy'
                confidence = probabilities[2]
            elif prediction == 0:  # Sell
                action = 'sell'
                confidence = probabilities[0]
            else:  # Hold
                action = 'hold'
                confidence = probabilities[1]
            
            # Prüfe Threshold
            if confidence < self.prediction_threshold:
                action = 'hold'
                confidence = 0.0
            
            # Berechne Positionsgröße
            position_size = self.calculate_position_size(confidence) if action != 'hold' else 0.0
            
            return {
                'action': action,
                'confidence': confidence,
                'position_size': position_size,
                'prediction_raw': prediction,
                'probabilities': {
                    'sell': probabilities[0],
                    'hold': probabilities[1],
                    'buy': probabilities[2]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei Signal-Generierung: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'position_size': 0.0,
                'error': str(e)
            }
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Bereitet Features für das Training vor
        
        Args:
            data: OHLCV-Daten
            
        Returns:
            DataFrame mit Features
        """
        df = data.copy()
        features = pd.DataFrame(index=df.index)
        
        try:
            # Price-based Features
            if self.feature_config.get('returns', True):
                # Returns verschiedener Perioden
                for period in [1, 2, 5, 10]:
                    features[f'return_{period}d'] = df['close'].pct_change(period)
                
                # Log Returns
                features['log_return'] = np.log(df['close'] / df['close'].shift(1))
                
                # Volatilität
                features['volatility_10d'] = features['return_1d'].rolling(10).std()
                features['volatility_20d'] = features['return_1d'].rolling(20).std()
            
            # Moving Averages
            ma_periods = self.feature_config.get('moving_averages', [5, 10, 20])
            for period in ma_periods:
                features[f'sma_{period}'] = df['close'].rolling(period).mean()
                features[f'sma_ratio_{period}'] = df['close'] / features[f'sma_{period}']
                
                # EMA
                features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                features[f'ema_ratio_{period}'] = df['close'] / features[f'ema_{period}']
            
            # RSI
            if self.feature_config.get('rsi', True):
                features['rsi'] = self._calculate_rsi(df['close'])
                features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
                features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
            
            # Bollinger Bands
            if self.feature_config.get('bollinger_bands', True):
                bb_period = 20
                bb_std = 2
                sma = df['close'].rolling(bb_period).mean()
                std = df['close'].rolling(bb_period).std()
                
                features['bb_upper'] = sma + (std * bb_std)
                features['bb_lower'] = sma - (std * bb_std)
                features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
                features['bb_squeeze'] = std / sma  # Bollinger Band Squeeze
            
            # Momentum Indicators
            if self.feature_config.get('momentum', True):
                # Price Momentum
                for period in [5, 10, 20]:
                    features[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
                
                # Rate of Change
                features['roc_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
            
            # Volume Features
            if self.feature_config.get('volume', True) and 'volume' in df.columns:
                # Volume Moving Averages
                features['volume_sma_10'] = df['volume'].rolling(10).mean()
                features['volume_sma_20'] = df['volume'].rolling(20).mean()
                features['volume_ratio'] = df['volume'] / features['volume_sma_20']
                
                # Price-Volume Features
                features['pv_trend'] = (df['close'].pct_change() * df['volume']).rolling(5).mean()
                
                # On Balance Volume (vereinfacht)
                price_change = df['close'].diff()
                features['obv'] = (np.where(price_change > 0, df['volume'], 
                                          np.where(price_change < 0, -df['volume'], 0))).cumsum()
                features['obv_sma'] = features['obv'].rolling(20).mean()
            
            # MACD
            if self.feature_config.get('macd', False):
                ema12 = df['close'].ewm(span=12).mean()
                ema26 = df['close'].ewm(span=26).mean()
                features['macd'] = ema12 - ema26
                features['macd_signal'] = features['macd'].ewm(span=9).mean()
                features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # Stochastic Oscillator
            if self.feature_config.get('stochastic', False):
                period = 14
                low_min = df['low'].rolling(period).min()
                high_max = df['high'].rolling(period).max()
                features['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
                features['stoch_d'] = features['stoch_k'].rolling(3).mean()
            
            # Market Microstructure
            if 'high' in df.columns and 'low' in df.columns:
                # True Range
                features['true_range'] = np.maximum(
                    df['high'] - df['low'],
                    np.maximum(
                        abs(df['high'] - df['close'].shift(1)),
                        abs(df['low'] - df['close'].shift(1))
                    )
                )
                features['atr'] = features['true_range'].rolling(14).mean()
                
                # Price Range
                features['price_range'] = (df['high'] - df['low']) / df['close']
                
                # Gap Features
                features['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
                features['gap_up'] = (features['gap'] > 0.01).astype(int)
                features['gap_down'] = (features['gap'] < -0.01).astype(int)
            
            # Time-based Features
            if hasattr(df.index, 'dayofweek'):
                features['day_of_week'] = df.index.dayofweek
                features['month'] = df.index.month
                features['quarter'] = df.index.quarter
            
            self.logger.debug(f"Features erstellt: {list(features.columns)}")
            return features
            
        except Exception as e:
            self.logger.error(f"Fehler bei Feature-Erstellung: {e}")
            return pd.DataFrame()
    
    def _prepare_features_single(self, df: pd.DataFrame) -> Optional[list]:
        """
        Bereitet Features für eine einzelne Vorhersage vor
        
        Args:
            df: DataFrame mit einer Zeile
            
        Returns:
            Liste mit Feature-Werten
        """
        try:
            # Für Single-Row Prediction müssen wir historische Daten simulieren
            # oder die Features anders berechnen
            
            # Vereinfachte Feature-Berechnung für Live-Daten
            row = df.iloc[0]
            
            features = []
            
            # Basis-Features (falls verfügbar aus den Daten)
            if 'return_1d' in row:
                features.extend([
                    row.get('return_1d', 0),
                    row.get('return_5d', 0),
                    row.get('volatility_20d', 0)
                ])
            
            # Moving Average Ratios
            for period in self.feature_config.get('moving_averages', [5, 10, 20]):
                sma_col = f'sma_{period}'
                if sma_col in row:
                    features.append(row['close'] / row[sma_col] if row[sma_col] != 0 else 1.0)
                else:
                    features.append(1.0)  # Neutral value
            
            # RSI
            if 'rsi' in row:
                rsi_val = row['rsi']
                features.extend([
                    rsi_val,
                    1 if rsi_val < 30 else 0,  # oversold
                    1 if rsi_val > 70 else 0   # overbought
                ])
            else:
                features.extend([50, 0, 0])  # Neutral RSI
            
            # Volume Ratio
            if 'volume_ratio' in row:
                features.append(row['volume_ratio'])
            else:
                features.append(1.0)
            
            # Erweitere auf erwartete Feature-Anzahl falls nötig
            expected_features = len(self.feature_names) if self.feature_names else 20
            while len(features) < expected_features:
                features.append(0.0)  # Padding mit neutralen Werten
            
            # Beschränke auf erwartete Anzahl
            features = features[:expected_features]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Fehler bei Single-Feature-Erstellung: {e}")
            return None
    
    def _create_targets(self, data: pd.DataFrame) -> pd.Series:
        """
        Erstellt Target-Labels für das Training
        
        Args:
            data: OHLCV-Daten
            
        Returns:
            Series mit Target-Labels (0=Sell, 1=Hold, 2=Buy)
        """
        df = data.copy()
        
        # Berechne Future Returns
        future_returns = df['close'].shift(-1) / df['close'] - 1
        
        # Definiere Thresholds
        buy_threshold = 0.01   # 1% Gewinn
        sell_threshold = -0.01  # 1% Verlust
        
        # Erstelle Labels
        targets = pd.Series(1, index=df.index)  # Default: Hold
        
        # Buy Signals
        targets[future_returns > buy_threshold] = 2
        
        # Sell Signals
        targets[future_returns < sell_threshold] = 0
        
        # Entferne letzten Wert (kein Future Return verfügbar)
        targets = targets[:-1]
        
        self.logger.info(f"Target Distribution - Sell: {(targets==0).sum()}, "
                        f"Hold: {(targets==1).sum()}, Buy: {(targets==2).sum()}")
        
        return targets.to_frame('target')['target']
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Berechnet RSI (Relative Strength Index)
        
        Args:
            prices: Preis-Series
            period: RSI-Periode
            
        Returns:
            RSI-Werte
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def optimize_hyperparameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimiert Hyperparameter mit Grid Search
        
        Args:
            data: Trainingsdaten
            
        Returns:
            Dict mit besten Parametern
        """
        self.logger.info("Starte Hyperparameter-Optimierung...")
        
        try:
            # Bereite Daten vor
            features_df = self._prepare_features(data)
            targets = self._create_targets(data)
            
            combined_df = features_df.join(targets, how='inner').dropna()
            X = combined_df.drop('target', axis=1)
            y = combined_df['target']
            
            # Skaliere Features
            X_scaled = self.scaler.fit_transform(X)
            
            # Parameter Grid
            if self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            else:  # Logistic Regression
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'lbfgs']
                }
            
            # Grid Search
            grid_search = GridSearchCV(
                self.model,
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_scaled, y)
            
            # Update Modell mit besten Parametern
            self.model = grid_search.best_estimator_
            
            self.logger.info(f"Beste Parameter: {grid_search.best_params_}")
            self.logger.info(f"Bester Score: {grid_search.best_score_:.3f}")
            
            return {
                'best_parameters': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei Hyperparameter-Optimierung: {e}")
            return {}
    
    def save_model(self, filepath: str) -> bool:
        """
        Speichert trainiertes Modell
        
        Args:
            filepath: Dateipfad zum Speichern
            
        Returns:
            True wenn erfolgreich
        """
        if not self.is_fitted:
            self.logger.error("Modell ist nicht trainiert")
            return False
            
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'parameters': self.parameters,
                'training_score': self.training_score
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Modell gespeichert: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Lädt gespeichertes Modell
        
        Args:
            filepath: Dateipfad zum Laden
            
        Returns:
            True wenn erfolgreich
        """
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.training_score = model_data.get('training_score', 0.0)
            
            self.is_fitted = True
            self.logger.info(f"Modell geladen: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden: {e}")
            return False
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Gibt Feature-Wichtigkeit zurück
        
        Returns:
            DataFrame mit Feature-Wichtigkeit
        """
        if not self.is_fitted or not hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame()
            
        try:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            self.logger.error(f"Fehler bei Feature-Importance: {e}")
            return pd.DataFrame()
    
    def predict_probabilities(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Gibt Wahrscheinlichkeiten für alle Klassen zurück
        
        Args:
            data: Eingabedaten
            
        Returns:
            DataFrame mit Wahrscheinlichkeiten
        """
        if not self.is_fitted:
            return pd.DataFrame()
            
        try:
            features = self._prepare_features(data)
            features_clean = features.dropna()
            
            if features_clean.empty:
                return pd.DataFrame()
            
            # Stelle sicher, dass Features korrekt sind
            if len(features_clean.columns) != len(self.feature_names):
                self.logger.warning("Feature-Anzahl stimmt nicht überein")
                return pd.DataFrame()
            
            features_scaled = self.scaler.transform(features_clean)
            probabilities = self.model.predict_proba(features_scaled)
            
            prob_df = pd.DataFrame(
                probabilities,
                columns=['sell_prob', 'hold_prob', 'buy_prob'],
                index=features_clean.index
            )
            
            return prob_df
            
        except Exception as e:
            self.logger.error(f"Fehler bei Wahrscheinlichkeits-Vorhersage: {e}")
            return pd.DataFrame()
    
    def backtest_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Führt Backtest der generierten Signale aus
        
        Args:
            data: Historische Daten
            
        Returns:
            Dict mit Backtest-Ergebnissen
        """
        if not self.is_fitted:
            raise ValueError("Modell muss trainiert sein")
            
        try:
            signals = []
            
            # Generiere Signale für alle Datenpunkte
            for i, (timestamp, row) in enumerate(data.iterrows()):
                if i < self.lookback_period:  # Skip initial period
                    continue
                    
                signal = self.generate_signal(row)
                signal['timestamp'] = timestamp
                signal['price'] = row['close']
                signals.append(signal)
            
            signals_df = pd.DataFrame(signals)
            
            # Berechne Statistiken
            total_signals = len(signals_df)
            buy_signals = len(signals_df[signals_df['action'] == 'buy'])
            sell_signals = len(signals_df[signals_df['action'] == 'sell'])
            hold_signals = len(signals_df[signals_df['action'] == 'hold'])
            
            avg_confidence = signals_df[signals_df['action'] != 'hold']['confidence'].mean()
            
            return {
                'total_signals': total_signals,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'hold_signals': hold_signals,
                'buy_ratio': buy_signals / total_signals if total_signals > 0 else 0,
                'sell_ratio': sell_signals / total_signals if total_signals > 0 else 0,
                'average_confidence': avg_confidence,
                'signals_dataframe': signals_df
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Signal-Backtest: {e}")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über das Modell zurück
        
        Returns:
            Dict mit Modell-Informationen
        """
        info = {
            'strategy_name': 'ML Strategy',
            'model_type': type(self.model).__name__ if self.model else 'None',
            'is_fitted': self.is_fitted,
            'training_score': self.training_score,
            'feature_count': len(self.feature_names),
            'lookback_period': self.lookback_period,
            'prediction_threshold': self.prediction_threshold,
            'parameters': self.parameters
        }
        
        # Modell-spezifische Informationen
        if self.is_fitted and self.model:
            if hasattr(self.model, 'n_estimators'):
                info['n_estimators'] = self.model.n_estimators
            if hasattr(self.model, 'max_depth'):
                info['max_depth'] = self.model.max_depth
            if hasattr(self.model, 'C'):
                info['C'] = self.model.C
        
        return info
    
    def update_parameters(self, new_parameters: Dict[str, Any]) -> None:
        """
        Aktualisiert Strategie-Parameter
        
        Args:
            new_parameters: Neue Parameter
        """
        self.parameters.update(new_parameters)
        
        # Update interne Parameter
        self.lookback_period = self.parameters.get('lookback_period', self.lookback_period)
        self.prediction_threshold = self.parameters.get('prediction_threshold', self.prediction_threshold)
        self.train_test_split_ratio = self.parameters.get('train_test_split', self.train_test_split_ratio)
        
        # Feature-Konfiguration aktualisieren
        if 'features' in new_parameters:
            self.feature_config = new_parameters['features']
        
        # Modell neu initialisieren falls Typ geändert wurde
        if 'model_type' in new_parameters:
            self.model_type = new_parameters['model_type']
            self._initialize_model()
            self.is_fitted = False  # Muss neu trainiert werden
        
        self.logger.info("Parameter aktualisiert")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Erweitert Basis-Info um ML-spezifische Informationen
        
        Returns:
            Dict mit detaillierten Informationen
        """
        info = super().get_info()
        info.update(self.get_model_info())
        
        if self.is_fitted:
            feature_importance = self.get_feature_importance()
            if not feature_importance.empty:
                info['top_features'] = feature_importance.head(5).to_dict('records')
        
        return info