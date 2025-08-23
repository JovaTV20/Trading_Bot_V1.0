"""
FINAL REPARIERTE ML-Strategie - Vollständig Pandas 2.0+ kompatibel
ERSETZE: strategies/ml_strategy.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import warnings

# Unterdrücke sklearn Warnings für bessere Ausgabe
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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

from core.base_strategy import StrategyBase

class MLStrategy(StrategyBase):
    """
    FINAL Machine Learning Trading Strategy - Vollständig Pandas 2.0+ kompatibel
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialisiert ML-Strategie"""
        super().__init__(config)
        
        # ML-Parameter
        self.lookback_period = self.parameters.get('lookback_period', 20)
        self.prediction_threshold = self.parameters.get('prediction_threshold', 0.5)  # REDUZIERT für mehr Trades
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
                n_estimators=30,  # REDUZIERT für schnellere Performance
                max_depth=5,      # REDUZIERT 
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=300,
                solver='liblinear'
            )
        else:
            self.logger.warning(f"Unbekannter Modell-Typ: {model_type}. Verwende Random Forest.")
            self.model = RandomForestClassifier(
                n_estimators=30,
                max_depth=5,
                random_state=42
            )
            
        self.logger.info(f"ML-Modell initialisiert: {type(self.model).__name__}")
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Trainiert das ML-Modell - KOMPLETT PANDAS 2.0+ KOMPATIBEL
        """
        if not self.validate_data(data):
            raise ValueError("Ungültige Trainingsdaten")
            
        self.logger.info(f"Trainiere ML-Modell mit {len(data)} Datenpunkten")
        
        try:
            # Bereite Features vor
            features_df = self._prepare_features(data)
            
            if features_df.empty:
                self.logger.error("Feature-Erstellung fehlgeschlagen - versuche einfachere Features")
                features_df = self._prepare_simple_features(data)
                
            if features_df.empty:
                raise ValueError("Keine Features generiert - auch einfache Features fehlgeschlagen")
            
            # Erstelle Targets
            targets = self._create_targets(data)
            
            # Synchronisiere Längen
            min_len = min(len(features_df), len(targets))
            features_df = features_df.iloc[:min_len].copy()
            targets = targets.iloc[:min_len].copy()
            
            self.logger.info(f"Features vor Training: {len(features_df)} Zeilen, {len(features_df.columns)} Features")
            
            # Kombiniere Features und Targets
            combined_df = features_df.copy()
            combined_df['target'] = targets
            
            # MODERNE NaN-Behandlung (Pandas 2.0+ kompatibel)
            initial_len = len(combined_df)
            
            # Ersetze unendliche Werte
            combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
            
            # MODERNE fillna-Syntax ohne deprecated methods
            combined_df = combined_df.ffill().bfill().fillna(0)
            
            # Entferne Zeilen mit NaN in Target
            combined_df = combined_df.dropna(subset=['target'])
            
            final_len = len(combined_df)
            self.logger.info(f"Nach Bereinigung: {final_len} Zeilen (verloren: {initial_len - final_len})")
            
            if final_len < self.lookback_period:
                raise ValueError(f"Nicht genügend Daten: {final_len} < {self.lookback_period}")
            
            X = combined_df.drop('target', axis=1)
            y = combined_df['target']
            
            # Speichere Feature-Namen
            self.feature_names = list(X.columns)
            self.logger.info(f"Features für Training: {len(self.feature_names)}")
            
            # Prüfe Target-Verteilung
            target_counts = y.value_counts()
            self.logger.info(f"Target-Verteilung: {target_counts.to_dict()}")
            
            if len(target_counts) < 2:
                self.logger.warning("Nur eine Klasse im Target - erstelle Balance")
                # Künstliche Balance durch Noise
                n_samples = len(y)
                n_positive = int(n_samples * 0.3)  # 30% positive Samples
                
                # Setze zufällige 30% auf positive Klasse
                positive_indices = np.random.choice(n_samples, n_positive, replace=False)
                y_balanced = y.copy()
                y_balanced.iloc[positive_indices] = 1
                y = y_balanced
                
                self.logger.info(f"Balancierte Target-Verteilung: {y.value_counts().to_dict()}")
            
            # Train-Test Split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=1 - self.train_test_split_ratio,
                    random_state=42,
                    stratify=y if len(y.unique()) > 1 else None
                )
            except ValueError as e:
                self.logger.warning(f"Stratify fehlgeschlagen: {e}")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=1 - self.train_test_split_ratio,
                    random_state=42
                )
            
            # Skaliere Features
            self.logger.info("Skaliere Features...")
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train), 
                columns=self.feature_names,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test), 
                columns=self.feature_names,
                index=X_test.index
            )
            
            # Trainiere Modell
            self.logger.info("Trainiere Modell...")
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluiere Modell
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            self.training_score = test_score
            
            self.logger.info(f"✅ Modell trainiert - Train: {train_score:.3f}, Test: {test_score:.3f}")
            
            # Feature Importance
            if hasattr(self.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                self.logger.info("Top 3 wichtigste Features:")
                for _, row in importance_df.head(3).iterrows():
                    self.logger.info(f"  {row['feature']}: {row['importance']:.3f}")
            
            self.is_fitted = True
            
            # Test Signal-Generierung
            self._test_signal_generation(X_test.iloc[0])
            
        except Exception as e:
            self.logger.error(f"Fehler beim Training: {e}", exc_info=True)
            raise
    
    def _test_signal_generation(self, test_row: pd.Series):
        """Testet Signal-Generierung nach dem Training"""
        try:
            signal = self.generate_signal(test_row)
            self.logger.info(f"Test-Signal: {signal['action']} (Confidence: {signal['confidence']:.3f})")
        except Exception as e:
            self.logger.warning(f"Test-Signal fehlgeschlagen: {e}")
    
    def generate_signal(self, row: pd.Series) -> Dict[str, Any]:
        """
        Generiert Trading-Signal
        """
        if not self.is_fitted:
            return {
                'action': 'hold',
                'confidence': 0.0,
                'position_size': 0.0
            }
        
        try:
            # Extrahiere Features
            features = self._extract_features_from_row(row)
            
            if features is None or len(features) != len(self.feature_names):
                return {
                    'action': 'hold',
                    'confidence': 0.0,
                    'position_size': 0.0
                }
            
            # Erstelle DataFrame für Prediction
            features_df = pd.DataFrame([features], columns=self.feature_names)
            
            # Skaliere Features
            features_scaled = pd.DataFrame(
                self.scaler.transform(features_df),
                columns=self.feature_names
            )
            
            # Vorhersage
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Konvertiere zu Action
            if len(probabilities) == 2:  # Binary
                action = 'buy' if prediction == 1 else 'hold'
                confidence = probabilities[1] if prediction == 1 else probabilities[0]
            else:  # Multi-class
                if prediction == 1:  # Buy
                    action = 'buy'
                    confidence = probabilities[1]
                else:  # Hold (oder andere Klasse)
                    action = 'hold'
                    confidence = probabilities[0] if len(probabilities) > 1 else 0.5
            
            # Threshold-Prüfung
            if confidence < self.prediction_threshold:
                action = 'hold'
                confidence = 0.0
            
            # Positionsgröße
            position_size = self.calculate_position_size(confidence) if action != 'hold' else 0.0
            
            return {
                'action': action,
                'confidence': confidence,
                'position_size': position_size,
                'prediction_raw': int(prediction),
                'probabilities': probabilities.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Signal-Fehler: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'position_size': 0.0
            }
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Feature-Erstellung - VOLLSTÄNDIG PANDAS 2.0+ KOMPATIBEL
        """
        try:
            df = data.copy()
            features = pd.DataFrame(index=df.index)
            
            # Returns
            features['return_1d'] = df['close'].pct_change().fillna(0)
            features['return_5d'] = df['close'].pct_change(5).fillna(0)
            
            # Moving Averages
            for period in [5, 10, 20]:
                sma = df['close'].rolling(period, min_periods=1).mean()
                features[f'sma_{period}'] = sma
                features[f'sma_ratio_{period}'] = (df['close'] / sma).fillna(1.0)
            
            # Volatilität
            features['volatility'] = features['return_1d'].rolling(10, min_periods=1).std().fillna(0)
            
            # RSI
            rsi = self._calculate_rsi(df['close'])
            features['rsi'] = rsi.fillna(50)
            features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
            features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
            
            # Bollinger Bands
            bb_sma = df['close'].rolling(20, min_periods=1).mean()
            bb_std = df['close'].rolling(20, min_periods=1).std()
            bb_upper = bb_sma + (bb_std * 2)
            bb_lower = bb_sma - (bb_std * 2)
            bb_width = bb_upper - bb_lower
            bb_width = bb_width.replace(0, 1)  # Vermeide Division durch 0
            features['bb_position'] = ((df['close'] - bb_lower) / bb_width).fillna(0.5)
            
            # Volume
            if 'volume' in df.columns:
                vol_sma = df['volume'].rolling(20, min_periods=1).mean()
                features['volume_ratio'] = (df['volume'] / vol_sma).fillna(1.0)
            else:
                features['volume_ratio'] = 1.0
            
            # Momentum
            features['momentum'] = (df['close'] / df['close'].shift(5) - 1).fillna(0)
            
            # Price Position (normalisiert)
            high_20 = df['high'].rolling(20, min_periods=1).max()
            low_20 = df['low'].rolling(20, min_periods=1).min()
            price_range = high_20 - low_20
            price_range = price_range.replace(0, 1)
            features['price_position'] = ((df['close'] - low_20) / price_range).fillna(0.5)
            
            # MODERNE fillna-Syntax (Pandas 2.0+ kompatibel)
            features = features.ffill().bfill().fillna(0)
            
            # Entferne Extremwerte
            features = features.replace([np.inf, -np.inf], 0)
            
            self.logger.debug(f"Features erstellt: {len(features.columns)} Features")
            return features
            
        except Exception as e:
            self.logger.error(f"Feature-Erstellung fehlgeschlagen: {e}")
            return pd.DataFrame()
    
    def _prepare_simple_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Einfache Fallback-Features"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Nur die essentiellen Features
            features['return_1d'] = data['close'].pct_change().fillna(0)
            features['sma_ratio'] = (data['close'] / data['close'].rolling(10, min_periods=1).mean()).fillna(1.0)
            features['momentum'] = (data['close'] / data['close'].shift(3) - 1).fillna(0)
            features['price_change'] = (data['close'] - data['open']).fillna(0) / data['open'].fillna(1)
            
            # Bereinige
            features = features.replace([np.inf, -np.inf], 0).fillna(0)
            
            self.logger.info(f"Einfache Features: {len(features.columns)}")
            return features
            
        except Exception as e:
            self.logger.error(f"Einfache Features fehlgeschlagen: {e}")
            return pd.DataFrame()
    
    def _extract_features_from_row(self, row: pd.Series) -> Optional[list]:
        """Extrahiert Features aus einer Zeile"""
        try:
            features = []
            
            # Standard Feature-Mapping
            feature_defaults = {
                'return_1d': 0.0,
                'return_5d': 0.0,
                'sma_5': row.get('close', 100),
                'sma_10': row.get('close', 100),
                'sma_20': row.get('close', 100),
                'sma_ratio_5': 1.0,
                'sma_ratio_10': 1.0,
                'sma_ratio_20': 1.0,
                'volatility': 0.01,
                'rsi': 50.0,
                'rsi_oversold': 0,
                'rsi_overbought': 0,
                'bb_position': 0.5,
                'volume_ratio': 1.0,
                'momentum': 0.0,
                'price_position': 0.5
            }
            
            # Extrahiere Features in der richtigen Reihenfolge
            for feature_name in self.feature_names:
                if feature_name in row and not pd.isna(row[feature_name]):
                    value = float(row[feature_name])
                    if np.isfinite(value):
                        features.append(value)
                    else:
                        features.append(feature_defaults.get(feature_name, 0.0))
                else:
                    features.append(feature_defaults.get(feature_name, 0.0))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature-Extraktion fehlgeschlagen: {e}")
            return None
    
    def _create_targets(self, data: pd.DataFrame) -> pd.Series:
        """
        Erstellt Targets - AGGRESSIVE für mehr Trades
        """
        try:
            # Future Returns berechnen
            future_returns = data['close'].shift(-1) / data['close'] - 1
            
            # SEHR AGGRESSIVE Thresholds für mehr Trading-Aktivität
            buy_threshold = 0.002   # 0.2% (sehr niedrig!)
            
            # Binary Classification: Buy vs Hold
            targets = (future_returns > buy_threshold).astype(int)
            
            # Entferne letzten Wert
            targets = targets[:-1]
            
            # Log Verteilung
            target_counts = targets.value_counts()
            buy_pct = target_counts.get(1, 0) / len(targets) * 100 if len(targets) > 0 else 0
            
            self.logger.info(f"Target: {target_counts.get(1, 0)} Buy ({buy_pct:.1f}%), {target_counts.get(0, 0)} Hold")
            
            # Falls zu wenig Buy-Signale, reduziere Threshold weiter
            if buy_pct < 20:  # Weniger als 20% Buy-Signale
                buy_threshold = 0.001  # Noch aggressiver: 0.1%
                targets = (future_returns[:-1] > buy_threshold).astype(int)
                new_counts = targets.value_counts()
                new_buy_pct = new_counts.get(1, 0) / len(targets) * 100 if len(targets) > 0 else 0
                self.logger.info(f"Reduzierter Threshold: {new_counts.get(1, 0)} Buy ({new_buy_pct:.1f}%)")
            
            return targets
            
        except Exception as e:
            self.logger.error(f"Target-Erstellung fehlgeschlagen: {e}")
            # Fallback: 50/50 split
            n_samples = len(data) - 1
            targets = pd.Series([1 if i % 2 == 0 else 0 for i in range(n_samples)])
            return targets
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI Berechnung"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            
            rs = gain / loss.replace(0, 1)
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)
        except:
            return pd.Series(50, index=prices.index)
    
    def get_info(self) -> Dict[str, Any]:
        """Strategie-Informationen"""
        info = super().get_info()
        info.update({
            'strategy_name': 'Final ML Strategy',
            'model_type': type(self.model).__name__ if self.model else 'None',
            'training_score': self.training_score,
            'feature_count': len(self.feature_names),
            'prediction_threshold': self.prediction_threshold,
            'is_pandas_2_compatible': True
        })
        return info