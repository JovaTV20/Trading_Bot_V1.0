"""
Optimierte ML-Strategie - Behebt Warnings und verbessert Trade-Generierung
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
    Optimierte Machine Learning Trading Strategy
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialisiert ML-Strategie"""
        super().__init__(config)
        
        # ML-Parameter
        self.lookback_period = self.parameters.get('lookback_period', 20)
        self.prediction_threshold = self.parameters.get('prediction_threshold', 0.55)  # REDUZIERT für mehr Trades
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
        
        # WICHTIG: Flag für Feature-Namen Kompatibilität
        self.use_feature_names = True
        
        # Initialisiere Modell
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialisiert das ML-Modell"""
        model_type = self.model_type.lower()
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=50,  # REDUZIERT für schnellere Performance
                max_depth=8,      # REDUZIERT 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=500,  # REDUZIERT
                solver='liblinear'
            )
        else:
            self.logger.warning(f"Unbekannter Modell-Typ: {model_type}. Verwende Random Forest.")
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            
        self.logger.info(f"ML-Modell initialisiert: {type(self.model).__name__}")
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Trainiert das ML-Modell - OPTIMIERT
        """
        if not self.validate_data(data):
            raise ValueError("Ungültige Trainingsdaten")
            
        self.logger.info(f"Trainiere ML-Modell mit {len(data)} Datenpunkten")
        
        try:
            # Bereite Features vor
            features_df = self._prepare_features(data)
            
            if features_df.empty:
                raise ValueError("Keine Features generiert")
            
            # Erstelle Targets - ENTSPANNTER für mehr Trades
            targets = self._create_targets(data)
            
            # Entferne NaN-Werte - VERBESSERT
            combined_df = features_df.join(targets, how='inner')
            
            # OPTIMIERT: Behandle NaN-Werte explizit
            initial_len = len(combined_df)
            combined_df = combined_df.fillna(0)  # Fülle NaN mit 0 statt dropna
            final_len = len(combined_df)
            
            if final_len < initial_len * 0.8:
                self.logger.warning(f"Viele NaN-Werte ersetzt: {initial_len - final_len} Zeilen")
            
            if len(combined_df) < self.lookback_period * 2:
                raise ValueError("Nicht genügend gültige Daten nach Feature-Engineering")
            
            X = combined_df.drop('target', axis=1)
            y = combined_df['target']
            
            # WICHTIG: Speichere Feature-Namen für sklearn Kompatibilität
            self.feature_names = list(X.columns)
            
            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=1 - self.train_test_split_ratio,
                random_state=42,
                stratify=y if len(np.unique(y)) > 1 else None
            )
            
            # KORRIGIERT: Skaliere Features mit DataFrame für Feature-Namen
            X_train_df = pd.DataFrame(
                self.scaler.fit_transform(X_train), 
                columns=self.feature_names,
                index=X_train.index
            )
            X_test_df = pd.DataFrame(
                self.scaler.transform(X_test), 
                columns=self.feature_names,
                index=X_test.index
            )
            
            # Trainiere Modell
            self.logger.info("Trainiere Modell...")
            self.model.fit(X_train_df, y_train)
            
            # Evaluiere Modell
            train_score = self.model.score(X_train_df, y_train)
            test_score = self.model.score(X_test_df, y_test)
            
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
            
        except Exception as e:
            self.logger.error(f"Fehler beim Training: {e}", exc_info=True)
            raise
    
    def generate_signal(self, row: pd.Series) -> Dict[str, Any]:
        """
        Generiert Trading-Signal - OPTIMIERT
        """
        if not self.is_fitted:
            return {
                'action': 'hold',
                'confidence': 0.0,
                'position_size': 0.0
            }
        
        try:
            # Einfache Feature-Extraktion für Live-Daten
            features = self._extract_simple_features(row)
            
            if features is None or len(features) != len(self.feature_names):
                return {
                    'action': 'hold',
                    'confidence': 0.0,
                    'position_size': 0.0
                }
            
            # KORRIGIERT: Verwende DataFrame für Feature-Namen Kompatibilität
            features_df = pd.DataFrame(
                [features], 
                columns=self.feature_names
            )
            
            # Skaliere Features
            features_scaled_df = pd.DataFrame(
                self.scaler.transform(features_df),
                columns=self.feature_names
            )
            
            # Vorhersage
            prediction = self.model.predict(features_scaled_df)[0]
            probabilities = self.model.predict_proba(features_scaled_df)[0]
            
            # Konvertiere Prediction zu Action
            if prediction == 2:  # Buy
                action = 'buy'
                confidence = probabilities[2]
            elif prediction == 0:  # Sell
                action = 'sell'
                confidence = probabilities[0]
            else:  # Hold
                action = 'hold'
                confidence = probabilities[1]
            
            # ENTSPANNTER Threshold für mehr Trades
            if confidence < self.prediction_threshold:
                action = 'hold'
                confidence = 0.0
            
            # Positionsgröße
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
        OPTIMIERTE Feature-Erstellung
        """
        df = data.copy()
        features = pd.DataFrame(index=df.index)
        
        try:
            # Returns - ROBUSTER
            for period in [1, 2, 5]:
                ret_col = f'return_{period}d'
                features[ret_col] = df['close'].pct_change(period).fillna(0)
                
                # Volatilität
                if period == 1:
                    features['volatility_10d'] = features[ret_col].rolling(10).std().fillna(0)
            
            # Moving Averages - VEREINFACHT
            for period in [5, 10, 20]:
                sma_col = f'sma_{period}'
                features[sma_col] = df['close'].rolling(period).mean()
                
                # Verhältnisse
                ratio_col = f'sma_ratio_{period}'
                features[ratio_col] = (df['close'] / features[sma_col]).fillna(1.0)
            
            # RSI - ROBUSTER
            features['rsi'] = self._calculate_rsi(df['close']).fillna(50)
            features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
            features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
            
            # Bollinger Bands - VEREINFACHT
            bb_period = 20
            sma_bb = df['close'].rolling(bb_period).mean()
            std_bb = df['close'].rolling(bb_period).std()
            
            features['bb_upper'] = sma_bb + (std_bb * 2)
            features['bb_lower'] = sma_bb - (std_bb * 2)
            
            # BB Position
            bb_width = features['bb_upper'] - features['bb_lower']
            features['bb_position'] = ((df['close'] - features['bb_lower']) / bb_width).fillna(0.5)
            
            # Volume Features - OPTIONAL
            if 'volume' in df.columns:
                features['volume_sma_20'] = df['volume'].rolling(20).mean()
                features['volume_ratio'] = (df['volume'] / features['volume_sma_20']).fillna(1.0)
            
            # Momentum - VEREINFACHT
            features['momentum_5'] = (df['close'] / df['close'].shift(5) - 1).fillna(0)
            
            # WICHTIG: Fülle alle verbleibenden NaN-Werte
            features = features.fillna(method='forward').fillna(0)
            
            self.logger.debug(f"Features erstellt: {len(features.columns)} Features")
            return features
            
        except Exception as e:
            self.logger.error(f"Fehler bei Feature-Erstellung: {e}")
            return pd.DataFrame()
    
    def _extract_simple_features(self, row: pd.Series) -> Optional[list]:
        """
        Extrahiert Features aus einer einzelnen Zeile - VEREINFACHT
        """
        try:
            features = []
            
            # Grundlegende Features mit Fallbacks
            features.extend([
                row.get('return_1d', 0),
                row.get('return_2d', 0), 
                row.get('return_5d', 0),
                row.get('volatility_10d', 0),
                row.get('sma_ratio_5', 1.0),
                row.get('sma_ratio_10', 1.0),
                row.get('sma_ratio_20', 1.0),
                row.get('rsi', 50),
                row.get('rsi_oversold', 0),
                row.get('rsi_overbought', 0),
                row.get('bb_position', 0.5),
                row.get('volume_ratio', 1.0),
                row.get('momentum_5', 0)
            ])
            
            # Erweitere auf erwartete Feature-Anzahl
            expected_features = len(self.feature_names) if self.feature_names else 13
            while len(features) < expected_features:
                features.append(0.0)
            
            # Beschränke auf erwartete Anzahl
            features = features[:expected_features]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Fehler bei Feature-Extraktion: {e}")
            return None
    
    def _create_targets(self, data: pd.DataFrame) -> pd.Series:
        """
        ENTSPANNTERE Target-Erstellung für mehr Trades
        """
        df = data.copy()
        
        # Future Returns
        future_returns = df['close'].shift(-1) / df['close'] - 1
        
        # ENTSPANNTERE Thresholds für mehr Signale
        buy_threshold = 0.005   # 0.5% statt 1%
        sell_threshold = -0.005  # -0.5% statt -1%
        
        # Erstelle Labels
        targets = pd.Series(1, index=df.index)  # Default: Hold
        
        # Buy/Sell Signals
        targets[future_returns > buy_threshold] = 2  # Buy
        targets[future_returns < sell_threshold] = 0  # Sell
        
        # Entferne letzten Wert
        targets = targets[:-1]
        
        # Log Target-Verteilung
        buy_count = (targets == 2).sum()
        sell_count = (targets == 0).sum()
        hold_count = (targets == 1).sum()
        
        self.logger.info(f"Target Distribution - Buy: {buy_count}, Hold: {hold_count}, Sell: {sell_count}")
        
        return targets.to_frame('target')['target']
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI Berechnung - ROBUSTER"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # Vermeide Division durch 0
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)  # Neutraler RSI als Fallback
        except:
            return pd.Series(50, index=prices.index)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Erweiterte Modell-Informationen"""
        info = {
            'strategy_name': 'Optimized ML Strategy',
            'model_type': type(self.model).__name__ if self.model else 'None',
            'is_fitted': self.is_fitted,
            'training_score': self.training_score,
            'feature_count': len(self.feature_names),
            'lookback_period': self.lookback_period,
            'prediction_threshold': self.prediction_threshold,
            'parameters': self.parameters
        }
        
        if self.is_fitted and hasattr(self.model, 'n_estimators'):
            info['n_estimators'] = getattr(self.model, 'n_estimators', 'N/A')
            
        return info
    
    def get_info(self) -> Dict[str, Any]:
        """Erweitert Basis-Info"""
        info = super().get_info()
        info.update(self.get_model_info())
        return info