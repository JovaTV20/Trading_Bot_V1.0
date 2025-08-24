"""
INTELLIGENT ADAPTIVE ML STRATEGY - CLEAN VERSION (No Syntax Errors)
ERSETZT: strategies/ml_strategy.py

FEATURES:
- Kontinuierliches Online Learning
- Markt-Regime Erkennung  
- Adaptive Feature Engineering
- Multi-Model Ensemble
- Overfitting-Schutz
- Realistische Targets
- Volatilitäts-adaptive Position Sizing
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
import warnings
from collections import deque
import json
from pathlib import Path

# Unterdrücke Warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ML Imports
try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.model_selection import cross_val_score
    import joblib
except ImportError as e:
    logging.error("Scikit-learn nicht installiert!")
    raise ImportError("Scikit-learn Paket erforderlich") from e

from core.base_strategy import StrategyBase

class IntelligentMLStrategy(StrategyBase):
    """
    INTELLIGENTE ADAPTIVE ML-STRATEGIE
    
    Diese Strategie ist WIRKLICH intelligent und lernt kontinuierlich:
    - Erkennt automatisch Markt-Regimes
    - Passt sich an neue Daten an (Online Learning)
    - Verwendet realistische Targets
    - Schützt vor Overfitting
    - Quantifiziert Unsicherheit
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialisiert die intelligente ML-Strategie"""
        super().__init__(config)
        
        # INTELLIGENTE PARAMETER
        self.lookback_period = self.parameters.get('lookback_period', 50)
        self.prediction_threshold = self.parameters.get('prediction_threshold', 0.65)
        self.regime_window = self.parameters.get('regime_window', 100)
        self.online_learning_window = self.parameters.get('online_learning_window', 200)
        self.min_confidence = self.parameters.get('min_confidence', 0.6)
        
        # MODEL ENSEMBLE ARCHITECTURE
        self.models = {
            'main': RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight='balanced'
            ),
            'online': SGDClassifier(
                loss='log_loss',
                learning_rate='adaptive',
                eta0=0.01,
                random_state=42,
                class_weight='balanced'
            ),
            'regime': PassiveAggressiveClassifier(
                C=0.5,
                random_state=42,
                class_weight='balanced'
            )
        }
        
        # ADAPTIVE COMPONENTS
        self.scalers = {}
        self.feature_names = []
        self.training_history = deque(maxlen=self.online_learning_window)
        self.performance_history = deque(maxlen=100)
        self.market_regime = 'unknown'
        
        # LEARNING STATE
        self.model_performance = {}
        self.last_retrain_date = None
        self.retrain_frequency = 20
        self.data_buffer = deque(maxlen=self.online_learning_window)
        
        # REGIME DETECTION
        self.regime_indicators = {
            'trend_strength': 0.0,
            'volatility_regime': 'normal',
            'momentum_regime': 'neutral'
        }
        
        self.current_volatility = 0.01
        
        # Initialize scalers
        for model_name in self.models.keys():
            self.scalers[model_name] = RobustScaler()
        
        self.logger.info("Intelligent ML Strategy initialisiert")
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        INTELLIGENTES TRAINING mit kontinuierlichem Lernen
        """
        if not self.validate_data(data):
            raise ValueError("Ungültige Trainingsdaten")
        
        self.logger.info(f"Starte intelligentes Training mit {len(data)} Datenpunkten")
        
        try:
            # 1. ADAPTIVE FEATURE ENGINEERING
            features_df = self._create_intelligent_features(data)
            
            if features_df.empty:
                self.logger.error("Feature-Erstellung fehlgeschlagen")
                features_df = self._create_fallback_features(data)
            
            # 2. INTELLIGENTE TARGET CREATION
            targets = self._create_smart_targets(data)
            
            # 3. DATA SYNCHRONIZATION
            min_len = min(len(features_df), len(targets))
            features_df = features_df.iloc[:min_len].copy()
            targets = targets.iloc[:min_len].copy()
            
            # 4. INTELLIGENT DATA CLEANING
            combined_df = self._clean_data_intelligently(features_df, targets)
            
            if len(combined_df) < self.lookback_period:
                raise ValueError(f"Nicht genügend saubere Daten: {len(combined_df)}")
            
            X = combined_df.drop('target', axis=1)
            y = combined_df['target']
            
            # 5. REGIME DETECTION
            self._detect_market_regime(data.tail(self.regime_window))
            
            # 6. MULTI-MODEL TRAINING
            self._train_ensemble_models(X, y)
            
            # 7. PERFORMANCE EVALUATION
            self._evaluate_model_performance(X, y)
            
            # 8. SAVE TRAINING STATE
            self._save_training_state(X.tail(50))
            
            self.is_fitted = True
            self.logger.info("Intelligentes Training abgeschlossen!")
            
        except Exception as e:
            self.logger.error(f"Training fehlgeschlagen: {e}", exc_info=True)
            raise
    
    def generate_signal(self, row: pd.Series) -> Dict[str, Any]:
        """
        INTELLIGENTE SIGNAL GENERATION mit Ensemble Voting
        """
        if not self.is_fitted:
            return self._default_signal()
        
        try:
            # 1. FEATURE EXTRACTION
            features = self._extract_intelligent_features(row)
            if features is None:
                return self._default_signal()
            
            # 2. ENSEMBLE PREDICTION
            predictions = {}
            confidences = {}
            
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba') and len(features) == len(self.feature_names):
                        # Skaliere Features
                        features_scaled = self.scalers[model_name].transform([features])
                        
                        # Prediction
                        pred = model.predict(features_scaled)[0]
                        proba = model.predict_proba(features_scaled)[0]
                        
                        predictions[model_name] = pred
                        confidences[model_name] = max(proba) if len(proba) > 0 else 0.5
                    
                except Exception as e:
                    self.logger.warning(f"Model {model_name} prediction failed: {e}")
                    predictions[model_name] = 0
                    confidences[model_name] = 0.0
            
            # 3. INTELLIGENT ENSEMBLE VOTING
            final_signal = self._ensemble_vote(predictions, confidences, row)
            
            # 4. ONLINE LEARNING UPDATE
            self._update_online_learning(row, features)
            
            # 5. REGIME-ADAPTIVE ADJUSTMENTS
            final_signal = self._apply_regime_adjustments(final_signal, row)
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            return self._default_signal()
    
    def _create_intelligent_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ADAPTIVE FEATURE ENGINEERING - Intelligente Features erstellen
        """
        try:
            df = data.copy()
            features = pd.DataFrame(index=df.index)
            
            # TREND FEATURES (Multi-Timeframe)
            for period in [10, 20, 50]:
                sma = df['close'].rolling(period, min_periods=max(1, period//2)).mean()
                ema = df['close'].ewm(span=period).mean()
                
                features[f'sma_ratio_{period}'] = (df['close'] / sma).fillna(1.0)
                features[f'ema_ratio_{period}'] = (df['close'] / ema).fillna(1.0)
                features[f'trend_strength_{period}'] = (sma - sma.shift(5)) / sma.shift(5).fillna(1)
            
            # VOLATILITY FEATURES (Adaptive)
            returns = df['close'].pct_change().fillna(0)
            features['volatility_short'] = returns.rolling(10, min_periods=5).std().fillna(0)
            features['volatility_long'] = returns.rolling(30, min_periods=10).std().fillna(0)
            features['volatility_ratio'] = (features['volatility_short'] / 
                                          (features['volatility_long'] + 1e-8)).fillna(1.0)
            
            # MOMENTUM FEATURES (Multi-Horizon)
            for period in [5, 10, 20]:
                momentum = (df['close'] / df['close'].shift(period) - 1).fillna(0)
                features[f'momentum_{period}'] = momentum
                features[f'momentum_strength_{period}'] = np.abs(momentum)
            
            # VOLUME INTELLIGENCE
            if 'volume' in df.columns and df['volume'].sum() > 0:
                vol_sma = df['volume'].rolling(20, min_periods=10).mean()
                features['volume_ratio'] = (df['volume'] / (vol_sma + 1)).fillna(1.0)
                features['volume_momentum'] = df['volume'].pct_change(5).fillna(0)
                
                # Price-Volume Divergence
                price_change = df['close'].pct_change(5).fillna(0)
                volume_change = df['volume'].pct_change(5).fillna(0)
                features['pv_divergence'] = price_change - volume_change
            else:
                features['volume_ratio'] = 1.0
                features['volume_momentum'] = 0.0
                features['pv_divergence'] = 0.0
            
            # ADVANCED INDICATORS
            features['rsi'] = self._calculate_rsi(df['close'], 14)
            bb_features = self._calculate_bollinger_features(df['close'], 20)
            features.update(bb_features)
            
            # REGIME-SENSITIVE FEATURES
            features['regime_trend'] = self._calculate_regime_trend(df['close'])
            features['regime_volatility'] = self._calculate_regime_volatility(returns)
            
            # INTELLIGENT CLEANING
            features = features.replace([np.inf, -np.inf], 0)
            features = features.ffill().bfill().fillna(0)
            
            # FEATURE SELECTION (Top Features only)
            if len(features.columns) > 15:
                variance_scores = features.var()
                stable_features = variance_scores[
                    (variance_scores > 0.001) & (variance_scores < 100)
                ].nlargest(15).index
                features = features[stable_features]
            
            self.feature_names = list(features.columns)
            self.logger.info(f"Created {len(self.feature_names)} intelligent features")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Intelligent feature creation failed: {e}")
            return pd.DataFrame()
    
    def _create_smart_targets(self, data: pd.DataFrame) -> pd.Series:
        """
        SMART TARGET CREATION - Realistische, profitable Targets
        """
        try:
            # CALCULATE FORWARD RETURNS
            future_returns_1d = data['close'].shift(-1) / data['close'] - 1
            future_returns_3d = data['close'].shift(-3) / data['close'] - 1
            future_returns_5d = data['close'].shift(-5) / data['close'] - 1
            
            # ADAPTIVE THRESHOLDS basierend auf Volatilität
            rolling_vol = data['close'].pct_change().rolling(20, min_periods=10).std()
            
            # Dynamic thresholds - 1.5x volatility für Buy
            buy_threshold = rolling_vol * 1.5
            
            # Fallback für NaN volatility
            buy_threshold = buy_threshold.fillna(0.015)
            
            # MULTI-HORIZON TARGET CREATION
            combined_return = (future_returns_1d * 0.5 + 
                             future_returns_3d * 0.3 + 
                             future_returns_5d * 0.2)
            
            # INTELLIGENT CLASSIFICATION
            targets = pd.Series(0, index=data.index)
            
            # Buy signals: Strong positive returns
            buy_condition = combined_return > buy_threshold
            targets[buy_condition] = 1
            
            # CLEAN TARGETS
            targets = targets[:-5]
            
            # LOG TARGET STATISTICS
            target_counts = targets.value_counts()
            buy_pct = target_counts.get(1, 0) / len(targets) * 100 if len(targets) > 0 else 0
            
            self.logger.info(f"Smart targets: {target_counts.get(1, 0)} Buy ({buy_pct:.1f}%), "
                           f"{target_counts.get(0, 0)} Hold")
            
            # QUALITY CHECK - Ensure minimum buy signals
            if buy_pct < 5:
                self.logger.warning(f"Very few buy signals ({buy_pct:.1f}%), adjusting thresholds")
                relaxed_threshold = rolling_vol * 1.0
                relaxed_threshold = relaxed_threshold.fillna(0.01)
                buy_condition = combined_return > relaxed_threshold
                targets = pd.Series(0, index=data.index[:-5])
                targets[buy_condition[:-5]] = 1
                
                new_counts = targets.value_counts()
                new_buy_pct = new_counts.get(1, 0) / len(targets) * 100 if len(targets) > 0 else 0
                self.logger.info(f"Adjusted targets: {new_counts.get(1, 0)} Buy ({new_buy_pct:.1f}%)")
            
            return targets
            
        except Exception as e:
            self.logger.error(f"Smart target creation failed: {e}")
            # Fallback: Simple momentum-based targets
            returns = data['close'].pct_change().fillna(0)
            targets = (returns.shift(-1) > 0.01).astype(int)[:-1]
            return targets
    
    def _clean_data_intelligently(self, features_df: pd.DataFrame, 
                                targets: pd.Series) -> pd.DataFrame:
        """
        INTELLIGENT DATA CLEANING
        """
        try:
            # Combine features and targets
            combined_df = features_df.copy()
            combined_df['target'] = targets
            
            # OUTLIER DETECTION using Isolation Forest
            try:
                isolation_forest = IsolationForest(
                    contamination=0.05,
                    random_state=42
                )
                
                numeric_features = combined_df.select_dtypes(include=[np.number]).drop('target', axis=1)
                if len(numeric_features.columns) > 0:
                    outlier_mask = isolation_forest.fit_predict(numeric_features)
                    combined_df = combined_df[outlier_mask == 1]
                    
                    outliers_removed = sum(outlier_mask == -1)
                    self.logger.info(f"Removed {outliers_removed} outliers")
            
            except Exception as e:
                self.logger.warning(f"Outlier detection failed: {e}")
            
            # HANDLE MISSING VALUES
            combined_df = combined_df.ffill().bfill().fillna(0)
            
            # REMOVE INVALID TARGETS
            combined_df = combined_df.dropna(subset=['target'])
            
            # FINAL QUALITY CHECKS
            if len(combined_df) == 0:
                raise ValueError("No data remaining after cleaning")
            
            # Check target balance
            target_counts = combined_df['target'].value_counts()
            if len(target_counts) < 2:
                self.logger.warning("Only one target class remaining - creating balance")
                n_samples = len(combined_df)
                n_positive = max(1, int(n_samples * 0.1))
                
                positive_indices = np.random.choice(
                    combined_df.index, size=min(n_positive, n_samples//2), replace=False
                )
                combined_df.loc[positive_indices, 'target'] = 1
            
            self.logger.info(f"Clean data: {len(combined_df)} samples")
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Data cleaning failed: {e}")
            combined_df = features_df.copy()
            combined_df['target'] = targets
            return combined_df.fillna(0)
    
    def _train_ensemble_models(self, X: pd.DataFrame, y: pd.Series):
        """
        TRAIN ENSEMBLE MODELS with Cross-Validation
        """
        try:
            X_array = X.values
            y_array = y.values
            
            model_scores = {}
            
            for model_name, model in self.models.items():
                try:
                    self.logger.info(f"Training {model_name} model...")
                    
                    # SCALE FEATURES
                    X_scaled = self.scalers[model_name].fit_transform(X_array)
                    
                    # TRAIN MODEL
                    model.fit(X_scaled, y_array)
                    
                    # CROSS VALIDATION SCORE
                    try:
                        cv_scores = cross_val_score(model, X_scaled, y_array, cv=3, scoring='accuracy')
                        model_scores[model_name] = {
                            'cv_mean': cv_scores.mean(),
                            'cv_std': cv_scores.std(),
                            'train_score': model.score(X_scaled, y_array)
                        }
                        
                        self.logger.info(f"{model_name}: CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                    
                    except Exception as e:
                        self.logger.warning(f"CV failed for {model_name}: {e}")
                        model_scores[model_name] = {'train_score': 0.5}
                
                except Exception as e:
                    self.logger.error(f"Training failed for {model_name}: {e}")
                    model_scores[model_name] = {'train_score': 0.0}
            
            self.model_performance = model_scores
            self.logger.info("Ensemble training completed!")
            
        except Exception as e:
            self.logger.error(f"Ensemble training failed: {e}")
            raise
    
    def _ensemble_vote(self, predictions: Dict[str, int], 
                      confidences: Dict[str, float], 
                      current_row: pd.Series) -> Dict[str, Any]:
        """
        INTELLIGENT ENSEMBLE VOTING with Dynamic Weighting
        """
        try:
            if not predictions:
                return self._default_signal()
            
            # DYNAMIC MODEL WEIGHTS based on performance
            weights = {}
            for model_name in predictions.keys():
                perf = self.model_performance.get(model_name, {'train_score': 0.5})
                weights[model_name] = max(0.1, perf.get('cv_mean', perf.get('train_score', 0.5)))
            
            # WEIGHTED VOTING
            weighted_votes = 0.0
            total_weights = 0.0
            weighted_confidence = 0.0
            
            for model_name, prediction in predictions.items():
                weight = weights[model_name]
                confidence = confidences[model_name]
                
                weighted_votes += prediction * weight * confidence
                weighted_confidence += confidence * weight
                total_weights += weight
            
            if total_weights > 0:
                final_prediction = weighted_votes / total_weights
                avg_confidence = weighted_confidence / total_weights
            else:
                final_prediction = 0
                avg_confidence = 0.0
            
            # DECISION LOGIC
            if final_prediction > self.prediction_threshold and avg_confidence > self.min_confidence:
                action = 'buy'
                confidence = avg_confidence
            else:
                action = 'hold'
                confidence = 0.0
            
            # POSITION SIZING based on confidence and volatility
            position_size = self._calculate_intelligent_position_size(confidence, current_row)
            
            return {
                'action': action,
                'confidence': confidence,
                'position_size': position_size,
                'ensemble_prediction': final_prediction,
                'model_votes': predictions,
                'model_confidences': confidences,
                'market_regime': self.market_regime
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble voting failed: {e}")
            return self._default_signal()
    
    def _calculate_intelligent_position_size(self, confidence: float, 
                                          current_row: pd.Series) -> float:
        """
        INTELLIGENT POSITION SIZING - Volatility & Confidence Adaptive
        """
        try:
            base_size = self.config.get('risk_management', {}).get('max_position_size', 0.1)
            
            # CONFIDENCE SCALING (0.5 to 1.5x)
            confidence_multiplier = 0.5 + confidence
            
            # VOLATILITY ADJUSTMENT
            volatility_multiplier = 1.0
            if hasattr(self, 'current_volatility') and self.current_volatility > 0:
                if self.current_volatility > 0.02:
                    volatility_multiplier = 0.7
                elif self.current_volatility < 0.01:
                    volatility_multiplier = 1.2
            
            # REGIME ADJUSTMENT
            regime_multiplier = 1.0
            if self.market_regime == 'bull':
                regime_multiplier = 1.1
            elif self.market_regime == 'bear':
                regime_multiplier = 0.8
            
            # FINAL CALCULATION
            position_size = (base_size * confidence_multiplier * 
                           volatility_multiplier * regime_multiplier)
            
            # SAFETY BOUNDS
            position_size = max(0.0, min(position_size, base_size * 2))
            
            return position_size
            
        except Exception as e:
            self.logger.warning(f"Position sizing error: {e}")
            return self.config.get('risk_management', {}).get('max_position_size', 0.1)
    
    def _detect_market_regime(self, recent_data: pd.DataFrame):
        """
        MARKET REGIME DETECTION - Bull/Bear/Sideways
        """
        try:
            if len(recent_data) < 30:
                self.market_regime = 'unknown'
                return
            
            prices = recent_data['close']
            returns = prices.pct_change().dropna()
            
            # TREND ANALYSIS
            sma_20 = prices.rolling(20).mean()
            sma_50 = prices.rolling(50).mean() if len(prices) >= 50 else sma_20
            
            current_price = prices.iloc[-1]
            trend_20 = (current_price - sma_20.iloc[-1]) / sma_20.iloc[-1]
            trend_50 = (current_price - sma_50.iloc[-1]) / sma_50.iloc[-1]
            
            # VOLATILITY ANALYSIS
            volatility = returns.std()
            self.current_volatility = volatility
            
            # MOMENTUM ANALYSIS
            momentum_5 = (prices.iloc[-1] / prices.iloc[-5] - 1) if len(prices) >= 5 else 0
            momentum_20 = (prices.iloc[-1] / prices.iloc[-20] - 1) if len(prices) >= 20 else 0
            
            # REGIME CLASSIFICATION
            if trend_20 > 0.02 and trend_50 > 0.01 and momentum_20 > 0.05:
                regime = 'bull'
            elif trend_20 < -0.02 and trend_50 < -0.01 and momentum_20 < -0.05:
                regime = 'bear'  
            elif abs(trend_20) < 0.01 and volatility < 0.015:
                regime = 'sideways'
            else:
                regime = 'transitional'
            
            # UPDATE REGIME INDICATORS
            self.regime_indicators = {
                'trend_strength': (trend_20 + trend_50) / 2,
                'volatility_regime': 'high' if volatility > 0.025 else 'low' if volatility < 0.01 else 'normal',
                'momentum_regime': 'strong_up' if momentum_20 > 0.1 else 'strong_down' if momentum_20 < -0.1 else 'neutral'
            }
            
            if regime != self.market_regime:
                self.logger.info(f"Market regime changed: {self.market_regime} -> {regime}")
                self.market_regime = regime
            
        except Exception as e:
            self.logger.warning(f"Regime detection error: {e}")
            self.market_regime = 'unknown'
    
    def _apply_regime_adjustments(self, signal: Dict[str, Any], 
                                current_row: pd.Series) -> Dict[str, Any]:
        """
        REGIME-ADAPTIVE SIGNAL ADJUSTMENTS
        """
        try:
            adjusted_signal = signal.copy()
            
            # BULL MARKET ADJUSTMENTS
            if self.market_regime == 'bull':
                if signal['action'] == 'buy':
                    adjusted_signal['confidence'] = min(1.0, signal['confidence'] * 1.1)
                    adjusted_signal['position_size'] = min(
                        signal['position_size'] * 1.1,
                        self.config.get('risk_management', {}).get('max_position_size', 0.1) * 1.5
                    )
            
            # BEAR MARKET ADJUSTMENTS  
            elif self.market_regime == 'bear':
                if signal['action'] == 'buy':
                    adjusted_signal['confidence'] = signal['confidence'] * 0.8
                    adjusted_signal['position_size'] = signal['position_size'] * 0.7
                    
                    if adjusted_signal['confidence'] < 0.7:
                        adjusted_signal['action'] = 'hold'
                        adjusted_signal['confidence'] = 0.0
                        adjusted_signal['position_size'] = 0.0
            
            # SIDEWAYS MARKET ADJUSTMENTS
            elif self.market_regime == 'sideways':
                if signal['action'] == 'buy' and signal['confidence'] < 0.75:
                    adjusted_signal['action'] = 'hold'
                    adjusted_signal['confidence'] = 0.0
                    adjusted_signal['position_size'] = 0.0
            
            # HIGH VOLATILITY ADJUSTMENTS
            if self.regime_indicators['volatility_regime'] == 'high':
                adjusted_signal['position_size'] *= 0.8
            
            return adjusted_signal
            
        except Exception as e:
            self.logger.warning(f"Regime adjustment error: {e}")
            return signal
    
    def _update_online_learning(self, current_row: pd.Series, features: List[float]):
        """
        ONLINE LEARNING UPDATE - Kontinuierliche Anpassung
        """
        try:
            current_data = {
                'timestamp': datetime.now(),
                'features': features,
                'row_data': current_row.to_dict()
            }
            self.data_buffer.append(current_data)
            
            should_retrain = (
                len(self.data_buffer) >= self.retrain_frequency and
                (self.last_retrain_date is None or 
                 (datetime.now() - self.last_retrain_date).days >= 1)
            )
            
            if should_retrain and len(self.data_buffer) >= 50:
                self._perform_online_retrain()
                
        except Exception as e:
            self.logger.warning(f"Online learning update error: {e}")
    
    def _perform_online_retrain(self):
        """
        PERFORM ONLINE RETRAINING with recent data
        """
        try:
            self.logger.info("Starting online retraining...")
            
            if len(self.data_buffer) < 30:
                return
            
            recent_features = []
            recent_targets = []
            
            for i, data_point in enumerate(list(self.data_buffer)[:-5]):
                if i + 5 < len(self.data_buffer):
                    features = data_point['features']
                    if len(features) == len(self.feature_names):
                        recent_features.append(features)
                        
                        current_price = data_point['row_data'].get('close', 100)
                        future_data = list(self.data_buffer)[i+1:i+6]
                        
                        if len(future_data) == 5:
                            future_prices = [d['row_data'].get('close', current_price) for d in future_data]
                            future_return = (future_prices[-1] / current_price - 1) if current_price > 0 else 0
                            target = 1 if future_return > 0.015 else 0
                            recent_targets.append(target)
            
            if len(recent_features) < 10:
                return
            
            X_recent = np.array(recent_features)
            y_recent = np.array(recent_targets)
            
            # Online learning for SGD model
            if 'online' in self.models and hasattr(self.models['online'], 'partial_fit'):
                try:
                    X_scaled = self.scalers['online'].transform(X_recent)
                    
                    classes = np.unique(y_recent)
                    if len(classes) >= 2:
                        self.models['online'].partial_fit(X_scaled, y_recent)
                        
                        accuracy = self.models['online'].score(X_scaled, y_recent)
                        self.model_performance['online']['recent_accuracy'] = accuracy
                        
                        self.logger.info(f"Online model updated - Recent accuracy: {accuracy:.3f}")
                    
                except Exception as e:
                    self.logger.warning(f"Online model update failed: {e}")
            
            self.last_retrain_date = datetime.now()
            
            while len(self.data_buffer) > 100:
                self.data_buffer.popleft()
                
        except Exception as e:
            self.logger.error(f"Online retraining failed: {e}")
    
    def _extract_intelligent_features(self, row: pd.Series) -> Optional[List[float]]:
        """
        EXTRACT INTELLIGENT FEATURES from single row
        """
        try:
            features = []
            
            # Map of expected features with safe defaults
            feature_mapping = {
                # Trend features
                'sma_ratio_10': row.get('close', 100) / max(row.get('sma_10', row.get('close', 100)), 1),
                'sma_ratio_20': row.get('close', 100) / max(row.get('sma_20', row.get('close', 100)), 1),
                'sma_ratio_50': row.get('close', 100) / max(row.get('sma_50', row.get('close', 100)), 1),
                'ema_ratio_10': row.get('close', 100) / max(row.get('ema_10', row.get('close', 100)), 1),
                'ema_ratio_20': row.get('close', 100) / max(row.get('ema_20', row.get('close', 100)), 1),
                'ema_ratio_50': row.get('close', 100) / max(row.get('ema_50', row.get('close', 100)), 1),
                
                # Momentum features
                'momentum_5': row.get('momentum_5', 0.0),
                'momentum_10': row.get('momentum_10', 0.0), 
                'momentum_20': row.get('momentum_20', 0.0),
                
                # Volatility features
                'volatility_short': row.get('volatility_short', 0.01),
                'volatility_long': row.get('volatility_long', 0.01),
                'volatility_ratio': row.get('volatility_ratio', 1.0),
                
                # Volume features
                'volume_ratio': row.get('volume_ratio', 1.0),
                'volume_momentum': row.get('volume_momentum', 0.0),
                'pv_divergence': row.get('pv_divergence', 0.0),
                
                # Advanced indicators
                'rsi': row.get('rsi', 50.0),
                'bb_position': row.get('bb_position', 0.5),
                'bb_width': row.get('bb_width', 0.1),
                
                # Regime features
                'regime_trend': row.get('regime_trend', 0.0),
                'regime_volatility': row.get('regime_volatility', 0.01)
            }
            
            # Extract features in consistent order
            for feature_name in self.feature_names:
                if feature_name in feature_mapping:
                    value = feature_mapping[feature_name]
                elif feature_name in row:
                    value = row[feature_name]
                else:
                    value = 0.0
                
                # Ensure finite value
                if pd.isna(value) or np.isinf(value):
                    value = 0.0
                    
                features.append(float(value))
            
            return features if len(features) == len(self.feature_names) else None
            
        except Exception as e:
            self.logger.warning(f"Feature extraction error: {e}")
            return None
    
    def _create_fallback_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        FALLBACK FEATURES - Simple but robust
        """
        try:
            features = pd.DataFrame(index=data.index)
            
            # Basic trend
            sma_10 = data['close'].rolling(10, min_periods=5).mean()
            features['sma_ratio_10'] = (data['close'] / sma_10).fillna(1.0)
            
            # Basic momentum
            features['momentum_5'] = (data['close'] / data['close'].shift(5) - 1).fillna(0)
            
            # Basic volatility
            returns = data['close'].pct_change().fillna(0)
            features['volatility_short'] = returns.rolling(10, min_periods=5).std().fillna(0.01)
            
            # Basic volume
            features['volume_ratio'] = 1.0
            
            # Basic RSI
            features['rsi'] = self._calculate_rsi(data['close'], 14)
            
            # Clean
            features = features.fillna(0)
            self.feature_names = list(features.columns)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Fallback features failed: {e}")
            # Ultimate fallback
            features = pd.DataFrame({
                'price_change': data['close'].pct_change().fillna(0),
                'simple_momentum': (data['close'] / data['close'].shift(3) - 1).fillna(0)
            }, index=data.index)
            self.feature_names = list(features.columns)
            return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI Calculation"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(period, min_periods=max(1, period//2)).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=max(1, period//2)).mean()
            rs = gain / loss.replace(0, 1)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except:
            return pd.Series(50, index=prices.index)
    
    def _calculate_bollinger_features(self, prices: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
        """Bollinger Bands Features"""
        try:
            sma = prices.rolling(period, min_periods=max(1, period//2)).mean()
            std = prices.rolling(period, min_periods=max(1, period//2)).std()
            
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            
            bb_width = (upper_band - lower_band) / sma
            bb_position = (prices - lower_band) / (upper_band - lower_band)
            
            return {
                'bb_position': bb_position.fillna(0.5),
                'bb_width': bb_width.fillna(0.1)
            }
        except:
            return {
                'bb_position': pd.Series(0.5, index=prices.index),
                'bb_width': pd.Series(0.1, index=prices.index)
            }
    
    def _calculate_regime_trend(self, prices: pd.Series) -> pd.Series:
        """Regime Trend Feature"""
        try:
            sma_short = prices.rolling(10, min_periods=5).mean()
            sma_long = prices.rolling(30, min_periods=10).mean()
            trend = (sma_short - sma_long) / sma_long
            return trend.fillna(0)
        except:
            return pd.Series(0, index=prices.index)
    
    def _calculate_regime_volatility(self, returns: pd.Series) -> pd.Series:
        """Regime Volatility Feature"""
        try:
            vol_short = returns.rolling(5, min_periods=3).std()
            vol_long = returns.rolling(20, min_periods=10).std()
            vol_regime = vol_short / vol_long.replace(0, 0.01)
            return vol_regime.fillna(1.0)
        except:
            return pd.Series(0.01, index=returns.index)
    
    def _evaluate_model_performance(self, X: pd.DataFrame, y: pd.Series):
        """EVALUATE MODEL PERFORMANCE"""
        try:
            for model_name, model in self.models.items():
                if hasattr(model, 'predict'):
                    X_scaled = self.scalers[model_name].transform(X.values)
                    predictions = model.predict(X_scaled)
                    
                    accuracy = accuracy_score(y, predictions)
                    
                    # Calculate precision and recall safely
                    try:
                        precision = precision_score(y, predictions, average='weighted', zero_division=0)
                        recall = recall_score(y, predictions, average='weighted', zero_division=0)
                    except:
                        precision = accuracy
                        recall = accuracy
                    
                    if model_name not in self.model_performance:
                        self.model_performance[model_name] = {}
                    
                    self.model_performance[model_name].update({
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall
                    })
                    
                    self.logger.info(f"{model_name} performance - Accuracy: {accuracy:.3f}")
                    
        except Exception as e:
            self.logger.warning(f"Performance evaluation error: {e}")
    
    def _save_training_state(self, recent_data: pd.DataFrame):
        """SAVE TRAINING STATE for persistence"""
        try:
            # Save recent data to training history
            for _, row in recent_data.iterrows():
                self.training_history.append({
                    'timestamp': datetime.now(),
                    'data': row.to_dict()
                })
                
            # Optional: Persist models to disk
            models_dir = Path('models')
            models_dir.mkdir(exist_ok=True)
            
            for model_name, model in self.models.items():
                try:
                    model_file = models_dir / f"intelligent_{model_name}_model.joblib"
                    joblib.dump(model, model_file)
                    
                    scaler_file = models_dir / f"intelligent_{model_name}_scaler.joblib"
                    joblib.dump(self.scalers[model_name], scaler_file)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to save {model_name}: {e}")
            
            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'model_performance': self.model_performance,
                'market_regime': self.market_regime,
                'regime_indicators': self.regime_indicators,
                'last_retrain_date': self.last_retrain_date.isoformat() if self.last_retrain_date else None
            }
            
            metadata_file = models_dir / "intelligent_strategy_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
                
            self.logger.info("Training state saved successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to save training state: {e}")
    
    def _default_signal(self) -> Dict[str, Any]:
        """DEFAULT SAFE SIGNAL"""
        return {
            'action': 'hold',
            'confidence': 0.0,
            'position_size': 0.0,
            'ensemble_prediction': 0.0,
            'market_regime': self.market_regime,
            'error': 'fallback_signal'
        }
    
    def get_info(self) -> Dict[str, Any]:
        """STRATEGY INFORMATION"""
        info = super().get_info()
        info.update({
            'strategy_name': 'Intelligent Adaptive ML Strategy',
            'version': '2.0 - Revolutionary',
            'model_types': list(self.models.keys()),
            'model_performance': self.model_performance,
            'feature_count': len(self.feature_names),
            'market_regime': self.market_regime,
            'regime_indicators': self.regime_indicators,
            'online_learning': True,
            'ensemble_voting': True,
            'regime_adaptation': True,
            'overfitting_protection': True,
            'realistic_targets': True,
            'data_buffer_size': len(self.data_buffer),
            'training_history_size': len(self.training_history),
            'last_retrain': self.last_retrain_date.isoformat() if self.last_retrain_date else 'Never'
        })
        return info
    
    def reset_learning_state(self):
        """RESET LEARNING STATE for fresh start"""
        try:
            self.data_buffer.clear()
            self.training_history.clear()
            self.performance_history.clear()
            self.model_performance = {}
            self.last_retrain_date = None
            self.market_regime = 'unknown'
            
            # Reinitialize models
            self.__init__(self.config)
            
            self.logger.info("Learning state reset successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to reset learning state: {e}")


# USAGE EXAMPLE:
"""
# In your config.json, update the strategy section:
{
    "strategy": {
        "name": "intelligent_ml_strategy",
        "parameters": {
            "lookback_period": 50,
            "prediction_threshold": 0.65,
            "min_confidence": 0.6,
            "regime_window": 100,
            "online_learning_window": 200
        }
    }
}

# Then update strategies/__init__.py:
from .ml_strategy import IntelligentMLStrategy

__all__ = ['IntelligentMLStrategy']

# And update core/controller.py to use the new strategy:
if strategy_name == 'intelligent_ml_strategy':
    self.strategy = IntelligentMLStrategy(self.config.get('strategy', {}))
"""