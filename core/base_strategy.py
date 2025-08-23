"""
Base Strategy Class
Definiert die Schnittstelle für alle Trading-Strategien
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional
import logging

class StrategyBase(ABC):
    """
    Basisklasse für alle Trading-Strategien
    
    Jede Strategie muss fit() und generate_signal() implementieren
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialisiert die Strategie mit Konfiguration
        
        Args:
            config: Dictionary mit Strategie-Parametern
        """
        self.config = config
        self.parameters = config.get('parameters', {})
        self.is_fitted = False
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """
        Trainiert die Strategie mit historischen Daten
        
        Args:
            data: DataFrame mit OHLCV-Daten (columns: open, high, low, close, volume)
        """
        pass
    
    @abstractmethod
    def generate_signal(self, row: pd.Series) -> Dict[str, Any]:
        """
        Generiert ein Trading-Signal für eine einzelne Kerze
        
        Args:
            row: Series mit OHLCV-Daten für einen Zeitpunkt
            
        Returns:
            Dict mit Signal-Informationen:
            {
                'action': 'buy'/'sell'/'hold',
                'confidence': float (0-1),
                'position_size': float (0-1, Anteil des verfügbaren Kapitals),
                'stop_loss': float (optional),
                'take_profit': float (optional)
            }
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validiert die Eingangsdaten
        
        Args:
            data: DataFrame mit OHLCV-Daten
            
        Returns:
            True wenn Daten valide sind
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in data.columns for col in required_columns):
            self.logger.error(f"Fehlende Spalten. Erwartet: {required_columns}")
            return False
            
        if data.empty:
            self.logger.error("Daten sind leer")
            return False
            
        if data.isnull().any().any():
            self.logger.warning("Daten enthalten NaN-Werte")
            
        return True
    
    def calculate_position_size(self, confidence: float, base_size: float = None) -> float:
        """
        Berechnet die Positionsgröße basierend auf Confidence und Risk Management
        
        Args:
            confidence: Vertrauen in das Signal (0-1)
            base_size: Basis-Positionsgröße (optional)
            
        Returns:
            Positionsgröße als Anteil des verfügbaren Kapitals (0-1)
        """
        if base_size is None:
            base_size = self.config.get('risk_management', {}).get('max_position_size', 0.1)
            
        # Skaliere Positionsgröße mit Confidence
        position_size = base_size * confidence
        
        # Begrenze auf Maximum
        max_size = self.config.get('risk_management', {}).get('max_position_size', 0.1)
        position_size = min(position_size, max_size)
        
        return position_size
    
    def get_stop_loss(self, entry_price: float, action: str) -> Optional[float]:
        """
        Berechnet Stop-Loss Level
        
        Args:
            entry_price: Einstiegspreis
            action: 'buy' oder 'sell'
            
        Returns:
            Stop-Loss Preis oder None
        """
        stop_loss_pct = self.config.get('risk_management', {}).get('stop_loss_pct', 0.02)
        
        if stop_loss_pct <= 0:
            return None
            
        if action == 'buy':
            return entry_price * (1 - stop_loss_pct)
        elif action == 'sell':
            return entry_price * (1 + stop_loss_pct)
        
        return None
    
    def get_take_profit(self, entry_price: float, action: str) -> Optional[float]:
        """
        Berechnet Take-Profit Level
        
        Args:
            entry_price: Einstiegspreis
            action: 'buy' oder 'sell'
            
        Returns:
            Take-Profit Preis oder None
        """
        take_profit_pct = self.config.get('risk_management', {}).get('take_profit_pct', 0.04)
        
        if take_profit_pct <= 0:
            return None
            
        if action == 'buy':
            return entry_price * (1 + take_profit_pct)
        elif action == 'sell':
            return entry_price * (1 - take_profit_pct)
        
        return None
    
    def get_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über die Strategie zurück
        
        Returns:
            Dict mit Strategie-Informationen
        """
        return {
            'name': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'parameters': self.parameters
        }