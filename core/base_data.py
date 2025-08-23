"""
Base Data Provider Class
Definiert die Schnittstelle für alle Datenquellen
"""

from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

class DataProviderBase(ABC):
    """
    Basisklasse für alle Datenanbieter
    
    Jeder Provider muss get_historical und get_latest implementieren
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialisiert den Datenanbieter
        
        Args:
            config: Konfigurationsdictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def get_historical(self, symbol: str, start_date: str, end_date: str, 
                      timeframe: str = '1Day') -> pd.DataFrame:
        """
        Lädt historische Kursdaten
        
        Args:
            symbol: Trading-Symbol (z.B. 'AAPL')
            start_date: Startdatum (YYYY-MM-DD)
            end_date: Enddatum (YYYY-MM-DD)
            timeframe: Zeitrahmen ('1Min', '5Min', '1Hour', '1Day')
            
        Returns:
            DataFrame mit OHLCV-Daten, Index=Datetime
            Spalten: open, high, low, close, volume
        """
        pass
    
    @abstractmethod
    def get_latest(self, symbol: str, timeframe: str = '1Day', limit: int = 1) -> pd.DataFrame:
        """
        Lädt die neuesten Kursdaten
        
        Args:
            symbol: Trading-Symbol
            timeframe: Zeitrahmen
            limit: Anzahl der gewünschten Kerzen
            
        Returns:
            DataFrame mit den neuesten OHLCV-Daten
        """
        pass
    
    @abstractmethod
    def is_market_open(self) -> bool:
        """
        Prüft ob der Markt geöffnet ist
        
        Returns:
            True wenn Markt geöffnet
        """
        pass
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validiert ein Trading-Symbol
        
        Args:
            symbol: Zu validierendes Symbol
            
        Returns:
            True wenn Symbol gültig ist
        """
        if not symbol or not isinstance(symbol, str):
            self.logger.error("Symbol muss ein nicht-leerer String sein")
            return False
            
        if len(symbol) < 1 or len(symbol) > 10:
            self.logger.error("Symbol muss zwischen 1 und 10 Zeichen lang sein")
            return False
            
        # Grundlegende Format-Validierung
        if not symbol.replace('.', '').replace('-', '').isalnum():
            self.logger.error("Symbol enthält ungültige Zeichen")
            return False
            
        return True
    
    def validate_timeframe(self, timeframe: str) -> bool:
        """
        Validiert einen Zeitrahmen
        
        Args:
            timeframe: Zu validierender Zeitrahmen
            
        Returns:
            True wenn Zeitrahmen gültig ist
        """
        valid_timeframes = ['1Min', '5Min', '15Min', '30Min', '1Hour', '1Day', '1Week', '1Month']
        return timeframe in valid_timeframes
    
    def standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardisiert DataFrame-Format
        
        Args:
            df: Eingabe-DataFrame
            
        Returns:
            Standardisiertes DataFrame
        """
        if df.empty:
            return df
        
        # Stelle sicher, dass Index datetime ist
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            elif 'time' in df.columns:
                df = df.set_index('time')
            else:
                self.logger.warning("Kein Timestamp-Index gefunden")
        
        # Standardisiere Spaltennamen (lowercase)
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Entferne Duplikate
        if df.index.duplicated().any():
            self.logger.warning("Duplikate im Index gefunden - entferne diese")
            df = df[~df.index.duplicated(keep='first')]
        
        # Sortiere nach Index
        df = df.sort_index()
        
        return df
    
    def get_market_hours(self) -> Dict[str, Any]:
        """
        Gibt Marktzeiten zurück
        
        Returns:
            Dict mit Marktzeiten-Informationen
        """
        # Standard US-Marktzeiten (kann von Subklassen überschrieben werden)
        return {
            'market_open': '09:30',
            'market_close': '16:00', 
            'timezone': 'US/Eastern',
            'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        }
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fügt grundlegende technische Indikatoren hinzu
        
        Args:
            df: DataFrame mit OHLCV-Daten
            
        Returns:
            DataFrame mit zusätzlichen Indikatoren
        """
        if df.empty or 'close' not in df.columns:
            return df
            
        try:
            # Simple Moving Averages
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            
            # Returns
            df['return_1d'] = df['close'].pct_change()
            df['return_5d'] = df['close'].pct_change(5)
            
            # Volatilität
            df['volatility_20d'] = df['return_1d'].rolling(20).std()
            
            # Volume Moving Average
            if 'volume' in df.columns:
                df['volume_sma_20'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
        except Exception as e:
            self.logger.warning(f"Fehler bei technischen Indikatoren: {e}")
            
        return df
    
    def get_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über den Datenanbieter zurück
        
        Returns:
            Dict mit Provider-Informationen
        """
        return {
            'provider': self.__class__.__name__,
            'config': self.config,
            'market_hours': self.get_market_hours()
        }