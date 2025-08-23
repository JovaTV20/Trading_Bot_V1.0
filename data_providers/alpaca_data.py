"""
Alpaca Data Provider - Implementierung für Alpaca Markets API
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, Any, Optional
import logging
import os
from dotenv import load_dotenv

# Alpaca API Import
try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
except ImportError as e:
    logging.error("alpaca-trade-api nicht installiert. Installiere mit: pip install alpaca-trade-api")
    raise ImportError("alpaca-trade-api Paket erforderlich") from e

from core.base_data import DataProviderBase

# Lade Environment-Variablen
load_dotenv()

class AlpacaDataProvider(DataProviderBase):
    """
    Alpaca Markets Data Provider
    
    Lädt historische und Live-Daten über die Alpaca API
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialisiert Alpaca Data Provider
        
        Args:
            config: Konfiguration
        """
        super().__init__(config)
        
        # API Credentials aus Environment
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API Credentials nicht gefunden. Prüfe .env Datei.")
        
        # Initialisiere API Client
        self.api = tradeapi.REST(
            key_id=self.api_key,
            secret_key=self.secret_key,
            base_url=self.base_url,
            api_version='v2'
        )
        
        # Timeframe-Mapping
        self.timeframe_mapping = {
            '1Min': TimeFrame.Minute,
            '5Min': TimeFrame(5, TimeFrameUnit.Minute),
            '15Min': TimeFrame(15, TimeFrameUnit.Minute),
            '30Min': TimeFrame(30, TimeFrameUnit.Minute),
            '1Hour': TimeFrame.Hour,
            '1Day': TimeFrame.Day,
            '1Week': TimeFrame.Week,
            '1Month': TimeFrame.Month
        }
        
        # Timezone
        self.market_timezone = pytz.timezone('US/Eastern')
        
        # Teste API-Verbindung
        self._test_connection()
        
    def _test_connection(self):
        """Testet die API-Verbindung"""
        try:
            account = self.api.get_account()
            self.logger.info(f"Alpaca API verbunden. Account Status: {account.status}")
            
            if account.trading_blocked:
                self.logger.warning("Trading ist blockiert!")
                
        except Exception as e:
            self.logger.error(f"Alpaca API Verbindung fehlgeschlagen: {e}")
            raise
    
    def get_historical(self, symbol: str, start_date: str, end_date: str, 
                      timeframe: str = '1Day') -> pd.DataFrame:
        """
        Lädt historische Kursdaten von Alpaca
        
        Args:
            symbol: Trading-Symbol (z.B. 'AAPL')
            start_date: Startdatum (YYYY-MM-DD)
            end_date: Enddatum (YYYY-MM-DD)
            timeframe: Zeitrahmen
            
        Returns:
            DataFrame mit OHLCV-Daten
        """
        if not self.validate_symbol(symbol):
            raise ValueError(f"Ungültiges Symbol: {symbol}")
            
        if not self.validate_timeframe(timeframe):
            raise ValueError(f"Ungültiger Timeframe: {timeframe}")
        
        try:
            self.logger.info(f"Lade historische Daten: {symbol} ({start_date} - {end_date}, {timeframe})")
            
            # Konvertiere Timeframe
            tf = self.timeframe_mapping.get(timeframe, TimeFrame.Day)
            
            # Lade Daten
            bars = self.api.get_bars(
                symbol,
                tf,
                start=start_date,
                end=end_date,
                adjustment='raw',
                limit=self.config.get('limit', 10000)
            ).df
            
            if bars.empty:
                self.logger.warning(f"Keine Daten für {symbol} erhalten")
                return pd.DataFrame()
            
            # Standardisiere DataFrame
            df = self.standardize_dataframe(bars)
            
            # Füge technische Indikatoren hinzu
            df = self.calculate_technical_indicators(df)
            
            self.logger.info(f"Daten geladen: {len(df)} Datenpunkte")
            return df
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden historischer Daten: {e}")
            raise
    
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
        if not self.validate_symbol(symbol):
            raise ValueError(f"Ungültiges Symbol: {symbol}")
        
        try:
            self.logger.debug(f"Lade aktuelle Daten: {symbol} ({timeframe}, limit={limit})")
            
            # Für aktuelle Daten verwenden wir einen kurzen Zeitraum
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=max(limit * 2, 30))).strftime('%Y-%m-%d')
            
            # Lade Daten
            tf = self.timeframe_mapping.get(timeframe, TimeFrame.Day)
            bars = self.api.get_bars(
                symbol,
                tf,
                start=start_date,
                end=end_date,
                adjustment='raw',
                limit=limit
            ).df
            
            if bars.empty:
                self.logger.warning(f"Keine aktuellen Daten für {symbol}")
                return pd.DataFrame()
            
            # Standardisiere und gib die letzten N Datenpunkte zurück
            df = self.standardize_dataframe(bars)
            df = self.calculate_technical_indicators(df)
            
            return df.tail(limit)
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden aktueller Daten: {e}")
            raise
    
    def is_market_open(self) -> bool:
        """
        Prüft ob der Markt geöffnet ist
        
        Returns:
            True wenn Markt geöffnet
        """
        try:
            clock = self.api.get_clock()
            return clock.is_open
            
        except Exception as e:
            self.logger.error(f"Fehler bei Markt-Status Abfrage: {e}")
            # Fallback: Prüfe Marktzeiten manuell
            return self._is_market_hours()
    
    def _is_market_hours(self) -> bool:
        """
        Fallback-Methode zur Marktzeiten-Prüfung
        
        Returns:
            True wenn in regulären Marktzeiten
        """
        now = datetime.now(self.market_timezone)
        
        # Prüfe Wochentag (0=Montag, 6=Sonntag)
        if now.weekday() >= 5:  # Samstag oder Sonntag
            return False
        
        # Reguläre Marktzeiten: 9:30 - 16:00 ET
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def get_market_calendar(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Lädt Marktkalender
        
        Args:
            start_date: Startdatum (optional)
            end_date: Enddatum (optional)
            
        Returns:
            DataFrame mit Marktzeiten
        """
        try:
            if not start_date:
                start_date = datetime.now().strftime('%Y-%m-%d')
            if not end_date:
                end_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
                
            calendar = self.api.get_calendar(start=start_date, end=end_date)
            
            if not calendar:
                return pd.DataFrame()
                
            # Konvertiere zu DataFrame
            cal_data = []
            for day in calendar:
                cal_data.append({
                    'date': day.date,
                    'open': day.open,
                    'close': day.close,
                    'session_open': day.session_open,
                    'session_close': day.session_close
                })
            
            return pd.DataFrame(cal_data)
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Marktkalenders: {e}")
            return pd.DataFrame()
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Lädt aktuelles Quote (Bid/Ask)
        
        Args:
            symbol: Trading-Symbol
            
        Returns:
            Dict mit Quote-Daten
        """
        try:
            quote = self.api.get_latest_quote(symbol)
            
            return {
                'symbol': symbol,
                'bid_price': float(quote.bid_price),
                'ask_price': float(quote.ask_price),
                'bid_size': quote.bid_size,
                'ask_size': quote.ask_size,
                'timestamp': quote.timestamp,
                'spread': float(quote.ask_price - quote.bid_price),
                'mid_price': float((quote.bid_price + quote.ask_price) / 2)
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Quotes: {e}")
            return {}
    
    def get_trade(self, symbol: str) -> Dict[str, Any]:
        """
        Lädt letzten Trade
        
        Args:
            symbol: Trading-Symbol
            
        Returns:
            Dict mit Trade-Daten
        """
        try:
            trade = self.api.get_latest_trade(symbol)
            
            return {
                'symbol': symbol,
                'price': float(trade.price),
                'size': trade.size,
                'timestamp': trade.timestamp,
                'conditions': trade.conditions
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden des letzten Trades: {e}")
            return {}
    
    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        """
        Lädt Market-Snapshot
        
        Args:
            symbol: Trading-Symbol
            
        Returns:
            Dict mit Snapshot-Daten
        """
        try:
            snapshot = self.api.get_snapshot(symbol)
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.now()
            }
            
            # Latest Quote
            if snapshot.latest_quote:
                result.update({
                    'bid_price': float(snapshot.latest_quote.bid_price),
                    'ask_price': float(snapshot.latest_quote.ask_price),
                    'bid_size': snapshot.latest_quote.bid_size,
                    'ask_size': snapshot.latest_quote.ask_size
                })
            
            # Latest Trade
            if snapshot.latest_trade:
                result.update({
                    'last_price': float(snapshot.latest_trade.price),
                    'last_size': snapshot.latest_trade.size
                })
            
            # Daily Bar
            if snapshot.daily_bar:
                result.update({
                    'open': float(snapshot.daily_bar.open),
                    'high': float(snapshot.daily_bar.high),
                    'low': float(snapshot.daily_bar.low),
                    'close': float(snapshot.daily_bar.close),
                    'volume': snapshot.daily_bar.volume,
                    'change': float(snapshot.daily_bar.close - snapshot.daily_bar.open),
                    'change_pct': float((snapshot.daily_bar.close - snapshot.daily_bar.open) / snapshot.daily_bar.open)
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Snapshots: {e}")
            return {}
    
    def search_assets(self, query: str) -> pd.DataFrame:
        """
        Sucht nach Assets
        
        Args:
            query: Suchbegriff
            
        Returns:
            DataFrame mit gefundenen Assets
        """
        try:
            assets = self.api.list_assets(status='active')
            
            # Filtere nach Suchbegriff
            filtered_assets = []
            query_lower = query.lower()
            
            for asset in assets:
                if (query_lower in asset.symbol.lower() or 
                    (asset.name and query_lower in asset.name.lower())):
                    filtered_assets.append({
                        'symbol': asset.symbol,
                        'name': asset.name,
                        'exchange': asset.exchange,
                        'asset_class': asset.asset_class,
                        'tradable': asset.tradable,
                        'marginable': asset.marginable,
                        'shortable': asset.shortable
                    })
            
            return pd.DataFrame(filtered_assets)
            
        except Exception as e:
            self.logger.error(f"Fehler bei Asset-Suche: {e}")
            return pd.DataFrame()
    
    def get_market_hours(self) -> Dict[str, Any]:
        """
        Gibt detaillierte Marktzeiten zurück
        
        Returns:
            Dict mit Marktzeiten-Informationen
        """
        try:
            calendar = self.get_market_calendar()
            
            if calendar.empty:
                # Fallback zu Standard-Zeiten
                return super().get_market_hours()
            
            today = calendar[calendar['date'] == datetime.now().strftime('%Y-%m-%d')]
            
            if today.empty:
                return {
                    'market_open': '09:30',
                    'market_close': '16:00',
                    'timezone': 'US/Eastern',
                    'is_trading_day': False
                }
            
            today_row = today.iloc[0]
            
            return {
                'market_open': today_row['open'].strftime('%H:%M'),
                'market_close': today_row['close'].strftime('%H:%M'),
                'session_open': today_row['session_open'].strftime('%H:%M'),
                'session_close': today_row['session_close'].strftime('%H:%M'),
                'timezone': 'US/Eastern',
                'is_trading_day': True,
                'date': today_row['date']
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Marktzeiten: {e}")
            return super().get_market_hours()
    
    def get_corporate_actions(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Lädt Corporate Actions (Splits, Dividenden)
        
        Args:
            symbol: Trading-Symbol
            start_date: Startdatum (optional)
            end_date: Enddatum (optional)
            
        Returns:
            DataFrame mit Corporate Actions
        """
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Alpaca bietet Corporate Actions über separate Endpoints
            # Hier würde man verschiedene Endpoints abfragen
            actions = []
            
            # Implementierung für Splits, Dividenden etc.
            # (Abhängig von verfügbaren Alpaca API Endpoints)
            
            return pd.DataFrame(actions)
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Corporate Actions: {e}")
            return pd.DataFrame()
    
    def get_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über den Alpaca Data Provider zurück
        
        Returns:
            Dict mit Provider-Informationen
        """
        info = super().get_info()
        
        try:
            account = self.api.get_account()
            
            info.update({
                'provider_type': 'Alpaca Markets',
                'base_url': self.base_url,
                'account_status': account.status,
                'trading_blocked': account.trading_blocked,
                'pattern_day_trader': account.pattern_day_trader,
                'available_timeframes': list(self.timeframe_mapping.keys()),
                'market_timezone': str(self.market_timezone)
            })
            
        except Exception as e:
            self.logger.warning(f"Konnte Account-Informationen nicht laden: {e}")
            info.update({
                'provider_type': 'Alpaca Markets',
                'base_url': self.base_url,
                'error': str(e)
            })
        
        return info