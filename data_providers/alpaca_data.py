"""
Alpaca Data Provider - KORRIGIERT für API Version 3.2+
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, Any, Optional
import logging
import os
from dotenv import load_dotenv

# Alpaca API Import - KORRIGIERT
try:
    import alpaca_trade_api as tradeapi
    
    # KORRIGIERT: Flexible TimeFrame-Imports
    try:
        # Neue API Version
        from alpaca_trade_api.common import TimeFrame, TimeFrameUnit
    except ImportError:
        try:
            # Ältere API Version
            from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
        except ImportError:
            # Fallback: Manuelle Definition
            class TimeFrameUnit:
                Minute = "minute"
                Hour = "hour" 
                Day = "day"
                Week = "week"
                Month = "month"
            
            class TimeFrame:
                Minute = "1Min"
                Hour = "1Hour"
                Day = "1Day"
                Week = "1Week"
                Month = "1Month"
                
                @classmethod
                def from_str(cls, timeframe_str):
                    return timeframe_str
            
            print("⚠️  Verwendung manueller TimeFrame-Definitionen")

    ALPACA_API_AVAILABLE = True
    print("✅ Alpaca Data API importiert")
    
except ImportError as e:
    print(f"❌ Alpaca API Import fehlgeschlagen: {e}")
    ALPACA_API_AVAILABLE = False
    
    # Dummy-Klassen für Entwicklung
    class TimeFrame:
        Minute = "1Min"
        Day = "1Day"

from core.base_data import DataProviderBase

# Lade Environment-Variablen
load_dotenv()

class AlpacaDataProvider(DataProviderBase):
    """
    Alpaca Markets Data Provider - KORRIGIERT für neue API
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialisiert Alpaca Data Provider"""
        super().__init__(config)
        
        if not ALPACA_API_AVAILABLE:
            raise ImportError("Alpaca Trade API nicht verfügbar. Installation: pip install alpaca-trade-api")
        
        # API Credentials aus Environment
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API Credentials nicht gefunden. Prüfe .env Datei.")
        
        # Initialisiere API Client
        try:
            self.api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url,
                api_version='v2'
            )
            print(f"✅ Alpaca Data Provider initialisiert: {self.base_url}")
        except Exception as e:
            raise ConnectionError(f"Alpaca Data Provider Initialisierung fehlgeschlagen: {e}")
        
        # Timeframe-Mapping - VEREINFACHT
        self.timeframe_mapping = {
            '1Min': '1Min',
            '5Min': '5Min', 
            '15Min': '15Min',
            '30Min': '30Min',
            '1Hour': '1Hour',
            '1Day': '1Day',
            '1Week': '1Week',
            '1Month': '1Month'
        }
        
        # Timezone
        self.market_timezone = pytz.timezone('US/Eastern')
        
        # Teste API-Verbindung
        self._test_connection()
        
    def _test_connection(self):
        """Testet die API-Verbindung"""
        try:
            account = self.api.get_account()
            self.logger.info(f"✅ Alpaca Data API verbunden. Account: {account.id}")
            self.logger.info(f"Account Status: {account.status}")
            
            if account.trading_blocked:
                self.logger.warning("⚠️  Trading ist blockiert!")
                
        except Exception as e:
            self.logger.error(f"❌ Alpaca Data API Verbindung fehlgeschlagen: {e}")
            raise
    
    def get_historical(self, symbol: str, start_date: str, end_date: str, 
                      timeframe: str = '1Day') -> pd.DataFrame:
        """
        Lädt historische Kursdaten von Alpaca - VEREINFACHT
        """
        if not self.validate_symbol(symbol):
            raise ValueError(f"Ungültiges Symbol: {symbol}")
            
        if not self.validate_timeframe(timeframe):
            self.logger.warning(f"Unbekannter Timeframe {timeframe}, verwende 1Day")
            timeframe = '1Day'
        
        try:
            self.logger.info(f"Lade historische Daten: {symbol} ({start_date} - {end_date}, {timeframe})")
            
            # VEREINFACHTER API-CALL ohne komplexe TimeFrame-Objekte
            try:
                # Neuer Ansatz: Direkte Timeframe-Strings
                bars = self.api.get_bars(
                    symbol,
                    timeframe,  # Verwende String direkt
                    start=start_date,
                    end=end_date,
                    adjustment='raw',
                    limit=self.config.get('limit', 10000)
                ).df
            except Exception as e:
                self.logger.warning(f"Direkter Timeframe-String fehlgeschlagen: {e}")
                
                # Fallback: Versuche verschiedene Ansätze
                try:
                    # Fallback 1: TimeFrame-Objekt falls verfügbar
                    if hasattr(TimeFrame, timeframe.replace('Min', 'Minute')):
                        tf_obj = getattr(TimeFrame, timeframe.replace('Min', 'Minute'))
                    else:
                        tf_obj = TimeFrame.Day  # Standard-Fallback
                    
                    bars = self.api.get_bars(
                        symbol,
                        tf_obj,
                        start=start_date,
                        end=end_date,
                        adjustment='raw',
                        limit=self.config.get('limit', 10000)
                    ).df
                    
                except Exception as e2:
                    self.logger.error(f"Alle Timeframe-Ansätze fehlgeschlagen: {e2}")
                    # Letzter Fallback: Standard Daily Bars
                    bars = self.api.get_bars(
                        symbol,
                        '1Day',
                        start=start_date,
                        end=end_date
                    ).df
            
            if bars.empty:
                self.logger.warning(f"Keine Daten für {symbol} erhalten")
                return pd.DataFrame()
            
            # Standardisiere DataFrame
            df = self.standardize_dataframe(bars)
            
            # Füge technische Indikatoren hinzu
            df = self.calculate_technical_indicators(df)
            
            self.logger.info(f"✅ Daten geladen: {len(df)} Datenpunkte")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Fehler beim Laden historischer Daten: {e}")
            # Gebe leeres DataFrame zurück statt Exception
            return pd.DataFrame()
    
    def get_latest(self, symbol: str, timeframe: str = '1Day', limit: int = 1) -> pd.DataFrame:
        """
        Lädt die neuesten Kursdaten - VEREINFACHT
        """
        if not self.validate_symbol(symbol):
            raise ValueError(f"Ungültiges Symbol: {symbol}")
        
        try:
            self.logger.debug(f"Lade aktuelle Daten: {symbol} ({timeframe}, limit={limit})")
            
            # Für aktuelle Daten: Kurzer Zeitraum
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=max(limit * 2, 30))).strftime('%Y-%m-%d')
            
            # Verwende get_historical mit kurzen Zeitraum
            df = self.get_historical(symbol, start_date, end_date, timeframe)
            
            if df.empty:
                self.logger.warning(f"Keine aktuellen Daten für {symbol}")
                return pd.DataFrame()
            
            # Gebe die letzten N Datenpunkte zurück
            return df.tail(limit)
            
        except Exception as e:
            self.logger.error(f"❌ Fehler beim Laden aktueller Daten: {e}")
            return pd.DataFrame()
    
    def is_market_open(self) -> bool:
        """
        Prüft ob der Markt geöffnet ist
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
        """
        try:
            now = datetime.now(self.market_timezone)
            
            # Prüfe Wochentag (0=Montag, 6=Sonntag)
            if now.weekday() >= 5:  # Samstag oder Sonntag
                return False
            
            # Reguläre Marktzeiten: 9:30 - 16:00 ET
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            return market_open <= now <= market_close
        except Exception as e:
            self.logger.error(f"Fehler bei Marktzeiten-Fallback: {e}")
            return False
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Lädt aktuelles Quote (Bid/Ask) - VEREINFACHT
        """
        try:
            # Versuche Latest Quote
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
    
    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        """
        Lädt Market-Snapshot - VEREINFACHT
        """
        try:
            snapshot = self.api.get_snapshot(symbol)
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.now()
            }
            
            # Latest Quote
            if hasattr(snapshot, 'latest_quote') and snapshot.latest_quote:
                result.update({
                    'bid_price': float(snapshot.latest_quote.bid_price),
                    'ask_price': float(snapshot.latest_quote.ask_price),
                    'bid_size': snapshot.latest_quote.bid_size,
                    'ask_size': snapshot.latest_quote.ask_size
                })
            
            # Latest Trade
            if hasattr(snapshot, 'latest_trade') and snapshot.latest_trade:
                result.update({
                    'last_price': float(snapshot.latest_trade.price),
                    'last_size': snapshot.latest_trade.size
                })
            
            # Daily Bar
            if hasattr(snapshot, 'daily_bar') and snapshot.daily_bar:
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
    
    def get_market_hours(self) -> Dict[str, Any]:
        """
        Gibt detaillierte Marktzeiten zurück
        """
        try:
            # Versuche Marktkalender zu laden
            today = datetime.now().strftime('%Y-%m-%d')
            calendar = self.api.get_calendar(start=today, end=today)
            
            if calendar and len(calendar) > 0:
                today_schedule = calendar[0]
                return {
                    'market_open': today_schedule.open.strftime('%H:%M'),
                    'market_close': today_schedule.close.strftime('%H:%M'),
                    'session_open': today_schedule.session_open.strftime('%H:%M'),
                    'session_close': today_schedule.session_close.strftime('%H:%M'),
                    'timezone': 'US/Eastern',
                    'is_trading_day': True,
                    'date': today
                }
            else:
                return {
                    'market_open': '09:30',
                    'market_close': '16:00',
                    'timezone': 'US/Eastern',
                    'is_trading_day': False
                }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Marktzeiten: {e}")
            return super().get_market_hours()
    
    def get_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über den Alpaca Data Provider zurück
        """
        info = {
            'provider_type': 'Alpaca Markets Data',
            'base_url': self.base_url,
            'api_available': ALPACA_API_AVAILABLE,
            'available_timeframes': list(self.timeframe_mapping.keys()),
            'market_timezone': str(self.market_timezone)
        }
        
        try:
            account = self.api.get_account()
            info.update({
                'account_status': account.status,
                'trading_blocked': account.trading_blocked,
                'pattern_day_trader': account.pattern_day_trader
            })
        except Exception as e:
            self.logger.warning(f"Konnte Account-Informationen nicht laden: {e}")
            info['connection_error'] = str(e)
        
        return info

# Test-Funktion
def test_alpaca_data():
    """Teste Alpaca Data Provider"""
    try:
        provider = AlpacaDataProvider({})
        
        # Test 1: Account Info
        info = provider.get_info()
        print(f"✅ Provider Info: {info['provider_type']}")
        
        # Test 2: Market Status
        is_open = provider.is_market_open()
        print(f"✅ Market Open: {is_open}")
        
        # Test 3: Historical Data (kurzer Zeitraum)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        
        data = provider.get_historical('AAPL', start_date, end_date, '1Day')
        if not data.empty:
            print(f"✅ Historical Data: {len(data)} Datenpunkte")
            print(f"   Letzter Preis: ${data.iloc[-1]['close']:.2f}")
        else:
            print("⚠️  Keine historischen Daten erhalten")
        
        return True
        
    except Exception as e:
        print(f"❌ Alpaca Data Test fehlgeschlagen: {e}")
        return False

if __name__ == "__main__":
    # Direkter Test
    success = test_alpaca_data()
    print(f"{'✅' if success else '❌'} Alpaca Data Provider Test: {'Erfolgreich' if success else 'Fehlgeschlagen'}")