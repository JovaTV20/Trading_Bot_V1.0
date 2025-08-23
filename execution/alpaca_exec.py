"""
Alpaca Execution Provider - KORRIGIERT für neue API Version 3.2+
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import os
import time
from dotenv import load_dotenv

# Alpaca API Import - KORRIGIERT für neue Version
try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.entity import Order
    
    # KORRIGIERT: Neue Import-Namen für API Version 3.2+
    try:
        # Versuche neue Import-Namen (API 3.2+)
        from alpaca_trade_api.common import TimeInForce, OrderSide, OrderType
    except ImportError:
        try:
            # Fallback für ältere Versionen
            from alpaca_trade_api.rest import TimeInForce, OrderSide, OrderType
        except ImportError:
            # Manuelle Definition falls Imports fehlen
            class TimeInForce:
                Day = "day"
                GTC = "gtc"
                IOC = "ioc"
                FOK = "fok"
            
            class OrderSide:
                Buy = "buy"
                Sell = "sell"
            
            class OrderType:
                Market = "market"
                Limit = "limit"
                Stop = "stop"
                StopLimit = "stop_limit"
            
            print("⚠️  Verwendung manueller Enum-Definitionen für Alpaca API")

    # Teste TimeFrame Import
    try:
        from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
    except ImportError:
        # Fallback für TimeFrame
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
            
            def __init__(self, amount, unit):
                self.amount = amount
                self.unit = unit
                
            def __str__(self):
                return f"{self.amount}{self.unit.capitalize()}"

    ALPACA_API_AVAILABLE = True
    print("✅ Alpaca Trade API erfolgreich importiert")
    
except ImportError as e:
    print(f"❌ Alpaca Trade API Import fehlgeschlagen: {e}")
    ALPACA_API_AVAILABLE = False
    
    # Dummy-Klassen für Entwicklung
    class TimeInForce:
        Day = "day"
        GTC = "gtc"
    
    class OrderSide:
        Buy = "buy"
        Sell = "sell"
    
    class OrderType:
        Market = "market"
        Limit = "limit"

from core.base_execution import ExecutionBase

# Lade Environment-Variablen
load_dotenv()

class AlpacaExecution(ExecutionBase):
    """
    Alpaca Markets Execution Provider - KORRIGIERT für neue API
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialisiert Alpaca Execution Provider
        """
        super().__init__(config)
        
        if not ALPACA_API_AVAILABLE:
            raise ImportError("Alpaca Trade API ist nicht verfügbar. Installation: pip install alpaca-trade-api")
        
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
            print(f"✅ Alpaca API Client initialisiert: {self.base_url}")
        except Exception as e:
            raise ConnectionError(f"Alpaca API Client Initialisierung fehlgeschlagen: {e}")
        
        # Order-Type Mapping - VEREINFACHT
        self.order_type_mapping = {
            'market': 'market',
            'limit': 'limit',
            'stop': 'stop',
            'stop_limit': 'stop_limit'
        }
        
        # Time-in-Force Mapping
        self.tif_mapping = {
            'day': 'day',
            'gtc': 'gtc',
            'ioc': 'ioc',
            'fok': 'fok'
        }
        
        # Side Mapping
        self.side_mapping = {
            'buy': 'buy',
            'sell': 'sell'
        }
        
        # Teste API-Verbindung
        self._test_connection()
        
    def _test_connection(self):
        """Testet die API-Verbindung"""
        try:
            account = self.api.get_account()
            self.logger.info(f"Alpaca Execution verbunden. Account: {account.id}")
            self.logger.info(f"Account Status: {account.status}")
            self.logger.info(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
            
            if account.trading_blocked:
                self.logger.error("⚠️  Trading ist blockiert!")
                
            if not account.trade_suspended_by_user:
                self.logger.info("✅ Trading aktiviert")
            else:
                self.logger.warning("⚠️  Trading vom Benutzer pausiert")
                
        except Exception as e:
            self.logger.error(f"Alpaca Execution Verbindung fehlgeschlagen: {e}")
            raise
    
    def place_order(self, symbol: str, side: str, quantity: float, 
                   order_type: str = "market", price: float = None,
                   stop_price: float = None, time_in_force: str = "day") -> Dict[str, Any]:
        """
        Platziert eine Order bei Alpaca - VEREINFACHT
        """
        # Validiere Parameter
        if not self.validate_order(symbol, side, quantity, order_type, price):
            raise ValueError("Ungültige Order-Parameter")
        
        try:
            self.logger.info(f"Platziere Order: {side} {quantity} {symbol} ({order_type})")
            
            # Bereite Order-Parameter vor - VEREINFACHT
            order_params = {
                'symbol': symbol.upper(),
                'qty': int(quantity) if not self.config.get('fractional_shares', False) else quantity,
                'side': side.lower(),
                'type': order_type.lower(),
                'time_in_force': time_in_force.lower()
            }
            
            # Füge Preis-Parameter hinzu
            if order_type.lower() in ['limit', 'stop_limit'] and price is not None:
                order_params['limit_price'] = str(price)
                
            if order_type.lower() in ['stop', 'stop_limit'] and stop_price is not None:
                order_params['stop_price'] = str(stop_price)
            
            # Platziere Order - VEREINFACHTER API-CALL
            order = self.api.submit_order(**order_params)
            
            # Konvertiere zu standardisiertem Format
            order_info = self._convert_order(order)
            
            # Logge Order
            self.log_order(order_info)
            
            self.logger.info(f"✅ Order platziert: {order.id} - Status: {order.status}")
            
            return order_info
            
        except Exception as e:
            self.logger.error(f"❌ Fehler beim Platzieren der Order: {e}")
            # Gebe detaillierten Fehler zurück statt Exception zu werfen
            return {
                'error': str(e),
                'status': 'failed',
                'symbol': symbol,
                'side': side,
                'quantity': quantity
            }
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Fragt Order-Status ab"""
        try:
            order = self.api.get_order(order_id)
            return self._convert_order(order)
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen des Order-Status: {e}")
            return {'error': str(e), 'order_id': order_id}
    
    def cancel_order(self, order_id: str) -> bool:
        """Storniert eine Order"""
        try:
            self.api.cancel_order(order_id)
            self.logger.info(f"Order storniert: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Stornieren der Order: {e}")
            return False
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Lädt alle offenen Positionen"""
        try:
            positions = self.api.list_positions()
            
            position_list = []
            for pos in positions:
                position_info = {
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'side': 'long' if float(pos.qty) > 0 else 'short',
                    'market_value': float(pos.market_value) if pos.market_value else 0,
                    'cost_basis': float(pos.cost_basis) if pos.cost_basis else 0,
                    'unrealized_pnl': float(pos.unrealized_pnl) if pos.unrealized_pnl else 0,
                    'unrealized_pnl_pct': float(pos.unrealized_plpc) if pos.unrealized_plpc else 0,
                    'avg_entry_price': float(pos.avg_entry_price) if pos.avg_entry_price else 0,
                    'current_price': float(pos.current_price) if pos.current_price else 0,
                    'last_day_price': float(pos.lastday_price) if pos.lastday_price else 0,
                    'change_today': float(pos.change_today) if pos.change_today else 0
                }
                position_list.append(position_info)
            
            return position_list
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Positionen: {e}")
            return []
    
    def get_account_info(self) -> Dict[str, Any]:
        """Lädt Account-Informationen"""
        try:
            account = self.api.get_account()
            
            return {
                'id': account.id,
                'account_number': account.account_number,
                'status': account.status,
                'currency': account.currency,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'long_market_value': float(account.long_market_value) if account.long_market_value else 0,
                'short_market_value': float(account.short_market_value) if account.short_market_value else 0,
                'daytrade_count': account.daytrade_count,
                'daytrading_buying_power': float(account.daytrading_buying_power),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'trade_suspended_by_user': account.trade_suspended_by_user,
                'multiplier': account.multiplier,
                'created_at': account.created_at
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Account-Informationen: {e}")
            return {
                'error': str(e),
                'equity': 0,
                'cash': 0,
                'buying_power': 0
            }
    
    def get_orders(self, status: str = 'all', limit: int = 100) -> List[Dict[str, Any]]:
        """Lädt Order-Historie"""
        try:
            orders = self.api.list_orders(status=status, limit=limit)
            return [self._convert_order(order) for order in orders]
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Orders: {e}")
            return []
    
    def _convert_order(self, alpaca_order) -> Dict[str, Any]:
        """Konvertiert Alpaca Order zu standardisiertem Format"""
        try:
            return {
                'order_id': alpaca_order.id,
                'client_order_id': getattr(alpaca_order, 'client_order_id', ''),
                'status': alpaca_order.status,
                'symbol': alpaca_order.symbol,
                'side': alpaca_order.side,
                'quantity': float(alpaca_order.qty),
                'filled_quantity': float(getattr(alpaca_order, 'filled_qty', 0)),
                'order_type': alpaca_order.order_type,
                'time_in_force': getattr(alpaca_order, 'time_in_force', 'day'),
                'price': float(getattr(alpaca_order, 'limit_price', 0)) if getattr(alpaca_order, 'limit_price') else None,
                'stop_price': float(getattr(alpaca_order, 'stop_price', 0)) if getattr(alpaca_order, 'stop_price') else None,
                'filled_avg_price': float(getattr(alpaca_order, 'filled_avg_price', 0)) if getattr(alpaca_order, 'filled_avg_price') else None,
                'timestamp': alpaca_order.created_at,
                'updated_at': getattr(alpaca_order, 'updated_at', None),
                'submitted_at': getattr(alpaca_order, 'submitted_at', None),
                'filled_at': getattr(alpaca_order, 'filled_at', None)
            }
        except Exception as e:
            self.logger.error(f"Fehler bei Order-Konvertierung: {e}")
            return {
                'order_id': str(alpaca_order.id) if hasattr(alpaca_order, 'id') else 'unknown',
                'status': 'conversion_error',
                'error': str(e)
            }
    
    def close_position(self, symbol: str) -> Dict[str, Any]:
        """Schließt komplette Position für ein Symbol"""
        position = self.get_position_for_symbol(symbol)
        if not position:
            self.logger.warning(f"Keine Position für {symbol} gefunden")
            return {'status': 'no_position'}
            
        quantity = abs(float(position.get('qty', 0)))
        if quantity == 0:
            return {'status': 'no_quantity'}
            
        # Bestimme Order-Seite (entgegengesetzt zur Position)
        current_side = position.get('side', 'long')
        order_side = 'sell' if current_side == 'long' else 'buy'
        
        # Platziere Market-Order zum Schließen
        return self.place_order(symbol, order_side, quantity, "market")
    
    def get_position_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Lädt Position für ein bestimmtes Symbol"""
        positions = self.get_positions()
        for pos in positions:
            if pos.get('symbol') == symbol:
                return pos
        return None
    
    def get_daily_pnl(self) -> float:
        """Berechnet täglichen P&L"""
        try:
            account = self.get_account_info()
            current_equity = float(account.get('equity', 0))
            last_equity = float(account.get('last_equity', current_equity))
            return current_equity - last_equity
        except Exception as e:
            self.logger.error(f"Fehler bei Daily P&L Berechnung: {e}")
            return 0.0
    
    def get_info(self) -> Dict[str, Any]:
        """Gibt Informationen über das Alpaca Execution System zurück"""
        info = {
            'provider_type': 'Alpaca Markets',
            'base_url': self.base_url,
            'api_available': ALPACA_API_AVAILABLE
        }
        
        try:
            account = self.get_account_info()
            if 'error' not in account:
                info.update({
                    'account_id': account.get('id'),
                    'account_status': account.get('status'),
                    'trading_blocked': account.get('trading_blocked'),
                    'portfolio_value': account.get('portfolio_value')
                })
        except Exception as e:
            info['connection_error'] = str(e)
        
        return info

# Zusätzliche Hilfs-Funktionen für Kompatibilität
def test_alpaca_connection():
    """Teste Alpaca-Verbindung direkt"""
    try:
        load_dotenv()
        
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not api_key or not secret_key:
            return False, "API-Keys nicht gefunden in .env"
        
        api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        account = api.get_account()
        
        return True, f"Verbindung erfolgreich - Portfolio: ${float(account.portfolio_value):,.2f}"
        
    except Exception as e:
        return False, f"Verbindung fehlgeschlagen: {e}"

if __name__ == "__main__":
    # Direkter Test
    success, message = test_alpaca_connection()
    print(f"{'✅' if success else '❌'} Alpaca Test: {message}")