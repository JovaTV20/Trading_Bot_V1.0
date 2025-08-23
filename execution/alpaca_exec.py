"""
Alpaca Execution Provider - Implementierung für Alpaca Markets Order Execution
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import os
import time
from dotenv import load_dotenv

# Alpaca API Import
try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.entity import Order
    from alpaca_trade_api.rest import TimeInForce, OrderSide, OrderType
except ImportError as e:
    logging.error("alpaca-trade-api nicht installiert. Installiere mit: pip install alpaca-trade-api")
    raise ImportError("alpaca-trade-api Paket erforderlich") from e

from core.base_execution import ExecutionBase

# Lade Environment-Variablen
load_dotenv()

class AlpacaExecution(ExecutionBase):
    """
    Alpaca Markets Execution Provider
    
    Führt Orders über die Alpaca API aus
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialisiert Alpaca Execution Provider
        
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
        
        # Order-Type Mapping
        self.order_type_mapping = {
            'market': OrderType.Market,
            'limit': OrderType.Limit,
            'stop': OrderType.Stop,
            'stop_limit': OrderType.StopLimit
        }
        
        # Time-in-Force Mapping
        self.tif_mapping = {
            'day': TimeInForce.Day,
            'gtc': TimeInForce.GTC,
            'ioc': TimeInForce.IOC,
            'fok': TimeInForce.FOK
        }
        
        # Side Mapping
        self.side_mapping = {
            'buy': OrderSide.Buy,
            'sell': OrderSide.Sell
        }
        
        # Teste API-Verbindung
        self._test_connection()
        
    def _test_connection(self):
        """Testet die API-Verbindung"""
        try:
            account = self.api.get_account()
            self.logger.info(f"Alpaca Execution verbunden. Account: {account.id}")
            
            if account.trading_blocked:
                self.logger.error("Trading ist blockiert!")
                raise ValueError("Trading blockiert")
                
            if not account.trade_suspended_by_user:
                self.logger.info("Trading aktiviert")
            else:
                self.logger.warning("Trading vom Benutzer pausiert")
                
        except Exception as e:
            self.logger.error(f"Alpaca Execution Verbindung fehlgeschlagen: {e}")
            raise
    
    def place_order(self, symbol: str, side: str, quantity: float, 
                   order_type: str = "market", price: float = None,
                   stop_price: float = None, time_in_force: str = "day") -> Dict[str, Any]:
        """
        Platziert eine Order bei Alpaca
        
        Args:
            symbol: Trading-Symbol
            side: 'buy' oder 'sell'
            quantity: Anzahl Aktien/Anteile
            order_type: Order-Typ ('market', 'limit', 'stop', 'stop_limit')
            price: Limit-Preis (bei Limit-Orders)
            stop_price: Stop-Preis (bei Stop-Orders)
            time_in_force: Gültigkeitsdauer ('day', 'gtc', 'ioc', 'fok')
            
        Returns:
            Dict mit Order-Informationen
        """
        # Validiere Parameter
        if not self.validate_order(symbol, side, quantity, order_type, price):
            raise ValueError("Ungültige Order-Parameter")
        
        try:
            self.logger.info(f"Platziere Order: {side} {quantity} {symbol} ({order_type})")
            
            # Konvertiere Parameter
            alpaca_side = self.side_mapping[side.lower()]
            alpaca_type = self.order_type_mapping[order_type.lower()]
            alpaca_tif = self.tif_mapping[time_in_force.lower()]
            
            # Bereite Order-Parameter vor
            order_params = {
                'symbol': symbol.upper(),
                'qty': quantity,
                'side': alpaca_side,
                'type': alpaca_type,
                'time_in_force': alpaca_tif
            }
            
            # Füge Preis-Parameter hinzu
            if order_type.lower() in ['limit', 'stop_limit'] and price is not None:
                order_params['limit_price'] = price
                
            if order_type.lower() in ['stop', 'stop_limit'] and stop_price is not None:
                order_params['stop_price'] = stop_price
            
            # Fractional Shares unterstützen
            if self.config.get('fractional_shares', False):
                order_params['qty'] = quantity  # Alpaca unterstützt Decimal-Quantities
            else:
                order_params['qty'] = int(quantity)  # Ganze Aktien
            
            # Platziere Order
            order = self.api.submit_order(**order_params)
            
            # Konvertiere zu standardisiertem Format
            order_info = self._convert_order(order)
            
            # Logge Order
            self.log_order(order_info)
            
            self.logger.info(f"Order platziert: {order.id} - Status: {order.status}")
            
            return order_info
            
        except Exception as e:
            self.logger.error(f"Fehler beim Platzieren der Order: {e}")
            raise
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Fragt Order-Status ab
        
        Args:
            order_id: Order-ID
            
        Returns:
            Dict mit Order-Status-Informationen
        """
        try:
            order = self.api.get_order(order_id)
            return self._convert_order(order)
            
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen des Order-Status: {e}")
            return {'error': str(e)}
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Storniert eine Order
        
        Args:
            order_id: Order-ID
            
        Returns:
            True wenn erfolgreich storniert
        """
        try:
            self.api.cancel_order(order_id)
            self.logger.info(f"Order storniert: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Stornieren der Order: {e}")
            return False
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Lädt alle offenen Positionen
        
        Returns:
            Liste mit Position-Informationen
        """
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
        """
        Lädt Account-Informationen
        
        Returns:
            Dict mit Account-Daten
        """
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
                'initial_margin': float(account.initial_margin) if account.initial_margin else 0,
                'maintenance_margin': float(account.maintenance_margin) if account.maintenance_margin else 0,
                'daytrade_count': account.daytrade_count,
                'daytrading_buying_power': float(account.daytrading_buying_power),
                'regt_buying_power': float(account.regt_buying_power),
                'sma': float(account.sma) if account.sma else 0,
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked,
                'trade_suspended_by_user': account.trade_suspended_by_user,
                'multiplier': account.multiplier,
                'created_at': account.created_at
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Account-Informationen: {e}")
            return {}
    
    def get_portfolio_history(self, period: str = '1D', timeframe: str = '1Min') -> pd.DataFrame:
        """
        Lädt Portfolio-Historie
        
        Args:
            period: Zeitraum ('1D', '7D', '1M', '3M', '1A', '2A', '5A', 'all')
            timeframe: Zeitrahmen ('1Min', '5Min', '15Min', '1H', '1D')
            
        Returns:
            DataFrame mit Portfolio-Historie
        """
        try:
            portfolio = self.api.get_portfolio_history(period=period, timeframe=timeframe)
            
            if not portfolio.timestamp:
                return pd.DataFrame()
            
            # Konvertiere zu DataFrame
            df = pd.DataFrame({
                'timestamp': [datetime.fromtimestamp(ts) for ts in portfolio.timestamp],
                'equity': portfolio.equity,
                'profit_loss': portfolio.profit_loss,
                'profit_loss_pct': portfolio.profit_loss_pct,
                'base_value': portfolio.base_value
            })
            
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Portfolio-Historie: {e}")
            return pd.DataFrame()
    
    def get_orders(self, status: str = 'all', limit: int = 100, 
                   start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """
        Lädt Order-Historie
        
        Args:
            status: Order-Status Filter ('open', 'closed', 'all')
            limit: Maximale Anzahl Orders
            start_date: Startdatum (optional)
            end_date: Enddatum (optional)
            
        Returns:
            Liste mit Order-Informationen
        """
        try:
            # Parameter für API-Call
            params = {
                'status': status,
                'limit': limit
            }
            
            if start_date:
                params['after'] = start_date
            if end_date:
                params['until'] = end_date
            
            orders = self.api.list_orders(**params)
            
            return [self._convert_order(order) for order in orders]
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Orders: {e}")
            return []
    
    def create_bracket_order(self, symbol: str, side: str, quantity: float,
                           stop_loss: float = None, take_profit: float = None) -> List[Dict[str, Any]]:
        """
        Erstellt Bracket-Order (mit OCO Orders)
        
        Args:
            symbol: Trading-Symbol
            side: Order-Seite
            quantity: Menge
            stop_loss: Stop-Loss Preis
            take_profit: Take-Profit Preis
            
        Returns:
            Liste mit Order-Dictionaries
        """
        try:
            self.logger.info(f"Erstelle Bracket-Order: {side} {quantity} {symbol}")
            
            orders = []
            
            # Hauptorder (Market)
            main_order = self.place_order(symbol, side, quantity, "market")
            orders.append(main_order)
            
            # Warte bis Hauptorder gefüllt ist
            if main_order.get('status') in ['new', 'pending_new']:
                time.sleep(1)  # Kurz warten
                main_order = self.get_order_status(main_order['order_id'])
            
            if main_order.get('status') == 'filled':
                # Bestimme Exit-Side
                exit_side = 'sell' if side == 'buy' else 'buy'
                
                # OCO Order erstellen (One-Cancels-Other)
                if stop_loss and take_profit:
                    try:
                        # Alpaca OCO Order
                        oco_order = self.api.submit_order(
                            symbol=symbol.upper(),
                            qty=quantity,
                            side=self.side_mapping[exit_side],
                            type=OrderType.Limit,
                            time_in_force=TimeInForce.GTC,
                            limit_price=take_profit,
                            stop_loss={
                                'stop_price': stop_loss
                            }
                        )
                        
                        orders.append(self._convert_order(oco_order))
                        
                    except:
                        # Fallback: Separate Orders
                        if stop_loss:
                            sl_order = self.place_order(symbol, exit_side, quantity, "stop", stop_price=stop_loss)
                            orders.append(sl_order)
                            
                        if take_profit:
                            tp_order = self.place_order(symbol, exit_side, quantity, "limit", price=take_profit)
                            orders.append(tp_order)
                            
                elif stop_loss:
                    sl_order = self.place_order(symbol, exit_side, quantity, "stop", stop_price=stop_loss)
                    orders.append(sl_order)
                    
                elif take_profit:
                    tp_order = self.place_order(symbol, exit_side, quantity, "limit", price=take_profit)
                    orders.append(tp_order)
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Fehler bei Bracket-Order: {e}")
            return [main_order] if 'main_order' in locals() else []
    
    def get_daily_pnl(self) -> float:
        """
        Berechnet täglichen P&L
        
        Returns:
            Täglicher Profit/Loss
        """
        try:
            account = self.api.get_account()
            
            # Berechne aus equity und last_equity
            current_equity = float(account.equity)
            last_equity = float(account.last_equity)
            
            return current_equity - last_equity
            
        except Exception as e:
            self.logger.error(f"Fehler bei Daily P&L Berechnung: {e}")
            return 0.0
    
    def close_all_positions(self) -> List[Dict[str, Any]]:
        """
        Schließt alle offenen Positionen
        
        Returns:
            Liste mit Close-Orders
        """
        try:
            self.logger.info("Schließe alle Positionen...")
            
            # Alpaca bietet einen speziellen Endpoint
            response = self.api.close_all_positions(cancel_orders=True)
            
            orders = []
            for order_info in response:
                if 'id' in order_info:  # Erfolgreiche Order
                    order = self.get_order_status(order_info['id'])
                    orders.append(order)
                    
            self.logger.info(f"{len(orders)} Positionen geschlossen")
            return orders
            
        except Exception as e:
            self.logger.error(f"Fehler beim Schließen aller Positionen: {e}")
            return []
    
    def cancel_all_orders(self) -> bool:
        """
        Storniert alle offenen Orders
        
        Returns:
            True wenn erfolgreich
        """
        try:
            self.api.cancel_all_orders()
            self.logger.info("Alle Orders storniert")
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Stornieren aller Orders: {e}")
            return False
    
    def get_watchlist(self, name: str = None) -> List[str]:
        """
        Lädt Watchlist
        
        Args:
            name: Watchlist-Name (optional)
            
        Returns:
            Liste mit Symbolen
        """
        try:
            watchlists = self.api.get_watchlists()
            
            if not watchlists:
                return []
                
            if name:
                # Suche spezifische Watchlist
                for wl in watchlists:
                    if wl.name == name:
                        return [asset.symbol for asset in wl.assets]
                return []
            else:
                # Erste Watchlist verwenden
                return [asset.symbol for asset in watchlists[0].assets]
                
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Watchlist: {e}")
            return []
    
    def add_to_watchlist(self, watchlist_name: str, symbol: str) -> bool:
        """
        Fügt Symbol zu Watchlist hinzu
        
        Args:
            watchlist_name: Watchlist-Name
            symbol: Trading-Symbol
            
        Returns:
            True wenn erfolgreich
        """
        try:
            # Finde Watchlist
            watchlists = self.api.get_watchlists()
            watchlist_id = None
            
            for wl in watchlists:
                if wl.name == watchlist_name:
                    watchlist_id = wl.id
                    break
            
            if not watchlist_id:
                # Erstelle neue Watchlist
                new_wl = self.api.create_watchlist(watchlist_name, [])
                watchlist_id = new_wl.id
            
            # Füge Symbol hinzu
            self.api.add_to_watchlist(watchlist_id, symbol)
            self.logger.info(f"Symbol {symbol} zu Watchlist {watchlist_name} hinzugefügt")
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Hinzufügen zu Watchlist: {e}")
            return False
    
    def _convert_order(self, alpaca_order: Order) -> Dict[str, Any]:
        """
        Konvertiert Alpaca Order zu standardisiertem Format
        
        Args:
            alpaca_order: Alpaca Order-Objekt
            
        Returns:
            Standardisiertes Order-Dictionary
        """
        return {
            'order_id': alpaca_order.id,
            'client_order_id': alpaca_order.client_order_id,
            'status': alpaca_order.status,
            'symbol': alpaca_order.symbol,
            'side': alpaca_order.side.value.lower(),
            'quantity': float(alpaca_order.qty),
            'filled_quantity': float(alpaca_order.filled_qty) if alpaca_order.filled_qty else 0,
            'order_type': alpaca_order.order_type.value.lower(),
            'time_in_force': alpaca_order.time_in_force.value.lower(),
            'price': float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
            'stop_price': float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
            'filled_avg_price': float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
            'timestamp': alpaca_order.created_at,
            'updated_at': alpaca_order.updated_at,
            'submitted_at': alpaca_order.submitted_at,
            'filled_at': alpaca_order.filled_at,
            'canceled_at': alpaca_order.canceled_at,
            'rejected_at': alpaca_order.rejected_at,
            'legs': []  # Für Multi-Leg Orders
        }
    
    def _wait_for_order_fill(self, order_id: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Wartet auf Order-Ausführung
        
        Args:
            order_id: Order-ID
            timeout: Timeout in Sekunden
            
        Returns:
            Order-Status
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            order = self.get_order_status(order_id)
            status = order.get('status', '')
            
            if status in ['filled', 'canceled', 'rejected']:
                return order
                
            time.sleep(0.5)  # Kurz warten
            
        # Timeout erreicht
        self.logger.warning(f"Timeout bei Order {order_id}")
        return self.get_order_status(order_id)
    
    def get_trading_calendar(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Lädt Handelskalender
        
        Args:
            start_date: Startdatum (optional)
            end_date: Enddatum (optional)
            
        Returns:
            DataFrame mit Handelstagen
        """
        try:
            if not start_date:
                start_date = datetime.now().strftime('%Y-%m-%d')
            if not end_date:
                end_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
                
            calendar = self.api.get_calendar(start=start_date, end=end_date)
            
            if not calendar:
                return pd.DataFrame()
                
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
            self.logger.error(f"Fehler beim Laden des Kalenders: {e}")
            return pd.DataFrame()
    
    def get_clock(self) -> Dict[str, Any]:
        """
        Lädt Markt-Clock Informationen
        
        Returns:
            Dict mit Clock-Informationen
        """
        try:
            clock = self.api.get_clock()
            
            return {
                'timestamp': clock.timestamp,
                'is_open': clock.is_open,
                'next_open': clock.next_open,
                'next_close': clock.next_close
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Clock: {e}")
            return {}
    
    def stream_trades(self, symbols: List[str], callback):
        """
        Startet Trade-Stream (für Live-Daten)
        
        Args:
            symbols: Liste der zu streamenden Symbole
            callback: Callback-Funktion für Trade-Updates
        """
        try:
            # Alpaca Streaming würde hier implementiert werden
            # Dies ist ein komplexeres Feature für Live-Trading
            self.logger.info(f"Trade-Streaming für {symbols} würde hier gestartet")
            pass
            
        except Exception as e:
            self.logger.error(f"Fehler beim Trade-Streaming: {e}")
    
    def get_asset_info(self, symbol: str) -> Dict[str, Any]:
        """
        Lädt Asset-Informationen
        
        Args:
            symbol: Trading-Symbol
            
        Returns:
            Dict mit Asset-Informationen
        """
        try:
            asset = self.api.get_asset(symbol)
            
            return {
                'symbol': asset.symbol,
                'name': asset.name,
                'exchange': asset.exchange,
                'asset_class': asset.asset_class,
                'status': asset.status,
                'tradable': asset.tradable,
                'marginable': asset.marginable,
                'shortable': asset.shortable,
                'easy_to_borrow': asset.easy_to_borrow,
                'fractionable': asset.fractionable
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Asset-Info: {e}")
            return {}
    
    def calculate_order_quantity(self, symbol: str, capital: float, 
                               position_size: float, current_price: float) -> float:
        """
        Berechnet Order-Menge unter Berücksichtigung von Alpaca-Regeln
        
        Args:
            symbol: Trading-Symbol
            capital: Verfügbares Kapital
            position_size: Gewünschte Positionsgröße (0-1)
            current_price: Aktueller Preis
            
        Returns:
            Anzahl Aktien/Anteile
        """
        if current_price <= 0 or capital <= 0 or position_size <= 0:
            return 0
            
        # Basis-Berechnung
        position_capital = capital * position_size
        quantity = position_capital / current_price
        
        # Alpaca-spezifische Regeln
        try:
            asset_info = self.get_asset_info(symbol)
            
            # Fractional Shares prüfen
            if asset_info.get('fractionable', False) and self.config.get('fractional_shares', False):
                # Runde auf 6 Dezimalstellen (Alpaca Maximum)
                quantity = round(quantity, 6)
            else:
                # Nur ganze Aktien
                quantity = int(quantity)
            
            # Minimum Order Value (meist $1)
            min_order_value = 1.0
            if quantity * current_price < min_order_value:
                return 0
                
        except Exception as e:
            self.logger.warning(f"Konnte Asset-Info nicht laden: {e}")
            # Fallback: Ganze Aktien
            quantity = int(quantity)
        
        return quantity
    
    def get_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über das Alpaca Execution System zurück
        
        Returns:
            Dict mit System-Informationen
        """
        info = super().get_info()
        
        try:
            account = self.get_account_info()
            clock = self.get_clock()
            
            info.update({
                'provider_type': 'Alpaca Markets',
                'base_url': self.base_url,
                'account_id': account.get('id'),
                'account_status': account.get('status'),
                'pattern_day_trader': account.get('pattern_day_trader'),
                'trading_blocked': account.get('trading_blocked'),
                'market_open': clock.get('is_open', False),
                'next_market_open': clock.get('next_open'),
                'next_market_close': clock.get('next_close'),
                'supported_order_types': list(self.order_type_mapping.keys()),
                'supported_time_in_force': list(self.tif_mapping.keys()),
                'fractional_shares': self.config.get('fractional_shares', False)
            })
            
        except Exception as e:
            self.logger.warning(f"Konnte System-Informationen nicht vollständig laden: {e}")
            info.update({
                'provider_type': 'Alpaca Markets',
                'base_url': self.base_url,
                'error': str(e)
            })
        
        return info