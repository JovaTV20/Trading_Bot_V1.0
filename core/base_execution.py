"""
Base Execution Class
Definiert die Schnittstelle für alle Order-Execution-Systeme
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional, List
from enum import Enum
import logging
from datetime import datetime

class OrderType(Enum):
    """Order-Typen"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order-Seiten"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order-Status"""
    NEW = "new"
    PENDING = "pending"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"

class TimeInForce(Enum):
    """Time-in-Force"""
    DAY = "day"
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill

class ExecutionBase(ABC):
    """
    Basisklasse für alle Order-Execution-Systeme
    
    Jedes System muss place_order implementieren
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialisiert das Execution-System
        
        Args:
            config: Konfigurationsdictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.orders_history: List[Dict[str, Any]] = []
        
    @abstractmethod
    def place_order(self, symbol: str, side: str, quantity: float, 
                   order_type: str = "market", price: float = None,
                   stop_price: float = None, time_in_force: str = "day") -> Dict[str, Any]:
        """
        Platziert eine Order
        
        Args:
            symbol: Trading-Symbol
            side: 'buy' oder 'sell'
            quantity: Anzahl Aktien/Anteile
            order_type: Order-Typ ('market', 'limit', 'stop', 'stop_limit')
            price: Limit-Preis (bei Limit-Orders)
            stop_price: Stop-Preis (bei Stop-Orders)
            time_in_force: Gültigkeitsdauer ('day', 'gtc', 'ioc', 'fok')
            
        Returns:
            Dict mit Order-Informationen:
            {
                'order_id': str,
                'status': str,
                'symbol': str,
                'side': str,
                'quantity': float,
                'filled_quantity': float,
                'price': float,
                'timestamp': datetime
            }
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Fragt Order-Status ab
        
        Args:
            order_id: Order-ID
            
        Returns:
            Dict mit Order-Status-Informationen
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Storniert eine Order
        
        Args:
            order_id: Order-ID
            
        Returns:
            True wenn erfolgreich storniert
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Lädt alle offenen Positionen
        
        Returns:
            Liste mit Position-Informationen
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """
        Lädt Account-Informationen
        
        Returns:
            Dict mit Account-Daten (Kapital, Kaufkraft, etc.)
        """
        pass
    
    def validate_order(self, symbol: str, side: str, quantity: float, 
                      order_type: str = "market", price: float = None) -> bool:
        """
        Validiert Order-Parameter
        
        Args:
            symbol: Trading-Symbol
            side: Order-Seite
            quantity: Menge
            order_type: Order-Typ
            price: Preis (optional)
            
        Returns:
            True wenn Order valide ist
        """
        # Symbol validieren
        if not symbol or not isinstance(symbol, str):
            self.logger.error("Ungültiges Symbol")
            return False
            
        # Side validieren
        if side not in ['buy', 'sell']:
            self.logger.error("Side muss 'buy' oder 'sell' sein")
            return False
            
        # Quantity validieren
        if quantity <= 0:
            self.logger.error("Quantity muss größer als 0 sein")
            return False
            
        # Order-Typ validieren
        valid_types = ['market', 'limit', 'stop', 'stop_limit']
        if order_type not in valid_types:
            self.logger.error(f"Order-Typ muss einer von {valid_types} sein")
            return False
            
        # Preis bei Limit-Orders validieren
        if order_type in ['limit', 'stop_limit'] and (price is None or price <= 0):
            self.logger.error("Limit-Orders benötigen einen gültigen Preis")
            return False
            
        return True
    
    def calculate_order_quantity(self, symbol: str, capital: float, 
                               position_size: float, current_price: float) -> float:
        """
        Berechnet Order-Menge basierend auf verfügbarem Kapital
        
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
            
        # Berechne verfügbares Kapital für Position
        position_capital = capital * position_size
        
        # Berechne Anzahl Aktien
        quantity = position_capital / current_price
        
        # Prüfe auf Fractional Shares
        fractional_enabled = self.config.get('execution', {}).get('fractional_shares', False)
        if not fractional_enabled:
            quantity = int(quantity)  # Runde auf ganze Aktien ab
            
        return quantity
    
    def calculate_risk_metrics(self, entry_price: float, stop_loss: float = None,
                             take_profit: float = None) -> Dict[str, float]:
        """
        Berechnet Risk/Reward Metriken
        
        Args:
            entry_price: Einstiegspreis
            stop_loss: Stop-Loss Preis (optional)
            take_profit: Take-Profit Preis (optional)
            
        Returns:
            Dict mit Risk-Metriken
        """
        metrics = {}
        
        if stop_loss:
            risk = abs(entry_price - stop_loss) / entry_price
            metrics['risk_pct'] = risk
            
        if take_profit:
            reward = abs(take_profit - entry_price) / entry_price
            metrics['reward_pct'] = reward
            
        if stop_loss and take_profit:
            if metrics['risk_pct'] > 0:
                metrics['risk_reward_ratio'] = metrics['reward_pct'] / metrics['risk_pct']
            else:
                metrics['risk_reward_ratio'] = float('inf')
                
        return metrics
    
    def create_bracket_order(self, symbol: str, side: str, quantity: float,
                           stop_loss: float = None, take_profit: float = None) -> List[Dict[str, Any]]:
        """
        Erstellt Bracket-Order (Hauptorder + Stop-Loss + Take-Profit)
        
        Args:
            symbol: Trading-Symbol
            side: Order-Seite
            quantity: Menge
            stop_loss: Stop-Loss Preis
            take_profit: Take-Profit Preis
            
        Returns:
            Liste mit Order-Dictionaries
        """
        orders = []
        
        # Hauptorder (Market)
        main_order = self.place_order(symbol, side, quantity, "market")
        orders.append(main_order)
        
        if main_order.get('status') in ['filled', 'pending']:
            # Stop-Loss Order
            if stop_loss:
                sl_side = 'sell' if side == 'buy' else 'buy'
                sl_order = self.place_order(symbol, sl_side, quantity, "stop", stop_price=stop_loss)
                orders.append(sl_order)
                
            # Take-Profit Order  
            if take_profit:
                tp_side = 'sell' if side == 'buy' else 'buy'
                tp_order = self.place_order(symbol, tp_side, quantity, "limit", price=take_profit)
                orders.append(tp_order)
        
        return orders
    
    def get_order_history(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Lädt Order-Historie
        
        Args:
            symbol: Filter nach Symbol (optional)
            limit: Maximale Anzahl Orders
            
        Returns:
            Liste mit historischen Orders
        """
        orders = self.orders_history.copy()
        
        if symbol:
            orders = [o for o in orders if o.get('symbol') == symbol]
            
        # Sortiere nach Timestamp (neueste zuerst)
        orders.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
        
        return orders[:limit]
    
    def get_daily_pnl(self) -> float:
        """
        Berechnet täglichen P&L
        
        Returns:
            Täglicher Profit/Loss
        """
        # Wird von Subklassen implementiert
        return 0.0
    
    def get_position_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Lädt Position für ein bestimmtes Symbol
        
        Args:
            symbol: Trading-Symbol
            
        Returns:
            Position-Dict oder None
        """
        positions = self.get_positions()
        for pos in positions:
            if pos.get('symbol') == symbol:
                return pos
        return None
    
    def close_position(self, symbol: str) -> Dict[str, Any]:
        """
        Schließt komplette Position für ein Symbol
        
        Args:
            symbol: Trading-Symbol
            
        Returns:
            Order-Informationen
        """
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
    
    def get_buying_power(self) -> float:
        """
        Lädt verfügbare Kaufkraft
        
        Returns:
            Verfügbare Kaufkraft
        """
        account = self.get_account_info()
        return float(account.get('buying_power', 0))
    
    def log_order(self, order_info: Dict[str, Any]) -> None:
        """
        Loggt Order-Informationen
        
        Args:
            order_info: Order-Dictionary
        """
        self.orders_history.append(order_info)
        
        # Begrenze Historie-Größe
        max_history = 1000
        if len(self.orders_history) > max_history:
            self.orders_history = self.orders_history[-max_history:]
            
        # Log Order
        self.logger.info(f"Order: {order_info.get('side')} {order_info.get('quantity')} "
                        f"{order_info.get('symbol')} @ {order_info.get('price', 'MARKET')}")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über das Execution-System zurück
        
        Returns:
            Dict mit System-Informationen
        """
        return {
            'provider': self.__class__.__name__,
            'config': self.config,
            'orders_count': len(self.orders_history)
        }