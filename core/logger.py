"""
Zentrales Logging-System für den TradingBot
"""

import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = 'TradingBot', level: str = 'INFO', 
                log_to_file: bool = True, log_to_console: bool = True) -> logging.Logger:
    """
    Konfiguriert den zentralen Logger
    
    Args:
        name: Logger-Name
        level: Log-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: In Datei loggen
        log_to_console: In Konsole loggen
        
    Returns:
        Konfigurierter Logger
    """
    
    # Erstelle Logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Verhindere doppelte Handler
    if logger.handlers:
        logger.handlers.clear()
    
    # Formatter definieren
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File Handler
    if log_to_file:
        # Erstelle logs-Verzeichnis falls nicht vorhanden
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Rotating File Handler (max 10MB pro Datei, 5 Backup-Dateien)
        log_file = log_dir / 'bot.log'
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Separater Handler für Errors
        error_file = log_dir / 'errors.log'
        error_handler = logging.handlers.RotatingFileHandler(
            filename=error_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
        
        # Trade-spezifischer Handler
        trade_file = log_dir / 'trades.log'
        trade_handler = logging.handlers.RotatingFileHandler(
            filename=trade_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=10,
            encoding='utf-8'
        )
        trade_handler.setLevel(logging.INFO)
        trade_handler.setFormatter(detailed_formatter)
        
        # Filter für Trade-Logs
        class TradeFilter(logging.Filter):
            def filter(self, record):
                return 'trade' in record.getMessage().lower() or 'order' in record.getMessage().lower()
        
        trade_handler.addFilter(TradeFilter())
        logger.addHandler(trade_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Gibt einen Logger für ein bestimmtes Modul zurück
    
    Args:
        name: Modul-Name
        
    Returns:
        Logger-Instanz
    """
    return logging.getLogger(f'TradingBot.{name}')

class PerformanceLogger:
    """
    Spezielle Logger-Klasse für Performance-Metriken
    """
    
    def __init__(self):
        self.logger = get_logger('Performance')
        
        # Performance Log-Datei
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        perf_file = log_dir / 'performance.log'
        handler = logging.handlers.RotatingFileHandler(
            filename=perf_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding='utf-8'
        )
        
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Separater Logger für Performance
        self.perf_logger = logging.getLogger('Performance_Metrics')
        self.perf_logger.setLevel(logging.INFO)
        self.perf_logger.addHandler(handler)
    
    def log_trade(self, symbol: str, side: str, quantity: float, price: float, 
                  pnl: float = None, strategy: str = None):
        """
        Loggt Trade-Informationen
        
        Args:
            symbol: Trading-Symbol
            side: buy/sell
            quantity: Menge
            price: Preis
            pnl: Profit/Loss (optional)
            strategy: Strategie-Name (optional)
        """
        trade_info = {
            'type': 'TRADE',
            'symbol': symbol,
            'side': side.upper(),
            'quantity': quantity,
            'price': price,
            'value': quantity * price,
            'pnl': pnl,
            'strategy': strategy or 'unknown'
        }
        
        message = (f"TRADE | {symbol} | {side.upper()} {quantity} @ {price:.2f} "
                  f"| Value: ${quantity * price:.2f}")
        
        if pnl is not None:
            message += f" | PnL: ${pnl:.2f}"
            
        if strategy:
            message += f" | Strategy: {strategy}"
        
        self.perf_logger.info(message)
    
    def log_portfolio_update(self, equity: float, cash: float, positions_value: float,
                           daily_pnl: float = None, total_return: float = None):
        """
        Loggt Portfolio-Update
        
        Args:
            equity: Gesamtequity
            cash: Verfügbares Cash
            positions_value: Wert der Positionen
            daily_pnl: Täglicher P&L
            total_return: Gesamtrendite
        """
        message = (f"PORTFOLIO | Equity: ${equity:.2f} | Cash: ${cash:.2f} "
                  f"| Positions: ${positions_value:.2f}")
        
        if daily_pnl is not None:
            message += f" | Daily PnL: ${daily_pnl:.2f}"
            
        if total_return is not None:
            message += f" | Total Return: {total_return:.2%}"
        
        self.perf_logger.info(message)
    
    def log_strategy_signal(self, symbol: str, action: str, confidence: float,
                          price: float, indicators: dict = None):
        """
        Loggt Strategie-Signal
        
        Args:
            symbol: Trading-Symbol
            action: buy/sell/hold
            confidence: Confidence-Level
            price: Aktueller Preis
            indicators: Technische Indikatoren (optional)
        """
        message = (f"SIGNAL | {symbol} | {action.upper()} | "
                  f"Confidence: {confidence:.2%} | Price: ${price:.2f}")
        
        if indicators:
            indicators_str = " | ".join([f"{k}: {v:.2f}" if isinstance(v, float) 
                                       else f"{k}: {v}" for k, v in indicators.items()])
            message += f" | {indicators_str}"
        
        self.perf_logger.info(message)

class EmailLogHandler(logging.Handler):
    """
    Custom Log Handler der kritische Errors per Email versendet
    """
    
    def __init__(self, email_alerter=None):
        super().__init__(level=logging.CRITICAL)
        self.email_alerter = email_alerter
        
    def emit(self, record):
        """
        Versendet Log-Record per Email
        
        Args:
            record: Log-Record
        """
        if not self.email_alerter:
            return
            
        try:
            subject = f"CRITICAL ERROR - TradingBot"
            message = self.format(record)
            
            # Füge Traceback hinzu falls vorhanden
            if record.exc_info:
                import traceback
                message += "\n\nTraceback:\n" + "".join(traceback.format_exception(*record.exc_info))
            
            self.email_alerter.send_error_alert("Critical Log Error", message)
            
        except Exception:
            # Verhindere Endlos-Schleifen bei Email-Fehlern
            pass

def setup_email_logging(email_alerter):
    """
    Aktiviert Email-Logging für kritische Fehler
    
    Args:
        email_alerter: EmailAlerter-Instanz
    """
    root_logger = logging.getLogger('TradingBot')
    email_handler = EmailLogHandler(email_alerter)
    
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    email_handler.setFormatter(formatter)
    
    root_logger.addHandler(email_handler)

def cleanup_old_logs(days: int = 30):
    """
    Löscht alte Log-Dateien
    
    Args:
        days: Anzahl Tage nach denen Logs gelöscht werden
    """
    log_dir = Path('logs')
    if not log_dir.exists():
        return
        
    cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
    deleted_count = 0
    
    for log_file in log_dir.glob('*.log*'):
        if log_file.stat().st_mtime < cutoff_time:
            try:
                log_file.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"Fehler beim Löschen von {log_file}: {e}")
    
    if deleted_count > 0:
        logger = get_logger('Cleanup')
        logger.info(f"{deleted_count} alte Log-Dateien gelöscht")

# Automatisches Setup beim Import
if not os.environ.get('DISABLE_AUTO_LOGGING'):
    setup_logger()