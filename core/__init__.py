# __init__.py für core/
"""
TradingBot Core Module
Zentrale Komponenten für Trading-Bot Funktionalität
"""

__version__ = "1.0.0"

from .base_strategy import StrategyBase
from .base_data import DataProviderBase
from .base_execution import ExecutionBase
from .controller import TradingController
from .backtester import Backtester
from .logger import setup_logger, get_logger

__all__ = [
    'StrategyBase',
    'DataProviderBase', 
    'ExecutionBase',
    'TradingController',
    'Backtester',
    'setup_logger',
    'get_logger'
]