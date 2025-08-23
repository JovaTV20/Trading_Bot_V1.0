#!/usr/bin/env python3
"""
TradingBot - Main Entry Point
Startet den Bot im Live- oder Backtest-Modus
"""

import sys
import argparse
import json
from pathlib import Path
from core.controller import TradingController
from core.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description='TradingBot für Alpaca')
    parser.add_argument('--mode', choices=['live', 'backtest'], default='backtest',
                       help='Betriebsmodus: live oder backtest')
    parser.add_argument('--symbol', default='AAPL',
                       help='Trading Symbol (default: AAPL)')
    parser.add_argument('--start-date', default='2023-01-01',
                       help='Startdatum für Backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-01-01',
                       help='Enddatum für Backtest (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=10000.0,
                       help='Startkapital für Backtest')
    parser.add_argument('--config', default='config.json',
                       help='Pfad zur Konfigurationsdatei')
    
    args = parser.parse_args()
    
    # Logger initialisieren
    logger = setup_logger()
    logger.info(f"TradingBot gestartet - Modus: {args.mode}")
    
    try:
        # Konfiguration laden
        if not Path(args.config).exists():
            logger.error(f"Konfigurationsdatei {args.config} nicht gefunden!")
            sys.exit(1)
            
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Controller initialisieren
        controller = TradingController(config)
        
        if args.mode == 'backtest':
            # Backtest ausführen
            logger.info(f"Starte Backtest für {args.symbol} ({args.start_date} - {args.end_date})")
            results = controller.run_backtest(
                symbol=args.symbol,
                start_date=args.start_date,
                end_date=args.end_date,
                initial_capital=args.capital
            )
            
            # Ergebnisse anzeigen
            print("\n" + "="*50)
            print("BACKTEST ERGEBNISSE")
            print("="*50)
            print(f"Symbol: {args.symbol}")
            print(f"Zeitraum: {args.start_date} - {args.end_date}")
            print(f"Startkapital: ${args.capital:,.2f}")
            print(f"Endkapital: ${results['final_capital']:,.2f}")
            print(f"Rendite: {results['total_return']:.2%}")
            print(f"Anzahl Trades: {results['total_trades']}")
            print(f"Gewinnrate: {results['win_rate']:.2%}")
            print(f"Max Drawdown: {results['max_drawdown']:.2%}")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            
        else:
            # Live-Trading starten
            logger.info(f"Starte Live-Trading für {args.symbol}")
            controller.run_live(args.symbol)
            
    except KeyboardInterrupt:
        logger.info("Bot durch Benutzer gestoppt")
    except Exception as e:
        logger.error(f"Unerwarteter Fehler: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()