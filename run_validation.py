#!/usr/bin/env python3
"""
Setup Validation Script für TradingBot
Prüft alle Komponenten vor dem ersten Start
"""

import sys
from pathlib import Path
import json

# Füge Projekt-Root zum Python-Path hinzu
sys.path.insert(0, str(Path(__file__).parent))

from utils.validator import validate_trading_setup, print_validation_report

def main():
    """Führt komplette Setup-Validierung aus"""
    print("🔍 TradingBot Setup-Validierung wird gestartet...\n")
    
    try:
        # Vollständige Validierung
        result = validate_trading_setup()
        
        # Detaillierter Report
        print_validation_report(result)
        
        # Empfehlungen
        if result['valid']:
            print("\n✅ SETUP ERFOLGREICH!")
            print("Empfohlene nächste Schritte:")
            print("1. Teste mit Paper Trading: python main.py --mode backtest --symbol AAPL")
            print("2. Starte Dashboard: cd dashboard && python app.py")
            print("3. Erste Live-Tests: python main.py --mode live --symbol AAPL")
        else:
            print("\n❌ SETUP UNVOLLSTÄNDIG!")
            print("\nBitte behebe die oben genannten Fehler vor dem Start.")
            print("\nHäufige Lösungen:")
            print("• Kopiere .env.template zu .env und fülle Credentials aus")
            print("• Installiere fehlende Pakete: pip install -r requirements.txt")
            print("• Erstelle Alpaca Account auf https://alpaca.markets")
            
        return 0 if result['valid'] else 1
        
    except Exception as e:
        print(f"\n💥 Validierung fehlgeschlagen: {e}")
        print("Bitte prüfe die Installation und versuche es erneut.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)