"""
Korrigierte Environment-Validierung für .env Datei
"""

import os
from pathlib import Path
from dotenv import load_dotenv

def validate_alpaca_credentials() -> dict:
    """Validiert Alpaca API Credentials"""
    result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # WICHTIG: Lade .env Datei explizit
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ .env Datei gefunden und geladen: {env_path.absolute()}")
    else:
        result['valid'] = False
        result['errors'].append(".env Datei nicht gefunden")
        return result
    
    # API Key prüfen
    api_key = os.getenv('ALPACA_API_KEY')
    if not api_key:
        result['valid'] = False
        result['errors'].append("ALPACA_API_KEY nicht gesetzt")
    elif len(api_key.strip()) < 20:  # Alpaca Keys sind länger
        result['warnings'].append("ALPACA_API_KEY scheint sehr kurz")
    else:
        print(f"✅ ALPACA_API_KEY gefunden: {api_key[:8]}...")
    
    # Secret Key prüfen
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    if not secret_key:
        result['valid'] = False
        result['errors'].append("ALPACA_SECRET_KEY nicht gesetzt")
    elif len(secret_key.strip()) < 40:  # Alpaca Secret Keys sind länger
        result['warnings'].append("ALPACA_SECRET_KEY scheint sehr kurz")
    else:
        print(f"✅ ALPACA_SECRET_KEY gefunden: {secret_key[:8]}...")
    
    # Base URL prüfen
    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    if 'paper-api' in base_url:
        result['warnings'].append("Paper Trading aktiviert (empfohlen für Tests)")
        print(f"✅ Paper Trading URL: {base_url}")
    else:
        print(f"⚠️  Live Trading URL: {base_url}")
    
    return result

def quick_test_env():
    """Schneller Test der .env Konfiguration"""
    print("\n🔍 ENVIRONMENT-VARIABLEN TEST:")
    print("="*50)
    
    # Lade .env
    env_path = Path('.env')
    if not env_path.exists():
        print("❌ .env Datei nicht gefunden!")
        return False
    
    load_dotenv(env_path)
    
    # Alle relevanten Variablen prüfen
    vars_to_check = [
        'ALPACA_API_KEY',
        'ALPACA_SECRET_KEY', 
        'ALPACA_BASE_URL',
        'EMAIL_ADDRESS',
        'EMAIL_PASSWORD',
        'ALERT_RECIPIENTS'
    ]
    
    all_good = True
    for var in vars_to_check:
        value = os.getenv(var)
        if value:
            # Sensible Daten nur teilweise anzeigen
            if 'KEY' in var or 'PASSWORD' in var:
                display_value = f"{value[:8]}..." if len(value) > 8 else "***"
            else:
                display_value = value
            print(f"✅ {var}: {display_value}")
        else:
            if var in ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY']:
                print(f"❌ {var}: NICHT GESETZT (ERFORDERLICH)")
                all_good = False
            else:
                print(f"⚠️  {var}: NICHT GESETZT (optional)")
    
    print("="*50)
    return all_good

if __name__ == "__main__":
    # Direkter Test
    success = quick_test_env()
    
    if success:
        print("\n✅ Alle erforderlichen Environment-Variablen sind gesetzt!")
        
        # Teste Alpaca-Verbindung
        print("\n🔌 TESTE ALPACA-VERBINDUNG:")
        try:
            import alpaca_trade_api as tradeapi
            
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
            
            api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
            account = api.get_account()
            
            print(f"✅ Alpaca-Verbindung erfolgreich!")
            print(f"   Account Status: {account.status}")
            print(f"   Trading Blocked: {account.trading_blocked}")
            print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
            
        except Exception as e:
            print(f"❌ Alpaca-Verbindung fehlgeschlagen: {e}")
            
    else:
        print("\n❌ Einige erforderliche Environment-Variablen fehlen!")