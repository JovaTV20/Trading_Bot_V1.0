# 🤖 TradingBot für Alpaca Markets

Ein professioneller Trading-Bot mit Machine Learning-Strategien, Web-Dashboard und umfassenden Risk-Management-Features.

## 🌟 Features

### Core Features
- ✅ **ML-basierte Trading-Strategien** (Random Forest, Logistic Regression)
- ✅ **Live-Trading & Backtesting** mit historischen Daten
- ✅ **Web-Dashboard** für Monitoring und Kontrolle
- ✅ **Alpaca Markets Integration** (Paper & Live Trading)
- ✅ **Email-Alerts** für Trades und Fehler
- ✅ **Risk-Management** mit Stop-Loss/Take-Profit
- ✅ **Umfangreiches Logging** und Performance-Tracking

### Advanced Features
- 📊 **Real-time Portfolio-Monitoring**
- 📈 **Interactive Charts** mit Plotly
- 🔧 **Parameter-Optimierung** mit GridSearch
- 📧 **Email-Benachrichtigungen** bei wichtigen Events
- 🐳 **Docker-Support** für einfache Deployment
- 🛡️ **Umfassende Validierung** aller Eingaben
- 📝 **Detaillierte Performance-Metriken**

## 🚀 Quick Start

### 1. Repository klonen
```bash
git clone <repository-url>
cd TradingBot
```

### 2. Virtual Environment erstellen
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Dependencies installieren
```bash
pip install -r requirements.txt
```

### 4. Environment konfigurieren
```bash
# .env Datei erstellen
cp .env.template .env

# Bearbeite .env mit deinen Credentials:
# - ALPACA_API_KEY
# - ALPACA_SECRET_KEY  
# - EMAIL_ADDRESS (optional)
# - EMAIL_PASSWORD (optional)
```

### 5. Setup validieren
```bash
python run_validation.py
```

### 6. Ersten Backtest ausführen
```bash
python main.py --mode backtest --symbol AAPL --start-date 2023-01-01 --end-date 2024-01-01
```

### 7. Dashboard starten
```bash
cd dashboard
python app.py
```
Dashboard ist verfügbar unter: http://127.0.0.1:5000

## 📁 Projektstruktur

```
TradingBot/
├── 📁 core/                    # Kern-Module
│   ├── __init__.py
│   ├── backtester.py          # Backtesting-Engine
│   ├── base_data.py           # Data Provider Interface
│   ├── base_execution.py      # Execution Interface
│   ├── base_strategy.py       # Strategy Interface
│   ├── controller.py          # Haupt-Controller
│   └── logger.py              # Logging-System
│
├── 📁 data_providers/          # Datenquellen
│   ├── __init__.py
│   └── alpaca_data.py         # Alpaca Markets API
│
├── 📁 execution/               # Order-Ausführung
│   ├── __init__.py
│   └── alpaca_exec.py         # Alpaca Trading API
│
├── 📁 strategies/              # Trading-Strategien
│   ├── __init__.py
│   └── ml_strategy.py         # ML-basierte Strategie
│
├── 📁 utils/                   # Hilfs-Module
│   ├── __init__.py
│   ├── email_alerts.py        # Email-System
│   └── validator.py           # Validierungen
│
├── 📁 dashboard/               # Web-Interface
│   ├── app.py                 # Flask-App
│   ├── templates/             # HTML-Templates
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── backtest.html
│   │   ├── trades.html
│   │   ├── settings.html
│   │   └── error.html
│   └── static/                # CSS/JS/Assets
│
├── 📁 logs/                    # Log-Dateien (auto-erstellt)
├── 📁 data/                    # Daten-Cache (auto-erstellt)
├── 📁 models/                  # ML-Modelle (auto-erstellt)
│
├── main.py                     # Haupt-Entry-Point
├── run_validation.py           # Setup-Validierung
├── config.json                 # Konfiguration
├── requirements.txt            # Python-Dependencies
├── .env.template               # Environment-Template
├── Dockerfile                  # Docker-Build
├── docker-compose.yml          # Docker-Compose
├── Makefile                    # Build-Automation
└── README.md                   # Diese Datei
```

## ⚙️ Konfiguration

### Environment Variables (.env)
```bash
# Alpaca API (ERFORDERLICH)
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper Trading

# Email-Alerts (OPTIONAL)
EMAIL_ADDRESS=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
ALERT_RECIPIENTS=recipient@gmail.com

# System
LOG_LEVEL=INFO
```

### Hauptkonfiguration (config.json)
```json
{
  "strategy": {
    "name": "ml_strategy",
    "parameters": {
      "lookback_period": 20,
      "prediction_threshold": 0.6
    }
  },
  "risk_management": {
    "max_position_size": 0.1,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.04
  }
}
```

## 🎯 Verwendung

### Backtesting
```bash
# Einfacher Backtest
python main.py --mode backtest --symbol AAPL

# Custom Backtest
python main.py --mode backtest \
  --symbol TSLA \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --capital 50000
```

### Live Trading
```bash
# Live Trading starten (Paper Mode empfohlen!)
python main.py --mode live --symbol AAPL
```

### Dashboard verwenden
```bash
cd dashboard
python app.py

# Browser öffnen: http://127.0.0.1:5000
```

### Makefile Commands
```bash
make setup          # Komplette Ersteinrichtung
make validate       # Setup validieren  
make backtest       # Standard-Backtest
make live           # Live-Trading starten
make dashboard      # Dashboard starten
make clean          # Aufräumen
make logs           # Logs anzeigen
```

## 🧠 ML-Strategie Details

### Features
Die ML-Strategie verwendet folgende technische Indikatoren:
- **Returns**: 1-Day, 5-Day, 10-Day Returns
- **Moving Averages**: SMA/EMA 5, 10, 20 Perioden
- **RSI**: Relative Strength Index
- **Bollinger Bands**: Oberes/Unteres Band, Position
- **Momentum**: Rate of Change, Momentum-Indikatoren
- **Volume**: Volume-Ratios, On Balance Volume

### Modelle
- **Random Forest Classifier** (Standard)
- **Logistic Regression** (Alternative)
- Automatische Hyperparameter-Optimierung
- Cross-Validation für Modell-Bewertung

### Signale
- **Buy Signal**: Hohe Kaufwahrscheinlichkeit (>60%)
- **Sell Signal**: Hohe Verkaufswahrscheinlichkeit (>60%) 
- **Hold**: Unsichere Vorhersagen

## 🛡️ Risk Management

### Position Sizing
- Maximum 10% des Portfolios pro Position (konfigurierbar)
- Confidence-basierte Positionsgrößen-Anpassung

### Stop-Loss/Take-Profit
- Automatische Stop-Loss Orders (Standard: 2%)
- Take-Profit Levels (Standard: 4%)
- Bracket-Orders für bessere Ausführung

### Limits
- Maximum Trades pro Tag (Standard: 5)
- Maximaler Drawdown-Schutz
- Position-Größen-Limits

## 📊 Monitoring & Alerts

### Dashboard Features
- **Real-time Portfolio-Übersicht**
- **Trade-Historie und -Analyse** 
- **Performance-Charts**
- **System-Status Monitoring**
- **Backtest-Tool**
- **Einstellungen-Panel**

### Email-Alerts
- Trade-Ausführungen
- System-Fehler
- Tägliche Performance-Zusammenfassung
- Kritische Events

### Logging
- Strukturierte Logs (JSON-Format)
- Separate Log-Files für Trades, Errors, Performance
- Automatische Log-Rotation
- Performance-Metriken-Tracking

## 🐳 Docker Deployment

### Einfacher Start
```bash
# Build
make docker-build

# Run
make docker-run
```

### Docker Compose (mit Services)
```bash
# Alle Services starten
docker-compose up -d

# Services:
# - tradingbot (Haupt-Bot)
# - dashboard (Web-Interface)
# - postgres (Datenbank) 
# - redis (Cache)
# - grafana (Monitoring)
```

## 🔧 Development

### Setup für Entwicklung
```bash
# Development-Dependencies
pip install -r requirements.txt
pip install black flake8 pytest

# Code formatieren
black .

# Linting
flake8 .

# Tests ausführen
pytest
```

### Neue Strategie hinzufügen
1. Erstelle `strategies/your_strategy.py`
2. Erbe von `StrategyBase`
3. Implementiere `fit()` und `generate_signal()`
4. Registriere in `strategies/__init__.py`

### Neuer Data Provider
1. Erstelle `data_providers/your_provider.py`
2. Erbe von `DataProviderBase`
3. Implementiere required methods
4. Registriere in `data_providers/__init__.py`

## 🚨 Troubleshooting

### Häufige Probleme

**1. "Alpaca API credentials not found"**
```bash
# Lösung: .env Datei erstellen und API-Keys eintragen
cp .env.template .env
# Bearbeite .env mit echten Credentials
```

**2. "Module not found errors"**
```bash
# Lösung: Virtual Environment aktivieren
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Dependencies neu installieren
pip install -r requirements.txt
```

**3. "Dashboard nicht erreichbar"**
```bash
# Lösung: Port prüfen und Flask-App starten
cd dashboard
python app.py
# Browser: http://127.0.0.1:5000
```

**4. "No data received"**
```bash
# Lösung: Marktzeiten und Symbol prüfen
# Verwende gültige US-Aktien-Symbole
# Prüfe ob Markt geöffnet (US Eastern Time)
```

### Log-Files prüfen
```bash
# Alle Logs anzeigen
make logs

# Spezifische Logs
tail -f logs/bot.log      # Haupt-Log
tail -f logs/errors.log   # Error-Log  
tail -f logs/trades.log   # Trade-Log
```

### Setup validieren
```bash
python run_validation.py
```

## ⚠️ Wichtige Hinweise

### Sicherheit
- **Verwende IMMER Paper Trading** für Tests
- Halte API-Keys geheim (nie in Git committen)
- Teste Strategien ausführlich vor Live-Trading
- Verwende angemessene Positionsgrößen

### Performance
- Backtests sind keine Garantie für zukünftige Performance
- Berücksichtige Slippage und Kommissionen
- Market-Timing ist extrem schwierig
- Diversifizierung ist wichtig

### Rechtliches
- Trading-Bots unterliegen lokalen Gesetzen
- Keine Anlageberatung - nutze auf eigene Verantwortung
- Verstehe die Risiken des algorithmischen Tradings
- Befolge die Terms of Service deines Brokers

## 📞 Support

### Dokumentation
- Vollständige API-Docs in den Docstrings
- Beispiele in den einzelnen Modulen
- Dashboard Help-Sektion

### Community
- Issues auf GitHub erstellen
- Diskussionen in GitHub Discussions
- Code-Reviews willkommen

### Professional Support
Für professionelle Unterstützung und Custom-Entwicklung kontaktiere das Entwicklungsteam.

---

**⚠️ Disclaimer**: Dieses Tool dient nur zu Bildungszwecken. Trading birgt erhebliche Risiken. Verwende nur Geld, das du dir leisten kannst zu verlieren. Keine Anlageberatung.

---

<div align="center">
Made with ❤️ for the Trading Community
</div>