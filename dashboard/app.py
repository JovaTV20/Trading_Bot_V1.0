"""
TradingBot Dashboard - Flask Web Interface
Zeigt Live-Performance, Trades und System-Status
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

# Füge das Projekt-Root-Verzeichnis zum Python-Path hinzu
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, render_template, jsonify, request, send_from_directory
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

from core.controller import TradingController
from core.logger import setup_logger, get_logger

# Flask App initialisieren
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = os.urandom(24)

# Logger
logger = get_logger('Dashboard')

# Globale Variablen
controller = None
config = {}

def load_config():
    """Lädt Konfiguration"""
    global config
    try:
        config_path = Path(__file__).parent.parent / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        return True
    except Exception as e:
        logger.error(f"Fehler beim Laden der Konfiguration: {e}")
        return False

def initialize_controller():
    """Initialisiert TradingController"""
    global controller
    try:
        if load_config():
            controller = TradingController(config)
            return True
        return False
    except Exception as e:
        logger.error(f"Fehler beim Initialisieren des Controllers: {e}")
        return False

@app.route('/')
def dashboard():
    """Haupt-Dashboard"""
    try:
        if not controller:
            if not initialize_controller():
                return render_template('error.html', 
                                     error="Controller konnte nicht initialisiert werden")
        
        # System-Status
        status = controller.get_status()
        
        return render_template('dashboard.html', 
                             status=status,
                             config=config)
    
    except Exception as e:
        logger.error(f"Dashboard-Fehler: {e}")
        return render_template('error.html', error=str(e))

@app.route('/api/status')
def api_status():
    """API: System-Status"""
    try:
        if not controller:
            return jsonify({'error': 'Controller nicht verfügbar'}), 500
        
        status = controller.get_status()
        return jsonify(status)
    
    except Exception as e:
        logger.error(f"Status-API Fehler: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio')
def api_portfolio():
    """API: Portfolio-Informationen"""
    try:
        if not controller:
            return jsonify({'error': 'Controller nicht verfügbar'}), 500
        
        # Account Info
        account_info = controller.execution.get_account_info()
        
        # Positionen
        positions = controller.execution.get_positions()
        
        # Portfolio-Historie (falls verfügbar)
        try:
            portfolio_history = controller.execution.get_portfolio_history(period='1D', timeframe='1Min')
            history_data = []
            
            if not portfolio_history.empty:
                for timestamp, row in portfolio_history.iterrows():
                    history_data.append({
                        'timestamp': timestamp.isoformat(),
                        'equity': float(row['equity']),
                        'profit_loss': float(row.get('profit_loss', 0))
                    })
        except Exception as e:
            logger.warning(f"Portfolio-Historie nicht verfügbar: {e}")
            history_data = []
        
        return jsonify({
            'account': account_info,
            'positions': positions,
            'history': history_data
        })
    
    except Exception as e:
        logger.error(f"Portfolio-API Fehler: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trades')
def api_trades():
    """API: Trade-Historie"""
    try:
        if not controller:
            return jsonify({'error': 'Controller nicht verfügbar'}), 500
        
        limit = request.args.get('limit', 100, type=int)
        
        # Hole Order-Historie
        orders = controller.execution.get_orders(limit=limit)
        
        return jsonify({'trades': orders})
    
    except Exception as e:
        logger.error(f"Trades-API Fehler: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance')
def api_performance():
    """API: Performance-Metriken"""
    try:
        if not controller:
            return jsonify({'error': 'Controller nicht verfügbar'}), 500
        
        # Hole Account-Info
        account_info = controller.execution.get_account_info()
        
        # Berechne Performance-Metriken
        current_equity = float(account_info.get('equity', 0))
        last_equity = float(account_info.get('last_equity', current_equity))
        
        daily_pnl = current_equity - last_equity
        daily_return = (daily_pnl / last_equity * 100) if last_equity > 0 else 0
        
        # Hole Positionen für weitere Berechnungen
        positions = controller.execution.get_positions()
        total_unrealized_pnl = sum(float(pos.get('unrealized_pnl', 0)) for pos in positions)
        
        performance = {
            'current_equity': current_equity,
            'daily_pnl': daily_pnl,
            'daily_return': daily_return,
            'total_unrealized_pnl': total_unrealized_pnl,
            'positions_count': len(positions),
            'buying_power': float(account_info.get('buying_power', 0)),
            'cash': float(account_info.get('cash', 0))
        }
        
        return jsonify(performance)
    
    except Exception as e:
        logger.error(f"Performance-API Fehler: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chart/<symbol>')
def api_chart(symbol):
    """API: Chart-Daten für Symbol"""
    try:
        if not controller:
            return jsonify({'error': 'Controller nicht verfügbar'}), 500
        
        # Parameter
        days = request.args.get('days', 30, type=int)
        timeframe = request.args.get('timeframe', '1Day')
        
        # Berechne Datumsbereich
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Lade Daten
        data = controller.data_provider.get_historical(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            timeframe=timeframe
        )
        
        if data.empty:
            return jsonify({'error': 'Keine Daten verfügbar'}), 404
        
        # Konvertiere zu JSON-Format
        chart_data = []
        for timestamp, row in data.iterrows():
            chart_data.append({
                'timestamp': timestamp.isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume'])
            })
        
        return jsonify({
            'symbol': symbol,
            'data': chart_data
        })
    
    except Exception as e:
        logger.error(f"Chart-API Fehler für {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """API: Backtest ausführen"""
    try:
        if not controller:
            return jsonify({'error': 'Controller nicht verfügbar'}), 500
        
        # Parameter aus Request
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        start_date = data.get('start_date', '2023-01-01')
        end_date = data.get('end_date', '2024-01-01')
        initial_capital = data.get('initial_capital', 10000)
        
        logger.info(f"Starte Backtest: {symbol} ({start_date} - {end_date})")
        
        # Führe Backtest aus
        results = controller.run_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        
        # Konvertiere Equity Curve für Chart
        equity_curve = results.get('equity_curve')
        if equity_curve is not None and not equity_curve.empty:
            equity_data = []
            for timestamp, row in equity_curve.iterrows():
                equity_data.append({
                    'timestamp': timestamp.isoformat(),
                    'equity': float(row['equity']),
                    'drawdown': float(row.get('drawdown', 0))
                })
            results['equity_data'] = equity_data
            
            # Entferne DataFrame (nicht JSON-serialisierbar)
            del results['equity_curve']
        
        # Konvertiere Trades
        if 'trades' in results:
            for trade in results['trades']:
                if 'entry_time' in trade and trade['entry_time']:
                    trade['entry_time'] = trade['entry_time'].isoformat()
                if 'exit_time' in trade and trade['exit_time']:
                    trade['exit_time'] = trade['exit_time'].isoformat()
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Backtest-API Fehler: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategy/info')
def api_strategy_info():
    """API: Strategie-Informationen"""
    try:
        if not controller:
            return jsonify({'error': 'Controller nicht verfügbar'}), 500
        
        info = controller.strategy.get_info()
        return jsonify(info)
    
    except Exception as e:
        logger.error(f"Strategy-Info-API Fehler: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/backtest')
def backtest_page():
    """Backtest-Seite"""
    return render_template('backtest.html')

@app.route('/trades')
def trades_page():
    """Trades-Seite"""
    return render_template('trades.html')

@app.route('/settings')
def settings_page():
    """Einstellungen-Seite"""
    return render_template('settings.html', config=config)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error='Seite nicht gefunden'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Server-Fehler: {error}")
    return render_template('error.html', error='Interner Server-Fehler'), 500

# Template-Funktionen
@app.template_filter('datetime')
def datetime_filter(timestamp):
    """Template-Filter für Datum/Zeit-Formatierung"""
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except:
            return timestamp
    
    if isinstance(timestamp, datetime):
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
    return str(timestamp)

@app.template_filter('currency')
def currency_filter(value):
    """Template-Filter für Währungsformatierung"""
    try:
        return f"${float(value):,.2f}"
    except:
        return str(value)

@app.template_filter('percentage')
def percentage_filter(value):
    """Template-Filter für Prozent-Formatierung"""
    try:
        return f"{float(value):.2%}"
    except:
        return str(value)

if __name__ == '__main__':
    # Initialisiere Logger
    setup_logger()
    
    try:
        # Teste Controller-Initialisierung
        if initialize_controller():
            logger.info("Dashboard gestartet - Controller verfügbar")
        else:
            logger.warning("Dashboard gestartet - Controller nicht verfügbar")
        
        # Starte Flask-App
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=True,
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("Dashboard durch Benutzer gestoppt")
    except Exception as e:
        logger.error(f"Dashboard-Start fehlgeschlagen: {e}")
        sys.exit(1)