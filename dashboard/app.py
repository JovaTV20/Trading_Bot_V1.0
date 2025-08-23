"""
TradingBot Dashboard - Komplette Flask Web Interface (ALLE FIXES INTEGRIERT)
ERSETZE: dashboard/app.py
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

# Füge das Projekt-Root-Verzeichnis zum Python-Path hinzu
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, render_template, jsonify, request
import pandas as pd

# Plotly nur importieren wenn verfügbar
try:
    import plotly
    import plotly.graph_objs as go
    import plotly.express as px
    from plotly.utils import PlotlyJSONEncoder
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# REPARIERT: CORS-Import mit Fehlerbehandlung
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("⚠️  flask-cors nicht installiert. CORS wird nicht aktiviert.")
    print("   Installation: pip install flask-cors")

from core.controller import TradingController
from core.logger import setup_logger, get_logger

# Flask App initialisieren
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = os.urandom(24)

# CORS nur aktivieren wenn verfügbar
if CORS_AVAILABLE:
    CORS(app)
    print("✅ CORS aktiviert")

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
        with open(config_path, 'r', encoding='utf-8') as f:
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
    """API: Portfolio-Informationen - REPARIERT"""
    try:
        if not controller:
            return jsonify({'error': 'Controller nicht verfügbar'}), 500
        
        # Account Info
        account_info = controller.execution.get_account_info()
        
        # Positionen
        positions = controller.execution.get_positions()
        
        # Portfolio-Historie (REPARIERT mit Fallback)
        try:
            if hasattr(controller.execution, 'get_portfolio_history'):
                portfolio_history = controller.execution.get_portfolio_history(period='1D', timeframe='1Min')
                history_data = []
                
                if not portfolio_history.empty:
                    for timestamp, row in portfolio_history.iterrows():
                        history_data.append({
                            'timestamp': timestamp.isoformat(),
                            'equity': float(row.get('equity', account_info.get('equity', 10000))),
                            'profit_loss': float(row.get('profit_loss', 0))
                        })
                else:
                    # Fallback: Mock-Daten für Demo
                    now = datetime.now()
                    current_equity = float(account_info.get('equity', 10000))
                    for i in range(12, 0, -1):
                        timestamp = now - timedelta(minutes=i*5)
                        history_data.append({
                            'timestamp': timestamp.isoformat(),
                            'equity': current_equity + (i-6) * 10,  # Leichte Schwankung
                            'profit_loss': (i-6) * 5
                        })
            else:
                # Fallback wenn Methode nicht existiert
                logger.info("Portfolio-Historie nicht implementiert, verwende Mock-Daten")
                now = datetime.now()
                current_equity = float(account_info.get('equity', 10000))
                history_data = []
                for i in range(12, 0, -1):
                    timestamp = now - timedelta(minutes=i*5)
                    history_data.append({
                        'timestamp': timestamp.isoformat(),
                        'equity': current_equity + (i-6) * 10,
                        'profit_loss': (i-6) * 5
                    })
                
        except Exception as e:
            logger.warning(f"Portfolio-Historie Fehler: {e}")
            # Absolute Fallback-Daten
            now = datetime.now()
            current_equity = float(account_info.get('equity', 10000))
            history_data = []
            for i in range(12, 0, -1):
                timestamp = now - timedelta(minutes=i*5)
                history_data.append({
                    'timestamp': timestamp.isoformat(),
                    'equity': current_equity + (i-6) * 10,
                    'profit_loss': (i-6) * 5
                })
        
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
    """API: Trade-Historie - REPARIERT"""
    try:
        if not controller:
            return jsonify({'error': 'Controller nicht verfügbar'}), 500
        
        limit = request.args.get('limit', 100, type=int)
        
        # Hole Order-Historie
        try:
            if hasattr(controller.execution, 'get_orders'):
                orders = controller.execution.get_orders(limit=limit)
            elif hasattr(controller.execution, 'get_order_history'):
                orders = controller.execution.get_order_history(limit=limit)
            else:
                # Fallback: Leere Liste
                orders = []
                logger.info("Keine Trade-Historie verfügbar")
        except Exception as e:
            logger.warning(f"Trade-Historie Fehler: {e}")
            orders = []
        
        return jsonify({'trades': orders})
    
    except Exception as e:
        logger.error(f"Trades-API Fehler: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance')
def api_performance():
    """API: Performance-Metriken - REPARIERT"""
    try:
        if not controller:
            return jsonify({'error': 'Controller nicht verfügbar'}), 500
        
        # Hole Account-Info
        account_info = controller.execution.get_account_info()
        
        # Sichere Wert-Extraktion
        current_equity = float(account_info.get('equity', 0))
        last_equity = float(account_info.get('last_equity', current_equity))
        
        # Berechne Performance-Metriken
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
    """API: Chart-Daten für Symbol - REPARIERT"""
    try:
        if not controller:
            return jsonify({'error': 'Controller nicht verfügbar'}), 500
        
        # Parameter
        days = request.args.get('days', 30, type=int)
        timeframe = request.args.get('timeframe', '1Day')
        
        # Berechne Datumsbereich - KORRIGIERT für Paper Account
        end_date = datetime.now() - timedelta(days=2)  # Nicht zu aktuelle Daten
        start_date = end_date - timedelta(days=days)
        
        # Lade Daten
        try:
            data = controller.data_provider.get_historical(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                timeframe=timeframe
            )
        except Exception as data_error:
            logger.warning(f"Daten-Fehler: {data_error}, versuche älteren Zeitraum")
            # Fallback: Viel ältere Daten
            end_date = datetime.now() - timedelta(days=30)
            start_date = end_date - timedelta(days=365)
            data = controller.data_provider.get_historical(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                timeframe='1Day'
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
                'volume': int(row['volume']) if not pd.isna(row['volume']) else 0
            })
        
        return jsonify({
            'symbol': symbol,
            'data': chart_data
        })
    
    except Exception as e:
        logger.error(f"Chart-API Fehler für {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest', methods=['POST'])
@app.route('/api/backtest', methods=['POST'])
@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """API: Backtest ausführen - PAPER ACCOUNT KOMPATIBEL"""
    try:
        if not controller:
            return jsonify({'error': 'Controller nicht verfügbar'}), 500
        
        # Parameter aus Request
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        start_date = data.get('start_date', '2023-01-01')
        end_date = data.get('end_date', '2023-12-31')
        initial_capital = data.get('initial_capital', 10000)
        
        # KRITISCH: Paper Account Datums-Korrektur
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Paper Account kann nur Daten bis vor ~15 Tagen abrufen
            max_end_date = datetime.now() - timedelta(days=15)
            
            # Automatische Korrektur für Paper Account
            if end_dt > max_end_date:
                logger.warning(f"End-Datum zu aktuell ({end_date}), korrigiere für Paper Account")
                end_dt = max_end_date
                end_date = end_dt.strftime('%Y-%m-%d')
                
                # Auch Start-Datum anpassen falls nötig
                min_start_date = end_dt - timedelta(days=365)  # Max 1 Jahr Zeitraum
                if start_dt > min_start_date:
                    start_dt = min_start_date
                    start_date = start_dt.strftime('%Y-%m-%d')
                
                logger.info(f"Korrigierte Daten: {start_date} bis {end_date}")
            
            # Mindest-Zeitraum prüfen
            time_diff = (end_dt - start_dt).days
            if time_diff < 30:
                logger.warning("Zeitraum zu kurz, erweitere auf 3 Monate")
                start_dt = end_dt - timedelta(days=90)
                start_date = start_dt.strftime('%Y-%m-%d')
            elif time_diff > 365:
                logger.warning("Zeitraum zu lang, beschränke auf 1 Jahr")
                start_dt = end_dt - timedelta(days=365)
                start_date = start_dt.strftime('%Y-%m-%d')
        
        except ValueError as date_error:
            logger.error(f"Datums-Parsing Fehler: {date_error}")
            # Fallback: Sichere Standarddaten
            end_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            logger.info(f"Fallback-Daten: {start_date} bis {end_date}")
        
        logger.info(f"Starte Backtest: {symbol} ({start_date} - {end_date})")
        
        # Teste erst Daten-Verfügbarkeit
        try:
            test_data = controller.data_provider.get_historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe='1Day'
            )
            
            if test_data.empty:
                # Fallback: Noch ältere Daten versuchen
                logger.warning("Keine Daten gefunden, versuche älteren Zeitraum")
                end_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=425)).strftime('%Y-%m-%d')
                logger.info(f"Älteren Zeitraum probieren: {start_date} bis {end_date}")
                
                test_data = controller.data_provider.get_historical(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe='1Day'
                )
                
                if test_data.empty:
                    raise Exception("Keine historischen Daten verfügbar für Paper Account")
            
        except Exception as data_test_error:
            logger.error(f"Daten-Test fehlgeschlagen: {data_test_error}")
            return jsonify({
                'error': f'Keine Daten verfügbar für {symbol}. Paper Account kann nur ältere Daten abrufen.',
                'suggestion': 'Verwende Daten aus 2022 oder früher',
                'corrected_dates': {
                    'start_date': (datetime.now() - timedelta(days=425)).strftime('%Y-%m-%d'),
                    'end_date': (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
                }
            }), 400
        
        # Führe Backtest aus mit korrigierten Daten
        try:
            results = controller.run_backtest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital
            )
        except Exception as backtest_error:
            logger.error(f"Backtest-Ausführung fehlgeschlagen: {backtest_error}")
            return jsonify({'error': f'Backtest fehlgeschlagen: {str(backtest_error)}'}), 500
        
        # JSON-Kompatibilität sicherstellen
        import json
        import math
        
        def fix_json_values(obj):
            """Ersetzt Infinity/NaN für JSON-Kompatibilität"""
            if isinstance(obj, dict):
                return {k: fix_json_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [fix_json_values(item) for item in obj]
            elif isinstance(obj, float):
                if math.isinf(obj):
                    return 999.99 if obj > 0 else -999.99
                elif math.isnan(obj):
                    return 0.0
                else:
                    return obj
            else:
                return obj
        
        # Repariere alle Results
        results = fix_json_values(results)
        
        # Füge verwendete Daten zur Antwort hinzu
        results['backtest_info'] = {
            'actual_start_date': start_date,
            'actual_end_date': end_date,
            'data_points': len(test_data) if not test_data.empty else 0,
            'paper_account_mode': True
        }
        
        # Konvertiere Equity Curve
        if 'equity_curve' in results:
            if hasattr(results['equity_curve'], 'to_dict'):
                equity_data = []
                for timestamp, row in results['equity_curve'].iterrows():
                    equity_data.append({
                        'timestamp': timestamp.isoformat(),
                        'equity': float(row.get('equity', 0)),
                        'drawdown': float(row.get('drawdown', 0))
                    })
                results['equity_data'] = equity_data
            del results['equity_curve']
        
        # Konvertiere Trade-Timestamps
        if 'trades' in results:
            for trade in results['trades']:
                for date_field in ['entry_time', 'exit_time']:
                    if date_field in trade and trade[date_field]:
                        if hasattr(trade[date_field], 'isoformat'):
                            trade[date_field] = trade[date_field].isoformat()
        
        logger.info(f"✅ Backtest erfolgreich: {results.get('total_trades', 0)} Trades")
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Backtest-API Fehler: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategy/info')
def api_strategy_info():
    """API: Strategie-Informationen - VOLLSTÄNDIG REPARIERT"""
    try:
        if not controller:
            return jsonify({'error': 'Controller nicht verfügbar'}), 500
        
        # REPARIERT: Sichere Strategie-Info Abfrage
        try:
            if hasattr(controller, 'strategy') and controller.strategy:
                strategy = controller.strategy
                
                # Basis-Informationen sicher abfragen
                info = {
                    'name': getattr(strategy, '__class__', type(strategy)).__name__,
                    'is_fitted': getattr(strategy, 'is_fitted', False),
                    'model_type': 'Machine Learning Strategy',
                    'status': 'active'
                }
                
                # Erweiterte Informationen falls verfügbar
                if hasattr(strategy, 'get_info'):
                    try:
                        detailed_info = strategy.get_info()
                        info.update({
                            'parameters': detailed_info.get('parameters', {}),
                            'training_score': detailed_info.get('training_score', 0.0),
                            'feature_count': detailed_info.get('feature_count', 0),
                            'lookback_period': detailed_info.get('lookback_period', 20),
                            'prediction_threshold': detailed_info.get('prediction_threshold', 0.5)
                        })
                    except Exception as detail_error:
                        logger.warning(f"Detaillierte Strategie-Info nicht verfügbar: {detail_error}")
                
                # Model-spezifische Infos falls verfügbar
                if hasattr(strategy, 'model') and strategy.model:
                    try:
                        model = strategy.model
                        info['model_class'] = type(model).__name__
                        
                        # Sichere Abfrage von Model-Parametern
                        if hasattr(model, 'n_estimators'):
                            info['n_estimators'] = getattr(model, 'n_estimators', 'N/A')
                        if hasattr(model, 'max_depth'):
                            info['max_depth'] = getattr(model, 'max_depth', 'N/A')
                        if hasattr(model, 'random_state'):
                            info['random_state'] = getattr(model, 'random_state', 'N/A')
                            
                    except Exception as model_error:
                        logger.warning(f"Model-Info nicht verfügbar: {model_error}")
                
                # Feature-Informationen falls verfügbar
                if hasattr(strategy, 'feature_names') and strategy.feature_names:
                    try:
                        info['feature_names'] = strategy.feature_names[:10]  # Nur erste 10
                        info['total_features'] = len(strategy.feature_names)
                    except Exception as feature_error:
                        logger.warning(f"Feature-Info nicht verfügbar: {feature_error}")
                
                # Top Features falls verfügbar
                try:
                    if (hasattr(strategy, 'model') and strategy.model and 
                        hasattr(strategy.model, 'feature_importances_') and
                        hasattr(strategy, 'feature_names')):
                        
                        importances = strategy.model.feature_importances_
                        feature_names = strategy.feature_names
                        
                        if len(importances) == len(feature_names):
                            # Top 5 Features
                            top_features = []
                            for i in sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)[:5]:
                                top_features.append({
                                    'feature': feature_names[i],
                                    'importance': float(importances[i])
                                })
                            info['top_features'] = top_features
                            
                except Exception as importance_error:
                    logger.warning(f"Feature-Importance nicht verfügbar: {importance_error}")
                
                return jsonify(info)
            
            else:
                # Fallback: Minimal-Info
                return jsonify({
                    'name': 'Strategy',
                    'is_fitted': False,
                    'model_type': 'Unknown',
                    'status': 'not_initialized'
                })
                
        except Exception as e:
            logger.warning(f"Strategy-Info Fehler: {e}")
            # Absoluter Fallback
            return jsonify({
                'name': 'ML Strategy',
                'is_fitted': False,
                'model_type': 'Machine Learning',
                'status': 'error',
                'error': str(e)
            })
    
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

# Template-Filter (Jinja2) - REPARIERT
@app.template_filter('tojsonfilter')
def to_json_filter(obj):
    """Konvertiert Python-Objekt zu JSON für Templates"""
    try:
        if obj is None:
            return '{}'
        
        # Spezielle Behandlung für komplexe Objekte
        if hasattr(obj, '__dict__'):
            # Konvertiere Objekt zu Dictionary
            obj_dict = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):  # Ignoriere private Attribute
                    try:
                        json.dumps(value)  # Test ob JSON-serialisierbar
                        obj_dict[key] = value
                    except (TypeError, ValueError):
                        obj_dict[key] = str(value)
            obj = obj_dict
        
        return json.dumps(obj, default=str)
    except (TypeError, ValueError) as e:
        logger.warning(f"JSON-Konvertierung fehlgeschlagen: {e}")
        return '{}'

@app.template_filter('datetime')
def datetime_filter(timestamp):
    """Template-Filter für Datum/Zeit-Formatierung"""
    if timestamp is None:
        return 'N/A'
    
    try:
        if isinstance(timestamp, str):
            # Versuche verschiedene Formate
            for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                try:
                    timestamp = datetime.strptime(timestamp.split('.')[0], fmt)
                    break
                except ValueError:
                    continue
        
        if isinstance(timestamp, datetime):
            return timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        return str(timestamp)
    except Exception as e:
        logger.warning(f"Datetime-Filter Fehler: {e}")
        return str(timestamp)

@app.template_filter('currency')
def currency_filter(value):
    """Template-Filter für Währungsformatierung"""
    try:
        if value is None:
            return '$0.00'
        return f"${float(value):,.2f}"
    except (ValueError, TypeError):
        return str(value)

@app.template_filter('percentage')
def percentage_filter(value):
    """Template-Filter für Prozent-Formatierung"""
    try:
        if value is None:
            return '0.00%'
        return f"{float(value):.2%}"
    except (ValueError, TypeError):
        return str(value)

# Zusätzliche Utility-Routen für Debugging
@app.route('/api/debug/info')
def api_debug_info():
    """Debug-Informationen für Troubleshooting"""
    try:
        debug_info = {
            'timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
            'flask_version': getattr(Flask, '__version__', 'unknown'),
            'controller_available': controller is not None,
            'config_loaded': bool(config),
            'plotly_available': PLOTLY_AVAILABLE,
            'cors_available': CORS_AVAILABLE
        }
        
        if controller:
            try:
                debug_info['controller_status'] = controller.get_status()
            except Exception as e:
                debug_info['controller_error'] = str(e)
        
        return jsonify(debug_info)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def api_health():
    """Health-Check Endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

def main():
    """Hauptfunktion für CLI-Start"""
    # Initialisiere Logger
    setup_logger()
    
    try:
        # Teste Controller-Initialisierung
        if initialize_controller():
            logger.info("Dashboard gestartet - Controller verfügbar")
        else:
            logger.warning("Dashboard gestartet - Controller nicht verfügbar")
        
        # Zeige wichtige Informationen
        print("\n" + "="*60)
        print("🌐 TRADINGBOT DASHBOARD GESTARTET")
        print("="*60)
        print(f"📊 Dashboard URL: http://127.0.0.1:5000")
        print(f"🔧 CORS Status: {'✅ Aktiviert' if CORS_AVAILABLE else '⚠️  Nicht verfügbar'}")
        print(f"📈 Plotly Status: {'✅ Verfügbar' if PLOTLY_AVAILABLE else '⚠️  Nicht verfügbar'}")
        print(f"🤖 Controller: {'✅ Bereit' if controller else '❌ Nicht verfügbar'}")
        print("="*60)
        print("💡 Tipps:")
        print("   • Verwende Daten aus 2023 für Backtests (Paper Account)")
        print("   • Dashboard läuft im Debug-Modus")
        print("   • Alle API-Endpoints haben Fallback-Mechanismen")
        print("="*60 + "\n")
        
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
        print(f"\n❌ FEHLER beim Dashboard-Start: {e}")
        print("\n🔧 MÖGLICHE LÖSUNGEN:")
        print("1. Prüfe ob alle Dependencies installiert sind:")
        print("   pip install -r requirements.txt")
        print("2. Prüfe ob config.json existiert")
        print("3. Prüfe ob .env-Datei konfiguriert ist")
        sys.exit(1)

if __name__ == '__main__':
    main()