"""
Email Alert System für TradingBot
Sendet Benachrichtigungen bei Trades, Fehlern und wichtigen Events
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from dotenv import load_dotenv

# Lade Environment-Variablen
load_dotenv()

class EmailAlerter:
    """
    Email-Alert System für Trading-Benachrichtigungen
    
    Unterstützt:
    - Trade-Alerts
    - Fehler-Benachrichtigungen
    - Tägliche Zusammenfassungen
    - System-Status Updates
    """
    
    def __init__(self):
        """
        Initialisiert Email-Alerter mit Konfiguration aus Environment
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # SMTP-Konfiguration aus Environment
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.email_address = os.getenv('EMAIL_ADDRESS')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        
        # Empfänger-Liste
        recipients_str = os.getenv('ALERT_RECIPIENTS', '')
        self.recipients = [email.strip() for email in recipients_str.split(',') if email.strip()]
        
        # Validiere Konfiguration
        if not self.email_address or not self.email_password:
            self.logger.error("Email-Credentials nicht konfiguriert! Prüfe .env Datei.")
            self.enabled = False
        elif not self.recipients:
            self.logger.warning("Keine Email-Empfänger konfiguriert!")
            self.enabled = False
        else:
            self.enabled = True
            self.logger.info(f"Email-Alerts aktiviert für {len(self.recipients)} Empfänger")
        
        # Email-Templates
        self.templates = {
            'trade': self._get_trade_template(),
            'error': self._get_error_template(),
            'daily_summary': self._get_daily_summary_template(),
            'system_status': self._get_system_status_template()
        }
    
    def send_alert(self, subject: str, message: str, recipients: List[str] = None,
                   html_message: str = None, attachments: List[str] = None) -> bool:
        """
        Sendet Email-Alert
        
        Args:
            subject: Email-Betreff
            message: Email-Text (Plain)
            recipients: Liste der Empfänger (optional, verwendet Standard-Liste)
            html_message: HTML-Version der Email (optional)
            attachments: Liste von Dateipfaden für Anhänge (optional)
            
        Returns:
            True wenn erfolgreich gesendet
        """
        if not self.enabled:
            self.logger.warning("Email-Alerts deaktiviert")
            return False
        
        try:
            # Verwende Standard-Empfänger falls keine angegeben
            if recipients is None:
                recipients = self.recipients
            
            # Erstelle Email
            msg = MIMEMultipart('alternative')
            msg['From'] = self.email_address
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[TradingBot] {subject}"
            
            # Füge Text-Teil hinzu
            text_part = MIMEText(message, 'plain')
            msg.attach(text_part)
            
            # Füge HTML-Teil hinzu falls vorhanden
            if html_message:
                html_part = MIMEText(html_message, 'html')
                msg.attach(html_part)
            
            # Füge Anhänge hinzu
            if attachments:
                for file_path in attachments:
                    if os.path.isfile(file_path):
                        self._attach_file(msg, file_path)
                    else:
                        self.logger.warning(f"Anhang nicht gefunden: {file_path}")
            
            # Sende Email
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.email_address, self.email_password)
                server.send_message(msg)
            
            self.logger.info(f"Email gesendet: {subject} an {len(recipients)} Empfänger")
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Email-Versand: {e}")
            return False
    
    def send_trade_alert(self, symbol: str, action: str, quantity: float, 
                        price: float, confidence: float = None, 
                        pnl: float = None, additional_info: Dict[str, Any] = None) -> bool:
        """
        Sendet Trade-Alert
        
        Args:
            symbol: Trading-Symbol
            action: buy/sell
            quantity: Menge
            price: Preis
            confidence: Confidence-Level (optional)
            pnl: Profit/Loss (optional)
            additional_info: Zusätzliche Informationen (optional)
            
        Returns:
            True wenn erfolgreich gesendet
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Basis-Informationen
            trade_info = {
                'timestamp': timestamp,
                'symbol': symbol,
                'action': action.upper(),
                'quantity': quantity,
                'price': price,
                'value': quantity * price,
                'confidence': confidence,
                'pnl': pnl
            }
            
            if additional_info:
                trade_info.update(additional_info)
            
            # Erstelle Subject
            subject = f"Trade Alert: {action.upper()} {symbol}"
            if pnl:
                pnl_indicator = "📈" if pnl > 0 else "📉"
                subject += f" {pnl_indicator} ${pnl:.2f}"
            
            # Text-Version
            text_message = self.templates['trade'].format(**trade_info)
            
            # HTML-Version
            html_message = self._create_trade_html(trade_info)
            
            return self.send_alert(subject, text_message, html_message=html_message)
            
        except Exception as e:
            self.logger.error(f"Fehler bei Trade-Alert: {e}")
            return False
    
    def send_error_alert(self, error_type: str, error_message: str, 
                        traceback_info: str = None) -> bool:
        """
        Sendet Fehler-Alert
        
        Args:
            error_type: Fehler-Typ/Quelle
            error_message: Fehlermeldung
            traceback_info: Traceback-Informationen (optional)
            
        Returns:
            True wenn erfolgreich gesendet
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            error_info = {
                'timestamp': timestamp,
                'error_type': error_type,
                'error_message': error_message,
                'traceback': traceback_info or 'Nicht verfügbar'
            }
            
            subject = f"🚨 ERROR: {error_type}"
            text_message = self.templates['error'].format(**error_info)
            html_message = self._create_error_html(error_info)
            
            return self.send_alert(subject, text_message, html_message=html_message)
            
        except Exception as e:
            self.logger.error(f"Fehler bei Error-Alert: {e}")
            return False
    
    def send_daily_summary(self, portfolio_value: float, daily_pnl: float,
                          trades_count: int, win_rate: float = None,
                          positions: List[Dict] = None, 
                          performance_chart: str = None) -> bool:
        """
        Sendet tägliche Zusammenfassung
        
        Args:
            portfolio_value: Aktueller Portfolio-Wert
            daily_pnl: Täglicher P&L
            trades_count: Anzahl Trades heute
            win_rate: Gewinnrate (optional)
            positions: Aktuelle Positionen (optional)
            performance_chart: Pfad zu Performance-Chart (optional)
            
        Returns:
            True wenn erfolgreich gesendet
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d')
            
            summary_info = {
                'date': timestamp,
                'portfolio_value': portfolio_value,
                'daily_pnl': daily_pnl,
                'pnl_pct': (daily_pnl / portfolio_value * 100) if portfolio_value > 0 else 0,
                'trades_count': trades_count,
                'win_rate': win_rate or 0,
                'positions_count': len(positions) if positions else 0
            }
            
            # Emoji für Performance
            pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
            subject = f"Daily Summary {timestamp} {pnl_emoji} ${daily_pnl:.2f}"
            
            # Text-Version
            text_message = self.templates['daily_summary'].format(**summary_info)
            
            # Füge Positionen hinzu
            if positions:
                text_message += "\n\nAktuelle Positionen:\n"
                for pos in positions:
                    text_message += f"• {pos.get('symbol', 'N/A')}: {pos.get('qty', 0)} @ ${pos.get('current_price', 0):.2f}\n"
            
            # HTML-Version
            html_message = self._create_summary_html(summary_info, positions)
            
            # Anhänge
            attachments = []
            if performance_chart and os.path.isfile(performance_chart):
                attachments.append(performance_chart)
            
            return self.send_alert(subject, text_message, html_message=html_message, 
                                 attachments=attachments)
            
        except Exception as e:
            self.logger.error(f"Fehler bei Daily-Summary: {e}")
            return False
    
    def send_system_status(self, status: str, uptime: str = None, 
                          memory_usage: float = None, last_trade: str = None) -> bool:
        """
        Sendet System-Status Update
        
        Args:
            status: System-Status (running/stopped/error)
            uptime: Laufzeit (optional)
            memory_usage: Speicherverbrauch (optional)
            last_trade: Letzter Trade (optional)
            
        Returns:
            True wenn erfolgreich gesendet
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            status_info = {
                'timestamp': timestamp,
                'status': status.upper(),
                'uptime': uptime or 'Unbekannt',
                'memory_usage': memory_usage or 0,
                'last_trade': last_trade or 'Keine Trades'
            }
            
            # Status-Emoji
            status_emojis = {
                'RUNNING': '✅',
                'STOPPED': '⏹️',
                'ERROR': '❌',
                'WARNING': '⚠️'
            }
            
            emoji = status_emojis.get(status.upper(), '🤖')
            subject = f"System Status: {status.upper()} {emoji}"
            
            text_message = self.templates['system_status'].format(**status_info)
            html_message = self._create_status_html(status_info)
            
            return self.send_alert(subject, text_message, html_message=html_message)
            
        except Exception as e:
            self.logger.error(f"Fehler bei System-Status: {e}")
            return False
    
    def test_connection(self) -> bool:
        """
        Testet Email-Verbindung
        
        Returns:
            True wenn Verbindung erfolgreich
        """
        if not self.enabled:
            return False
            
        try:
            test_message = f"""
TradingBot Email-Test

Diese Test-Email wurde um {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} gesendet.

Wenn Sie diese Email erhalten, ist die Email-Konfiguration korrekt.

Konfiguration:
- SMTP Server: {self.smtp_server}:{self.smtp_port}
- Absender: {self.email_address}
- Empfänger: {len(self.recipients)} konfiguriert
            """
            
            return self.send_alert("Email Test", test_message)
            
        except Exception as e:
            self.logger.error(f"Email-Test fehlgeschlagen: {e}")
            return False
    
    def _attach_file(self, msg: MIMEMultipart, file_path: str):
        """
        Fügt Datei als Anhang hinzu
        
        Args:
            msg: Email-Message
            file_path: Pfad zur Datei
        """
        try:
            with open(file_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            
            filename = os.path.basename(file_path)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {filename}'
            )
            
            msg.attach(part)
            
        except Exception as e:
            self.logger.error(f"Fehler beim Anhängen der Datei {file_path}: {e}")
    
    def _get_trade_template(self) -> str:
        """Trade-Alert Template"""
        return """
Trading Alert - {timestamp}

Symbol: {symbol}
Action: {action}
Quantity: {quantity}
Price: ${price:.2f}
Total Value: ${value:.2f}
Confidence: {confidence:.2%}

P&L: ${pnl:.2f}

This is an automated message from your TradingBot.
        """.strip()
    
    def _get_error_template(self) -> str:
        """Error-Alert Template"""
        return """
🚨 ERROR ALERT - {timestamp}

Error Type: {error_type}
Error Message: {error_message}

Traceback:
{traceback}

Please check the system immediately.

This is an automated error notification from your TradingBot.
        """.strip()
    
    def _get_daily_summary_template(self) -> str:
        """Daily Summary Template"""
        return """
Daily Summary - {date}

Portfolio Performance:
• Portfolio Value: ${portfolio_value:,.2f}
• Daily P&L: ${daily_pnl:.2f} ({pnl_pct:+.2f}%)
• Total Trades: {trades_count}
• Win Rate: {win_rate:.1%}
• Active Positions: {positions_count}

This is your daily trading summary from TradingBot.
        """.strip()
    
    def _get_system_status_template(self) -> str:
        """System Status Template"""
        return """
System Status Update - {timestamp}

Status: {status}
Uptime: {uptime}
Memory Usage: {memory_usage:.1f}%
Last Trade: {last_trade}

This is an automated system status update from your TradingBot.
        """.strip()
    
    def _create_trade_html(self, trade_info: Dict[str, Any]) -> str:
        """Erstellt HTML-Version für Trade-Alert"""
        pnl_color = "green" if trade_info.get('pnl', 0) >= 0 else "red"
        confidence_color = "green" if trade_info.get('confidence', 0) > 0.7 else "orange" if trade_info.get('confidence', 0) > 0.5 else "red"
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 20px;">
            <h2 style="color: #2E86AB;">🤖 Trading Alert</h2>
            <p><strong>Timestamp:</strong> {trade_info['timestamp']}</p>
            
            <table style="border-collapse: collapse; width: 100%; margin: 20px 0;">
                <tr style="background-color: #f2f2f2;">
                    <td style="border: 1px solid #ddd; padding: 8px;"><strong>Symbol</strong></td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{trade_info['symbol']}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;"><strong>Action</strong></td>
                    <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold; color: {'green' if trade_info['action'] == 'BUY' else 'red'};">{trade_info['action']}</td>
                </tr>
                <tr style="background-color: #f2f2f2;">
                    <td style="border: 1px solid #ddd; padding: 8px;"><strong>Quantity</strong></td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{trade_info['quantity']}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;"><strong>Price</strong></td>
                    <td style="border: 1px solid #ddd; padding: 8px;">${trade_info['price']:.2f}</td>
                </tr>
                <tr style="background-color: #f2f2f2;">
                    <td style="border: 1px solid #ddd; padding: 8px;"><strong>Total Value</strong></td>
                    <td style="border: 1px solid #ddd; padding: 8px;">${trade_info['value']:.2f}</td>
                </tr>
                {"<tr><td style='border: 1px solid #ddd; padding: 8px;'><strong>Confidence</strong></td><td style='border: 1px solid #ddd; padding: 8px; color: " + confidence_color + ";'>" + f"{trade_info['confidence']:.2%}" + "</td></tr>" if trade_info.get('confidence') else ""}
                {"<tr style='background-color: #f2f2f2;'><td style='border: 1px solid #ddd; padding: 8px;'><strong>P&L</strong></td><td style='border: 1px solid #ddd; padding: 8px; font-weight: bold; color: " + pnl_color + ";'>" + f"${trade_info['pnl']:.2f}" + "</td></tr>" if trade_info.get('pnl') is not None else ""}
            </table>
            
            <p style="color: #666; font-size: 12px;">
                This is an automated message from your TradingBot.
            </p>
        </body>
        </html>
        """
    
    def _create_error_html(self, error_info: Dict[str, Any]) -> str:
        """Erstellt HTML-Version für Error-Alert"""
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 20px;">
            <h2 style="color: #D32F2F;">🚨 Error Alert</h2>
            <p><strong>Timestamp:</strong> {error_info['timestamp']}</p>
            
            <div style="background-color: #ffebee; border-left: 4px solid #f44336; padding: 10px; margin: 20px 0;">
                <h3 style="color: #D32F2F; margin-top: 0;">Error Details</h3>
                <p><strong>Type:</strong> {error_info['error_type']}</p>
                <p><strong>Message:</strong> {error_info['error_message']}</p>
            </div>
            
            <div style="background-color: #f5f5f5; border: 1px solid #ddd; padding: 10px; margin: 20px 0;">
                <h4>Traceback:</h4>
                <pre style="white-space: pre-wrap; font-size: 12px;">{error_info['traceback']}</pre>
            </div>
            
            <p style="color: #666; font-size: 12px;">
                Please check the system immediately. This is an automated error notification from your TradingBot.
            </p>
        </body>
        </html>
        """
    
    def _create_summary_html(self, summary_info: Dict[str, Any], positions: List[Dict] = None) -> str:
        """Erstellt HTML-Version für Daily Summary"""
        pnl_color = "green" if summary_info['daily_pnl'] >= 0 else "red"
        pnl_arrow = "↗" if summary_info['daily_pnl'] >= 0 else "↘"
        
        positions_html = ""
        if positions:
            positions_html = "<h3>Current Positions:</h3><ul>"
            for pos in positions:
                unrealized_pnl = pos.get('unrealized_pnl', 0)
                pnl_color_pos = "green" if unrealized_pnl >= 0 else "red"
                positions_html += f"""
                <li>
                    <strong>{pos.get('symbol', 'N/A')}</strong>: 
                    {pos.get('qty', 0)} shares @ ${pos.get('current_price', 0):.2f}
                    <span style="color: {pnl_color_pos};">(${unrealized_pnl:.2f})</span>
                </li>
                """
            positions_html += "</ul>"
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 20px;">
            <h2 style="color: #2E86AB;">📊 Daily Summary - {summary_info['date']}</h2>
            
            <div style="display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0;">
                <div style="background-color: #e3f2fd; border-radius: 8px; padding: 15px; flex: 1; min-width: 200px;">
                    <h3 style="margin-top: 0; color: #1976d2;">Portfolio Value</h3>
                    <p style="font-size: 24px; font-weight: bold; margin: 0;">${summary_info['portfolio_value']:,.2f}</p>
                </div>
                
                <div style="background-color: {'#e8f5e8' if summary_info['daily_pnl'] >= 0 else '#ffebee'}; border-radius: 8px; padding: 15px; flex: 1; min-width: 200px;">
                    <h3 style="margin-top: 0; color: {pnl_color};">Daily P&L {pnl_arrow}</h3>
                    <p style="font-size: 24px; font-weight: bold; margin: 0; color: {pnl_color};">
                        ${summary_info['daily_pnl']:.2f} ({summary_info['pnl_pct']:+.2f}%)
                    </p>
                </div>
            </div>
            
            <div style="display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0;">
                <div style="background-color: #fff3e0; border-radius: 8px; padding: 15px; flex: 1; min-width: 150px;">
                    <h4 style="margin-top: 0; color: #f57c00;">Total Trades</h4>
                    <p style="font-size: 20px; font-weight: bold; margin: 0;">{summary_info['trades_count']}</p>
                </div>
                
                <div style="background-color: #f3e5f5; border-radius: 8px; padding: 15px; flex: 1; min-width: 150px;">
                    <h4 style="margin-top: 0; color: #7b1fa2;">Win Rate</h4>
                    <p style="font-size: 20px; font-weight: bold; margin: 0;">{summary_info['win_rate']:.1%}</p>
                </div>
                
                <div style="background-color: #e0f2f1; border-radius: 8px; padding: 15px; flex: 1; min-width: 150px;">
                    <h4 style="margin-top: 0; color: #00695c;">Positions</h4>
                    <p style="font-size: 20px; font-weight: bold; margin: 0;">{summary_info['positions_count']}</p>
                </div>
            </div>
            
            {positions_html}
            
            <p style="color: #666; font-size: 12px; margin-top: 30px;">
                This is your daily trading summary from TradingBot.
            </p>
        </body>
        </html>
        """
    
    def _create_status_html(self, status_info: Dict[str, Any]) -> str:
        """Erstellt HTML-Version für System Status"""
        status_colors = {
            'RUNNING': '#4caf50',
            'STOPPED': '#ff9800', 
            'ERROR': '#f44336',
            'WARNING': '#ff5722'
        }
        
        status_color = status_colors.get(status_info['status'], '#666')
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 20px;">
            <h2 style="color: #2E86AB;">🤖 System Status Update</h2>
            <p><strong>Timestamp:</strong> {status_info['timestamp']}</p>
            
            <div style="background-color: #f5f5f5; border-left: 4px solid {status_color}; padding: 15px; margin: 20px 0;">
                <h3 style="color: {status_color}; margin-top: 0;">Status: {status_info['status']}</h3>
                
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Uptime:</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">{status_info['uptime']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Memory Usage:</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">{status_info['memory_usage']:.1f}%</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px;"><strong>Last Trade:</strong></td>
                        <td style="padding: 8px;">{status_info['last_trade']}</td>
                    </tr>
                </table>
            </div>
            
            <p style="color: #666; font-size: 12px;">
                This is an automated system status update from your TradingBot.
            </p>
        </body>
        </html>
        """
    
    def add_recipient(self, email: str) -> bool:
        """
        Fügt Empfänger hinzu
        
        Args:
            email: Email-Adresse
            
        Returns:
            True wenn erfolgreich hinzugefügt
        """
        try:
            if email not in self.recipients:
                self.recipients.append(email)
                self.logger.info(f"Empfänger hinzugefügt: {email}")
                return True
            else:
                self.logger.info(f"Empfänger bereits vorhanden: {email}")
                return False
                
        except Exception as e:
            self.logger.error(f"Fehler beim Hinzufügen des Empfängers: {e}")
            return False
    
    def remove_recipient(self, email: str) -> bool:
        """
        Entfernt Empfänger
        
        Args:
            email: Email-Adresse
            
        Returns:
            True wenn erfolgreich entfernt
        """
        try:
            if email in self.recipients:
                self.recipients.remove(email)
                self.logger.info(f"Empfänger entfernt: {email}")
                return True
            else:
                self.logger.info(f"Empfänger nicht gefunden: {email}")
                return False
                
        except Exception as e:
            self.logger.error(f"Fehler beim Entfernen des Empfängers: {e}")
            return False
    
    def get_recipients(self) -> List[str]:
        """
        Gibt aktuelle Empfänger-Liste zurück
        
        Returns:
            Liste der Email-Adressen
        """
        return self.recipients.copy()
    
    def is_enabled(self) -> bool:
        """
        Prüft ob Email-Alerts aktiviert sind
        
        Returns:
            True wenn aktiviert
        """
        return self.enabled
    
    def get_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über Email-System zurück
        
        Returns:
            Dict mit System-Informationen
        """
        return {
            'enabled': self.enabled,
            'smtp_server': self.smtp_server,
            'smtp_port': self.smtp_port,
            'sender': self.email_address,
            'recipients_count': len(self.recipients),
            'recipients': self.recipients if self.enabled else [],
            'templates_available': list(self.templates.keys())
        }