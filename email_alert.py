#!/usr/bin/env python3
"""
Email Alert Service - Backup for SMS
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def send_email_alert(event_type, latitude, longitude, to_email="your-email@gmail.com"):
    """
    Send email alert as backup when SMS fails
    """
    try:
        # Gmail SMTP settings (you'll need to configure these)
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = "your-sender-email@gmail.com"  # Change this
        sender_password = "your-app-password"  # Change this

        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = f"🚨 VIOLENCE DETECTION ALERT - {event_type}"

        body = f"""
        🚨 VIOLENCE DETECTION ALERT 🚨

        Event: {event_type}
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Location: {latitude}, {longitude}
        Maps: https://maps.google.com/?q={latitude},{longitude}

        This is an automated security alert from your Violence Detection System.

        Please respond immediately.
        """

        msg.attach(MIMEText(body, 'plain'))

        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, to_email, text)
        server.quit()

        print(f"✅ Email alert sent to {to_email}")
        return True

    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

if __name__ == "__main__":
    # Test email
    send_email_alert("Fighting", "11.0055", "76.9661", "6382941185@airtelmail.com")
