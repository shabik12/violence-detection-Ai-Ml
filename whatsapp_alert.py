import requests
import json
import os
from datetime import datetime

# Default configuration (can be overridden by environment variables)
INSTANCE_ID = os.getenv('WHATSAPP_INSTANCE_ID', 'instance164765')
TOKEN = os.getenv('WHATSAPP_TOKEN', '')
RECIPIENT = os.getenv('WHATSAPP_RECIPIENT', '6382941185')

def send_whatsapp(message_body, recipient=None):
    """
    Send a WhatsApp message using UltraMsg API.
    """
    to_number = recipient if recipient else RECIPIENT
    url = f"https://api.ultramsg.com/{INSTANCE_ID}/messages/chat"
    
    payload = {
        "token": TOKEN,
        "to": to_number,
        "body": message_body
    }
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        return response.text
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    # Test script
    test_message = f"Test WhatsApp message from Violence Detection System at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    print(f"Sending test message to {RECIPIENT}...")
    result = send_whatsapp(test_message)
    print(f"Result: {result}")