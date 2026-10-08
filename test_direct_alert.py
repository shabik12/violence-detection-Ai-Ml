#!/usr/bin/env python3
"""
Direct SMS Alert Test - Bypass the app to test alerts directly
"""

import os
from alert_service import handle_alert, get_alert_status, send_sms_alert, get_location

def test_direct_sms():
    print("🧪 DIRECT SMS ALERT TEST")
    print("=" * 40)

    # Check configuration
    status = get_alert_status()
    print(f"Alert Enabled: {status['enabled']}")
    print(f"SMS Configured: {status['sms_configured']}")
    print(f"Threshold: {status['confidence_threshold']}")
    print(f"Cooldown: {status['cooldown_seconds']}s")

    # Test 1: Direct SMS
    print("\n1. Testing Direct SMS...")
    lat, lon = get_location()
    result = send_sms_alert("DIRECT TEST", lat, lon)
    print(f"Direct SMS Result: {result}")

    # Test 2: Handle Alert with Fighting
    print("\n2. Testing Handle Alert with Fighting...")
    result = handle_alert("Fighting", 0.85)
    print(f"Handle Alert Result: {result}")

    # Test 3: Handle Alert with Assault
    print("\n3. Testing Handle Alert with Assault...")
    result = handle_alert("Assault", 0.75)
    print(f"Handle Alert Result: {result}")

    # Test 4: Handle Alert with Normal (should not send)
    print("\n4. Testing Handle Alert with Normal (should not send)...")
    result = handle_alert("Normal", 0.95)
    print(f"Handle Alert Result: {result}")

    print("\n✅ Direct SMS Test Complete!")
    print("Check your phone for SMS messages.")

if __name__ == "__main__":
    test_direct_sms()
