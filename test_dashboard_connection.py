
import time
import json
import os
from alert_service import handle_alert, save_location_to_cache

def test_dashboard_to_alert_connection():
    print("🚀 SITUATION: Simulating Frontend Dashboard GPS update...")

    # 1. Simulate the HTML Dashboard sending a specific GPS location (e.g., Madurai)
    test_lat = "9.9252"
    test_lon = "78.1198"

    print(f"📡 Step 1: Updating cache with Dashboard CSV: {test_lat}, {test_lon}")
    save_location_to_cache(test_lat, test_lon)

    print("⏳ Waiting for cache to settle...")
    time.sleep(1)

    # 2. Trigger an alert with "default" arguments to see if it picks up the Dashboard GPS
    print("\n🚨 Step 2: Triggering manual alert with 'None' coordinates...")
    print("   (The system should now PREFER the Dashboard GPS we just saved)")

    # We pass None for lat/lon so it's forced to look at the cache/dashboard
    success = handle_alert("Fighting", 0.95, latitude=None, longitude=None)

    if success:
        print("\n✅ SUCCESS: Alert system processed the request.")
        print(f"📝 Check your WhatsApp! It should show location: {test_lat}, {test_lon}")
        print("🔗 Look for 'Live Dashboard: main_working.html' in the message.")
    else:
        print("\n❌ FAILED: Alert was not sent. Check console logs for errors.")

if __name__ == "__main__":
    test_dashboard_to_alert_connection()
