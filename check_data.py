#!/usr/bin/env python3
"""
Quick ThingSpeak Data Check
"""

import urllib.request
import json

def check_thingspeak_data():
    """Check if data is available on ThingSpeak"""
    print("🔍 Checking ThingSpeak Data")
    print("=" * 40)

    try:
        # Read latest data
        url = "https://api.thingspeak.com/channels/2374983/feeds.json?results=5"
        print(f"📥 Reading from: {url}")

        response = urllib.request.urlopen(url, timeout=10)
        data = json.loads(response.read().decode())

        print(f"✅ Channel ID: {data.get('channel', {}).get('id', 'N/A')}")
        print(f"✅ Channel Name: {data.get('channel', {}).get('name', 'N/A')}")
        print(f"✅ Total Feeds: {len(data.get('feeds', []))}")

        feeds = data.get('feeds', [])
        if feeds:
            print("\n📊 Latest 5 Entries:")
            for i, feed in enumerate(feeds, 1):
                timestamp = feed.get('created_at', 'N/A')
                field2 = feed.get('field2', 'N/A')
                field3 = feed.get('field3', 'N/A')
                field4 = feed.get('field4', 'N/A')

                print(f"   {i}. {timestamp}")
                print(f"      Status: {field2}")
                print(f"      Location: {field3}, {field4}")
        else:
            print("⚠️ No feeds found")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    check_thingspeak_data()
