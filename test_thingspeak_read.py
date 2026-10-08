#!/usr/bin/env python3
"""
Test ThingSpeak Data Reading
"""

import urllib.request
import json

def test_thingspeak_read():
    """Test reading data from ThingSpeak"""
    print("📖 Testing ThingSpeak Data Reading")
    print("=" * 50)

    channel_id = "2374983"

    try:
        # Read recent data
        url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json?results=10"
        print(f"📥 Reading from: {url}")

        response = urllib.request.urlopen(url, timeout=10)
        data = json.loads(response.read().decode())

        print(f"✅ Successfully read data!")
        print(f"   Channel ID: {data.get('channel', {}).get('id', 'N/A')}")
        print(f"   Channel Name: {data.get('channel', {}).get('name', 'N/A')}")
        print(f"   Total Feeds: {len(data.get('feeds', []))}")

        # Display recent feeds
        feeds = data.get('feeds', [])
        if feeds:
            print("\n📊 Recent Feeds:")
            for i, feed in enumerate(feeds[:5], 1):
                timestamp = feed.get('created_at', 'N/A')
                field2 = feed.get('field2', 'N/A')
                field3 = feed.get('field3', 'N/A')
                field4 = feed.get('field4', 'N/A')

                print(f"   {i}. {timestamp}")
                print(f"      Status: {field2}")
                print(f"      Lat: {field3}, Lon: {field4}")
        else:
            print("⚠️ No feeds found")

        return True

    except urllib.error.HTTPError as e:
        if e.code == 400:
            print("❌ HTTP 400: Channel may be private or invalid")
        elif e.code == 404:
            print("❌ HTTP 404: Channel not found")
        else:
            print(f"❌ HTTP {e.code}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading data: {e}")
        return False

def test_dashboard_html():
    """Test if dashboard HTML can access the data"""
    print("\n🌐 Testing Dashboard HTML Access")
    print("=" * 50)

    # Create a simple HTML test
    html_test = """
<!DOCTYPE html>
<html>
<head>
    <title>ThingSpeak Test</title>
</head>
<body>
    <h1>ThingSpeak Data Test</h1>
    <div id="status">Loading...</div>
    <div id="data"></div>

    <script>
        async function testData() {
            try {
                const response = await fetch('https://api.thingspeak.com/channels/2374983/feeds.json?results=5');
                const data = await response.json();

                document.getElementById('status').innerHTML = '✅ Connected successfully!';

                let html = '<h2>Recent Data:</h2>';
                data.feeds.forEach((feed, index) => {
                    html += `<p>${index + 1}. ${feed.created_at}: ${feed.field2 || 'N/A'} at (${feed.field3 || 'N/A'}, ${feed.field4 || 'N/A'})</p>`;
                });

                document.getElementById('data').innerHTML = html;
                console.log('Data loaded:', data);

            } catch (error) {
                document.getElementById('status').innerHTML = `❌ Error: ${error.message}`;
                console.error('Error:', error);
            }
        }

        testData();
        setInterval(testData, 5000); // Test every 5 seconds
    </script>
</body>
</html>
    """

    with open('test_dashboard.html', 'w') as f:
        f.write(html_test)

    print("✅ Created test_dashboard.html")
    print("💡 Open this file in browser to test ThingSpeak access")
    print("📝 It will show if the dashboard can read data from ThingSpeak")

if __name__ == "__main__":
    print("🧪 ThingSpeak Dashboard Test Suite")
    print("=" * 60)

    # Test reading data
    read_works = test_thingspeak_read()

    # Create HTML test
    test_dashboard_html()

    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"• ThingSpeak Read: {'✅ Working' if read_works else '❌ Failed'}")
    print("• HTML Test: ✅ Created test_dashboard.html")

    if read_works:
        print("\n🎉 ThingSpeak is accessible! Check main.html for any JavaScript issues.")
    else:
        print("\n⚠️ ThingSpeak access failed. Dashboard won't show data.")
