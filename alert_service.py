"""
AI-Based Real-Time Violence Detection and Automated Alert System
Alert Service Module

This module provides SMS alert functionality for violence and weapon detection.
It integrates with the existing detection system without modifying core functionality.

Author: Final Year Engineering Project
"""

import requests
import os
import time
from datetime import datetime
from typing import Tuple, Optional
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


# Import WhatsApp module
try:
    from whatsapp_alert import send_whatsapp
except ImportError:
    send_whatsapp = None


# Configuration constants
CONFIDENCE_THRESHOLD = 0.1  # Minimum confidence for triggering alerts (lowered for testing)
ALERT_COOLDOWN = 0  # Seconds between alerts (disabled for testing)
DEFAULT_LATITUDE = "8.7139"  # Tirunelveli default coordinates
DEFAULT_LONGITUDE = "77.7567"
CHENNAI_LAT = "13.0827" # Common IP Geolocation fallback for India
CHENNAI_LON = "80.2707"
LOCATION_CACHE_FILE = "location_cache.json"

def save_location_to_cache(lat: str, lon: str):
    """Save location to a local file cache for cross-module persistence. Always overwrites old cache."""
    try:
        with open(LOCATION_CACHE_FILE, 'w') as f:
            json.dump({'lat': str(lat), 'lon': str(lon), 'timestamp': time.time()}, f)
        print(f"📂 Location cached to file (overwritten): {lat}, {lon}")
    except Exception as e:
        print(f"❌ Error caching location: {e}")

def get_cached_location() -> Optional[Tuple[str, str, float]]:
    """Retrieve location and timestamp from the local file cache."""
    try:
        if os.path.exists(LOCATION_CACHE_FILE):
            with open(LOCATION_CACHE_FILE, 'r') as f:
                data = json.load(f)
                return data['lat'], data['lon'], data.get('timestamp', 0)
    except Exception as e:
        print(f"❌ Error reading location cache: {e}")
    return None

def resolve_coordinates(provided_lat: Optional[str], provided_lon: Optional[str]) -> Tuple[str, str]:
    """
    Resolve the best available coordinates prioritizing real-time GPS from frontend.
    Priority:
    1. Fresh cache (from Frontend/HTML) - within 5 minutes
    2. Provided coordinates if they are NOT defaults
    3. Old cache if available
    4. Fallback/IP-based location
    """
    cached = get_cached_location()

    # 1. Fresh cache check
    if cached:
        c_lat, c_lon, c_time = cached
        # Is it a real GPS or just the Chennai IP-pool default?
        is_chennai = str(c_lat).startswith("13.082") and str(c_lon).startswith("80.270")

        # If it's a fresh (5m) Non-Chennai coordinate, use it!
        if time.time() - c_time < 300 and not is_chennai:
            # print(f"📍 Using fresh GPS from Frontend: {c_lat}, {c_lon}")
            return str(c_lat), str(c_lon)

    # 2. Provided coordinates check
    if provided_lat and provided_lon:
        # Check if they are NOT the Tirunelveli defaults or Chennai defaults
        p_lat_s, p_lon_s = str(provided_lat), str(provided_lon)
        is_default = (p_lat_s == DEFAULT_LATITUDE and p_lon_s == DEFAULT_LONGITUDE)
        is_chennai = p_lat_s.startswith("13.082") and p_lon_s.startswith("80.270")

        if not is_default and not is_chennai:
            # print(f"📍 Using provided real-time coordinates: {provided_lat}, {provided_lon}")
            return str(provided_lat), str(provided_lon)

    # 3. If we have ANY cache that isn't Chennai, even if old, it's probably better than Chennai IP
    if cached:
        c_lat, c_lon, c_time = cached
        is_chennai = str(c_lat).startswith("13.082") and str(c_lon).startswith("80.270")
        if not is_chennai:
            # print(f"📍 Using (possibly old) real GPS from Cache: {c_lat}, {c_lon}")
            return str(c_lat), str(c_lon)

    # 4. Fallback to get_location (which checks IP)
    return get_location()



# Global variables for alert management
_last_alert_time = 0  # Track last alert timestamp
_last_dashboard_alert_time = 0 # Track last dashboard sync alert
_alert_enabled = True  # Master switch for alerts
_gps_server_thread = None # Storage for the background GPS server
DASHBOARD_ALERT_COOLDOWN = 60 # Cooldown for dashboard sync WhatsApps (1 minute)





# WhatsApp Configuration (UltraMsg)
WHATSAPP_INSTANCE_ID = os.getenv('WHATSAPP_INSTANCE_ID', 'instance164765')
WHATSAPP_TOKEN = os.getenv('WHATSAPP_TOKEN', '')
WHATSAPP_RECIPIENT = os.getenv('WHATSAPP_RECIPIENT', '6382941185') # Recipient phone number



def get_location() -> Tuple[str, str]:
    """
    Fetch current latitude and longitude using IP-based geolocation.

    This function uses multiple geolocation services for reliability:
    1. Primary: ipinfo.io
    2. Fallback: ip-api.com
    3. Default: Tirunelveli coordinates

    Returns:
        tuple: (latitude, longitude) as strings

    Example:
        >>> lat, lon = get_location()
        >>> print(f"Location: {lat}, {lon}")
    """
    try:
        # 1. Try file cache first. We increase expiration to 1 hour (3600s)
        # so manual GPS coordinates stick throughout a monitoring session.
        try:
            cached_data = get_cached_location()
            if cached_data:
                lat, lon, timestamp = cached_data
                age = time.time() - timestamp
                if age < 3600: # 1 hour threshold
                    # print(f"✓ Using RECENT cached coordinates (Age: {age:.1f}s): {lat}, {lon}")
                    return lat, lon
                else:
                    print(f"ℹ️ Cached coordinates expired (Age: {age:.1f}s). Fetching fresh location...")
        except Exception as e:
            print(f"⚠ Cache check failed: {e}")


        # 2. Try primary service: ipinfo.io
        try:
            response = requests.get('https://ipinfo.io/json', timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('loc'):
                    lat, lon = data['loc'].split(',')
                    print(f"✓ Fresh location obtained from ipinfo.io: {lat}, {lon}")
                    # Update cache with fresh IP-based location
                    save_location_to_cache(lat, lon)
                    return lat, lon
        except Exception as e:
            print(f"⚠ ipinfo.io failed: {e}")


        # 3. Try fallback service: ip-api.com
        try:
            response = requests.get('http://ip-api.com/json', timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('lat') and data.get('lon'):
                    lat, lon = str(data['lat']), str(data['lon'])
                    print(f"✓ Fresh location obtained from ip-api.com: {lat}, {lon}")
                    # Update cache with fresh IP-based location
                    save_location_to_cache(lat, lon)
                    return lat, lon
        except Exception as e:
            print(f"⚠ ip-api.com failed: {e}")


        # Use default coordinates if all services fail
        print(f"⚠ All location services failed, using default: {DEFAULT_LATITUDE}, {DEFAULT_LONGITUDE}")
        return DEFAULT_LATITUDE, DEFAULT_LONGITUDE

    except Exception as e:
        print(f"❌ Critical error in get_location(): {e}")
        return DEFAULT_LATITUDE, DEFAULT_LONGITUDE


def send_sms_alert(event_type: str, latitude: str, longitude: str) -> bool:
    """
    Send SMS alert using Twilio API when violence or weapon is detected.

    Args:
        event_type: Type of event (e.g., "Fighting", "Weapon", "Assault")
        latitude: Latitude coordinate
        longitude: Longitude coordinate

    Returns:
        bool: True if SMS sent successfully, False otherwise

    Example:
        >>> success = send_sms_alert("Fighting", "8.7139", "77.7567")
        >>> if success:
        ...     print("Alert sent successfully")
    """
    # Check if Twilio credentials are configured






def send_whatsapp_alert(event_type: str, latitude: str, longitude: str) -> bool:
    """
    Send WhatsApp alert using UltraMsg API when violence or weapon is detected.

    Args:
        event_type: Type of event (e.g., "Fighting", "Weapon", "Assault")
        latitude: Latitude coordinate
        longitude: Longitude coordinate

    Returns:
        bool: True if WhatsApp sent successfully, False otherwise
    """
    # Create professional alert message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine location source label for the message
    is_gps = False
    cached = get_cached_location()
    if cached and latitude == cached[0] and longitude == cached[1]:
        is_gps = True

    loc_type = "📍 Precise GPS Location" if is_gps else "🌐 Estimated IP Location"

    message_body = (
        f"🚨 VIOLENCE DETECTION ALERT 🚨\n\n"
        f"Event: {event_type}\n"
        f"Time: {timestamp}\n"
        f"{loc_type}: {latitude}, {longitude}\n"
        f"Maps: https://maps.google.com/?q={latitude},{longitude}\n\n"
        f"Live Dashboard: main_working.html\n"
        f"This is an automated security alert. Please respond immediately."
    )

    if send_whatsapp:
        try:
            result = send_whatsapp(message_body, WHATSAPP_RECIPIENT)
            print(f"✓ WhatsApp alert response: {result}")
            return "true" in result.lower() or "ok" in result.lower() or "success" in result.lower() or "sent" in result.lower()
        except Exception as e:
            print(f"❌ Error using whatsapp_alert module: {e}")
            # Fall through to manual implementation

    # Fallback/Manual implementation
    if not all([WHATSAPP_INSTANCE_ID, WHATSAPP_TOKEN, WHATSAPP_RECIPIENT]):
        print("❌ WhatsApp (UltraMsg) credentials not configured.")
        return False

    try:
        url = f"https://api.ultramsg.com/{WHATSAPP_INSTANCE_ID}/messages/chat"
        payload = {
            "token": WHATSAPP_TOKEN,
            "to": WHATSAPP_RECIPIENT,
            "body": message_body
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=payload, headers=headers, timeout=15)

        if response.status_code == 200:
            print(f"✓ WhatsApp alert sent successfully! Response: {response.text}")
            return True
        else:
            print(f"❌ Failed to send WhatsApp alert. Status: {response.status_code}, Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error sending WhatsApp alert: {e}")
        return False



def is_violent_event(event_type: str) -> bool:

    """
    Check if the event type qualifies as a violent/weapon event.

    Args:
        event_type: The detected event type

    Returns:
        bool: True if event should trigger alert, False otherwise
    """
    # Define violent and weapon-related events
    violent_events = {
        'fighting', 'assault', 'abuse', 'arrest', 'burglary',
        'weapon', 'knife', 'gun', 'pistol', 'rifle', 'weapon_detected'
    }

    return event_type.lower() in violent_events


def handle_multi_person_alert(person_count: int, latitude: str = None, longitude: str = None) -> bool:
    """
    Send SMS alert when multiple persons are detected (crowd/crowding alert).

    Args:
        person_count: Number of persons detected
        latitude: Latitude coordinate (optional, will fetch if not provided)
        longitude: Longitude coordinate (optional, will fetch if not provided)

    Returns:
        bool: True if alert was sent, False otherwise

    Example:
        >>> if person_count > 1:
        ...     handle_multi_person_alert(person_count, lat, lon)
    """
    try:
        # Check if alert system is enabled
        if not _alert_enabled:
            print("🔕 Alert system is disabled")
            return False

        # Only trigger for more than one person
        if person_count <= 1:
            return False

        # Check cooldown period
        if not check_alert_cooldown():
            return False

        # Resolve best coordinates
        latitude, longitude = resolve_coordinates(latitude, longitude)
        print(f"📍 Location for multi-person alert: {latitude}, {longitude}")

        # Update cache with resolved coordinates
        save_location_to_cache(latitude, longitude)

        # Send SMS alert for multi-person detection
        event_type = f"Multiple Persons ({person_count})"
        print(f"🚨 Sending multi-person alert: {person_count} persons detected")

        # Send SMS
        sms_success = send_sms_alert(event_type, latitude, longitude)

        # Send WhatsApp
        whatsapp_success = send_whatsapp_alert(event_type, latitude, longitude)


        if sms_success or whatsapp_success:
            print(f"✅ Multi-person alert sent (SMS: {sms_success}, WhatsApp: {whatsapp_success})")
            return True
        else:
            print(f"❌ Failed to send multi-person alerts")
            return False


    except Exception as e:
        print(f"❌ Error in handle_multi_person_alert(): {e}")
        return False


def check_alert_cooldown() -> bool:
    """
    Check if enough time has passed since the last alert to prevent spam.

    Returns:
        bool: True if cooldown period has passed, False otherwise
    """
    global _last_alert_time
    current_time = time.time()

    if current_time - _last_alert_time >= ALERT_COOLDOWN:
        _last_alert_time = current_time
        return True

    remaining_time = ALERT_COOLDOWN - (current_time - _last_alert_time)
    print(f"⏰ Alert cooldown active. {remaining_time:.0f} seconds remaining.")
    return False


def handle_alert(event_type: str, confidence_score: float, latitude: str = None, longitude: str = None) -> bool:
    """
    Main alert handler function - call this from your detection loop.


    This function checks all conditions and sends SMS alert if warranted:
    1. Check if alerts are enabled
    2. Verify event is violent/weapon related
    3. Check confidence threshold
    4. Verify cooldown period
    5. Fetch location and send SMS

    Args:
        event_type: Type of event detected (e.g., "Fighting", "Normal")
        confidence_score: Confidence score from detection (0.0 to 1.0)
        latitude: Optional latitude (uses IP-based if not provided)
        longitude: Optional longitude (uses IP-based if not provided)

    Returns:
        bool: True if alert was sent, False otherwise

    Example:
        >>> # In your detection loop
        >>> if handle_alert(prediction_label, confidence, lat, lon):
        ...     print("Alert sent!")
        >>> else:
        ...     print("No alert needed")
    """
    try:
        # Check if alert system is enabled
        if not _alert_enabled:
            print("🔕 Alert system is disabled")
            return False

        # Check if event is violent/weapon related
        if not is_violent_event(event_type):
            print(f"ℹ️ Non-violent event detected: {event_type} - No alert needed")
            return False

        # Check confidence threshold
        if confidence_score < CONFIDENCE_THRESHOLD:
            print(f"📊 Confidence too low: {confidence_score:.3f} < {CONFIDENCE_THRESHOLD}")
            return False

        # Check cooldown period
        if not check_alert_cooldown():
            return False

        # Resolve best coordinates
        latitude, longitude = resolve_coordinates(latitude, longitude)
        print(f"📍 Location for alert: {latitude}, {longitude}")

        # Sync the chosen coordinates back to cache for consistency
        save_location_to_cache(latitude, longitude)



        # Send alerts

        print(f"🚨 Sending alert for {event_type} (confidence: {confidence_score:.3f})")

        # Send SMS
        sms_success = send_sms_alert(event_type, latitude, longitude)

        # Send WhatsApp
        whatsapp_success = send_whatsapp_alert(event_type, latitude, longitude)

        if sms_success or whatsapp_success:
            print(f"✅ Alert successfully sent (SMS: {sms_success}, WhatsApp: {whatsapp_success})")
            return True
        else:
            print(f"❌ Failed to send alerts for {event_type}")
            return False

    except Exception as e:
        print(f"❌ Critical error in handle_alert(): {e}")
        return False


def send_dashboard_alert(status: str, latitude: str, longitude: str) -> bool:
    """
    Send a WhatsApp alert for dashboard synchronization/status updates.
    Includes a cooldown to prevent spamming the user's WhatsApp.
    """
    global _last_dashboard_alert_time
    current_time = time.time()

    # Don't send if system disabled
    if not _alert_enabled:
        return False

    # Skip if it's a violent event - those are handled by the main handle_alert function
    # to avoid double-notifying the user
    if is_violent_event(status):
        return False

    # Apply cooldown to DASHBOARD alerts (Normal/System updates) to avoid spam
    if current_time - _last_dashboard_alert_time < DASHBOARD_ALERT_COOLDOWN:
        return False

    # Resolve best coordinates
    latitude, longitude = resolve_coordinates(latitude, longitude)

    # Determine location source label for the message
    is_gps = False
    cached = get_cached_location()
    if cached and latitude == str(cached[0]) and longitude == str(cached[1]):
        is_gps = True

    loc_type = "📍 Precise GPS Location" if is_gps else "🌐 Estimated IP Location"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message_body = (
        f"📊 DASHBOARD SYNC UPDATE 📊\n\n"
        f"Status: {status}\n"
        f"Time: {timestamp}\n"
        f"{loc_type}: {latitude}, {longitude}\n"
        f"Maps: https://maps.google.com/?q={latitude},{longitude}\n\n"
        f"Live Dashboard: main_working.html"
    )

    if send_whatsapp:
        try:
            return send_whatsapp(message_body, WHATSAPP_RECIPIENT)
        except Exception as e:
            print(f"❌ WhatsApp sync alert failed: {e}")
            return False
    return False


def enable_alerts():

    """Enable the alert system."""
    global _alert_enabled
    _alert_enabled = True
    print("✅ Alert system enabled")


def disable_alerts():
    """Disable the alert system."""
    global _alert_enabled
    _alert_enabled = False
    print("🔕 Alert system disabled")


def is_alert_enabled() -> bool:
    """Check if alert system is currently enabled."""
    return _alert_enabled


def set_confidence_threshold(threshold: float):
    """
    Set the minimum confidence threshold for triggering alerts.

    Args:
        threshold: Confidence threshold between 0.0 and 1.0
    """
    global CONFIDENCE_THRESHOLD
    if 0.0 <= threshold <= 1.0:
        CONFIDENCE_THRESHOLD = threshold
        print(f"✅ Confidence threshold set to {threshold}")
    else:
        print(f"❌ Invalid threshold: {threshold}. Must be between 0.0 and 1.0")


def set_alert_cooldown(seconds: int):
    """
    Set the cooldown period between alerts to prevent spam.

    Args:
        seconds: Number of seconds between alerts
    """
    global ALERT_COOLDOWN
    if seconds >= 0:
        ALERT_COOLDOWN = seconds
        print(f"✅ Alert cooldown set to {seconds} seconds")
    else:
        print(f"❌ Invalid cooldown: {seconds}. Must be non-negative")


def test_alert_system():
    """
    Test the complete alert system with a simulated event.

    Returns:
        bool: True if test passes, False otherwise
    """
    print("🧪 Starting alert system test...")

    try:
        # Test location service
        print("1. Testing location service...")
        lat, lon = get_location()
        print(f"   ✓ Location: {lat}, {lon}")

        # Test event classification
        print("2. Testing event classification...")
        assert is_violent_event("Fighting") == True
        assert is_violent_event("Normal") == False
        print("   ✓ Event classification working")

        # Test confidence threshold
        print("3. Testing confidence threshold...")
        set_confidence_threshold(0.7)
        print(f"   ✓ Threshold set to {CONFIDENCE_THRESHOLD}")

        # Test cooldown
        print("4. Testing cooldown system...")
        set_alert_cooldown(60)
        print(f"   ✓ Cooldown set to {ALERT_COOLDOWN} seconds")
        # Test SMS configuration (without sending)


        # Test WhatsApp configuration
        print("6. Testing WhatsApp configuration...")
        if not all([WHATSAPP_INSTANCE_ID, WHATSAPP_TOKEN, WHATSAPP_RECIPIENT]):
            print("   ⚠ WhatsApp not configured - check WHATSAPP_ environment variables")
        else:
            print("   ✓ WhatsApp credentials configured")


        print("✅ Alert system test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Alert system test failed: {e}")
        return False


def get_alert_status() -> dict:
    """
    Get current status of the alert system.

    Returns:
        dict: Current configuration and status
    """
    return {
        'enabled': _alert_enabled,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'cooldown_seconds': ALERT_COOLDOWN,
        'last_alert_time': _last_alert_time,
        'sms_configured': all([os.getenv('TWILIO_ACCOUNT_SID'), os.getenv('TWILIO_AUTH_TOKEN'), os.getenv('TWILIO_PHONE_NUMBER'), os.getenv('ALERT_PHONE_NUMBER')]),
        'whatsapp_configured': all([WHATSAPP_INSTANCE_ID, WHATSAPP_TOKEN, WHATSAPP_RECIPIENT]),
        'default_location': f"{DEFAULT_LATITUDE}, {DEFAULT_LONGITUDE}"

    }



# ================= HTTP GPS RECEIVER ===================

class GPSReceiverHandler(BaseHTTPRequestHandler):
    """Handles POST requests from frontend to update coordinates."""

    def log_message(self, format, *args):
        # Suppress server logging to keep terminal clean
        return

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        """Handle GPS coordinates update via POST."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)

            lat = data.get('lat')
            lon = data.get('lon')

            if lat and lon:
                print(f"🛰️ [GPS RECEIVER] Sync from Dashboard: {lat}, {lon}")
                save_location_to_cache(str(lat), str(lon))

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "lat": lat, "lon": lon}).encode())
            else:
                self.send_response(400)
                self.end_headers()
        except Exception as e:
            print(f"❌ Error in GPS receiver: {e}")
            self.send_response(500)
            self.end_headers()

def start_gps_server(port=5000):
    """Start the GPS receiver server in a background thread."""
    global _gps_server_thread

    if _gps_server_thread is not None and _gps_server_thread.is_alive():
        return

    def _run_server():
        try:
            server_address = ('', port)
            httpd = HTTPServer(server_address, GPSReceiverHandler)
            print(f"🛰️ GPS Receiver active on port {port} (Listening for Frontend updates)")
            httpd.serve_forever()
        except Exception as e:
            if "Address already in use" in str(e):
                # Silently fail if port already taken (likely by our other app instance)
                pass
            else:
                print(f"⚠ GPS Server error: {e}")

    _gps_server_thread = threading.Thread(target=_run_server, daemon=True)
    _gps_server_thread.start()

# Module initialization

if __name__ == "__main__":
    print("=" * 60)
    print("AI-BASED VIOLENCE DETECTION ALERT SERVICE")
    print("=" * 60)
    print("\n🔧 Testing alert system...")
    test_alert_system()

    print("\n📊 Current alert status:")
    status = get_alert_status()
    for key, value in status.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")

    print("\n📝 Usage Instructions:")
    print("1. Set Twilio environment variables:")
    print("   export TWILIO_ACCOUNT_SID='your_account_sid'")
    print("   export TWILIO_AUTH_TOKEN='your_auth_token'")
    print("   export TWILIO_PHONE_NUMBER='+1234567890'")
    print("   export ALERT_PHONE_NUMBER='+0987654321'")
    print("\n2. In your detection loop, call:")
    print("   from alert_service import handle_alert")
    print("   handle_alert(prediction_label, confidence_score)")
    print("\n" + "=" * 60)
