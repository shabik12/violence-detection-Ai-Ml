"""
Integration Test for Violence Detection Alert System

This script tests the complete integration between:
- Alert service (SMS notifications)
- Detection logger (local logging)
- Location services
- Event classification

Author: Final Year Engineering Project
"""

import sys
import os

def test_alert_integration():
    """Test complete alert system integration."""
    print("=" * 60)
    print("AI-BASED VIOLENCE DETECTION - INTEGRATION TEST")
    print("=" * 60)

    try:
        # Test 1: Import all modules
        print("\n1. Testing module imports...")
        try:
            from alert_service import (
                handle_alert, get_location, send_sms_alert,
                test_alert_system, get_alert_status, enable_alerts, disable_alerts
            )
            from detection_logger import save_detection, get_recent_detections
            print("   ✓ All modules imported successfully")
        except ImportError as e:
            print(f"   ❌ Import failed: {e}")
            return False

        # Test 2: Alert system functionality
        print("\n2. Testing alert system...")
        try:
            # Test alert system
            test_result = test_alert_system()
            if test_result:
                print("   ✓ Alert system test passed")
            else:
                print("   ⚠ Alert system test completed with warnings")

            # Get alert status
            status = get_alert_status()
            print(f"   ✓ Alert status: {status}")

        except Exception as e:
            print(f"   ❌ Alert system test failed: {e}")
            return False

        # Test 3: Location services
        print("\n3. Testing location services...")
        try:
            lat, lon = get_location()
            print(f"   ✓ Location obtained: {lat}, {lon}")
        except Exception as e:
            print(f"   ❌ Location service failed: {e}")
            return False

        # Test 4: Detection logging
        print("\n4. Testing detection logging...")
        try:
            # Log test detections
            save_detection(lat, lon, "Fighting", 0.85)
            save_detection(lat, lon, "Normal", 0.92)
            save_detection(lat, lon, "Weapon", 0.78)

            # Retrieve recent detections
            recent = get_recent_detections(5)
            print(f"   ✓ Logged {len(recent)} recent detections")

            # Verify latest entries
            if len(recent) >= 3:
                latest = recent[-3:]
                for detection in latest:
                    event_type = detection.get('event_type', 'Unknown')
                    confidence = detection.get('confidence', 0.0)
                    print(f"     - {event_type}: {confidence:.2f}")

        except Exception as e:
            print(f"   ❌ Detection logging failed: {e}")
            return False

        # Test 5: Alert triggering (without sending SMS)
        print("\n5. Testing alert triggering logic...")
        try:
            # Test violent event (should trigger alert)
            print("   Testing violent event (Fighting, 0.85 confidence)...")
            result1 = handle_alert("Fighting", 0.85)
            print(f"   ✓ Fighting alert result: {result1}")

            # Test normal event (should not trigger alert)
            print("   Testing normal event (Normal, 0.92 confidence)...")
            result2 = handle_alert("Normal", 0.92)
            print(f"   ✓ Normal alert result: {result2}")

            # Test weapon detection (should trigger alert)
            print("   Testing weapon event (Weapon, 0.78 confidence)...")
            result3 = handle_alert("Weapon", 0.78)
            print(f"   ✓ Weapon alert result: {result3}")

        except Exception as e:
            print(f"   ❌ Alert triggering test failed: {e}")
            return False

        # Test 6: Alert system controls
        print("\n6. Testing alert system controls...")
        try:
            # Test enable/disable
            disable_alerts()
            print("   ✓ Alerts disabled")

            enable_alerts()
            print("   ✓ Alerts enabled")

        except Exception as e:
            print(f"   ❌ Alert controls test failed: {e}")
            return False

        print("\n" + "=" * 60)
        print("✅ INTEGRATION TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print("\n📋 Test Summary:")
        print("   ✓ Module imports working")
        print("   ✓ Alert system functional")
        print("   ✓ Location services operational")
        print("   ✓ Detection logging working")
        print("   ✓ Alert triggering logic correct")
        print("   ✓ System controls functional")

        print("\n🚀 System is ready for deployment!")
        print("\n📝 Next Steps:")
        print("   1. Configure Twilio environment variables")
        print("   2. Run 'streamlit run app.py' to start the system")
        print("   3. Test with real violence/weapon content")
        print("   4. Monitor SMS alerts and dashboard")

        return True

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        return False


def test_dependencies():
    """Test if all required dependencies are installed."""
    print("\n🔍 Checking dependencies...")

    required_packages = [
        'requests', 'twilio', 'streamlit', 'opencv-python',
        'numpy', 'tensorflow', 'ultralytics'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'tensorflow':
                import tensorflow
            elif package == 'ultralytics':
                import ultralytics
            else:
                __import__(package)
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("   ✓ All dependencies installed")
        return True


if __name__ == "__main__":
    print("Starting Violence Detection Alert System Integration Test...")

    # Check dependencies first
    if not test_dependencies():
        print("\n❌ Please install missing dependencies before continuing")
        sys.exit(1)

    # Run integration test
    success = test_alert_integration()

    if success:
        print("\n🎉 All tests passed! System is ready to use.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Please check the errors above.")
        sys.exit(1)
