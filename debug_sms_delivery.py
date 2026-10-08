#!/usr/bin/env python3
"""
Check Twilio SMS delivery status
"""

from twilio.rest import Client
from alert_service import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, ALERT_PHONE_NUMBER

def check_sms_status(message_sid):
    """Check the delivery status of a sent message"""
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages(message_sid).fetch()

        print(f"📱 Message SID: {message.sid}")
        print(f"📊 Status: {message.status}")
        print(f"📤 To: {message.to}")
        print(f"📥 From: {message.from_}")
        print(f"📝 Body: {message.body[:50]}...")
        print(f"💰 Price: {message.price}")
        print(f"🌍 Error Code: {message.error_code}")
        print(f"❌ Error Message: {message.error_message}")

        return message.status
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return None

def send_and_check_test():
    """Send a test SMS and check its status"""
    from alert_service import send_sms_alert, get_location

    print("🚀 Sending test SMS...")
    lat, lon = get_location()

    # Get the client and send message
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    try:
        message = client.messages.create(
            body=f"🚨 TEST ALERT\n\nEvent: TEST\nLocation: {lat}, {lon}\nTime: Test message",
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )

        print(f"✅ Message sent with SID: {message.sid}")
        print(f"📊 Initial Status: {message.status}")

        # Check status after a few seconds
        import time
        time.sleep(2)

        print(f"\n🔍 Checking delivery status...")
        updated_message = client.messages(message.sid).fetch()
        print(f"📊 Updated Status: {updated_message.status}")

        if updated_message.status in ['delivered', 'sent']:
            print("✅ Message delivered successfully!")
        elif updated_message.status == 'failed':
            print(f"❌ Message failed: {updated_message.error_message}")
        else:
            print(f"⏳ Message status: {updated_message.status}")

        return updated_message.status

    except Exception as e:
        print(f"❌ Failed to send: {e}")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("TWILIO SMS DELIVERY DEBUG")
    print("=" * 60)

    # Send test and check status
    status = send_and_check_test()

    print(f"\n📋 Possible reasons for non-delivery:")
    print("1. Twilio trial account - verify your account")
    print("2. Phone number format - should be +91XXXXXXXXXX")
    print("3. Carrier filtering - some carriers block unknown senders")
    print("4. Phone settings - check spam/DND settings")
    print("5. Twilio restrictions for India - may need verification")
    print(f"\nYour number: {ALERT_PHONE_NUMBER}")
