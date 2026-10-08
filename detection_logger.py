"""
Detection Logger Module

This module handles logging detection events to maintain compatibility
with the existing HTML dashboard system without using a database.

Author: Final Year Engineering Project
"""

import os
import json
from datetime import datetime
from typing import Optional


def save_detection(latitude: str, longitude: str, event_type: str, confidence: float):
    """
    Save detection event to maintain compatibility with existing system.

    This function creates a simple JSON log file for detection events.
    The existing HTML dashboard reads from ThingSpeak, so this function
    primarily serves as a local backup and debugging tool.

    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        event_type: Type of event detected
        confidence: Confidence score (0.0 to 1.0)

    Example:
        >>> save_detection("8.7139", "77.7567", "Fighting", 0.85)
    """
    try:
        # Create detections directory if it doesn't exist
        detections_dir = "detections"
        if not os.path.exists(detections_dir):
            os.makedirs(detections_dir)

        # Create detection entry
        detection_entry = {
            "timestamp": datetime.now().isoformat(),
            "latitude": latitude,
            "longitude": longitude,
            "event_type": event_type,
            "confidence": confidence,
            "status": "logged"
        }

        # Log file path
        log_file = os.path.join(detections_dir, "detections.json")

        # Read existing detections
        detections = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    detections = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                detections = []

        # Add new detection
        detections.append(detection_entry)

        # Keep only last 100 detections to prevent file from growing too large
        if len(detections) > 100:
            detections = detections[-100:]

        # Write updated detections
        with open(log_file, 'w') as f:
            json.dump(detections, f, indent=2)

        print(f"✓ Detection logged: {event_type} at {latitude}, {longitude}")

    except Exception as e:
        print(f"❌ Failed to save detection: {e}")


def get_recent_detections(count: int = 10) -> list:
    """
    Get recent detection events from local log.

    Args:
        count: Number of recent detections to retrieve

    Returns:
        list: Recent detection entries
    """
    try:
        log_file = os.path.join("detections", "detections.json")
        if not os.path.exists(log_file):
            return []

        with open(log_file, 'r') as f:
            detections = json.load(f)

        return detections[-count:] if detections else []

    except Exception as e:
        print(f"❌ Failed to read detections: {e}")
        return []


def clear_detection_log():
    """Clear the detection log file."""
    try:
        log_file = os.path.join("detections", "detections.json")
        if os.path.exists(log_file):
            os.remove(log_file)
            print("✓ Detection log cleared")
        else:
            print("ℹ No detection log found")
    except Exception as e:
        print(f"❌ Failed to clear detection log: {e}")


if __name__ == "__main__":
    # Test the detection logger
    print("Testing detection logger...")
    save_detection("8.7139", "77.7567", "Fighting", 0.85)
    save_detection("8.7140", "77.7568", "Normal", 0.92)

    recent = get_recent_detections()
    print(f"Recent detections: {len(recent)}")
    for detection in recent:
        event_type = detection.get('event_type', 'Unknown')
        confidence = detection.get('confidence', 0.0)
        print(f"  - {event_type}: {confidence:.2f}")
