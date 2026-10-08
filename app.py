
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from ultralytics import YOLO
import tempfile
import urllib.request
import requests
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import json
from datetime import datetime

# ConvNeXt preprocessing - normalize to [-1, 1] range (ImageNet normalization)
def preprocess_input(x):
    """Preprocess input for ConvNeXt - normalize to [-1, 1] range."""
    return (x / 127.5) - 1.0

# Import alert service and detection logger
try:
    from alert_service import handle_alert, test_alert_system, get_alert_status
    from detection_logger import save_detection
    ALERT_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠ Alert system not available: {e}")
    ALERT_SYSTEM_AVAILABLE = False

st.set_page_config(page_title="Violence Detection System", layout="wide")

# Constants
IMG_SIZE = 224  # ConvNeXt input size
DATADIR = "Dataset"
CATEGORIES = sorted(os.listdir(DATADIR))
abnormal_labels = [label.lower() for label in CATEGORIES if label.lower() != "normal"]
font = cv2.FONT_HERSHEY_SIMPLEX

# Version information for debugging
st.sidebar.markdown(f"**TensorFlow Version:** {tf.__version__}")
try:
    keras_version = tf.keras.__version__
except AttributeError:
    keras_version = "Built-in with TensorFlow"
st.sidebar.markdown(f"**Keras Version:** {keras_version}")

# ================= LOAD MODELS ===================
@st.cache_resource
def load_models():
    """Load CNN and YOLO models using official APIs only."""
    models = {}

    # Load CNN model using safe loader
    try:
        from load_cnn_safe import load_cnn_model_safely
        cnn_model = load_cnn_model_safely("CNN_video.h5")

        if cnn_model is not None:
            st.success("Successfully loaded CNN model")
            models['cnn'] = cnn_model
        else:
            st.error("Failed to load CNN model")
            models['cnn'] = None
    except Exception as e:
        st.error(f"Error loading CNN model: {str(e)}")
        st.warning("Model compatibility issue detected. Please retrain model with current TensorFlow version.")
        models['cnn'] = None

    # Load YOLO model using simple inference approach
    try:
        print("Loading YOLO model for inference...")
        from simple_yolo import load_yolo_model
        yolo_model = load_yolo_model('yolov8n.pt')

        if yolo_model:
            models['yolo'] = yolo_model
            st.success("YOLO model loaded")
        else:
            st.warning("YOLO model loading failed. CNN detection will still work.")
            models['yolo'] = None

    except Exception as e:
        st.error(f"Error loading YOLO model: {str(e)}")
        st.warning("YOLO model loading failed. CNN detection will still work.")
        models['yolo'] = None
        print(f"❌ Failed to load YOLO model: {str(e)}")

    return models

# Load models
models = load_models()
cnn_model = models.get('cnn')
yolo_model = models.get('yolo')

# ================= HELPER FUNCTIONS ===================
def detect_objects_yolo(frame, model):
    """
    Run YOLO inference on a single frame using simple detector.

    Args:
        frame: numpy array (BGR format from OpenCV)
        model: loaded YOLO model

    Returns:
        results: YOLO detection results or None
    """
    if model is None:
        return None

    try:
        from simple_yolo import detect_objects
        detections = detect_objects(frame, model)

        if detections:
            print(f"🎯 YOLO detected {len(detections)} objects:")
            for det in detections:
                print(f"   - {det['label']}: {det['confidence']:.3f}")

        return detections

    except Exception as e:
        print(f"❌ YOLO inference failed: {str(e)}")
        return None

def preprocess_frame(frame, target_size=(IMG_SIZE, IMG_SIZE)):
    """Preprocess frame for ConvNeXt model."""
    # Convert to RGB if grayscale
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:  # RGBA to RGB
        frame = frame[..., :3]

    # Resize and preprocess for ConvNeXt
    frame = cv2.resize(frame, target_size)
    frame = preprocess_input(frame)  # Using ConvNeXt's preprocessing
    return np.expand_dims(frame, axis=0)  # Add batch dimension

def detect_cnn(file_path, threshold=0.3):  # Lowered threshold from 0.5 to 0.3
    """Detect violence in an image using the CNN model.

    Args:
        file_path: Path to the image file
        threshold: Confidence threshold for detection

    Returns:
        tuple: (predicted_class_index, confidence, is_abnormal)
    """
    try:
        # Read and preprocess the image
        frame = cv2.imread(file_path)
        if frame is None:
            raise ValueError(f"Could not read image from {file_path}")

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Make prediction
        predictions = cnn_model.predict(processed_frame, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        predicted_label = CATEGORIES[predicted_class]

        # Debug output
        print(f"DEBUG: Predicted={predicted_label}, Confidence={confidence:.3f}, Threshold={threshold}")

        # Determine if the prediction is above threshold
        is_abnormal = (confidence >= threshold and
                      predicted_label.lower() in abnormal_labels)

        return predicted_class, confidence, is_abnormal

    except Exception as e:
        st.error(f"Error in detection: {str(e)}")
        return None, 0.0, False

def fetch_location():
    try:
        # Try multiple location services for better accuracy
        try:
            # First try ipinfo.io
            response = requests.get('https://ipinfo.io/json', timeout=10)
            data = response.json()
            if data.get('loc'):
                lat, lon = data['loc'].split(',')
                print(f"IPInfo Location: {lat}, {lon}")
                return lat, lon
        except:
            pass

        # Fallback to ip-api.com
        try:
            response = requests.get('http://ip-api.com/json', timeout=10)
            data = response.json()
            if data.get('lat') and data.get('lon'):
                lat, lon = str(data['lat']), str(data['lon'])
                print(f"IP-API Location: {lat}, {lon}")
                return lat, lon
        except:
            pass

        # If all else fails, use Tirunelveli coordinates
        print("Using Tirunelveli default coordinates")
        return "8.7139", "77.7567"  # Tirunelveli city center

    except Exception as e:
        print(f"Location fetch error: {e}")
        # Return Tirunelveli coordinates as default
        return "8.7139", "77.7567"  # Tirunelveli city center

def send_to_thingspeak(status, lat, lon):
    import urllib.parse
    api_key = os.getenv("THINGSPEAK_WRITE_API_KEY")
    # Ensure we have valid coordinates
    if lat == "N/A" or lon == "N/A":
        lat, lon = "8.7139", "77.7567"  # Tirunelveli coordinates

    # URL-encode the status to handle spaces and special characters
    encoded_status = urllib.parse.quote(status)
    url = f"https://api.thingspeak.com/update?api_key={api_key}&field2={encoded_status}&field3={lat}&field4={lon}"
    try:
        urllib.request.urlopen(url, timeout=10)
        print(f"Data sent to ThingSpeak: {status}, {lat}, {lon}")
    except Exception as e:
        print(f"Could not send data to ThingSpeak: {e}")

def send_normal_status():
    """Send normal status to keep dashboard active"""
    lat, lon = fetch_location()
    send_to_thingspeak("Normal", lat, lon)

# ================= STREAMLIT INTERFACE ===================
st.title("🚨 Violence Detection System")
st.markdown("### Powered by ConvNeXt-Tiny & YOLOv8")

# Add detection explanation
st.sidebar.markdown("### Detection Guide")
st.sidebar.markdown("""
**📍 Location Detection:**
- **Normal**: No violence detected
- **Fighting**: Punching, hitting, physical combat
- **Assault**: Holding someone aggressively, threatening behavior
- **Arrest**: Someone being restrained
- **Abuse**: Verbal or physical abuse
- **Burglary**: Breaking in, theft attempts
""")

# Sidebar options
st.sidebar.title("Options")
option = st.sidebar.radio("Choose Input", ["Live Camera", "Upload Video/Image"])

# Display model status
st.sidebar.markdown("### Model Status")
cnn_status = "✅ Loaded" if cnn_model else "❌ Not Available"
yolo_status = "✅ Loaded" if yolo_model else "❌ Not Available"
st.sidebar.text(f"CNN Model: {cnn_status}")
st.sidebar.text(f"YOLO Model: {yolo_status}")

# Add location options
st.sidebar.markdown("### Location Settings")
location_method = st.sidebar.radio("Location Method", ["Real-time GPS", "Auto (IP-based)", "Manual Entry"])

# Add alert system controls
if ALERT_SYSTEM_AVAILABLE:
    st.sidebar.markdown("### 🚨 Alert System")
    alert_status = get_alert_status()

    # Display alert status
    status_color = "🟢" if alert_status['enabled'] else "🔴"
    st.sidebar.text(f"{status_color} Status: {'Enabled' if alert_status['enabled'] else 'Disabled'}")
    st.sidebar.text(f"📊 Threshold: {alert_status['confidence_threshold']}")
    st.sidebar.text(f"⏰ Cooldown: {alert_status['cooldown_seconds']}s")
    st.sidebar.text(f"📱 SMS Configured: {'Yes' if alert_status['sms_configured'] else 'No'}")

    # Alert controls
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Enable", key="enable_alert"):
            from alert_service import enable_alerts
            enable_alerts()
            st.rerun()
    with col2:
        if st.button("Disable", key="disable_alert"):
            from alert_service import disable_alerts
            disable_alerts()
            st.rerun()

    # Test alert system
    if st.sidebar.button("🧪 Test Alert System"):
        with st.spinner("Testing alert system..."):
            test_result = test_alert_system()
            if test_result:
                st.sidebar.success("✅ Alert system test passed!")
            else:
                st.sidebar.error("❌ Alert system test failed!")

    # Confidence threshold slider
    new_threshold = st.sidebar.slider(
        "Alert Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=alert_status['confidence_threshold'],
        step=0.05
    )
    if new_threshold != alert_status['confidence_threshold']:
        from alert_service import set_confidence_threshold
        set_confidence_threshold(new_threshold)
        st.rerun()
else:
    st.sidebar.markdown("### 🚨 Alert System")
    st.sidebar.error("❌ Alert system not available")
    st.sidebar.caption("Install alert_service.py to enable")

# Initialize session state for location
if 'current_lat' not in st.session_state:
    st.session_state.current_lat = "8.7139"  # Tirunelveli default
if 'current_lon' not in st.session_state:
    st.session_state.current_lon = "77.7567"  # Tirunelveli default
if 'location_source' not in st.session_state:
    st.session_state.location_source = "Default"

# Real-time GPS tracking
if location_method == "Real-time GPS":
    st.sidebar.markdown("#### 📍 Real-time GPS Tracking")

    # HTML/JavaScript for real-time GPS
    gps_html = """
    <div id="gps-status"></div>
    <script>
    let watchId;
    let updateCount = 0;

    function updateLocation(position) {
        const lat = position.coords.latitude.toFixed(6);
        const lon = position.coords.longitude.toFixed(6);
        const accuracy = position.coords.accuracy.toFixed(0);

        // Send location to Streamlit
        const data = {
            type: 'gps_update',
            lat: lat,
            lon: lon,
            accuracy: accuracy,
            timestamp: new Date().toISOString()
        };

        // Use Streamlit's component communication
        window.parent.postMessage(data, '*');

        // Update status
        document.getElementById('gps-status').innerHTML =
            `🟢 GPS Active<br>
            Lat: ${lat}<br>
            Lon: ${lon}<br>
            Accuracy: ±${accuracy}m<br>
            Updates: ${++updateCount}`;
    }

    function gpsError(error) {
        document.getElementById('gps-status').innerHTML =
            `🔴 GPS Error: ${error.message}`;
    }

    // Start watching position
    if (navigator.geolocation) {
        watchId = navigator.geolocation.watchPosition(
            updateLocation,
            gpsError,
            {
                enableHighAccuracy: true,
                maximumAge: 0,
                timeout: 5000
            }
        );
        document.getElementById('gps-status').innerHTML = '🟡 Starting GPS...';
    } else {
        document.getElementById('gps-status').innerHTML = '🔴 GPS not supported';
    }

    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        if (watchId) {
            navigator.geolocation.clearWatch(watchId);
        }
    });
    </script>
    """

    # Display GPS component
    gps_component = st.components.v1.html(gps_html, height=120)

    # Listen for GPS updates
    if 'gps_update' in st.query_params:
        gps_data = st.query_params['gps_update']
        if len(gps_data) >= 2:
            st.session_state.current_lat = gps_data[0]
            st.session_state.current_lon = gps_data[1]
            st.session_state.location_source = "Real-time GPS"
            st.rerun()

    st.sidebar.info("📍 GPS tracking active in browser")
    st.sidebar.caption("Allow location access when prompted")

# Manual location entry
elif location_method == "Manual Entry":
    manual_lat = st.sidebar.text_input("Latitude", value=st.session_state.current_lat)
    manual_lon = st.sidebar.text_input("Longitude", value=st.session_state.current_lon)
    if st.sidebar.button("Use Manual Location"):
        st.session_state.current_lat = manual_lat
        st.session_state.current_lon = manual_lon
        st.session_state.location_source = "Manual Entry"
        st.sidebar.success(f"Using manual location: {manual_lat}, {manual_lon}")
        st.rerun()
else:
    # Auto location (IP-based)
    if st.sidebar.button("Refresh Auto Location"):
        auto_lat, auto_lon = fetch_location()
        st.session_state.current_lat = auto_lat
        st.session_state.current_lon = auto_lon
        st.session_state.location_source = "IP-based"
        st.sidebar.info(f"Auto location: {auto_lat}, {auto_lon}")
        st.rerun()

# Use session state coordinates
lat = st.session_state.current_lat
lon = st.session_state.current_lon

# Add location validation
st.sidebar.markdown("### Current Location")
st.sidebar.code(f"Source: {st.session_state.get('location_source', 'Unknown')}\nLatitude: {lat}\nLongitude: {lon}\n\nClick 'Send Normal Status' to test")

# Add test button for ThingSpeak
st.sidebar.markdown("### ThingSpeak Test")
if st.sidebar.button("Send Normal Status"):
    send_to_thingspeak("Normal", lat, lon)
    st.sidebar.success(f"Normal status sent from {st.session_state.get('location_source', 'unknown')} location!")
    st.sidebar.balloons()

output_placeholder = st.empty()
cnn_result_placeholder = st.empty()
yolo_result_placeholder = st.empty()
illegal_result_placeholder = st.empty()

# ======== UPLOAD IMAGE/VIDEO ========
if option == "Upload Video/Image":
    uploaded_file = st.file_uploader("Choose a file", type=["jpg","png","mp4","avi"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        file_path = tfile.name

        if uploaded_file.type.startswith("image"):
            frame = cv2.imread(file_path)

            # Get CNN prediction
            cnn_class_idx, cnn_confidence, is_abnormal = detect_cnn(file_path, 0.3)

            # Use safe YOLO inference
            results = detect_objects_yolo(frame, yolo_model)
            if results:
                annotated_frame = results[0].plot().copy()
            else:
                annotated_frame = frame.copy()

            abnormal_detected = False
            illegal_detected = False

            # Check YOLO detections if model is available
            yolo_labels = []
            if results and yolo_model is not None:
                for det in results[0].boxes:
                    cls_id = int(det.cls[0])
                    conf = float(det.conf[0])
                    label = yolo_model.names[cls_id].lower()
                    if conf > 0.5:
                        yolo_labels.append(label)
                        if label in ["cell phone", "person"]:
                            x1, y1, x2, y2 = map(int, det.xyxy[0])
                            cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), (0,0,255), 2)
                            cv2.putText(annotated_frame, f"{label} {conf:.2f}",
                                      (x1, y1-5), font, 0.7, (0,0,255), 2)

            # Get CNN prediction with threshold
            cnn_class_idx, cnn_confidence, is_abnormal = detect_cnn(file_path, 0.3)  # Lowered threshold

            # Update detection status
            if cnn_class_idx is not None:
                cnn_label = CATEGORIES[cnn_class_idx]
                cnn_result_placeholder.success(f"CNN Detection: {cnn_label} (Confidence: {cnn_confidence*100:.1f}%)")

                # Send ALL detections to ThingSpeak (normal and abnormal)
                send_to_thingspeak(cnn_label, lat, lon)

                # Check if detection is above threshold and abnormal
                if is_abnormal and cnn_confidence >= 0.3:  # Lowered threshold
                    abnormal_detected = True
                    # Save the detection
                    save_detection(lat, lon, cnn_label, cnn_confidence)

                    # Trigger alert system if available
                    if ALERT_SYSTEM_AVAILABLE:
                        alert_sent = handle_alert(cnn_label, cnn_confidence)
                        if alert_sent:
                            cnn_result_placeholder.success("🚨 SMS Alert Sent!")
                        else:
                            cnn_result_placeholder.info("ℹ️ Alert conditions not met")

            # Check for illegal activities (cell phone usage)
            if yolo_model is not None and "cell phone" in yolo_labels and "person" in yolo_labels:
                illegal_detected = True
                abnormal_detected = True
                illegal_result_placeholder.error("⚠️ Illegal activity detected: Cell phone usage in restricted area!")
                cnn_result = "Illegal"

                # Save illegal activity detection
                save_detection(lat, lon, "Illegal Activity", 1.0)

                # Trigger alert for illegal activity
                if ALERT_SYSTEM_AVAILABLE:
                    alert_sent = handle_alert("Illegal Activity", 1.0)
                    if alert_sent:
                        illegal_result_placeholder.success("🚨 SMS Alert Sent for Illegal Activity!")

            cv2.putText(annotated_frame, f"CNN: {cnn_result} ({cnn_conf*100:.1f}%)", (10,40), font, 1, (255,255,0), 2, cv2.LINE_AA)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            output_placeholder.image(annotated_frame, channels="RGB", width=400)

            #output_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

            # Display results separately
            cnn_result_placeholder.markdown(f"**CNN Prediction:** {cnn_result} ({cnn_conf*100:.1f}%)")
            yolo_result_placeholder.markdown(f"**YOLO Detections:** {', '.join(yolo_labels) if yolo_labels else 'None'}")
            if illegal_detected:
                illegal_result_placeholder.markdown(f"**Illegal Activity Detected!** ❌", unsafe_allow_html=True)

            if abnormal_detected:
                send_to_thingspeak(cnn_result, lat, lon)

# ======== LIVE CAMERA ========
elif option == "Live Camera":
    stframe = st.empty()
    run = st.checkbox("Run Camera")

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open camera. Please check your camera connection.")
        run = False

    # Frame processing function
    def process_frame(frame):
        # Make a copy for display
        display_frame = frame.copy()

        # Prepare frame for CNN
        processed_frame = preprocess_frame(frame)

        # Get CNN prediction
        predictions = cnn_model.predict(processed_frame, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = float(predictions[predicted_class])
        label = CATEGORIES[predicted_class]

        # Debug output
        print(f"LIVE DEBUG: Predicted={label}, Confidence={confidence:.3f}")

        is_abnormal = (confidence >= 0.3 and  # Lowered threshold
                      label.lower() in abnormal_labels)

        # Process with YOLO if available
        if yolo_model is not None:
            results = detect_objects_yolo(frame, yolo_model)
            yolo_labels = []

            # Draw YOLO detections
            if results:
                for det in results[0].boxes:
                    cls_id = int(det.cls[0])
                    conf = float(det.conf[0])
                    yolo_label = yolo_model.names[cls_id].lower()
                    if conf > 0.5:  # Confidence threshold for YOLO
                        yolo_labels.append(yolo_label)
                        x1, y1, x2, y2 = map(int, det.xyxy[0])

                        # Draw bounding box
                        color = (0, 0, 255)  # Red for detections
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display_frame,
                                  f"{yolo_label} {conf:.2f}",
                                  (x1, y1-10), font, 0.7, color, 2)

            # Check for illegal activities
            if "cell phone" in yolo_labels and "person" in yolo_labels:
                is_abnormal = True
                cv2.putText(display_frame, "ILLEGAL: Cell phone detected!",
                           (20, 80), font, 1, (0, 0, 255), 2)

                # Save illegal activity detection
                save_detection(lat, lon, "Illegal Activity", 1.0)

                # Trigger alert for illegal activity
                if ALERT_SYSTEM_AVAILABLE:
                    alert_sent = handle_alert("Illegal Activity", 1.0)
                    if alert_sent:
                        print("🚨 SMS Alert sent for illegal activity in live camera!")

        # Display CNN prediction
        color = (0, 0, 255) if is_abnormal else (0, 255, 0)
        status = "ABNORMAL" if is_abnormal else "NORMAL"
        cv2.putText(display_frame, f"Status: {status}", (20, 30),
                   font, 0.8, color, 2)
        cv2.putText(display_frame, f"{label}: {confidence*100:.1f}%",
                   (20, 70), font, 0.8, color, 2)

        # Send ALL detections to ThingSpeak (normal and abnormal)
        send_to_thingspeak(label, lat, lon)

        # Save detection if abnormal
        if is_abnormal and confidence >= 0.3:  # Lowered threshold
            save_detection(lat, lon, label, confidence)

            # Trigger alert system if available
            if ALERT_SYSTEM_AVAILABLE:
                alert_sent = handle_alert(label, confidence)
                if alert_sent:
                    print("🚨 SMS Alert sent for live camera detection!")

        return display_frame

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture camera frame.")
            break

        try:
            # Process frame with detection models
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = process_frame(frame_rgb)

            # Display the processed frame
            stframe.image(processed_frame, channels="RGB")

        except Exception as e:
            st.error(f"Error processing frame: {str(e)}")
            break

    # Release resources
    if 'cap' in locals() and cap is not None:
        cap.release()
