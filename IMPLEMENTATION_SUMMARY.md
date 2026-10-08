# AI-Based Real-Time Violence Detection and Automated WhatsApp Alert System

## 1. Project Overview

The **AI-Based Real-Time Violence Detection and Automated WhatsApp Alert System** is a computer-vision application designed to monitor a live camera feed and automatically notify a configured WhatsApp recipient when violence or weapon-related activity is detected.

The system combines **CNN and YOLO-based detection**, real-time video processing, dynamic camera/device location retrieval, confidence-based filtering, alert cooldown protection, local detection logging, and the **UltraMsg WhatsApp API**.

### Core Workflow

```text
Camera ON
    ↓
Real-Time Video Capture
    ↓
CNN / YOLO Detection
    ↓
Violence / Weapon Detected
    ↓
Confidence Threshold Check
    ↓
Alert Cooldown Check
    ↓
Fetch Current Camera Location
    ↓
Generate Alert
    ↓
UltraMsg API
    ↓
WhatsApp Notification
    ↓
Local Detection Logging
```

The system is designed so that failure of the external alert or location service does not stop the core AI detection pipeline.

---

# 2. Core Features

### Real-Time Violence and Weapon Detection

The system continuously analyzes the live camera feed using the project's trained CNN and YOLO models.

Depending on the configured models, it can identify events such as:

- Fighting
- Violence
- Assault
- Weapons
- Knife
- Gun
- Other configured threat classes

The actual detection classes depend on the trained model and configuration used by the application.

---

### Real-Time Camera Monitoring

When the application starts, the connected camera is used as the video source.

```text
Camera
  ↓
Frame Capture
  ↓
Preprocessing
  ↓
AI Model
  ↓
Detection Result
  ↓
Event + Confidence
```

The alert system is triggered only when a configured threat is detected.

---

# 3. Dynamic Real-World Location

The project does **not use a hardcoded/default location**.

When a qualifying violence or weapon event is detected, the application obtains the **current location associated with the device/camera running the application**.

The location is dynamically retrieved when required for an alert.

### Location Workflow

```text
Camera/Application Running
          ↓
Current Device/Network Location
          ↓
Latitude + Longitude
          ↓
Google Maps Location
          ↓
WhatsApp Alert
```

The location represents the **camera system's current location**, not the location of the person or object detected in the video.

If the location service cannot provide a location, the application should handle the failure gracefully and continue the detection process.

---

# 4. UltraMsg WhatsApp Alert System

The project uses **UltraMsg API** to deliver automated WhatsApp notifications.

When the detection satisfies the configured conditions, `alert_service.py` prepares an alert and communicates with the UltraMsg API.

### Alert Workflow

```text
Threat Detected
      ↓
Confidence ≥ Threshold?
      ↓
Cooldown Available?
      ↓
Fetch Current Location
      ↓
Create Alert Message
      ↓
UltraMsg API
      ↓
WhatsApp Recipient
```

---

# 5. WhatsApp Alert Format

A typical alert can contain:

```text
🚨 VIOLENCE DETECTION ALERT 🚨

Event: Knife Detection
Confidence: 87%

Time: 2026-10-08 10:30:15

Camera Location:
Latitude: XX.XXXX
Longitude: YY.YYYY

Google Maps:
https://maps.google.com/?q=XX.XXXX,YY.YYYY

This is an automated security alert.
Please respond immediately.
```

The exact message format can be customized in the alert service.

---

# 6. Confidence Threshold

The system uses a confidence threshold to prevent low-confidence detections from generating unnecessary alerts.

Example:

```text
Detection confidence = 0.87
Threshold = 0.60

0.87 ≥ 0.60
      ↓
Alert generated
```

Whereas:

```text
Detection confidence = 0.42
Threshold = 0.60

0.42 < 0.60
      ↓
No alert
```

### Default

```text
Confidence Threshold = 0.60
```

The threshold can be adjusted according to the model and application requirements.

---

# 7. Alert Cooldown

Continuous video processing may detect the same event across many consecutive frames.

Without cooldown protection, this could generate repeated WhatsApp messages.

The system therefore uses an alert cooldown.

Example:

```text
10:30:00 → Violence detected → WhatsApp alert
10:30:01 → Violence detected → Ignored
10:30:02 → Violence detected → Ignored
10:30:03 → Violence detected → Ignored
...
10:35:00 → New alert allowed
```

### Default

```text
Cooldown = 300 seconds / 5 minutes
```

The value can be configured.

---

# 8. Project Components

The following components represent the actual project functionality. Experimental, debugging, and test-only scripts are not considered part of the normal application workflow.

## Main Application

### `app.py`

The primary application entry point.

Responsible for:

- Starting the Streamlit application
- Accessing the camera/video source
- Processing video frames
- Running the detection pipeline
- Displaying detection results
- Connecting detection results to the alert system
- Providing application controls

---

## Alert System

### `alert_service.py`

The central alert-management module.

Responsible for:

- Processing detection events
- Checking confidence thresholds
- Applying alert cooldown
- Obtaining the current camera/device location
- Creating the alert message
- Communicating with UltraMsg
- Handling external API failures
- Managing alert configuration

---

### `whatsapp_alert.py`

WhatsApp notification functionality used by the project.

Responsible for:

- Preparing WhatsApp requests
- Communicating with the configured WhatsApp API functionality
- Supporting WhatsApp notification delivery

Sensitive credentials must be supplied through environment variables rather than hardcoded into the source code.

---

## Detection Logging

### `detection_logger.py`

Provides local detection logging.

Responsible for:

- Recording detected events
- Maintaining local detection history
- Supporting debugging and monitoring
- Storing detection information without requiring an external database

---

# 9. AI Detection Components

The project contains the CNN/YOLO detection and model-loading components required by the application's actual detection pipeline.

These may include the project's model and YOLO processing modules such as:

```text
model.py
yolo_detector.py
yolo_enhanced.py
yolo_safe_load.py
yolo_simple.py
yolo_minimal.py
yolo_with_boxes.py
load_cnn_safe.py
load_yolo_safe.py
safe_yolo_loader.py
clean_yolo_loader.py
clean_yolo_inference.py
```

These files should be retained **only when they are part of the active model/detection pipeline used by the final application**. Alternative or obsolete implementations should not be presented as required runtime components.

---

# 10. Video Processing

### `video.py`

Provides video-processing functionality used by the project where applicable.

It handles functionality related to:

- Video input
- Frame processing
- Detection workflow integration
- Camera/video processing utilities

---

# 11. Configuration

### `settings.yaml`

Contains non-sensitive application configuration.

Sensitive API credentials should **not** be stored in this file.

---

# 12. Dependencies

### `requirements.txt`

Contains the Python packages required by the application.

A new user can install the dependencies using:

```bash
pip install -r requirements.txt
```

---

# 13. Environment Variables

API credentials must be configured locally.

Create a `.env` file containing the required UltraMsg configuration.

Example:

```env
ULTRAMSG_INSTANCE_ID=your_instance_id
ULTRAMSG_TOKEN=your_ultramsg_token
ULTRAMSG_RECIPIENT=your_recipient_number
```

The actual credentials must never be committed to GitHub.

The repository should contain:

```text
.env.example
```

instead of the real `.env`.

Example:

```env
ULTRAMSG_INSTANCE_ID=
ULTRAMSG_TOKEN=
ULTRAMSG_RECIPIENT=
```

This tells another developer exactly which configuration values are required.

---

# 14. Required Model Files

The trained model weights required by the active CNN/YOLO pipeline must be available locally.

Large model files should generally not be committed directly to GitHub if they exceed repository limits.

The README should clearly explain:

- Which model files are required
- Where they should be placed
- How to obtain them
- Which model configuration is being used

The application should not depend on undocumented local files.

---

# 15. What Someone Needs to Run the Project

A person cloning the project needs the following.

### Software

- Python
- Required Python packages
- Compatible webcam/camera
- Internet connection
- Git, if cloning from GitHub

### Project Components

- Main application
- Active CNN/YOLO detection modules
- Model-loading modules
- Video-processing module
- Alert service
- WhatsApp integration
- Detection logger
- Configuration files
- Required model weights
- `requirements.txt`

### External Services

- UltraMsg account
- UltraMsg instance
- UltraMsg API token
- WhatsApp recipient
- Internet connectivity
- A supported method for obtaining the camera/device's current location

---

# 16. Installation

Clone the repository:

```bash
git clone https://github.com/shabik12/violence-detection-Ai-Ml.git
```

Enter the project:

```bash
cd violence-detection-Ai-Ml
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 17. Configure UltraMsg

Create a local `.env` file:

```env
ULTRAMSG_INSTANCE_ID=your_instance_id
ULTRAMSG_TOKEN=your_ultramsg_token
ULTRAMSG_RECIPIENT=your_recipient_number
```

Use your own UltraMsg credentials.

Do not copy credentials from another developer.

---

# 18. Camera Setup

Connect a compatible webcam or camera to the computer.

The camera must be accessible by the application.

```text
Camera Connected
       ↓
Application Started
       ↓
Live Camera Feed
       ↓
AI Detection
```

---

# 19. Location Setup

The application must be able to obtain the **current location of the camera/device running the application**.

No hardcoded location should be used.

The location system should dynamically provide:

```text
Latitude
Longitude
```

These coordinates are then converted into a Google Maps link and included in the WhatsApp alert.

If location retrieval fails, the application should report the issue and continue AI detection instead of terminating.

---

# 20. Run the Application

Start the application with:

```bash
streamlit run app.py
```

The application then follows:

```text
Camera ON
    ↓
Live Video
    ↓
AI Detection
    ↓
Violence/Weapon Detected
    ↓
Confidence Check
    ↓
Cooldown Check
    ↓
Current Location
    ↓
Alert Message
    ↓
UltraMsg API
    ↓
WhatsApp
```

---

# 21. Complete System Architecture

```text
                         ┌─────────────────┐
                         │     CAMERA      │
                         │   Webcam / USB  │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │     app.py      │
                         │ Main Application│
                         └────────┬────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │   CNN + YOLO        │
                       │ Detection Pipeline  │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Event + Confidence  │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  Alert Service      │
                       │ alert_service.py    │
                       └──────────┬──────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
       Confidence Check    Current Location    Cooldown Check
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │   Alert Message     │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │    UltraMsg API     │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │      WhatsApp       │
                       │     Recipient       │
                       └─────────────────────┘

                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Detection Logger    │
                       │ Local JSON Storage  │
                       └─────────────────────┘
```

---

# 22. Error Handling

The external services should not stop the AI detection system.

### Location Failure

```text
Violence Detected
       ↓
Location unavailable
       ↓
Log/handle location error
       ↓
Detection continues
```

### UltraMsg Failure

```text
Violence Detected
       ↓
UltraMsg API unavailable
       ↓
Log alert failure
       ↓
Detection continues
```

This provides a fail-safe architecture where the core detection system remains operational even when external services are temporarily unavailable.

---

# 23. Security

The following must never be committed to GitHub:

```text
.env
UltraMsg token
UltraMsg instance credentials
Private API keys
Sensitive configuration
Personal authentication credentials
```

Use:

```text
.env.example
```

to document required environment variables.

All real credentials should remain local to the developer/deployment environment.

---

# 24. Files Not Required for the Normal Application

The repository may contain development artifacts from model experimentation, but these should not be presented as required runtime components.

Examples include:

```text
test_*.py
debug_*.py
compare_*.py
*_test.py
*_diagnosis.py
*_results.txt
*_out.txt
```

Training/evaluation scripts such as:

```text
train_weapons_yolo.py
retrain_model.py
evaluate_model.py
model_evaluation.py
evaluation_metrics.py
```

are useful during development/model training but are not required for normal application execution once the trained models are available.

Similarly, old application versions and scripts containing credentials should remain excluded.

---

# 25. Final Working Flow

The complete intended operation is:

```text
1. Connect Camera
        ↓
2. Start Application
        ↓
3. Capture Live Video
        ↓
4. CNN/YOLO Processes Frames
        ↓
5. Violence/Weapon Detected
        ↓
6. Check Confidence
        ↓
7. Check Alert Cooldown
        ↓
8. Fetch Current Camera/Device Location
        ↓
9. Generate Alert Message
        ↓
10. Send Through UltraMsg API
        ↓
11. WhatsApp Recipient Receives Alert
        ↓
12. Detection Is Logged Locally
```

## Final Objective

The project provides an automated real-time safety monitoring system:

**Camera → AI Detection → Violence/Weapon Detection → Confidence Filtering → Current Location → UltraMsg API → WhatsApp Alert**

The system provides the recipient with information about **what was detected, when it was detected, and the current location of the camera system at the time of the event**, while maintaining continuous AI-based monitoring.
