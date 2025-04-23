import os
import cv2
import base64
import time
import pickle
import numpy as np
import logging
import uuid
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText
from google.auth.transport.requests import Request

# Flask setup
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Create uploads directory
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Gmail API setup
SCOPES = ['https://www.googleapis.com/auth/gmail.send']
def get_gmail_service():
    try:
        creds = None
        token_path = 'token.pickle'
        if os.path.exists(token_path):
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
        else:
            logger.error(f"❌ {token_path} not found. Run generate_token.py to create it.")
            raise Exception(f"{token_path} not found")

        if creds and creds.valid:
            logger.info("✅ Using valid credentials")
        elif creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired credentials")
            creds.refresh(Request())
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)
            logger.info("✅ Credentials refreshed and saved")
        else:
            logger.error("❌ No valid credentials or refresh token. Regenerate token.pickle.")
            raise Exception("No valid credentials or refresh token")

        service = build('gmail', 'v1', credentials=creds)
        logger.info("✅ Gmail service initialized")
        return service
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gmail service: {e}", exc_info=True)
        return None

def send_email(subject, body, to_email):
    try:
        service = get_gmail_service()
        if not service:
            logger.error("❌ Email sending failed: Gmail service not available")
            return False
        message = MIMEText(body)
        message['to'] = to_email
        message['subject'] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        service.users().messages().send(userId='me', body={'raw': raw}).execute()
        logger.info(f"📧 Email sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"❌ Email sending failed to {to_email}: {e}", exc_info=True)
        return False

# Generate token.pickle (run once locally if needed)
def generate_token():
    try:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
        logger.info("✅ Generated token.pickle")
    except Exception as e:
        logger.error(f"❌ Failed to generate token.pickle: {e}")
        raise

# Load YOLO Model
MODEL_PATH = "model/best.pt"
logger.info(f"✅ Loading model from: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    logger.info("✅ Model loaded successfully.")
    CLASS_LABELS = model.names if hasattr(model, 'names') else [
        "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
        "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle"
    ]
    logger.info(f"Model classes: {CLASS_LABELS}")
except Exception as e:
    logger.error(f"❌ Failed to load YOLO model: {e}")
    raise

# Global variables
video_source = 0
violation_detected = False
last_violation_time = 0
EMAIL_COOLDOWN = 120
webcam_active = False
current_source_type = 'webcam'

# PPE Detection Function
def detect_ppe(frame):
    global violation_detected, last_violation_time
    try:
        results = model(frame)
        person_detected = False
        ppe_violation_this_frame = False
        annotated_frame = frame.copy()

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                if box.xyxy is None or box.conf is None or box.cls is None:
                    continue
                if len(box.xyxy) == 0 or len(box.conf) == 0 or len(box.cls) == 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                if class_id < 0 or class_id >= len(CLASS_LABELS):
                    logger.warning(f"⚠️ Invalid class_id: {class_id}. Skipping...")
                    continue
                label = CLASS_LABELS[class_id]
                color = (0, 255, 0) if class_id in [0, 1, 7] else (0, 0, 255) if class_id in [2, 3, 4] else (255, 0, 0) if class_id == 5 else (0, 255, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if label == "Person":
                    person_detected = True
                if label in ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]:
                    ppe_violation_this_frame = True

        new_violation_status = person_detected and ppe_violation_this_frame
        if new_violation_status and not violation_detected:
            logger.info("⚠️ PPE violation detected: Person missing required PPE")
            current_time = time.time()
            if (current_time - last_violation_time) > EMAIL_COOLDOWN:
                subject = "PPE Violation Detected"
                body = f"A PPE violation was detected at {time.strftime('%Y-%m-%d %H:%M:%S')}: a person is potentially missing Hardhat, Mask, or Safety Vest."
                to_email = "myselfkarthik1@gmail.com"
                logger.info(f"Attempting to send violation email to {to_email}")
                if send_email(subject, body, to_email):
                    last_violation_time = current_time
                    logger.info("✅ Email sent successfully")
                else:
                    logger.error("❌ Email sending failed during violation")
            else:
                logger.info(f"Violation detected, but within cooldown period ({int(EMAIL_COOLDOWN - (current_time - last_violation_time))}s remaining)")

        violation_detected = new_violation_status
        return annotated_frame
    except Exception as e:
        logger.error(f"❌ Error in detect_ppe: {e}")
        return frame

# Generate Frames for Video Streaming
def generate_frames():
    global video_source, webcam_active, current_source_type
    cap = None
    logger.info(f"Starting generate_frames. Source: {video_source}, Type: {current_source_type}, Webcam Active: {webcam_active}")

    while True:
        if current_source_type == 'webcam' and not webcam_active:
            logger.info("Webcam selected but toggled off. Yielding placeholder.")
            yield get_placeholder_frame_bytes()
            time.sleep(0.5)
            continue

        if video_source is None:
            logger.info("No video source set. Yielding placeholder.")
            yield get_placeholder_frame_bytes()
            time.sleep(0.5)
            continue

        if cap is None or not cap.isOpened():
            try:
                logger.info(f"Attempting to open video source: {video_source}")
                cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG if isinstance(video_source, str) else None)
                if isinstance(video_source, str):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                    cap.set(cv2.CAP_PROP_FPS, 15)
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
                if not cap.isOpened():
                    logger.error(f"Cannot open video source: {video_source}")
                    cap = None
                    yield get_placeholder_frame_bytes()
                    time.sleep(1)
                    continue
                logger.info(f"Successfully opened video source: {video_source}")
            except Exception as e:
                logger.error(f"Exception while opening video source {video_source}: {e}")
                cap = None
                yield get_placeholder_frame_bytes()
                time.sleep(1)
                continue

        success, frame = cap.read()
        if not success:
            logger.warning(f"End of video stream or cannot read frame from source: {video_source}")
            if current_source_type != 'webcam':
                logger.info(f"Source {video_source} finished or disconnected. Releasing capture.")
                cap.release()
                cap = None
                yield get_placeholder_frame_bytes()
                break
            else:
                logger.info("Webcam stream failed. Will attempt to reopen.")
                cap.release()
                cap = None
                yield get_placeholder_frame_bytes()
                time.sleep(0.5)
                continue

        processed_frame = detect_ppe(frame)
        try:
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                logger.warning("Error encoding frame to JPEG")
        except Exception as e:
            logger.error(f"Exception during frame encoding: {e}")
            break

        time.sleep(0.033)  # Approximate 30 FPS

    if cap and cap.isOpened():
        cap.release()
    logger.info("generate_frames loop finished.")

# Helper function for placeholder image
def get_placeholder_frame_bytes():
    placeholder_path = 'static/placeholder.jpg'
    if os.path.exists(placeholder_path):
        img = cv2.imread(placeholder_path)
    else:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Stream Paused / No Source", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    ret, buffer = cv2.imencode('.jpg', img)
    if ret:
        return (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')

# Toggle Webcam Endpoint
@app.route('/toggle_webcam', methods=['POST'])
def toggle_webcam():
    global webcam_active, current_source_type, video_source
    if current_source_type == 'webcam':
        webcam_active = request.json.get('active', False)
        logger.info(f"Webcam toggled: {'ON' if webcam_active else 'OFF'}")
        return jsonify({'status': 'success', 'webcam_active': webcam_active})
    logger.info("Toggle request ignored: Current source is not webcam")
    return jsonify({'status': 'ignored', 'message': 'Toggle only applies when source is webcam', 'webcam_active': webcam_active})

# Violation Status Endpoint
@app.route('/violation_status')
def violation_status():
    return jsonify({'violation': violation_detected})

# Test Email Endpoint
@app.route('/test_email', methods=['GET'])
def test_email():
    logger.info("Testing email functionality")
    subject = "Test Email from PPE Detection App"
    body = "This is a test email to verify Gmail API functionality."
    to_email = "myselfkarthik1@gmail.com"
    if send_email(subject, body, to_email):
        logger.info("Test email sent successfully")
        return jsonify({'status': 'success', 'message': 'Test email sent'})
    logger.error("Test email failed to send")
    return jsonify({'status': 'error', 'message': 'Test email failed'}), 500

# Refresh Token Check Endpoint
@app.route('/refresh_token', methods=['GET'])
def refresh_token():
    logger.info("Checking token validity")
    try:
        service = get_gmail_service()
        if service:
            logger.info("Token is valid or successfully refreshed")
            return jsonify({'status': 'success', 'message': 'Token is valid'})
        else:
            logger.error("Token is invalid or could not be refreshed")
            return jsonify({'status': 'error', 'message': 'Token is invalid. Regenerate token.pickle.'}), 500
    except Exception as e:
        logger.error(f"Error checking token: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'Error checking token: {e}'}), 500

# Index Route
@app.route('/')
def index():
    return render_template('index.html',
                           current_source_type=current_source_type,
                           webcam_active=webcam_active,
                           video_source=video_source if isinstance(video_source, str) else '')

# About Route (Placeholder)
@app.route('/about')
def about():
    return render_template('about.html')  # Create about.html or redirect to index

# Video Feed Route
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Upload Video Route
@app.route('/upload', methods=['POST'])
def upload_video():
    global video_source, current_source_type, webcam_active
    if 'video' not in request.files:
        logger.error("Upload failed: No file part in request")
        return jsonify({'status': 'error', 'message': 'No file part in request'}), 400
    file = request.files['video']
    if file.filename == '':
        logger.error("Upload failed: No selected file")
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    if not file or not allowed_file(file.filename):
        logger.error(f"Upload failed: Invalid file type for {file.filename}")
        return jsonify({'status': 'error', 'message': 'Invalid file type. Allowed: mp4, avi, mov'}), 400

    original_filename = file.filename
    file_extension = os.path.splitext(original_filename)[1].lower()
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    logger.info(f"Uploading file: {original_filename} as {unique_filename}")

    try:
        file.save(file_path)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        logger.info(f"File saved locally: {file_path}, Size: {file_size:.2f} MB")

        # Verify video file
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logger.error(f"Invalid video file: {unique_filename}")
            os.remove(file_path)
            return jsonify({'status': 'error', 'message': 'Invalid video file'}), 400
        cap.release()

        video_source = file_path
        current_source_type = 'uploaded'
        webcam_active = False
        logger.info(f"Source changed to Uploaded Video: {video_source}")
        return jsonify({'status': 'success', 'message': 'Video uploaded successfully', 'url': file_path})
    except Exception as e:
        logger.error(f"Error saving file {unique_filename}: {e}", exc_info=True)
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'status': 'error', 'message': f'Error uploading file: {str(e)}'}), 500

# Helper function for allowed file types
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Change Source Route
@app.route('/change_source', methods=['POST'])
def change_source():
    global video_source, current_source_type, webcam_active
    source_type = request.form.get('source_type')
    logger.info(f"Request received to change source to: {source_type}")
    if source_type == 'webcam':
        video_source = 0
        current_source_type = 'webcam'
        webcam_active = False  # Reset webcam to off
        logger.info(f"Source changed to Webcam ({video_source}). Webcam active state: {webcam_active}")
    elif source_type == 'cctv':
        cctv_url = request.form.get('cctv_url')
        if cctv_url:
            video_source = cctv_url
            current_source_type = 'cctv'
            webcam_active = False
            logger.info(f"Source changed to CCTV: {video_source}")
        else:
            logger.error("CCTV source selected but URL is missing")
            return jsonify({'status': 'error', 'message': 'CCTV URL cannot be empty'}), 400
    elif source_type == 'uploaded':
        logger.info("'uploaded' source type selected - requires file upload via /upload route")
        return jsonify({'status': 'error', 'message': 'Please upload a video first'}), 400
    return jsonify({'status': 'success', 'message': 'Source changed'})

# Main Execution Block
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Flask app on http://0.0.0.0:{port}")
    app.run(debug=True, host='0.0.0.0', port=port, threaded=True)