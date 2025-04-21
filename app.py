import os
import cv2
import base64
import time
import pickle
import numpy as np
import logging
import uuid
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from ultralytics import YOLO
from google.cloud import storage
from google.cloud.storage.retry import DEFAULT_RETRY
from google.cloud.exceptions import NotFound
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

# Google Cloud Storage setup
try:
    if os.getenv('CLOUD_RUN', False):
        storage_client = storage.Client()
    else:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account-key.json"
        storage_client = storage.Client()
    bucket_name = "ppe-detection-uploads"
    bucket = storage_client.bucket(bucket_name)
    token_blob = bucket.blob("tokens/token.pickle")
    logger.info(f"✅ Configured Google Cloud Storage Bucket: {bucket_name}")
except Exception as e:
    logger.error(f"❌ Failed to initialize Google Cloud Storage: {e}")
    raise

# Download token.pickle from GCS
def download_token_from_gcs():
    try:
        if token_blob.exists():
            token_blob.download_to_filename('token.pickle')
            logger.info("✅ Downloaded token.pickle from GCS")
        else:
            logger.error("❌ token.pickle not found in GCS")
            raise Exception("token.pickle not found in GCS")
    except Exception as e:
        logger.error(f"❌ Failed to download token.pickle from GCS: {e}")
        raise

# Upload token.pickle to GCS
def upload_token_to_gcs():
    try:
        if os.path.exists('token.pickle'):
            token_blob.upload_from_filename('token.pickle', retry=DEFAULT_RETRY)
            logger.info("✅ Uploaded refreshed token.pickle to GCS")
        else:
            logger.error("❌ token.pickle not found locally for upload")
    except Exception as e:
        logger.error(f"❌ Failed to upload token.pickle to GCS: {e}")

# Gmail API setup
SCOPES = ['https://www.googleapis.com/auth/gmail.send']
def get_gmail_service():
    try:
        creds = None
        token_path = 'token.pickle'
        if os.getenv('CLOUD_RUN', False):
            download_token_from_gcs()
        if os.path.exists(token_path):
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
        else:
            logger.error(f"❌ {token_path} not found. Ensure token is in GCS or run generate_token.py locally.")
            raise Exception(f"{token_path} not found")

        if creds and creds.valid:
            logger.info("✅ Using valid credentials")
        elif creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired credentials")
            creds.refresh(Request())
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)
            if os.getenv('CLOUD_RUN', False):
                upload_token_to_gcs()
            logger.info("✅ Credentials refreshed and saved")
        else:
            logger.error("❌ No valid credentials or refresh token. Regenerate token.pickle locally and upload to GCS.")
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

        if cap is None or not cap.isOpened():
            if video_source is None:
                logger.info("No video source set. Yielding placeholder.")
                yield get_placeholder_frame_bytes()
                time.sleep(0.5)
                continue

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

        if cap and cap.isOpened():
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

        else:
            logger.info("No capture device available. Yielding placeholder.")
            yield get_placeholder_frame_bytes()
            time.sleep(0.5)

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
            return jsonify({'status': 'error', 'message': 'Token is invalid. Regenerate token.pickle locally and upload to GCS.'}), 500
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
        return "No file part", 400
    file = request.files['video']
    if file.filename == '':
        logger.error("Upload failed: No selected file")
        return "No selected file", 400
    if file:
        original_filename = file.filename
        file_extension = os.path.splitext(original_filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        logger.info(f"Uploading file: {original_filename} as {unique_filename}")
        try:
            blob = bucket.blob(f"uploads/{unique_filename}")
            blob.upload_from_file(file, retry=DEFAULT_RETRY)
            blob.make_public()
            public_url = blob.public_url
            logger.info(f"File uploaded to GCS: {public_url}")
            video_source = public_url
            current_source_type = 'uploaded'
            webcam_active = False
            logger.info(f"Source changed to Uploaded Video: {video_source}")
            return redirect(url_for('index'))
        except Exception as e:
            logger.error(f"Error uploading file {unique_filename} to GCS: {e}")
            return f"Error uploading file: {e}", 500
    return "Upload failed", 400

# Change Source Route
@app.route('/change_source', methods=['POST'])
def change_source():
    global video_source, current_source_type, webcam_active
    source_type = request.form.get('source_type')
    logger.info(f"Request received to change source to: {source_type}")
    if source_type == 'webcam':
        video_source = 0
        current_source_type = 'webcam'
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
            return "CCTV URL cannot be empty", 400
    elif source_type == 'uploaded':
        logger.info("'uploaded' source type selected - requires file upload via /upload route")
    return redirect(url_for('index'))

# Main Execution Block
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Flask app on http://0.0.0.0:{port}")
    app.run(debug=True, host='0.0.0.0', port=port, threaded=True)