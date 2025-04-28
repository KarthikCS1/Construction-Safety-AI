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
from googleapiclient.discovery import build
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from PIL import Image
import io

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = 'Uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

SCOPES = ['https://www.googleapis.com/auth/gmail.send']

# List of email recipients for alerts (configurable)
EMAIL_RECIPIENTS = ['myselfkarthik1@gmail.com']  # Add more emails here or update via /update_recipients

def get_gmail_service():
    try:
        token_path = 'token.pickle'
        if not os.path.exists(token_path):
            logger.error(f"❌ {token_path} not found. Please provide token.pickle.")
            raise Exception(f"{token_path} not found")
        
        # Load credentials from token.pickle
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
        logger.info("✅ Loaded token.pickle")

        service = build('gmail', 'v1', credentials=creds)
        logger.info("✅ Gmail service initialized")
        return service
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gmail service: {e}", exc_info=True)
        return None

def send_email(subject, body, to_emails, screenshot=None):
    try:
        service = get_gmail_service()
        if not service:
            logger.error("❌ Email sending failed: Gmail service not available")
            return False

        # Create a single email with multiple recipients
        message = MIMEMultipart()
        message['to'] = ', '.join(to_emails)  # Join multiple recipients with commas
        message['subject'] = subject
        text_part = MIMEText(body)
        message.attach(text_part)

        if screenshot is not None:
            ret, buffer = cv2.imencode('.jpg', screenshot, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                img_data = buffer.tobytes()
                image_part = MIMEImage(img_data, name='violation_screenshot.jpg')
                message.attach(image_part)
                logger.info("📸 Screenshot attached to email")
            else:
                logger.warning("⚠️ Failed to encode screenshot for email")

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        service.users().messages().send(userId='me', body={'raw': raw}).execute()
        logger.info(f"📧 Email sent to {', '.join(to_emails)}")
        return True
    except Exception as e:
        logger.error(f"❌ Email sending failed to {', '.join(to_emails)}: {e}", exc_info=True)
        return False

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

video_source = 0  # Default to webcam
violation_detected = False
last_violation_time = 0
EMAIL_COOLDOWN = 120
webcam_active = False
mobile_camera_active = False
mobile_facing_mode = 'user'
current_source_type = 'webcam'
violation_screenshot = None

def detect_ppe(frame):
    global violation_detected, last_violation_time, violation_screenshot
    try:
        logger.info("Processing frame for PPE detection")
        results = model(frame)
        person_detected = False
        ppe_violation_this_frame = False
        annotated_frame = frame.copy()
        for result in results:
            if result.boxes is None:
                logger.warning("No boxes detected in frame")
                continue
            for box in result.boxes:
                if box.xyxy is None or box.conf is None or box.cls is None:
                    logger.warning("Invalid box data")
                    continue
                if len(box.xyxy) == 0 or len(box.conf) == 0 or len(box.cls) == 0:
                    logger.warning("Empty box data")
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                if class_id < 0 or class_id >= len(CLASS_LABELS):
                    logger.warning(f"Invalid class_id: {class_id}. Skipping...")
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
            violation_screenshot = annotated_frame.copy()
            current_time = time.time()
            if (current_time - last_violation_time) > EMAIL_COOLDOWN:
                subject = "PPE Violation Detected"
                body = f"A PPE violation was detected at {time.strftime('%Y-%m-%d %H:%M:%S')}: a person is potentially missing Hardhat, Mask, or Safety Vest."
                logger.info(f"Attempting to send violation email to {', '.join(EMAIL_RECIPIENTS)}")
                if send_email(subject, body, EMAIL_RECIPIENTS, violation_screenshot):
                    last_violation_time = current_time
                    logger.info("✅ Email sent successfully")
                else:
                    logger.error("❌ Email sending failed during violation")
            else:
                logger.info(f"Violation detected, but within cooldown period ({int(EMAIL_COOLDOWN - (current_time - last_violation_time))}s remaining)")
        violation_detected = new_violation_status
        return annotated_frame
    except Exception as e:
        logger.error(f"❌ Error in detect_ppe: {e}", exc_info=True)
        return frame

def generate_frames():
    global video_source, webcam_active, mobile_camera_active, current_source_type
    cap = None
    logger.info(f"Starting generate_frames. Source: {video_source}, Type: {current_source_type}, Webcam Active: {webcam_active}, Mobile Active: {mobile_camera_active}")
    while True:
        if current_source_type == 'webcam' and not webcam_active:
            logger.info("Webcam selected but toggled off. Yielding placeholder.")
            yield get_placeholder_frame_bytes()
            time.sleep(0.5)
            continue
        if current_source_type == 'mobile':
            logger.info("Mobile source selected. Frames handled via WebRTC.")
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
                cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG if isinstance(video_source, str) else cv2.CAP_DSHOW)
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
        time.sleep(0.033)
    if cap and cap.isOpened():
        cap.release()
    logger.info("generate_frames loop finished.")

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

@app.route('/toggle_webcam', methods=['POST'])
def toggle_webcam():
    global webcam_active, mobile_camera_active, current_source_type, video_source
    data = request.json
    active = data.get('active', False)
    source_type = data.get('source_type', current_source_type)
    logger.info(f"Toggle request: source_type={source_type}, active={active}")
    if source_type == 'webcam':
        webcam_active = active
        mobile_camera_active = False
        current_source_type = 'webcam'
        video_source = 0  # Explicitly set webcam source
        logger.info(f"Webcam toggled: {'ON' if webcam_active else 'OFF'}, video_source={video_source}")
        return jsonify({'status': 'success', 'webcam_active': webcam_active, 'mobile_camera_active': mobile_camera_active, 'source_type': current_source_type})
    elif source_type == 'mobile':
        mobile_camera_active = active
        webcam_active = False
        current_source_type = 'mobile'
        video_source = None
        logger.info(f"Mobile camera toggled: {'ON' if mobile_camera_active else 'OFF'}, video_source={video_source}")
        return jsonify({'status': 'success', 'mobile_camera_active': mobile_camera_active, 'webcam_active': webcam_active, 'source_type': current_source_type})
    logger.info("Toggle request ignored: Invalid source type")
    return jsonify({'status': 'error', 'message': 'Invalid source type', 'webcam_active': webcam_active, 'mobile_camera_active': mobile_camera_active, 'source_type': current_source_type}), 400

@app.route('/process_mobile_frame', methods=['POST'])
def process_mobile_frame():
    global violation_detected, last_violation_time, mobile_camera_active, violation_screenshot
    logger.info(f"Processing mobile frame. Source: {current_source_type}, Mobile Active: {mobile_camera_active}")
    try:
        if current_source_type != 'mobile' or not mobile_camera_active:
            logger.error(f"Mobile camera not active or incorrect source type. Source: {current_source_type}, Active: {mobile_camera_active}")
            return jsonify({'status': 'error', 'message': 'Mobile camera not active or incorrect source type'}), 400
        data = request.json
        if not data or 'frame' not in data:
            logger.error(f"Invalid request data: {data}")
            return jsonify({'status': 'error', 'message': 'No frame data provided'}), 400
        frame_data = data['frame']
        logger.info("Received mobile frame data")
        if not frame_data.startswith('data:image/jpeg;base64,'):
            logger.error("Frame data does not start with correct MIME type")
            return jsonify({'status': 'error', 'message': 'Invalid frame data format'}), 400
        frame_data = frame_data.split(',')[1]
        try:
            frame_bytes = base64.b64decode(frame_data)
            logger.info("Successfully decoded base64 frame")
        except Exception as e:
            logger.error(f"Base64 decoding failed: {e}")
            return jsonify({'status': 'error', 'message': 'Invalid base64 frame data'}), 400
        try:
            image = Image.open(io.BytesIO(frame_bytes))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            logger.info("Converted frame to OpenCV format")
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return jsonify({'status': 'error', 'message': 'Failed to process image'}), 400
        processed_frame = detect_ppe(frame)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            logger.error("Failed to encode processed frame")
            return jsonify({'status': 'error', 'message': 'Failed to encode frame'}), 500
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        processed_frame_data_url = f"data:image/jpeg;base64,{processed_frame_b64}"
        logger.info("Returning processed mobile frame")
        return jsonify({
            'status': 'success',
            'processed_frame': processed_frame_data_url,
            'violation': violation_detected
        })
    except Exception as e:
        logger.error(f"Error processing mobile frame: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'Error processing frame: {str(e)}'}), 500

@app.route('/violation_status')
def violation_status():
    logger.info(f"Violation status checked: {violation_detected}")
    return jsonify({'status': 'success', 'violation': violation_detected})

@app.route('/update_recipients', methods=['POST'])
def update_recipients():
    global EMAIL_RECIPIENTS
    data = request.json
    new_recipients = data.get('recipients', [])
    if not new_recipients or not all(isinstance(email, str) and '@' in email for email in new_recipients):
        logger.error("Invalid email recipients provided")
        return jsonify({'status': 'error', 'message': 'Invalid email recipients'}), 400
    EMAIL_RECIPIENTS = new_recipients
    logger.info(f"Updated email recipients: {', '.join(EMAIL_RECIPIENTS)}")
    return jsonify({'status': 'success', 'recipients': EMAIL_RECIPIENTS})

@app.route('/get_recipients', methods=['GET'])
def get_recipients():
    logger.info(f"Current email recipients: {', '.join(EMAIL_RECIPIENTS)}")
    return jsonify({'status': 'success', 'recipients': EMAIL_RECIPIENTS})

@app.route('/')
def index():
    logger.info(f"Rendering index.html with source_type={current_source_type}, webcam_active={webcam_active}, mobile_camera_active={mobile_camera_active}")
    return render_template('index.html',
                           current_source_type=current_source_type,
                           webcam_active=webcam_active,
                           mobile_camera_active=mobile_camera_active,
                           mobile_facing=mobile_facing_mode,
                           video_source=video_source if isinstance(video_source, str) else '')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    logger.info("Starting video feed")
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_video():
    global video_source, current_source_type, webcam_active, mobile_camera_active
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
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"File saved locally: {file_path}, Size: {file_size:.2f} MB")
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logger.error(f"Invalid video file: {unique_filename}")
            os.remove(file_path)
            return jsonify({'status': 'error', 'message': 'Invalid video file'}), 400
        cap.release()
        video_source = file_path
        current_source_type = 'uploaded'
        webcam_active = False
        mobile_camera_active = False
        logger.info(f"Source changed to Uploaded Video: {video_source}")
        return jsonify({'status': 'success', 'message': 'Video uploaded successfully', 'url': file_path})
    except Exception as e:
        logger.error(f"Error saving file {unique_filename}: {e}", exc_info=True)
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'status': 'error', 'message': f'Error uploading file: {str(e)}'}), 500

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/change_source', methods=['POST'])
def change_source():
    global video_source, current_source_type, webcam_active, mobile_camera_active, mobile_facing_mode
    source_type = request.form.get('source_type')
    logger.info(f"Request received to change source to: {source_type}")
    if source_type == 'webcam':
        video_source = 0
        current_source_type = 'webcam'
        webcam_active = False
        mobile_camera_active = False
        logger.info(f"Source changed to Webcam ({video_source}). Webcam active state: {webcam_active}")
    elif source_type == 'mobile':
        video_source = None
        current_source_type = 'mobile'
        webcam_active = False
        mobile_camera_active = False
        mobile_facing_mode = 'user'
        logger.info(f"Source changed to Mobile Camera. Mobile active state: {mobile_camera_active}")
    elif source_type == 'cctv':
        cctv_url = request.form.get('cctv_url')
        if cctv_url:
            video_source = cctv_url
            current_source_type = 'cctv'
            webcam_active = False
            mobile_camera_active = False
            logger.info(f"Source changed to CCTV: {video_source}")
        else:
            logger.error("CCTV source selected but URL is missing")
            return jsonify({'status': 'error', 'message': 'CCTV URL cannot be empty'}), 400
    elif source_type == 'uploaded':
        logger.info("'uploaded' source type selected - requires file upload via /upload route")
        return jsonify({'status': 'error', 'message': 'Please upload a video first'}), 400
    return jsonify({'status': 'success', 'message': 'Source changed'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Flask app on http://0.0.0.0:{port}")
    app.run(debug=True, host='0.0.0.0', port=port, threaded=True)