# Construction Safety AI

![Construction Safety AI](static/placeholder.jpg)

Construction Safety AI is an advanced system designed to enhance workplace safety on construction sites by detecting Personal Protective Equipment (PPE) violations in real-time. Utilizing the state-of-the-art **YOLOv11** model, the system identifies missing PPE (hardhats, masks, safety vests) and sends instant email alerts via Gmail API. Built with a **Flask-based frontend and backend**, it supports video uploads, live webcam detection, and CCTV stream processing, ensuring robust safety compliance monitoring.

## Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Team](#team)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
Construction Safety AI addresses the critical need for safety compliance in construction environments. By leveraging **YOLOv11**, trained on curated datasets from **Roboflow** and **Kaggle**, the system detects PPE violations in real-time video feeds. The Flask frontend provides an intuitive interface for uploading videos, toggling webcam feeds, and configuring CCTV streams. Upon detecting violations, the system sends email alerts using the Gmail API, ensuring prompt action by site supervisors.

Key objectives:
- Real-time PPE violation detection.
- Seamless integration of video sources (webcam, CCTV, uploaded videos).
- Automated email notifications for safety breaches.
- User-friendly Flask-based interface.

## System Architecture
The system follows a modular architecture:

1. **Frontend (Flask)**:
   - Built with Flask templates (`index.html`, `about.html`) and CSS (`style.css`).
   - Features a dark-themed UI with Inter font (About page) and Roboto Mono (main page).
   - Supports video upload, live detection display, webcam toggle, and CCTV URL input.

2. **Backend (Flask)**:
   - Handles video processing, source switching, and API endpoints (`/upload`, `/video_feed`, `/toggle_webcam`, `/test_email`).
   - Integrates Gmail API for sending violation alerts to `myselfkarthik1@gmail.com`.

3. **AI Model (YOLOv11)**:
   - Trained on PPE detection datasets from Roboflow and Kaggle.
   - Processes video frames to detect missing hardhats, masks, or safety vests.
   - Outputs annotated frames displayed in the `#video-feed` element.

4. **Video Processing**:
   - Supports multiple sources: uploaded videos (stored in `uploads/`), webcam, and CCTV streams.
   - Streams processed frames via Flask’s `/video_feed` endpoint.

5. **Email Notifications**:
   - Gmail API sends alerts when violations are detected.
   - Configured with `credentials.json` and `token.pickle` for authentication.

**Architecture Diagram**:
```
[Video Source: Webcam/CCTV/Upload] --> [Flask Backend: Video Processing]
       |                                    |
       v                                    v
[YOLOv11: PPE Detection] <--> [Flask Frontend: Display & Controls]
       |                                    |
       v                                    v
[Gmail API: Email Alerts] <--> [User: Safety Supervisor]
```

## Features
- **Real-Time Detection**: Identifies PPE violations using YOLOv11.
- **Video Source Flexibility**: Supports webcam, CCTV, and uploaded videos (`.mp4`, `.avi`, `.mov`).
- **Email Alerts**: Sends notifications via Gmail API for detected violations.
- **User-Friendly Interface**: Flask frontend with toggle controls and violation alerts.
- **Responsive Design**: Adapts to desktop and mobile devices.

## Dataset
The YOLOv11 model was trained on open-source datasets from:
- **Roboflow**: Curated PPE detection datasets with annotated images of construction workers.
- **Kaggle**: Additional PPE-related datasets for enhanced model robustness.

**Team Contributions**:
- **Data Collection**: Karthik C S, Karthickkumaran A, Tharunrajan S.
- **Preprocessing & Augmentation**: Karthik C S, Tharunrajan S.
- **Model Training**: Karthik C S, Karthickkumaran A.

## Technologies Used
- **Frontend**: Flask, HTML, CSS, JavaScript
- **Backend**: Flask, Python
- **AI Model**: YOLOv11 (Ultralytics)
- **Email Integration**: Gmail API
- **Dependencies**: OpenCV, NumPy, PyTorch, Flask
- **Fonts**: Inter (About page), Roboto Mono (main page)
- **Icons**: Font Awesome

## Team
The project was developed under the guidance of:
- **Mrs. T. Dhivya** (Project Mentor, Assistant Professor)

Team members and their contributions:
- **Karthik C S** (Team Lead):
  - Data Collection, Preprocessing, Augmentation
  - AI Model Training
  - Front End Development
- **Karthickkumaran A** (Team Member):
  - Front End UI
  - Data Collection
  - AI Model Training
- **Tharunrajan S** (Team Member):
  - Data Collection
  - Testing AI Model
  - Testing Front End
  - Augmentation

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/ppe-detection-app.git
   cd ppe-detection-app
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Gmail API**:
   - Download `credentials.json` from [Google Cloud Console](https://console.cloud.google.com/).
   - Place it in the project root.
   - Run `generate_token.py` to create `token.pickle`:
     ```bash
     python generate_token.py
     ```

5. **Download YOLOv11 Model**:
   - Place the trained `best.pt` model in `model/`.

6. **Directory Structure**:
   ```plaintext
   ppe-detection-app/
   ├── app.py
   ├── templates/
   │   ├── index.html
   │   └── about.html
   ├── static/
   │   ├── style.css
   │   ├── favicon.ico
   │   └── placeholder.jpg
   ├── model/
   │   └── best.pt
   ├── uploads/
   ├── credentials.json
   ├── token.pickle
   ├── requirements.txt
   ├── generate_token.py
   ```

## Usage
1. **Run the Application**:
   ```bash
   python app.py
   ```

2. **Access the Web Interface**:
   - Open `http://127.0.0.1:8080` in a browser.
   - Upload a video, toggle the webcam, or enter a CCTV URL.
   - View live detections in the `#video-feed` section.
   - Check `myselfkarthik1@gmail.com` for violation alerts.

3. **Test Email Functionality**:
   ```bash
   curl http://127.0.0.1:8080/test_email
   ```

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.