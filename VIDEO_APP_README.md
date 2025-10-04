# Video Accident Detection Application (app1.py)

## 🚨 Overview
This application allows users to upload video files and automatically scan them for accidents using AI-powered detection. The system combines YOLO object detection with Gemini AI analysis to provide comprehensive accident analysis.

## ✨ Features

### 🎥 Video Upload & Processing
- **Multiple Format Support**: MP4, AVI, MOV, MKV, WMV, FLV, WEBM
- **Real-time Progress**: Progress bar and status updates during processing
- **Frame-by-Frame Analysis**: Scans every 5th frame for optimal performance
- **Automatic Output**: Saves results to organized folders

### 🤖 AI-Powered Detection
- **YOLO Model**: Uses custom-trained `i1-yolov8s.pt` for accident detection
- **Gemini AI Analysis**: Analyzes accident frames with detailed insights
- **Confidence Scoring**: Provides confidence levels for each detection

### 📊 Comprehensive Analysis
- **Vehicle Details**: Identifies vehicle types, colors, and damage
- **License Plates**: Attempts to read license plate numbers
- **Severity Assessment**: Rates accident severity (minor/moderate/severe)
- **Emergency Response**: Recommends appropriate response levels
- **Environmental Factors**: Analyzes road and weather conditions

## 🚀 How to Use

### 1. Launch the Application
```bash
python app1.py
```

### 2. Select Video File
- Click "📁 Select Video File" button
- Choose your video file from the file dialog
- Supported formats: MP4, AVI, MOV, MKV, WMV, FLV, WEBM

### 3. Start Analysis
- Click "🚀 Start Analysis" button
- Watch the progress bar and status updates
- The system will process the video frame by frame

### 4. View Results
- Analysis results appear in the text area
- Detailed reports are saved to `video_accident_detections/` folder
- Each accident gets individual analysis files

## 📁 Output Structure

```
video_accident_detections/
└── [video_name]_[timestamp]/
    ├── comprehensive_video_report.txt
    ├── accident_frame_[frame]_[timestamp].jpg
    ├── analysis_frame_[frame].txt
    └── ... (for each detected accident)
```

## 📋 Sample Output

### Video Analysis Report
```
COMPREHENSIVE VIDEO ACCIDENT DETECTION REPORT
============================================================
Video File: demo_video.mp4
Analysis Time: 2025-10-04 10:45:00
Total Frames: 300
Frames Processed: 300
Video Duration: 10.00 seconds
FPS: 30.00
Accidents Detected: 1
Detection Rate: 0.33%
Model Used: D:\camera\Accident-Detection-Web-App\server\models\i1-yolov8s.pt
AI Analysis: Gemini 2.5 Flash
```

### AI Analysis Example
```
ACCIDENT FRAME ANALYSIS
==================================================
Frame Number: 150
Time in Video: 5.00s
Detection Time: 2025-10-04 10:45:15
Confidence: 0.85
Image File: accident_frame_000150_20251004_104515.jpg

GEMINI AI ANALYSIS:
==================================================
Based on the analysis of this accident scene:

1. Number of vehicles involved: 2 vehicles
2. Vehicle types: Both appear to be passenger cars
3. Vehicle colors: Red car and blue car visible
4. License plate numbers: Not clearly visible due to angle
5. Vehicle damage assessment: Moderate damage visible on both vehicles
6. Accident severity: Moderate - appears to be a side collision
7. Road conditions: Dry road surface, good visibility
8. Weather conditions: Clear weather, good lighting
9. Emergency vehicles present: None visible in this frame
10. Recommended emergency response level: Immediate response required

This appears to be a moderate severity accident requiring emergency services.
```

## 🛠️ Technical Details

### Dependencies
- **OpenCV**: Video processing and image manipulation
- **Ultralytics YOLO**: Object detection model
- **Google Gemini AI**: Advanced image analysis
- **Tkinter**: GUI interface
- **PIL**: Image processing

### Performance
- **Processing Speed**: ~5-10 frames per second (depending on hardware)
- **Memory Usage**: Moderate (loads video frame by frame)
- **Accuracy**: High accuracy with custom-trained model
- **AI Analysis**: ~2-3 seconds per accident frame

### Model Information
- **YOLO Model**: `i1-yolov8s.pt` (custom-trained for accidents)
- **Gemini Model**: `gemini-2.5-flash` (latest multimodal AI)
- **Confidence Threshold**: 0.5 (adjustable in code)

## 🎯 Use Cases

### Law Enforcement
- **Traffic Investigation**: Analyze accident footage
- **Evidence Collection**: Automated accident detection
- **Report Generation**: Detailed AI analysis reports

### Insurance Companies
- **Claim Processing**: Automated accident verification
- **Damage Assessment**: AI-powered damage analysis
- **Fraud Detection**: Identify suspicious patterns

### Transportation Safety
- **Road Safety Analysis**: Identify accident-prone areas
- **Traffic Monitoring**: Real-time accident detection
- **Safety Improvements**: Data-driven safety measures

## 🔧 Configuration

### Model Path
Update the model path in `app1.py`:
```python
self.model_path = r'D:\camera\Accident-Detection-Web-App\server\models\i1-yolov8s.pt'
```

### Gemini API Key
Update the API key in `app1.py`:
```python
self.gemini_api_key = "YOUR_GEMINI_API_KEY"
```

### Processing Settings
Adjust processing parameters:
```python
# Process every Nth frame (default: 5)
if frame_count % 5 == 0:

# Confidence threshold (default: 0.5)
confidence_threshold = 0.5
```

## 📈 Performance Tips

### For Large Videos
- **Reduce Frame Rate**: Process every 10th frame instead of 5th
- **Lower Resolution**: Resize frames before processing
- **Batch Processing**: Process videos in smaller chunks

### For Better Accuracy
- **Higher Resolution**: Use higher quality source videos
- **Good Lighting**: Ensure videos have adequate lighting
- **Stable Footage**: Avoid shaky or blurry videos

## 🚨 Troubleshooting

### Common Issues

1. **"Model not found" Error**
   - Check if `i1-yolov8s.pt` exists in the specified path
   - Verify the model path in the code

2. **"API Error" Messages**
   - Verify Gemini API key is correct
   - Check internet connection
   - Ensure API quota is not exceeded

3. **Slow Processing**
   - Reduce frame processing frequency
   - Close other applications
   - Use SSD storage for better I/O

4. **GUI Not Responding**
   - Processing runs in background thread
   - Wait for completion or restart application

## 🎉 Demo Video

A demo video (`demo_video.mp4`) is included for testing:
- **Duration**: 10 seconds
- **Accident Scene**: Frames 150-200 (5-6.7 seconds)
- **Content**: Simulated car accident with emergency vehicles

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure model files are in correct locations
4. Check API keys and internet connectivity

---

**🎯 Ready to detect accidents in your videos with AI-powered analysis!**
