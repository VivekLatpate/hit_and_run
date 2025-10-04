# Accident Detection System with OpenCV

This repository contains scripts to run real-time accident detection using your custom trained YOLO model with OpenCV.

## Model Available

- `i1-yolov8s.pt` - Custom trained YOLOv8 Small model for accident detection

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Model Path

Make sure the model path in the scripts points to your actual model file:
- `D:\camera\Accident-Detection-Web-App\server\models\i1-yolov8s.pt`

## Usage

### Quick Start - Real-time Accident Detection

Run the accident detection system:

```bash
python test_yolo_models.py
```

This will:
1. Test the accident detection model
2. Start real-time webcam detection
3. Show bounding boxes around detected accidents

### Advanced Usage

For more control, use the main integration script:

```bash
python yolo_opencv_integration.py
```

## Features

### YOLOOpenCVProcessor Class

The main class provides these methods:

- `load_model()` - Load the accident detection model
- `detect_objects(image)` - Detect accidents in an image
- `draw_detections(image, results)` - Draw bounding boxes and labels
- `process_video(video_path)` - Process video files for accidents

### Example Usage

```python
from yolo_opencv_integration import YOLOOpenCVProcessor

# Initialize processor with accident detection model
processor = YOLOOpenCVProcessor('path/to/i1-yolov8s.pt')

# Process image
import cv2
image = cv2.imread('your_image.jpg')
results = processor.detect_objects(image)
annotated_image = processor.draw_detections(image, results)

# Display result
cv2.imshow('Accident Detection', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Real-time Accident Detection

The system supports real-time accident detection:

1. Run the script
2. The webcam will start automatically
3. Press 'q' to quit
4. Press 's' to save current frame with detections

## Video Processing

To process video files for accident detection:

```python
processor = YOLOOpenCVProcessor('path/to/i1-yolov8s.pt')
processor.process_video('input_video.mp4', 'accident_output.mp4')
```

## Model Information

- **Model**: Custom trained YOLOv8s for accident detection
- **Classes**: 1 class - "Accident"
- **Performance**: ~50-60ms per frame (real-time capable)
- **Purpose**: Specialized for detecting vehicle accidents

## Troubleshooting

### Common Issues

1. **Model not found**: Check file path in scripts
2. **CUDA errors**: Install PyTorch with CUDA support if using GPU
3. **Webcam not working**: Check camera permissions and availability

### Performance Tips

- The model is optimized for accident detection
- Runs efficiently on CPU and GPU
- Adjust confidence threshold based on your needs (default: 0.5)

## Requirements

- Python 3.8+
- OpenCV 4.8+
- Ultralytics YOLO
- PyTorch
- NumPy

See `requirements.txt` for complete dependency list.
