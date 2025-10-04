"""
Simple example script to demonstrate YOLO model usage with OpenCV
This script shows how to use both YOLO models in your models directory
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def test_model(model_path, model_name):
    """Test a YOLO model with a sample image"""
    print(f"\n=== Testing {model_name} ===")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        # Load the model
        model = YOLO(model_path)
        print(f"✅ Successfully loaded {model_name}")
        
        # Create a sample image
        sample_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(sample_image, f"Testing {model_name}", (50, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(sample_image, "Press any key to continue", (50, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Run detection
        results = model(sample_image)
        
        # Print model info
        print(f"Model classes: {len(model.names)}")
        print(f"Class names: {list(model.names.values())[:5]}...")  # Show first 5 classes
        
        # Display the image
        cv2.imshow(f'{model_name} Test', sample_image)
        cv2.waitKey(2000)  # Show for 2 seconds
        cv2.destroyAllWindows()
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        return False

def webcam_detection(model_path, model_name):
    """Run real-time detection using webcam"""
    print(f"\n=== Webcam Detection with {model_name} ===")
    print("Press 'q' to quit, 's' to save current frame")
    
    # Load model
    model = YOLO(model_path)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection every 5th frame for better performance
        if frame_count % 5 == 0:
            results = model(frame)
            
            # Draw detections
            annotated_frame = frame.copy()
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if confidence > 0.5:  # Confidence threshold
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # Draw label
                        class_name = model.names[class_id]
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow(f'{model_name} Detection', annotated_frame)
        else:
            cv2.imshow(f'{model_name} Detection', frame)
        
        frame_count += 1
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'captured_frame_{model_name}.jpg', frame)
            print(f"Frame saved as captured_frame_{model_name}.jpg")
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function to demonstrate accident detection model"""
    
    # Define model path - only using accident detection model
    model_path = r'D:\camera\Accident-Detection-Web-App\server\models\i1-yolov8s.pt'
    model_name = 'Accident Detection Model'
    
    print("🚀 Accident Detection System with OpenCV")
    print("=" * 50)
    
    # Test the accident detection model
    if test_model(model_path, model_name):
        print(f"\n📹 Starting webcam detection with {model_name}")
        print("Press 'q' to quit, 's' to save current frame")
        
        try:
            webcam_detection(model_path, model_name)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("❌ Accident detection model could not be loaded")

if __name__ == "__main__":
    main()
