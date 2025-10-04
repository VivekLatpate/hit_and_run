"""
Simple Accident Detection System with Camera and Popup Alerts
- Camera turns on automatically
- Detects accidents in real-time
- Shows popup alert when accident happens
- Green border = Safe, Red border = Accident detected
"""

import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox
import threading
import time
import os

class AccidentDetectionSystem:
    def __init__(self, model_path):
        """Initialize the accident detection system"""
        self.model_path = model_path
        self.model = None
        self.camera_running = False
        self.load_model()
    
    def load_model(self):
        """Load the accident detection model"""
        try:
            self.model = YOLO(self.model_path)
            print("✅ Accident Detection Model Loaded Successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def show_accident_alert(self):
        """Show popup alert when accident is detected"""
        try:
            # Create a popup window
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            
            # Show the alert message
            messagebox.showerror(
                "🚨 ACCIDENT DETECTED! 🚨", 
                "EMERGENCY ALERT!\n\nAccident has been detected!\n\nPlease check the camera feed immediately.\n\nEmergency services may need to be contacted."
            )
            
            root.destroy()
        except Exception as e:
            print(f"Alert error: {e}")
    
    def detect_accidents(self, frame):
        """Detect accidents in the frame"""
        if self.model is None:
            return False, frame
        
        try:
            # Run detection
            results = self.model(frame)
            
            # Check if any accidents detected
            accident_detected = False
            confidence_threshold = 0.5
            
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    confidence = box.conf[0].cpu().numpy()
                    if confidence >= confidence_threshold:
                        accident_detected = True
                        
                        # Draw red bounding box around accident
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                        
                        # Add accident label
                        label = f"ACCIDENT: {confidence:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            return accident_detected, frame
            
        except Exception as e:
            print(f"Detection error: {e}")
            return False, frame
    
    def run_camera_detection(self):
        """Run the camera detection system"""
        print("🚀 Starting Accident Detection System")
        print("=" * 50)
        print("📹 Camera: ON")
        print("🎯 Detection: ACTIVE")
        print("🔊 Alerts: ENABLED")
        print("=" * 50)
        print("Press 'q' to quit")
        print("=" * 50)
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            print("Trying alternative camera sources...")
            
            # Try different camera indices
            for i in range(1, 5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"✅ Camera opened at index {i}")
                    break
            
            if not cap.isOpened():
                print("❌ No camera found! Please check your camera connection.")
                return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.camera_running = True
        frame_count = 0
        accident_count = 0
        last_alert_time = 0
        
        print("🎥 Camera is now running...")
        
        while self.camera_running:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Cannot read from camera!")
                break
            
            # Run detection every 5th frame for better performance
            if frame_count % 5 == 0:
                accident_detected, processed_frame = self.detect_accidents(frame)
                
                if accident_detected:
                    accident_count += 1
                    
                    # Show popup alert (with cooldown to avoid spam)
                    current_time = time.time()
                    if current_time - last_alert_time > 3.0:  # 3 second cooldown
                        # Show alert in a separate thread to avoid blocking
                        alert_thread = threading.Thread(target=self.show_accident_alert)
                        alert_thread.daemon = True
                        alert_thread.start()
                        last_alert_time = current_time
                        print("🚨 ACCIDENT DETECTED! Alert popup shown!")
            else:
                processed_frame = frame
            
            # Add status information to frame
            status_color = (0, 255, 0) if not accident_detected else (0, 0, 255)
            status_text = "SAFE" if not accident_detected else "ACCIDENT DETECTED!"
            
            # Draw status bar
            cv2.rectangle(processed_frame, (0, 0), (640, 50), (0, 0, 0), -1)
            cv2.putText(processed_frame, f"Status: {status_text}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Draw border around frame
            border_color = (0, 255, 0) if not accident_detected else (0, 0, 255)
            cv2.rectangle(processed_frame, (0, 0), (639, 479), border_color, 5)
            
            # Add frame counter
            cv2.putText(processed_frame, f"Frame: {frame_count}", (500, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('🚨 Accident Detection System', processed_frame)
            
            frame_count += 1
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Show final statistics
        print("\n" + "=" * 50)
        print("📊 Detection Statistics:")
        print(f"Total frames processed: {frame_count}")
        print(f"Accidents detected: {accident_count}")
        if frame_count > 0:
            print(f"Detection rate: {(accident_count/frame_count)*100:.2f}%")
        print("=" * 50)
        print("👋 System stopped")

def main():
    """Main function to run the accident detection system"""
    
    # Model path
    model_path = r'D:\camera\Accident-Detection-Web-App\server\models\i1-yolov8s.pt'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please check the model path!")
        return
    
    # Create detection system
    detector = AccidentDetectionSystem(model_path)
    
    if detector.model is None:
        print("❌ Failed to load accident detection model")
        return
    
    try:
        # Start the camera detection
        detector.run_camera_detection()
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
