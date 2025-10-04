"""
Enhanced Accident Detection System with Frame Capture and Timestamps
- Saves accident frames automatically
- Shows exact time when accident happened
- Captures 2-4 frames around accident detection
- Provides detailed accident report
"""

import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox
import threading
import time
import os
from datetime import datetime

class EnhancedAccidentDetectionSystem:
    def __init__(self, model_path):
        """Initialize the enhanced accident detection system"""
        self.model_path = model_path
        self.model = None
        self.camera_running = False
        self.accident_frames = []
        self.frame_buffer = []
        self.buffer_size = 10  # Keep last 10 frames
        self.load_model()
        
        # Create accident folder
        self.accident_folder = "accident_detections"
        if not os.path.exists(self.accident_folder):
            os.makedirs(self.accident_folder)
    
    def load_model(self):
        """Load the accident detection model"""
        try:
            self.model = YOLO(self.model_path)
            print("✅ Accident Detection Model Loaded Successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def show_accident_alert(self, accident_time, frame_count):
        """Show popup alert with accident details"""
        try:
            root = tk.Tk()
            root.withdraw()
            
            messagebox.showerror(
                "🚨 ACCIDENT DETECTED! 🚨", 
                f"EMERGENCY ALERT!\n\n"
                f"Accident detected at:\n"
                f"Time: {accident_time}\n"
                f"Frame: {frame_count}\n\n"
                f"Frames have been saved to:\n"
                f"{self.accident_folder}/ folder\n\n"
                f"Please check immediately!"
            )
            root.destroy()
        except Exception as e:
            print(f"Alert error: {e}")
    
    def save_accident_frames(self, accident_time, frame_count):
        """Save accident frames with timestamps"""
        try:
            timestamp_str = accident_time.strftime("%Y%m%d_%H%M%S")
            accident_folder = os.path.join(self.accident_folder, f"accident_{timestamp_str}")
            
            if not os.path.exists(accident_folder):
                os.makedirs(accident_folder)
            
            # Save frames from buffer (2-4 frames around accident)
            frames_to_save = min(4, len(self.frame_buffer))
            saved_frames = []
            
            for i, (frame, frame_num) in enumerate(self.frame_buffer[-frames_to_save:]):
                filename = f"frame_{frame_num:06d}_{timestamp_str}.jpg"
                filepath = os.path.join(accident_folder, filename)
                
                # Add timestamp and frame info to image
                annotated_frame = frame.copy()
                cv2.putText(annotated_frame, f"Accident Detection", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Time: {accident_time.strftime('%H:%M:%S')}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Frame: {frame_num}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imwrite(filepath, annotated_frame)
                saved_frames.append(filepath)
            
            # Create accident report
            report_path = os.path.join(accident_folder, "accident_report.txt")
            with open(report_path, 'w') as f:
                f.write("ACCIDENT DETECTION REPORT\n")
                f.write("=" * 50 + "\n")
                f.write(f"Detection Time: {accident_time}\n")
                f.write(f"Frame Number: {frame_count}\n")
                f.write(f"Frames Captured: {len(saved_frames)}\n")
                f.write(f"Model Used: {self.model_path}\n")
                f.write("\nSaved Files:\n")
                for i, filepath in enumerate(saved_frames, 1):
                    f.write(f"{i}. {os.path.basename(filepath)}\n")
                f.write("\n" + "=" * 50 + "\n")
            
            print(f"📸 Accident frames saved to: {accident_folder}")
            print(f"📄 Report saved: {report_path}")
            
            return accident_folder, saved_frames
            
        except Exception as e:
            print(f"Error saving frames: {e}")
            return None, []
    
    def detect_accidents(self, frame):
        """Detect accidents in the frame"""
        if self.model is None:
            return False, frame, 0.0
        
        try:
            results = self.model(frame)
            
            accident_detected = False
            max_confidence = 0.0
            confidence_threshold = 0.5
            
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    confidence = box.conf[0].cpu().numpy()
                    if confidence >= confidence_threshold:
                        accident_detected = True
                        max_confidence = max(max_confidence, confidence)
                        
                        # Draw red bounding box around accident
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                        
                        # Add accident label
                        label = f"ACCIDENT: {confidence:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            return accident_detected, frame, max_confidence
            
        except Exception as e:
            print(f"Detection error: {e}")
            return False, frame, 0.0
    
    def run_camera_detection(self):
        """Run the enhanced camera detection system"""
        print("🚀 Starting Enhanced Accident Detection System")
        print("=" * 60)
        print("📹 Camera: ON")
        print("🎯 Detection: ACTIVE")
        print("🔊 Alerts: ENABLED")
        print("📸 Auto-save: ENABLED")
        print("=" * 60)
        print("Press 'q' to quit")
        print("Press 's' to manually save current frame")
        print("=" * 60)
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.camera_running = True
        frame_count = 0
        accident_count = 0
        last_alert_time = 0
        
        print("🎥 Camera is now running...")
        print("📁 Accident frames will be saved to:", self.accident_folder)
        
        while self.camera_running:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Cannot read from camera!")
                break
            
            # Add frame to buffer
            self.frame_buffer.append((frame.copy(), frame_count))
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)
            
            # Run detection every 3rd frame for better performance
            if frame_count % 3 == 0:
                accident_detected, processed_frame, confidence = self.detect_accidents(frame)
                
                if accident_detected:
                    accident_count += 1
                    current_time = datetime.now()
                    
                    # Show popup alert (with cooldown)
                    if time.time() - last_alert_time > 3.0:
                        # Save accident frames
                        accident_folder, saved_files = self.save_accident_frames(current_time, frame_count)
                        
                        # Show alert in separate thread
                        alert_thread = threading.Thread(
                            target=self.show_accident_alert, 
                            args=(current_time, frame_count)
                        )
                        alert_thread.daemon = True
                        alert_thread.start()
                        
                        last_alert_time = time.time()
                        print(f"🚨 ACCIDENT DETECTED!")
                        print(f"   Time: {current_time.strftime('%H:%M:%S')}")
                        print(f"   Frame: {frame_count}")
                        print(f"   Confidence: {confidence:.2f}")
                        print(f"   Frames saved: {len(saved_files)}")
            else:
                processed_frame = frame
                accident_detected = False
                confidence = 0.0
            
            # Add status information to frame
            status_color = (0, 255, 0) if not accident_detected else (0, 0, 255)
            status_text = "SAFE" if not accident_detected else f"ACCIDENT! ({confidence:.2f})"
            
            # Draw status bar
            cv2.rectangle(processed_frame, (0, 0), (640, 80), (0, 0, 0), -1)
            cv2.putText(processed_frame, f"Status: {status_text}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(processed_frame, f"Frame: {frame_count}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (10, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw border around frame
            border_color = (0, 255, 0) if not accident_detected else (0, 0, 255)
            cv2.rectangle(processed_frame, (0, 0), (639, 479), border_color, 5)
            
            # Display the frame
            cv2.imshow('🚨 Enhanced Accident Detection System', processed_frame)
            
            frame_count += 1
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Manually save current frame
                timestamp = datetime.now()
                filename = f"manual_save_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                filepath = os.path.join(self.accident_folder, filename)
                cv2.imwrite(filepath, processed_frame)
                print(f"📸 Manual save: {filepath}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Show final statistics
        print("\n" + "=" * 60)
        print("📊 Detection Statistics:")
        print(f"Total frames processed: {frame_count}")
        print(f"Accidents detected: {accident_count}")
        if frame_count > 0:
            print(f"Detection rate: {(accident_count/frame_count)*100:.2f}%")
        print(f"Accident folder: {self.accident_folder}")
        print("=" * 60)
        print("👋 System stopped")

def main():
    """Main function to run the enhanced accident detection system"""
    
    # Model path
    model_path = r'D:\camera\Accident-Detection-Web-App\server\models\i1-yolov8s.pt'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please check the model path!")
        return
    
    # Create detection system
    detector = EnhancedAccidentDetectionSystem(model_path)
    
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

