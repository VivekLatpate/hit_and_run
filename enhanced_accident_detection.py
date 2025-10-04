"""
Enhanced Accident Detection System with Visual Alerts
- Green outline: No accident detected
- Red outline: Accident detected
- Audio alerts and visual notifications
- Works with webcam or sample images
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import threading

class EnhancedAccidentDetector:
    def __init__(self, model_path):
        """
        Initialize Enhanced Accident Detection System
        
        Args:
            model_path (str): Path to the accident detection model file (.pt)
        """
        self.model_path = model_path
        self.model = None
        self.accident_detected = False
        self.alert_cooldown = 0
        self.last_alert_time = 0
        self.load_model()
    
    def load_model(self):
        """Load the accident detection model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Successfully loaded Accident Detection Model: {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model {self.model_path}: {e}")
            self.model = None
    
    def detect_accidents(self, image):
        """
        Detect accidents in an image
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            results: YOLO detection results
        """
        if self.model is None:
            return None
        
        try:
            results = self.model(image)
            return results
        except Exception as e:
            print(f"Error during detection: {e}")
            return None
    
    def play_alert_sound(self):
        """Play alert sound for accident detection"""
        try:
            # Try to play Windows system sound
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except:
            print("🔊 ALERT: ACCIDENT DETECTED!")
    
    def draw_enhanced_detections(self, image, results, confidence_threshold=0.5, simulate_accident=False):
        """
        Draw enhanced bounding boxes with color-coded alerts
        
        Args:
            image: OpenCV image
            results: YOLO detection results
            confidence_threshold: Minimum confidence for detections
            simulate_accident: Force accident detection for testing
            
        Returns:
            annotated_image: Image with enhanced detections
        """
        annotated_image = image.copy()
        current_time = time.time()
        
        # Default to green outline (no accident)
        outline_color = (0, 255, 0)  # Green
        status_text = "SAFE - No Accident Detected"
        status_color = (0, 255, 0)
        
        # Check for real detections
        real_accident_detected = False
        accident_count = 0
        
        if results is not None and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None:
                for box in boxes:
                    confidence = box.conf[0].cpu().numpy()
                    if confidence >= confidence_threshold:
                        accident_count += 1
                        real_accident_detected = True
        
        # Use simulation if no real accident detected
        if simulate_accident and not real_accident_detected:
            accident_count = 1
            real_accident_detected = True
        
        if real_accident_detected or simulate_accident:
            # Accident detected - Red outline
            self.accident_detected = True
            outline_color = (0, 0, 255)  # Red
            status_text = f"⚠️ ALERT: {accident_count} ACCIDENT(S) DETECTED!"
            status_color = (0, 0, 255)
            
            # Play alert sound (with cooldown to avoid spam)
            if current_time - self.last_alert_time > 2.0:  # 2 second cooldown
                self.play_alert_sound()
                self.last_alert_time = current_time
            
            # Draw red bounding boxes for accidents
            if results is not None and len(results) > 0 and not simulate_accident:
                # Draw real detections
                result = results[0]
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        if confidence >= confidence_threshold:
                            # Draw thick red bounding box
                            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), 
                                        (0, 0, 255), 4)
                            
                            # Draw red label background
                            label = f"ACCIDENT: {confidence:.2f}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                            cv2.rectangle(annotated_image, (int(x1), int(y1) - label_size[1] - 15), 
                                        (int(x1) + label_size[0], int(y1)), (0, 0, 255), -1)
                            
                            # Draw white text
                            cv2.putText(annotated_image, label, (int(x1), int(y1) - 8), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                # Draw simulated accident box
                cv2.rectangle(annotated_image, (200, 200), (440, 280), (0, 0, 255), 4)
                cv2.putText(annotated_image, "SIMULATED ACCIDENT", (210, 250), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            self.accident_detected = False
        
        # Draw status bar at the top
        cv2.rectangle(annotated_image, (0, 0), (image.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(annotated_image, status_text, (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Draw border around entire frame
        cv2.rectangle(annotated_image, (0, 0), (image.shape[1]-1, image.shape[0]-1), 
                     outline_color, 8)
        
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_image, timestamp, (image.shape[1] - 200, image.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_image
    
    def create_sample_images(self):
        """Create sample images to demonstrate the system"""
        print("📸 Creating sample images for demonstration...")
        
        # Sample 1: Safe scene (green outline) - Normal road scene
        safe_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a simple road
        cv2.rectangle(safe_image, (0, 300), (640, 480), (100, 100, 100), -1)  # Road
        cv2.rectangle(safe_image, (0, 0), (640, 300), (135, 206, 235), -1)  # Sky
        
        # Draw some cars (normal driving)
        cv2.rectangle(safe_image, (100, 320), (200, 380), (0, 0, 255), -1)  # Red car
        cv2.rectangle(safe_image, (300, 320), (400, 380), (0, 255, 0), -1)  # Green car
        cv2.rectangle(safe_image, (500, 320), (600, 380), (255, 0, 0), -1)  # Blue car
        
        cv2.putText(safe_image, "NORMAL TRAFFIC FLOW", (150, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Sample 2: Accident scene (red outline) - Simulated accident
        accident_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a road
        cv2.rectangle(accident_image, (0, 300), (640, 480), (100, 100, 100), -1)  # Road
        cv2.rectangle(accident_image, (0, 0), (640, 300), (135, 206, 235), -1)  # Sky
        
        # Draw crashed cars (overlapping/angled)
        cv2.rectangle(accident_image, (200, 320), (350, 380), (0, 0, 255), -1)  # Red car
        cv2.rectangle(accident_image, (300, 300), (450, 400), (255, 0, 0), -1)  # Blue car (overlapping)
        
        # Add some debris/sparks effect
        for i in range(10):
            x = np.random.randint(150, 500)
            y = np.random.randint(300, 400)
            cv2.circle(accident_image, (x, y), 3, (0, 255, 255), -1)  # Yellow sparks
        
        cv2.putText(accident_image, "VEHICLE COLLISION DETECTED!", (80, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        
        return safe_image, accident_image
    
    def run_demo_mode(self):
        """Run demo mode with sample images"""
        print("🚀 Starting Enhanced Accident Detection System - DEMO MODE")
        print("=" * 60)
        print("📹 Camera Status: DEMO MODE")
        print("🎯 Detection: ACTIVE")
        print("🔊 Alerts: ENABLED")
        print("=" * 60)
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("  - Press 'n' for next sample")
        print("  - Press 't' to toggle accident simulation")
        print("=" * 60)
        
        # Create sample images
        safe_image, accident_image = self.create_sample_images()
        samples = [safe_image, accident_image]
        sample_names = ["Safe Scene", "Accident Scene"]
        current_sample = 0
        simulate_accident = False
        
        frame_count = 0
        accident_frames = 0
        
        while True:
            current_image = samples[current_sample]
            
            # Run detection
            results = self.detect_accidents(current_image)
            
            # Use simulation for accident scene or when toggled
            use_simulation = (current_sample == 1) or simulate_accident
            annotated_frame = self.draw_enhanced_detections(current_image, results, simulate_accident=use_simulation)
            
            # Count accident frames
            if self.accident_detected:
                accident_frames += 1
            
            # Add sample info
            cv2.putText(annotated_frame, f"Sample: {sample_names[current_sample]}", 
                       (10, annotated_frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add simulation status
            sim_text = "SIMULATION: ON" if use_simulation else "SIMULATION: OFF"
            sim_color = (0, 255, 255) if use_simulation else (255, 255, 255)
            cv2.putText(annotated_frame, sim_text, 
                       (10, annotated_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, sim_color, 2)
            
            # Display frame
            cv2.imshow('🚨 Enhanced Accident Detection System - DEMO', annotated_frame)
            
            frame_count += 1
            
            # Handle key presses
            key = cv2.waitKey(1000) & 0xFF  # Wait 1 second between frames
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'accident_detection_demo_{int(time.time())}.jpg'
                cv2.imwrite(filename, annotated_frame)
                print(f"📸 Frame saved as: {filename}")
            elif key == ord('n'):
                current_sample = (current_sample + 1) % len(samples)
                print(f"🔄 Switching to: {sample_names[current_sample]}")
            elif key == ord('t'):
                simulate_accident = not simulate_accident
                print(f"🔄 Accident simulation: {'ON' if simulate_accident else 'OFF'}")
        
        # Cleanup and statistics
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("📊 Demo Statistics:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Frames with accidents: {accident_frames}")
        if frame_count > 0:
            print(f"  Accident detection rate: {(accident_frames/frame_count)*100:.2f}%")
        print("=" * 60)
        print("👋 Demo complete")
    
    def run_webcam_detection(self):
        """Run real-time accident detection with enhanced alerts"""
        print("🚀 Starting Enhanced Accident Detection System")
        print("=" * 60)
        print("📹 Camera Status: ON")
        print("🎯 Detection: ACTIVE")
        print("🔊 Alerts: ENABLED")
        print("=" * 60)
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("  - Press 'r' to reset alert cooldown")
        print("=" * 60)
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open webcam - Switching to DEMO MODE")
            self.run_demo_mode()
            return
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        accident_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Could not read from webcam - Switching to DEMO MODE")
                cap.release()
                self.run_demo_mode()
                return
            
            # Run detection every 3rd frame for better performance
            if frame_count % 3 == 0:
                results = self.detect_accidents(frame)
                annotated_frame = self.draw_enhanced_detections(frame, results)
            else:
                # Use previous detection result for smooth display
                annotated_frame = self.draw_enhanced_detections(frame, None)
            
            # Count accident frames
            if self.accident_detected:
                accident_frames += 1
            
            # Display frame
            cv2.imshow('🚨 Enhanced Accident Detection System', annotated_frame)
            
            frame_count += 1
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'accident_detection_frame_{int(time.time())}.jpg'
                cv2.imwrite(filename, annotated_frame)
                print(f"📸 Frame saved as: {filename}")
            elif key == ord('r'):
                self.last_alert_time = 0
                print("🔄 Alert cooldown reset")
        
        # Cleanup and statistics
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("📊 Detection Statistics:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Frames with accidents: {accident_frames}")
        if frame_count > 0:
            print(f"  Accident detection rate: {(accident_frames/frame_count)*100:.2f}%")
        print("=" * 60)
        print("👋 System shutdown complete")

def main():
    """Main function to run enhanced accident detection"""
    
    # Model path
    model_path = r'D:\camera\Accident-Detection-Web-App\server\models\i1-yolov8s.pt'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Initialize enhanced detector
    detector = EnhancedAccidentDetector(model_path)
    
    if detector.model is None:
        print("❌ Failed to load accident detection model")
        return
    
    # Ask user for mode
    print("🎯 Choose Detection Mode:")
    print("1. Webcam Detection (Real-time)")
    print("2. Demo Mode (Sample Images)")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            detector.run_webcam_detection()
        elif choice == "2":
            detector.run_demo_mode()
        else:
            print("Invalid choice - Starting Demo Mode")
            detector.run_demo_mode()
            
    except KeyboardInterrupt:
        print("\n🛑 Detection stopped by user")
    except Exception as e:
        print(f"❌ Error during detection: {e}")

if __name__ == "__main__":
    main()