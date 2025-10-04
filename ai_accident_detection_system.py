"""
Enhanced Accident Detection System with Gemini AI Analysis
- Captures frames around accident detection
- Analyzes accident frames with Gemini API
- Identifies car details, license plates, vehicle types
- Provides detailed accident analysis
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
import base64
import requests
import json
import google.generativeai as genai

class GeminiAccidentAnalyzer:
    def __init__(self, api_key):
        """Initialize Gemini API analyzer"""
        self.api_key = api_key
        # Configure the API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def analyze_accident_frame(self, image, accident_time, frame_number):
        """Analyze accident frame using Gemini API"""
        try:
            # Convert image to PIL format for Gemini
            from PIL import Image
            import numpy as np
            
            # Convert OpenCV image to PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            prompt = f"""
            Analyze this accident scene image taken at {accident_time}. Frame number: {frame_number}.
            
            Please provide detailed analysis including:
            1. Number of vehicles involved
            2. Vehicle types (car, truck, motorcycle, etc.)
            3. Vehicle colors
            4. License plate numbers (if visible)
            5. Vehicle damage assessment
            6. Accident severity (minor, moderate, severe)
            7. Road conditions
            8. Weather conditions (if visible)
            9. Any emergency vehicles present
            10. Recommended emergency response level
            
            Format your response as a structured analysis.
            """
            
            # Generate content with image and text
            response = self.model.generate_content([prompt, pil_image])
            
            if response.text:
                return response.text
            else:
                return "No analysis available from Gemini API"
                
        except Exception as e:
            print(f"Error analyzing frame: {e}")
            return f"Analysis error: {str(e)}"

class EnhancedAccidentDetectionWithAI:
    def __init__(self, model_path, gemini_api_key):
        """Initialize Enhanced Accident Detection System with AI Analysis"""
        self.model_path = model_path
        self.model = None
        self.gemini_analyzer = GeminiAccidentAnalyzer(gemini_api_key)
        self.camera_running = False
        self.accident_frames = []
        self.frame_buffer = []
        self.buffer_size = 15  # Keep more frames for better analysis
        self.load_model()
        
        # Create accident folder
        self.accident_folder = "accident_detections_with_ai"
        if not os.path.exists(self.accident_folder):
            os.makedirs(self.accident_folder)
    
    def load_model(self):
        """Load the accident detection model"""
        try:
            self.model = YOLO(self.model_path)
            print("✅ Accident Detection Model Loaded Successfully!")
            print("✅ Gemini AI Analyzer Initialized!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def show_accident_alert(self, accident_time, frame_count, analysis=None):
        """Show popup alert with accident details and AI analysis"""
        try:
            root = tk.Tk()
            root.withdraw()
            
            message = f"EMERGENCY ALERT!\n\nAccident detected at:\nTime: {accident_time}\nFrame: {frame_count}\n\n"
            
            if analysis:
                message += f"AI Analysis Preview:\n{analysis[:200]}...\n\n"
            
            message += f"Detailed analysis saved to:\n{self.accident_folder}/ folder\n\nPlease check immediately!"
            
            messagebox.showerror("🚨 ACCIDENT DETECTED! 🚨", message)
            root.destroy()
        except Exception as e:
            print(f"Alert error: {e}")
    
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
    
    def save_accident_frames_with_ai_analysis(self, accident_time, frame_count):
        """Save accident frames and analyze with Gemini AI"""
        try:
            timestamp_str = accident_time.strftime("%Y%m%d_%H%M%S")
            accident_folder = os.path.join(self.accident_folder, f"accident_{timestamp_str}")
            
            if not os.path.exists(accident_folder):
                os.makedirs(accident_folder)
            
            # Save frames from buffer (around accident detection)
            frames_to_save = min(6, len(self.frame_buffer))  # Save more frames for analysis
            saved_frames = []
            ai_analyses = []
            
            print(f"🔍 Analyzing {frames_to_save} frames with Gemini AI...")
            
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
                
                # Analyze frame with Gemini AI
                print(f"🤖 Analyzing frame {frame_num}...")
                analysis = self.gemini_analyzer.analyze_accident_frame(
                    frame, accident_time.strftime('%Y-%m-%d %H:%M:%S'), frame_num
                )
                
                if analysis:
                    ai_analyses.append({
                        'frame_number': frame_num,
                        'filename': filename,
                        'analysis': analysis
                    })
                    
                    # Save individual analysis
                    analysis_file = os.path.join(accident_folder, f"analysis_frame_{frame_num:06d}.txt")
                    with open(analysis_file, 'w', encoding='utf-8') as f:
                        f.write(f"ACCIDENT FRAME ANALYSIS\n")
                        f.write("=" * 50 + "\n")
                        f.write(f"Frame Number: {frame_num}\n")
                        f.write(f"Detection Time: {accident_time}\n")
                        f.write(f"Image File: {filename}\n")
                        f.write("\n" + "=" * 50 + "\n")
                        f.write("GEMINI AI ANALYSIS:\n")
                        f.write("=" * 50 + "\n")
                        f.write(analysis)
                        f.write("\n" + "=" * 50 + "\n")
            
            # Create comprehensive accident report
            report_path = os.path.join(accident_folder, "comprehensive_accident_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("COMPREHENSIVE ACCIDENT DETECTION REPORT\n")
                f.write("=" * 60 + "\n")
                f.write(f"Detection Time: {accident_time}\n")
                f.write(f"Frame Number: {frame_count}\n")
                f.write(f"Frames Captured: {len(saved_frames)}\n")
                f.write(f"Model Used: {self.model_path}\n")
                f.write(f"AI Analysis: Gemini 1.5 Flash\n")
                f.write(f"Analysis Timestamp: {datetime.now()}\n")
                f.write("\n" + "=" * 60 + "\n")
                f.write("SAVED FILES:\n")
                f.write("=" * 60 + "\n")
                for i, filepath in enumerate(saved_frames, 1):
                    f.write(f"{i}. {os.path.basename(filepath)}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("AI ANALYSIS SUMMARY:\n")
                f.write("=" * 60 + "\n")
                
                for analysis_data in ai_analyses:
                    f.write(f"\nFRAME {analysis_data['frame_number']} ANALYSIS:\n")
                    f.write("-" * 40 + "\n")
                    f.write(analysis_data['analysis'])
                    f.write("\n" + "-" * 40 + "\n")
            
            print(f"📸 Accident frames and AI analysis saved to: {accident_folder}")
            print(f"📄 Comprehensive report saved: {report_path}")
            
            return accident_folder, saved_frames, ai_analyses
            
        except Exception as e:
            print(f"Error saving frames and analysis: {e}")
            return None, [], []
    
    def run_camera_detection(self):
        """Run the enhanced camera detection system with AI analysis"""
        print("🚀 Starting Enhanced Accident Detection System with AI Analysis")
        print("=" * 70)
        print("📹 Camera: ON")
        print("🎯 Detection: ACTIVE")
        print("🔊 Alerts: ENABLED")
        print("📸 Auto-save: ENABLED")
        print("🤖 AI Analysis: GEMINI API")
        print("=" * 70)
        print("Press 'q' to quit")
        print("Press 's' to manually save current frame")
        print("=" * 70)
        
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
        print("📁 Accident frames and AI analysis will be saved to:", self.accident_folder)
        
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
                    if time.time() - last_alert_time > 5.0:  # 5 second cooldown for AI analysis
                        print(f"🚨 ACCIDENT DETECTED! Starting AI analysis...")
                        print(f"   Time: {current_time.strftime('%H:%M:%S')}")
                        print(f"   Frame: {frame_count}")
                        print(f"   Confidence: {confidence:.2f}")
                        
                        # Save frames and analyze with AI
                        accident_folder, saved_files, ai_analyses = self.save_accident_frames_with_ai_analysis(current_time, frame_count)
                        
                        # Show alert with AI analysis preview
                        analysis_preview = ""
                        if ai_analyses and len(ai_analyses) > 0:
                            analysis_preview = ai_analyses[-1]['analysis'][:200] + "..."
                        
                        # Show alert in separate thread
                        alert_thread = threading.Thread(
                            target=self.show_accident_alert, 
                            args=(current_time, frame_count, analysis_preview)
                        )
                        alert_thread.daemon = True
                        alert_thread.start()
                        
                        last_alert_time = time.time()
                        print(f"✅ AI analysis completed for {len(saved_files)} frames")
            else:
                processed_frame = frame
                accident_detected = False
                confidence = 0.0
            
            # Add status information to frame
            status_color = (0, 255, 0) if not accident_detected else (0, 0, 255)
            status_text = "SAFE" if not accident_detected else f"ACCIDENT! ({confidence:.2f})"
            
            # Draw status bar
            cv2.rectangle(processed_frame, (0, 0), (640, 100), (0, 0, 0), -1)
            cv2.putText(processed_frame, f"Status: {status_text}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(processed_frame, f"Frame: {frame_count}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (10, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"AI: Gemini Ready", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Draw border around frame
            border_color = (0, 255, 0) if not accident_detected else (0, 0, 255)
            cv2.rectangle(processed_frame, (0, 0), (639, 479), border_color, 5)
            
            # Display the frame
            cv2.imshow('🚨 AI-Powered Accident Detection System', processed_frame)
            
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
        print("\n" + "=" * 70)
        print("📊 Detection Statistics:")
        print(f"Total frames processed: {frame_count}")
        print(f"Accidents detected: {accident_count}")
        if frame_count > 0:
            print(f"Detection rate: {(accident_count/frame_count)*100:.2f}%")
        print(f"Accident folder: {self.accident_folder}")
        print("🤖 AI Analysis: Gemini API")
        print("=" * 70)
        print("👋 System stopped")

def main():
    """Main function to run the AI-enhanced accident detection system"""
    
    # Model path
    model_path = r'D:\camera\Accident-Detection-Web-App\server\models\i1-yolov8s.pt'
    
    # Gemini API key
    gemini_api_key = "AIzaSyC7kfChFFqncVELG4AooyD7jBCD1YP2v1s"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please check the model path!")
        return
    
    # Create detection system
    detector = EnhancedAccidentDetectionWithAI(model_path, gemini_api_key)
    
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
