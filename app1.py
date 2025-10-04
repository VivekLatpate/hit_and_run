"""
Video Upload and Accident Detection Application
- User can upload a video file
- System scans the video for accidents using AI
- Provides detailed analysis with Gemini AI
- Saves accident frames and analysis reports
"""

import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import threading
import time
import os
from datetime import datetime
import google.generativeai as genai
from PIL import Image
import json

class VideoAccidentDetector:
    def __init__(self):
        """Initialize the Video Accident Detection System"""
        self.model_path = r'D:\camera\Accident-Detection-Web-App\server\models\i1-yolov8s.pt'
        self.gemini_api_key = "AIzaSyC7kfChFFqncVELG4AooyD7jBCD1YP2v1s"
        
        # Initialize components
        self.model = None
        self.gemini_model = None
        self.video_path = None
        self.output_folder = "video_accident_detections"
        
        # Create output folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        # Initialize models
        self.load_models()
        
        # GUI components
        self.root = None
        self.progress_var = None
        self.status_label = None
        self.result_text = None
        
    def load_models(self):
        """Load YOLO and Gemini models"""
        try:
            # Load YOLO model
            self.model = YOLO(self.model_path)
            print("✅ YOLO Model loaded successfully!")
            
            # Configure Gemini API
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            print("✅ Gemini AI Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            messagebox.showerror("Error", f"Failed to load models: {e}")
    
    def analyze_frame_with_ai(self, frame, frame_number, timestamp):
        """Analyze frame using Gemini AI"""
        try:
            # Convert OpenCV image to PIL
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            prompt = f"""
            Analyze this video frame #{frame_number} taken at {timestamp}.
            
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
            response = self.gemini_model.generate_content([prompt, pil_image])
            
            if response.text:
                return response.text
            else:
                return "No analysis available from Gemini AI"
                
        except Exception as e:
            print(f"Error analyzing frame: {e}")
            return f"Analysis error: {str(e)}"
    
    def detect_accidents_in_frame(self, frame):
        """Detect accidents in a single frame"""
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
    
    def process_video(self):
        """Process the uploaded video for accident detection"""
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showerror("Error", "Please select a valid video file!")
            return
        
        try:
            # Update status
            self.status_label.config(text="Processing video...")
            self.root.update()
            
            # Open video
            cap = cv2.VideoCapture(self.video_path)
            
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open video file!")
                return
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"Video Info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")
            
            # Create output folder for this video
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = os.path.join(self.output_folder, f"{video_name}_{timestamp}")
            os.makedirs(output_folder, exist_ok=True)
            
            # Process frames
            frame_count = 0
            accident_count = 0
            accident_frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Update progress
                progress = (frame_count / total_frames) * 100
                self.progress_var.set(progress)
                self.status_label.config(text=f"Processing frame {frame_count}/{total_frames}")
                self.root.update()
                
                # Detect accidents every 5th frame for better performance
                if frame_count % 5 == 0:
                    accident_detected, processed_frame, confidence = self.detect_accidents_in_frame(frame)
                    
                    if accident_detected:
                        accident_count += 1
                        current_time = datetime.now()
                        
                        # Save accident frame
                        frame_filename = f"accident_frame_{frame_count:06d}_{timestamp}.jpg"
                        frame_path = os.path.join(output_folder, frame_filename)
                        cv2.imwrite(frame_path, processed_frame)
                        
                        # Analyze with AI
                        self.status_label.config(text=f"Analyzing accident frame {frame_count} with AI...")
                        self.root.update()
                        
                        analysis = self.analyze_frame_with_ai(
                            frame, frame_count, 
                            f"{frame_count/fps:.2f}s" if fps > 0 else f"Frame {frame_count}"
                        )
                        
                        # Save analysis
                        analysis_filename = f"analysis_frame_{frame_count:06d}.txt"
                        analysis_path = os.path.join(output_folder, analysis_filename)
                        
                        with open(analysis_path, 'w', encoding='utf-8') as f:
                            f.write(f"ACCIDENT FRAME ANALYSIS\n")
                            f.write("=" * 50 + "\n")
                            f.write(f"Frame Number: {frame_count}\n")
                            f.write(f"Time in Video: {frame_count/fps:.2f}s\n" if fps > 0 else f"Frame: {frame_count}\n")
                            f.write(f"Detection Time: {current_time}\n")
                            f.write(f"Confidence: {confidence:.2f}\n")
                            f.write(f"Image File: {frame_filename}\n")
                            f.write("\n" + "=" * 50 + "\n")
                            f.write("GEMINI AI ANALYSIS:\n")
                            f.write("=" * 50 + "\n")
                            f.write(analysis)
                            f.write("\n" + "=" * 50 + "\n")
                        
                        accident_frames.append({
                            'frame_number': frame_count,
                            'time_in_video': frame_count/fps if fps > 0 else frame_count,
                            'confidence': confidence,
                            'frame_file': frame_filename,
                            'analysis_file': analysis_filename,
                            'analysis': analysis
                        })
                        
                        print(f"🚨 Accident detected at frame {frame_count} (confidence: {confidence:.2f})")
            
            cap.release()
            
            # Create comprehensive report
            report_path = os.path.join(output_folder, "comprehensive_video_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("COMPREHENSIVE VIDEO ACCIDENT DETECTION REPORT\n")
                f.write("=" * 60 + "\n")
                f.write(f"Video File: {self.video_path}\n")
                f.write(f"Analysis Time: {datetime.now()}\n")
                f.write(f"Total Frames: {total_frames}\n")
                f.write(f"Frames Processed: {frame_count}\n")
                f.write(f"Video Duration: {duration:.2f} seconds\n")
                f.write(f"FPS: {fps:.2f}\n")
                f.write(f"Accidents Detected: {accident_count}\n")
                f.write(f"Detection Rate: {(accident_count/frame_count)*100:.2f}%\n")
                f.write(f"Model Used: {self.model_path}\n")
                f.write(f"AI Analysis: Gemini 2.5 Flash\n")
                f.write("\n" + "=" * 60 + "\n")
                f.write("ACCIDENT SUMMARY:\n")
                f.write("=" * 60 + "\n")
                
                for i, accident in enumerate(accident_frames, 1):
                    f.write(f"\nACCIDENT #{i}:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Frame: {accident['frame_number']}\n")
                    f.write(f"Time: {accident['time_in_video']:.2f}s\n")
                    f.write(f"Confidence: {accident['confidence']:.2f}\n")
                    f.write(f"Frame File: {accident['frame_file']}\n")
                    f.write(f"Analysis File: {accident['analysis_file']}\n")
                    f.write("\nAI Analysis Preview:\n")
                    f.write(accident['analysis'][:300] + "...\n")
                    f.write("-" * 30 + "\n")
            
            # Update GUI with results
            self.status_label.config(text="Analysis complete!")
            self.progress_var.set(100)
            
            # Display results
            result_text = f"""
VIDEO ANALYSIS COMPLETE!

📹 Video: {os.path.basename(self.video_path)}
📊 Total Frames: {total_frames:,}
⏱️ Duration: {duration:.2f} seconds
🚨 Accidents Detected: {accident_count}
📈 Detection Rate: {(accident_count/frame_count)*100:.2f}%

📁 Results saved to: {output_folder}

"""
            
            if accident_count > 0:
                result_text += "🚨 ACCIDENT DETAILS:\n"
                result_text += "=" * 40 + "\n"
                for i, accident in enumerate(accident_frames, 1):
                    result_text += f"\nAccident #{i}:\n"
                    result_text += f"• Frame: {accident['frame_number']}\n"
                    result_text += f"• Time: {accident['time_in_video']:.2f}s\n"
                    result_text += f"• Confidence: {accident['confidence']:.2f}\n"
                    result_text += f"• AI Analysis: {accident['analysis'][:100]}...\n"
            else:
                result_text += "✅ No accidents detected in this video.\n"
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, result_text)
            
            # Show completion message
            messagebox.showinfo("Analysis Complete", 
                              f"Video analysis completed!\n\n"
                              f"Accidents detected: {accident_count}\n"
                              f"Results saved to: {output_folder}")
            
        except Exception as e:
            print(f"Error processing video: {e}")
            messagebox.showerror("Error", f"Error processing video: {e}")
            self.status_label.config(text="Error occurred!")
    
    def select_video(self):
        """Open file dialog to select video"""
        file_types = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
            ("MP4 files", "*.mp4"),
            ("AVI files", "*.avi"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=file_types
        )
        
        if filename:
            self.video_path = filename
            self.status_label.config(text=f"Selected: {os.path.basename(filename)}")
            print(f"Selected video: {filename}")
    
    def start_processing(self):
        """Start video processing in a separate thread"""
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file first!")
            return
        
        # Start processing in a separate thread
        thread = threading.Thread(target=self.process_video)
        thread.daemon = True
        thread.start()
    
    def create_gui(self):
        """Create the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Video Accident Detection System")
        self.root.geometry("800x700")
        self.root.configure(bg='#f0f0f0')
        
        # Title
        title_label = tk.Label(
            self.root, 
            text="🚨 Video Accident Detection System", 
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#d32f2f'
        )
        title_label.pack(pady=20)
        
        # Video selection frame
        select_frame = tk.Frame(self.root, bg='#f0f0f0')
        select_frame.pack(pady=10)
        
        tk.Button(
            select_frame,
            text="📁 Select Video File",
            command=self.select_video,
            font=("Arial", 12),
            bg='#2196f3',
            fg='white',
            padx=20,
            pady=10
        ).pack(side=tk.LEFT, padx=10)
        
        # Process button
        tk.Button(
            select_frame,
            text="🚀 Start Analysis",
            command=self.start_processing,
            font=("Arial", 12, "bold"),
            bg='#4caf50',
            fg='white',
            padx=20,
            pady=10
        ).pack(side=tk.LEFT, padx=10)
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Please select a video file to analyze",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='#666666'
        )
        self.status_label.pack(pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            self.root,
            variable=self.progress_var,
            maximum=100,
            length=400
        )
        progress_bar.pack(pady=10)
        
        # Results text area
        results_frame = tk.Frame(self.root, bg='#f0f0f0')
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(
            results_frame,
            text="Analysis Results:",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0'
        ).pack(anchor=tk.W)
        
        self.result_text = tk.Text(
            results_frame,
            height=15,
            width=80,
            font=("Consolas", 10),
            bg='white',
            fg='black',
            wrap=tk.WORD
        )
        
        scrollbar = tk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Instructions
        instructions = """
INSTRUCTIONS:
1. Click 'Select Video File' to choose your video
2. Click 'Start Analysis' to begin processing
3. The system will scan the video for accidents
4. AI will analyze any detected accidents
5. Results will be saved to the output folder

SUPPORTED FORMATS: MP4, AVI, MOV, MKV, WMV, FLV, WEBM
        """
        
        self.result_text.insert(1.0, instructions)
        
        # Start GUI
        self.root.mainloop()

def main():
    """Main function"""
    print("🚀 Starting Video Accident Detection Application")
    print("=" * 50)
    
    # Check if model exists
    model_path = r'D:\camera\Accident-Detection-Web-App\server\models\i1-yolov8s.pt'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please check the model path!")
        return
    
    # Create and run the application
    app = VideoAccidentDetector()
    app.create_gui()

if __name__ == "__main__":
    main()
