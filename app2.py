"""
Enhanced Video Accident Detection Application (app2.py)
- Advanced accident description with detailed vehicle analysis
- License plate detection and recognition
- Car color identification
- Frame proof display with visual evidence
- Enhanced AI analysis with specific details
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
from PIL import Image, ImageTk
import json
import re

class EnhancedAccidentDetector:
    def __init__(self):
        """Initialize the Enhanced Accident Detection System"""
        self.model_path = r'D:\camera\Accident-Detection-Web-App\server\models\i1-yolov8s.pt'
        self.gemini_api_key = "AIzaSyC7kfChFFqncVELG4AooyD7jBCD1YP2v1s"
        
        # Initialize components
        self.model = None
        self.gemini_model = None
        self.video_path = None
        self.output_folder = "enhanced_accident_detections"
        
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
        self.proof_frames = []
        self.accident_data = []
        
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
    
    def extract_license_plate(self, frame, accident_region):
        """Extract and enhance license plate region"""
        try:
            x1, y1, x2, y2 = accident_region
            
            # Expand region slightly to capture license plates
            margin = 50
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(frame.shape[1], x2 + margin)
            y2 = min(frame.shape[0], y2 + margin)
            
            # Extract region
            region = frame[int(y1):int(y2), int(x1):int(x2)]
            
            # Convert to grayscale
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # Apply contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # Apply threshold
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return thresh, region
            
        except Exception as e:
            print(f"Error extracting license plate: {e}")
            return None, None
    
    def detect_car_colors(self, frame, accident_regions):
        """Detect dominant car colors in accident regions"""
        colors = []
        
        for region in accident_regions:
            try:
                x1, y1, x2, y2 = region
                car_region = frame[int(y1):int(y2), int(x1):int(x2)]
                
                # Convert to HSV for better color detection
                hsv = cv2.cvtColor(car_region, cv2.COLOR_BGR2HSV)
                
                # Define color ranges
                color_ranges = {
                    'Red': [(0, 50, 50), (10, 255, 255)],
                    'Blue': [(100, 50, 50), (130, 255, 255)],
                    'Green': [(40, 50, 50), (80, 255, 255)],
                    'Yellow': [(20, 50, 50), (40, 255, 255)],
                    'White': [(0, 0, 200), (180, 30, 255)],
                    'Black': [(0, 0, 0), (180, 255, 50)],
                    'Silver': [(0, 0, 100), (180, 30, 200)],
                    'Gray': [(0, 0, 50), (180, 30, 150)]
                }
                
                # Count pixels for each color
                color_counts = {}
                for color_name, (lower, upper) in color_ranges.items():
                    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                    count = cv2.countNonZero(mask)
                    color_counts[color_name] = count
                
                # Get dominant color
                dominant_color = max(color_counts, key=color_counts.get)
                colors.append(dominant_color)
                
            except Exception as e:
                print(f"Error detecting car color: {e}")
                colors.append("Unknown")
        
        return colors
    
    def analyze_accident_with_details(self, frame, frame_number, timestamp, accident_regions):
        """Enhanced accident analysis with detailed vehicle information"""
        try:
            # Convert OpenCV image to PIL
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Extract license plate regions
            license_plate_images = []
            for region in accident_regions:
                plate_img, region_img = self.extract_license_plate(frame, region)
                if plate_img is not None:
                    license_plate_images.append(plate_img)
            
            # Detect car colors
            car_colors = self.detect_car_colors(frame, accident_regions)
            
            prompt = f"""
            Analyze this accident scene frame #{frame_number} taken at {timestamp}.
            
            Please provide EXTREMELY DETAILED analysis including:
            
            VEHICLE DETAILS:
            1. Exact number of vehicles involved
            2. Vehicle types (sedan, SUV, truck, motorcycle, bus, etc.)
            3. Vehicle makes/models if identifiable (Toyota, Honda, BMW, etc.)
            4. Vehicle colors (be very specific: metallic red, navy blue, etc.)
            5. Vehicle sizes (compact, mid-size, full-size, etc.)
            6. Vehicle conditions (new, old, damaged areas)
            
            LICENSE PLATES:
            7. License plate numbers (if visible, provide exact text)
            8. License plate states/countries if identifiable
            9. License plate colors and styles
            
            DAMAGE ASSESSMENT:
            10. Specific damage locations (front bumper, side door, rear, etc.)
            11. Damage severity (minor scratches, major dents, severe damage)
            12. Visible debris or parts scattered around
            13. Fluid leaks (oil, coolant, etc.)
            
            ACCIDENT ANALYSIS:
            14. Accident type (head-on, rear-end, side collision, rollover, etc.)
            15. Accident severity (minor, moderate, severe, fatal)
            16. Likely cause of accident
            17. Speed estimation based on damage
            18. Direction of impact
            
            ENVIRONMENTAL FACTORS:
            19. Road conditions (dry, wet, icy, etc.)
            20. Weather conditions (sunny, cloudy, rainy, etc.)
            21. Lighting conditions
            22. Traffic signs or signals visible
            23. Road type (highway, city street, intersection, etc.)
            
            EMERGENCY RESPONSE:
            24. Any emergency vehicles present
            25. People visible (injured, witnesses, etc.)
            26. Recommended emergency response level
            27. Immediate safety concerns
            
            Provide your analysis in a structured format with clear sections.
            """
            
            # Generate content with image and text
            response = self.gemini_model.generate_content([prompt, pil_image])
            
            if response.text:
                # Add detected colors to analysis
                analysis = response.text
                if car_colors:
                    analysis += f"\n\nDETECTED CAR COLORS: {', '.join(car_colors)}"
                
                return analysis
            else:
                return "No analysis available from Gemini AI"
                
        except Exception as e:
            print(f"Error analyzing accident: {e}")
            return f"Analysis error: {str(e)}"
    
    def detect_accidents_in_frame(self, frame):
        """Detect accidents in a single frame with enhanced visualization"""
        if self.model is None:
            return False, frame, 0.0, []
        
        try:
            results = self.model(frame)
            
            accident_detected = False
            max_confidence = 0.0
            accident_regions = []
            confidence_threshold = 0.5
            
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    confidence = box.conf[0].cpu().numpy()
                    if confidence >= confidence_threshold:
                        accident_detected = True
                        max_confidence = max(max_confidence, confidence)
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        accident_regions.append((x1, y1, x2, y2))
                        
                        # Draw enhanced bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                        
                        # Add accident label with confidence
                        label = f"ACCIDENT: {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                        
                        # Draw label background
                        cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 15), 
                                    (int(x1) + label_size[0], int(y1)), (0, 0, 255), -1)
                        
                        # Draw label text
                        cv2.putText(frame, label, (int(x1), int(y1) - 8), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
                        # Add frame number and timestamp
                        timestamp_text = f"Frame: {len(accident_regions)}"
                        cv2.putText(frame, timestamp_text, (int(x1), int(y2) + 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return accident_detected, frame, max_confidence, accident_regions
            
        except Exception as e:
            print(f"Detection error: {e}")
            return False, frame, 0.0, []
    
    def create_proof_frames_display(self, accident_data):
        """Create visual proof frames for display"""
        proof_frames = []
        
        for accident in accident_data:
            try:
                # Load the accident frame
                frame_path = accident['frame_file']
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    
                    # Resize for display
                    display_frame = cv2.resize(frame, (300, 200))
                    
                    # Convert to PIL for tkinter
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    proof_frames.append({
                        'photo': photo,
                        'frame_number': accident['frame_number'],
                        'time': accident['time_in_video'],
                        'confidence': accident['confidence']
                    })
                    
            except Exception as e:
                print(f"Error creating proof frame: {e}")
        
        return proof_frames
    
    def process_video(self):
        """Process the uploaded video for enhanced accident detection"""
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showerror("Error", "Please select a valid video file!")
            return
        
        try:
            # Update status
            self.status_label.config(text="Processing video with enhanced analysis...")
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
            self.accident_data = []
            
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
                
                # Detect accidents every 3rd frame for better coverage
                if frame_count % 3 == 0:
                    accident_detected, processed_frame, confidence, accident_regions = self.detect_accidents_in_frame(frame)
                    
                    if accident_detected:
                        accident_count += 1
                        current_time = datetime.now()
                        
                        # Save accident frame
                        frame_filename = f"accident_frame_{frame_count:06d}_{timestamp}.jpg"
                        frame_path = os.path.join(output_folder, frame_filename)
                        cv2.imwrite(frame_path, processed_frame)
                        
                        # Enhanced AI analysis
                        self.status_label.config(text=f"Enhanced AI analysis for frame {frame_count}...")
                        self.root.update()
                        
                        analysis = self.analyze_accident_with_details(
                            frame, frame_count, 
                            f"{frame_count/fps:.2f}s" if fps > 0 else f"Frame {frame_count}",
                            accident_regions
                        )
                        
                        # Save enhanced analysis
                        analysis_filename = f"enhanced_analysis_frame_{frame_count:06d}.txt"
                        analysis_path = os.path.join(output_folder, analysis_filename)
                        
                        with open(analysis_path, 'w', encoding='utf-8') as f:
                            f.write(f"ENHANCED ACCIDENT FRAME ANALYSIS\n")
                            f.write("=" * 60 + "\n")
                            f.write(f"Frame Number: {frame_count}\n")
                            f.write(f"Time in Video: {frame_count/fps:.2f}s\n" if fps > 0 else f"Frame: {frame_count}\n")
                            f.write(f"Detection Time: {current_time}\n")
                            f.write(f"Confidence: {confidence:.2f}\n")
                            f.write(f"Accident Regions: {len(accident_regions)}\n")
                            f.write(f"Image File: {frame_filename}\n")
                            f.write("\n" + "=" * 60 + "\n")
                            f.write("ENHANCED GEMINI AI ANALYSIS:\n")
                            f.write("=" * 60 + "\n")
                            f.write(analysis)
                            f.write("\n" + "=" * 60 + "\n")
                        
                        # Store accident data
                        accident_info = {
                            'frame_number': frame_count,
                            'time_in_video': frame_count/fps if fps > 0 else frame_count,
                            'confidence': confidence,
                            'frame_file': frame_path,
                            'analysis_file': analysis_filename,
                            'analysis': analysis,
                            'accident_regions': accident_regions
                        }
                        
                        self.accident_data.append(accident_info)
                        
                        print(f"🚨 Enhanced accident detected at frame {frame_count} (confidence: {confidence:.2f})")
            
            cap.release()
            
            # Create comprehensive enhanced report
            report_path = os.path.join(output_folder, "enhanced_comprehensive_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("ENHANCED COMPREHENSIVE VIDEO ACCIDENT DETECTION REPORT\n")
                f.write("=" * 70 + "\n")
                f.write(f"Video File: {self.video_path}\n")
                f.write(f"Analysis Time: {datetime.now()}\n")
                f.write(f"Total Frames: {total_frames}\n")
                f.write(f"Frames Processed: {frame_count}\n")
                f.write(f"Video Duration: {duration:.2f} seconds\n")
                f.write(f"FPS: {fps:.2f}\n")
                f.write(f"Accidents Detected: {accident_count}\n")
                f.write(f"Detection Rate: {(accident_count/frame_count)*100:.2f}%\n")
                f.write(f"Model Used: {self.model_path}\n")
                f.write(f"AI Analysis: Gemini 2.5 Flash Enhanced\n")
                f.write("\n" + "=" * 70 + "\n")
                f.write("ENHANCED ACCIDENT SUMMARY:\n")
                f.write("=" * 70 + "\n")
                
                for i, accident in enumerate(self.accident_data, 1):
                    f.write(f"\nACCIDENT #{i} - DETAILED ANALYSIS:\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"Frame: {accident['frame_number']}\n")
                    f.write(f"Time: {accident['time_in_video']:.2f}s\n")
                    f.write(f"Confidence: {accident['confidence']:.2f}\n")
                    f.write(f"Accident Regions: {len(accident['accident_regions'])}\n")
                    f.write(f"Frame File: {accident['frame_file']}\n")
                    f.write(f"Analysis File: {accident['analysis_file']}\n")
                    f.write("\nENHANCED AI ANALYSIS:\n")
                    f.write(accident['analysis'])
                    f.write("\n" + "-" * 50 + "\n")
            
            # Create proof frames for display
            self.proof_frames = self.create_proof_frames_display(self.accident_data)
            
            # Update GUI with enhanced results
            self.status_label.config(text="Enhanced analysis complete!")
            self.progress_var.set(100)
            
            # Display enhanced results
            result_text = f"""
ENHANCED VIDEO ANALYSIS COMPLETE!

📹 Video: {os.path.basename(self.video_path)}
📊 Total Frames: {total_frames:,}
⏱️ Duration: {duration:.2f} seconds
🚨 Accidents Detected: {accident_count}
📈 Detection Rate: {(accident_count/frame_count)*100:.2f}%

📁 Enhanced results saved to: {output_folder}

"""
            
            if accident_count > 0:
                result_text += "🚨 ENHANCED ACCIDENT DETAILS:\n"
                result_text += "=" * 50 + "\n"
                for i, accident in enumerate(self.accident_data, 1):
                    result_text += f"\nAccident #{i}:\n"
                    result_text += f"• Frame: {accident['frame_number']}\n"
                    result_text += f"• Time: {accident['time_in_video']:.2f}s\n"
                    result_text += f"• Confidence: {accident['confidence']:.2f}\n"
                    result_text += f"• Regions: {len(accident['accident_regions'])}\n"
                    
                    # Extract key details from analysis
                    analysis_preview = accident['analysis'][:200]
                    if "VEHICLE DETAILS:" in analysis_preview:
                        vehicle_section = analysis_preview.split("VEHICLE DETAILS:")[1].split("\n")[0:3]
                        result_text += f"• Vehicle Info: {' '.join(vehicle_section)}\n"
                    
                    result_text += f"• Analysis: {analysis_preview}...\n"
            else:
                result_text += "✅ No accidents detected in this video.\n"
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, result_text)
            
            # Show completion message
            messagebox.showinfo("Enhanced Analysis Complete", 
                              f"Enhanced video analysis completed!\n\n"
                              f"Accidents detected: {accident_count}\n"
                              f"Enhanced results saved to: {output_folder}\n"
                              f"Proof frames available for display")
            
        except Exception as e:
            print(f"Error processing video: {e}")
            messagebox.showerror("Error", f"Error processing video: {e}")
            self.status_label.config(text="Error occurred!")
    
    def show_proof_frames(self):
        """Display proof frames in a new window"""
        if not self.proof_frames:
            messagebox.showwarning("No Proof Frames", "No accident frames available to display!")
            return
        
        # Create proof frames window
        proof_window = tk.Toplevel(self.root)
        proof_window.title("Accident Proof Frames")
        proof_window.geometry("800x600")
        
        # Create scrollable frame
        canvas = tk.Canvas(proof_window)
        scrollbar = ttk.Scrollbar(proof_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Display proof frames
        for i, proof in enumerate(self.proof_frames):
            frame_frame = ttk.Frame(scrollable_frame)
            frame_frame.pack(pady=10, padx=10, fill="x")
            
            # Frame image
            img_label = tk.Label(frame_frame, image=proof['photo'])
            img_label.pack(side="left", padx=10)
            
            # Frame details
            details_frame = ttk.Frame(frame_frame)
            details_frame.pack(side="left", fill="both", expand=True, padx=10)
            
            tk.Label(details_frame, text=f"Accident #{i+1}", font=("Arial", 12, "bold")).pack(anchor="w")
            tk.Label(details_frame, text=f"Frame: {proof['frame_number']}").pack(anchor="w")
            tk.Label(details_frame, text=f"Time: {proof['time']:.2f}s").pack(anchor="w")
            tk.Label(details_frame, text=f"Confidence: {proof['confidence']:.2f}").pack(anchor="w")
            
            # Analysis preview
            if i < len(self.accident_data):
                analysis_preview = self.accident_data[i]['analysis'][:150] + "..."
                tk.Label(details_frame, text="Analysis Preview:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,0))
                tk.Label(details_frame, text=analysis_preview, wraplength=400, justify="left").pack(anchor="w")
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
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
        """Create the enhanced GUI interface"""
        self.root = tk.Tk()
        self.root.title("Enhanced Video Accident Detection System")
        self.root.geometry("900x800")
        self.root.configure(bg='#f0f0f0')
        
        # Title
        title_label = tk.Label(
            self.root, 
            text="🚨 Enhanced Video Accident Detection System", 
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
            text="🚀 Start Enhanced Analysis",
            command=self.start_processing,
            font=("Arial", 12, "bold"),
            bg='#4caf50',
            fg='white',
            padx=20,
            pady=10
        ).pack(side=tk.LEFT, padx=10)
        
        # Proof frames button
        tk.Button(
            select_frame,
            text="🖼️ Show Proof Frames",
            command=self.show_proof_frames,
            font=("Arial", 12),
            bg='#ff9800',
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
            length=500
        )
        progress_bar.pack(pady=10)
        
        # Results text area
        results_frame = tk.Frame(self.root, bg='#f0f0f0')
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(
            results_frame,
            text="Enhanced Analysis Results:",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0'
        ).pack(anchor=tk.W)
        
        self.result_text = tk.Text(
            results_frame,
            height=18,
            width=90,
            font=("Consolas", 10),
            bg='white',
            fg='black',
            wrap=tk.WORD
        )
        
        scrollbar = tk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Enhanced instructions
        instructions = """
ENHANCED ACCIDENT DETECTION FEATURES:

🎯 DETAILED VEHICLE ANALYSIS:
• Vehicle types, makes, models
• Exact car colors and finishes
• Vehicle sizes and conditions
• License plate detection and recognition

🚨 COMPREHENSIVE DAMAGE ASSESSMENT:
• Specific damage locations
• Damage severity levels
• Debris and fluid analysis
• Accident type classification

📊 ENHANCED AI ANALYSIS:
• Speed estimation
• Impact direction analysis
• Environmental factors
• Emergency response recommendations

🖼️ VISUAL PROOF:
• Frame-by-frame evidence
• Enhanced visualization
• Proof frame display
• Detailed analysis reports

INSTRUCTIONS:
1. Click 'Select Video File' to choose your video
2. Click 'Start Enhanced Analysis' to begin processing
3. Click 'Show Proof Frames' to view accident evidence
4. Results include detailed vehicle and damage analysis

SUPPORTED FORMATS: MP4, AVI, MOV, MKV, WMV, FLV, WEBM
        """
        
        self.result_text.insert(1.0, instructions)
        
        # Start GUI
        self.root.mainloop()

def main():
    """Main function"""
    print("🚀 Starting Enhanced Video Accident Detection Application")
    print("=" * 60)
    
    # Check if model exists
    model_path = r'D:\camera\Accident-Detection-Web-App\server\models\i1-yolov8s.pt'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please check the model path!")
        return
    
    # Create and run the enhanced application
    app = EnhancedAccidentDetector()
    app.create_gui()

if __name__ == "__main__":
    main()
