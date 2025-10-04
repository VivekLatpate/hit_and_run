import cv2
import numpy as np
from ultralytics import YOLO
import os

class YOLOOpenCVProcessor:
    def __init__(self, model_path):
        """
        Initialize YOLO model for OpenCV integration
        
        Args:
            model_path (str): Path to the YOLO model file (.pt)
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"Successfully loaded model: {self.model_path}")
        except Exception as e:
            print(f"Error loading model {self.model_path}: {e}")
            self.model = None
    
    def detect_objects(self, image):
        """
        Detect objects in an image using YOLO
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            results: YOLO detection results
        """
        if self.model is None:
            print("Model not loaded!")
            return None
        
        try:
            # Run YOLO detection
            results = self.model(image)
            return results
        except Exception as e:
            print(f"Error during detection: {e}")
            return None
    
    def draw_detections(self, image, results, confidence_threshold=0.5):
        """
        Draw bounding boxes and labels on the image
        
        Args:
            image: OpenCV image
            results: YOLO detection results
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            annotated_image: Image with drawn detections
        """
        annotated_image = image.copy()
        
        if results is None or len(results) == 0:
            return annotated_image
        
        # Get the first result (assuming single image)
        result = results[0]
        
        # Extract boxes, confidences, and class IDs
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Filter by confidence
                if confidence >= confidence_threshold:
                    # Get class name
                    class_name = self.model.names[class_id]
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated_image, (int(x1), int(y1) - label_size[1] - 10), 
                                (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                    cv2.putText(annotated_image, label, (int(x1), int(y1) - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_image
    
    def process_video(self, video_path, output_path=None, confidence_threshold=0.5):
        """
        Process a video file with YOLO detection
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            confidence_threshold: Minimum confidence for detections
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            results = self.detect_objects(frame)
            
            # Draw detections
            annotated_frame = self.draw_detections(frame, results, confidence_threshold)
            
            # Display frame
            cv2.imshow('YOLO Detection', annotated_frame)
            
            # Write frame if output is specified
            if out:
                out.write(annotated_frame)
            
            frame_count += 1
            print(f"Processed frame {frame_count}", end='\r')
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")

def main():
    """Example usage of YOLOOpenCVProcessor - Accident Detection Only"""
    
    # Path to accident detection model only
    model_path = r'D:\camera\Accident-Detection-Web-App\server\models\i1-yolov8s.pt'
    
    # Check if model exists
    if os.path.exists(model_path):
        print(f"✓ Found Accident Detection Model: {model_path}")
    else:
        print(f"✗ Missing Accident Detection Model: {model_path}")
        return
    
    # Example 1: Process image
    print("\n=== Example 1: Accident Detection Image Processing ===")
    processor = YOLOOpenCVProcessor(model_path)
    
    # Create a sample image for testing (you can replace this with your own image)
    sample_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(sample_image, "Accident Detection Test", (50, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(sample_image, "Model: i1-yolov8s.pt", (50, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # Detect accidents
    results = processor.detect_objects(sample_image)
    annotated_image = processor.draw_detections(sample_image, results)
    
    # Display result
    cv2.imshow('Accident Detection Test', annotated_image)
    cv2.waitKey(3000)  # Show for 3 seconds
    cv2.destroyAllWindows()
    
    # Example 2: Process webcam
    print("\n=== Example 2: Real-time Accident Detection ===")
    print("Press 'q' to quit webcam processing")
    
    cap = cv2.VideoCapture(0)  # Use webcam
    
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect accidents
            results = processor.detect_objects(frame)
            
            # Draw detections
            annotated_frame = processor.draw_detections(frame, results)
            
            # Display frame
            cv2.imshow('Real-time Accident Detection', annotated_frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Could not open webcam")
    
    # Example 3: Process video file
    print("\n=== Example 3: Video Accident Detection ===")
    print("To process a video file for accident detection, use:")
    print("processor.process_video('path/to/video.mp4', 'accident_output.mp4')")

if __name__ == "__main__":
    main()
