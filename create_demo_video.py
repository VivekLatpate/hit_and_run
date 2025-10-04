"""
Create a demo video for testing the accident detection system
"""

import cv2
import numpy as np
import os

def create_demo_video():
    """Create a demo video with simulated accident scenes"""
    
    # Video properties
    width, height = 640, 480
    fps = 30
    duration = 10  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('demo_video.mp4', fourcc, fps, (width, height))
    
    print(f"Creating demo video: {total_frames} frames, {duration} seconds")
    
    for frame_num in range(total_frames):
        # Create base frame (road scene)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw road
        cv2.rectangle(frame, (0, height//2), (width, height), (100, 100, 100), -1)
        
        # Draw lane markings
        for i in range(0, width, 80):
            cv2.rectangle(frame, (i, height//2 + 20), (i + 40, height//2 + 30), (255, 255, 255), -1)
        
        # Add some cars
        if frame_num < 150:  # Normal driving
            # Car 1
            cv2.rectangle(frame, (200, height//2 - 60), (280, height//2 - 20), (0, 0, 255), -1)
            cv2.putText(frame, "CAR", (210, height//2 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Car 2
            cv2.rectangle(frame, (400, height//2 - 60), (480, height//2 - 20), (255, 0, 0), -1)
            cv2.putText(frame, "CAR", (410, height//2 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        elif 150 <= frame_num < 200:  # Accident scene
            # Crashed cars
            cv2.rectangle(frame, (250, height//2 - 60), (330, height//2 - 20), (0, 0, 255), -1)
            cv2.putText(frame, "CRASH!", (260, height//2 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add some debris
            cv2.circle(frame, (300, height//2 + 10), 15, (128, 128, 128), -1)
            cv2.circle(frame, (320, height//2 + 20), 10, (128, 128, 128), -1)
            
            # Add emergency text
            cv2.putText(frame, "ACCIDENT DETECTED!", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        else:  # After accident
            # Emergency vehicles
            cv2.rectangle(frame, (200, height//2 - 60), (280, height//2 - 20), (0, 255, 255), -1)
            cv2.putText(frame, "AMBULANCE", (205, height//2 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            cv2.rectangle(frame, (350, height//2 - 60), (430, height//2 - 20), (255, 255, 0), -1)
            cv2.putText(frame, "POLICE", (365, height//2 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {frame_num/fps:.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        if frame_num % 30 == 0:
            print(f"Created frame {frame_num}/{total_frames}")
    
    # Release video writer
    out.release()
    
    print("Demo video created: demo_video.mp4")
    print(f"Video contains accident scene at frames 150-200 (5-6.7 seconds)")

if __name__ == "__main__":
    create_demo_video()
