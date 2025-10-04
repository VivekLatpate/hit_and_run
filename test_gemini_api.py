"""
Test script to verify Gemini API is working correctly
"""

import google.generativeai as genai
from PIL import Image
import cv2
import numpy as np

def test_gemini_api():
    """Test Gemini API with a simple image"""
    
    # API key
    api_key = "AIzaSyC7kfChFFqncVELG4AooyD7jBCD1YP2v1s"
    
    # Configure the API
    genai.configure(api_key=api_key)
    
    try:
        # Create a simple test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:] = (100, 150, 200)  # Blue background
        
        # Add some text
        cv2.putText(test_image, "TEST IMAGE", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Convert to PIL
        image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Initialize model
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("Using gemini-2.5-flash model")
        
        # Test prompt
        prompt = """
        Analyze this test image. Describe what you see in detail.
        """
        
        print("Testing Gemini API...")
        print("Created test image")
        print("Connecting to Gemini API...")
        
        # Generate content
        response = model.generate_content([prompt, pil_image])
        
        if response.text:
            print("SUCCESS! Gemini API is working!")
            print("Response:")
            print("-" * 50)
            print(response.text)
            print("-" * 50)
            return True
        else:
            print("No response from Gemini API")
            return False
            
    except Exception as e:
        print(f"Error testing Gemini API: {e}")
        return False

if __name__ == "__main__":
    test_gemini_api()
