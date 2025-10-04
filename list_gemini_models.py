"""
Test script to list available Gemini models
"""

import google.generativeai as genai

def list_available_models():
    """List all available Gemini models"""
    
    # API key
    api_key = "AIzaSyC7kfChFFqncVELG4AooyD7jBCD1YP2v1s"
    
    # Configure the API
    genai.configure(api_key=api_key)
    
    try:
        print("Listing available Gemini models...")
        
        # List models
        models = genai.list_models()
        
        print("Available models:")
        print("-" * 50)
        
        for model in models:
            print(f"Model: {model.name}")
            print(f"Display Name: {model.display_name}")
            print(f"Description: {model.description}")
            print(f"Supported Methods: {model.supported_generation_methods}")
            print("-" * 50)
            
        return True
        
    except Exception as e:
        print(f"Error listing models: {e}")
        return False

if __name__ == "__main__":
    list_available_models()

