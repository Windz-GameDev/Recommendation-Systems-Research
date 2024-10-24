
import google.generativeai as genai
from dotenv import load_dotenv
import PIL.Image
import os


# Load environment variables from the .env file
load_dotenv()

# Access the API key from the environment
google_api_key = os.getenv('GOOGLE_API_KEY') 

# Check if the API key is loaded correctly
if not google_api_key:
    raise ValueError("API key not found. Please check your .env file.")

# Configure the API key in the generative AI library
genai.configure(api_key=google_api_key)

# Create the model
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Code for having the API describe an image

# Open the image
img = PIL.Image.open('./images/Gym 1.jpg')

# Check if the image is loaded correctly
print(f"Image loaded: {img}")

# Generate content using the image and a prompt
try:
    response = model.generate_content(["What is in this photo?", img])
    print(response.text)
except Exception as e:
    print(f"Error during API call: {e}")

'''
# Code for having the API respond to a text prompt

try:
    response = model.generate_content("Write a story about a magic backpack.")
    print(response.text)
except Exception as e:
    print(f"Error during API call: {e}")
'''