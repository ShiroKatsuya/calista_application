
from google import genai
from PIL import Image # For handling image files
import os

import dotenv
# Load environment variables from .env file
dotenv.load_dotenv()

model_id = os.getenv("GEMINI_MODEL")


img = Image.open('')


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
query = "Jelaskan secara singkat dengan fokus pada informasi yang disampaikan oleh gambar"


explanation_response = client.models.generate_content(
        model=model_id,
        # config=generation_config,
        contents=[
            img,
            query
        ]
    )
    # Check if explanation_response and explanation_response.text are not None

cleaned_response = explanation_response.text.replace('*', '').replace('\n\n', '\n')
print(cleaned_response)

