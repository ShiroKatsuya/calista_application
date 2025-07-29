import os
import base64
import io
import json
from PIL import Image
from openai import OpenAI
import dotenv
# Load environment variables from .env file
dotenv.load_dotenv()

# Load environment variables
nebius_api_key = os.getenv("NEBIUS_API_KEY")
image_generation_model = os.getenv("IMAGE_GENERATION")

if not nebius_api_key:
    raise ValueError("NEBIUS_API_KEY environment variable is not set.")
if not image_generation_model or not isinstance(image_generation_model, str):
    raise ValueError("IMAGE_GENERATION environment variable is not set or is not a string.")

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=nebius_api_key
)

# Define the image generation prompt
prompt = "draw a picture of stars orbiting a supermassive black hole"

try:
    response = client.images.generate(
        model=image_generation_model,
        response_format="b64_json",
        extra_body={
            "response_extension": "png",
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 30,
            "negative_prompt": "",
            "seed": -1
        },
        prompt=prompt
    )

    # Parse the JSON response
    image_data_json = response.to_json()
    image_data = json.loads(image_data_json)

    # Extract and decode the base64 image
    b64_image = image_data['data'][0]['b64_json']
    decoded_image_bytes = base64.b64decode(b64_image)
    image_stream = io.BytesIO(decoded_image_bytes)

    # Save image to static\image_generation
    output_dir = os.path.join("static", "image_generation")
    os.makedirs(output_dir, exist_ok=True)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"generated_image_{timestamp}.png")
    image = Image.open(image_stream)
    image.save(output_path)
except Exception as e:
    print(f"An error occurred during image generation: {e}")


