
import os
import uuid
import time
from flask import request, jsonify, Response, stream_with_context
import ollama
from app import app # Import the app instance
import json

API_KEYS_FILE = "api_keys.json"

# Load API keys from file if exists
def load_api_keys():
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, "r") as f:
            data = json.load(f)
            for k, v in data.items():
                v["last_reset"] = float(v.get("last_reset", time.time()))
            return data
    return {}

# Save API keys to file
def save_api_keys():
    with open(API_KEYS_FILE, "w") as f:
        json.dump(id_api_keys, f)

# In-memory store for API keys and their usage. In a real application, use a database.
id_api_keys = load_api_keys()

# Add a simple in-memory rate limit store for API key creation by IP
api_key_creation_timestamps = {}

# Function to generate a new API key
@app.route('/generate_api_key', methods=['POST'])
def generate_api_key():
    # Server-side rate limit: 3 per minute per IP
    ip = request.remote_addr
    now = time.time()
    last_time = api_key_creation_timestamps.get(ip, 0)
    if now - last_time < 180:
        return jsonify({"error": "Terlalu sering membuat API Key. Silakan tunggu 3 menit sebelum mencoba lagi."}), 429
    api_key_creation_timestamps[ip] = now

    new_key = str(uuid.uuid4())
    id_api_keys[new_key] = {"uses": 0, "last_reset": time.time(), "limit": 10}
    save_api_keys()
    return jsonify({"api_key": new_key, "message": "API Key generated successfully. Limit: 10 responses."})

# Decorator for API key validation and rate limiting
def api_key_required(f):
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('Authorization')
        if not api_key or not api_key.startswith('Bearer '):
            return jsonify({"error": "Unauthorized: Bearer token missing"}), 401
        
        key = api_key.split(' ')[1]
        key_data = id_api_keys.get(key)

        if not key_data:
            return jsonify({"error": "Unauthorized: Invalid API Key"}), 401

        # Daily reset logic
        now = time.time()
        if now - key_data["last_reset"] > 86400:  # 24 hours
            key_data["uses"] = 0
            key_data["last_reset"] = now
            print(f"API Key {key} usage reset after 24 hours.")

        if key_data["uses"] >= key_data["limit"]:
            return jsonify({"error": "API Key limit exceeded. Max 10 responses per day."}), 429
        
        key_data["uses"] += 1
        save_api_keys()
        print(f"API Key {key} used {key_data['uses']} times (limit {key_data['limit']})")
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__ # Preserve original function name
    return decorated_function

# Ollama non-streaming endpoint
@app.route('/api/ollama/generate', methods=['POST'])
@api_key_required
def ollama_generate():
    data = request.get_json()
    model = data.get('model', 'llama3-2.3b:latest')
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            stream=False
        )
        return jsonify({"response": response['response']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ollama streaming endpoint
@app.route('/api/ollama/stream', methods=['POST'])
@api_key_required
def ollama_stream():
    data = request.get_json()
    model = data.get('model', 'llama3-2.3b:latest')
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    def generate():
        try:
            for chunk in ollama.generate(
                model=model,
                prompt=prompt,
                stream=True
            ):
                yield json.dumps({"response": chunk['response']}) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return Response(stream_with_context(generate()), mimetype='application/jsonlines')