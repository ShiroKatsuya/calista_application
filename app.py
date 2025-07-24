import os
import logging
from flask import Flask

# # Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

# Import routes after app creation to avoid circular imports

from routes import *
from routes_aplication import *

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
