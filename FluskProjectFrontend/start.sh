#!/bin/bash

# Set environment variables if needed
export FLASK_APP=app/views.py
export FLASK_ENV=development

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt

# Change directory to the app folder
cd /app

# Run your Flask application
flask run --host=0.0.0.0 --port=8000