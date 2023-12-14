# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container at /app
COPY . .
COPY music_svm_model.pkl /app/models/

# Copy the start.sh script into the container at /app
COPY start.sh .
RUN chmod +x start.sh

# Expose the port Flask is running on
EXPOSE 8000

# Define environment variable for Flask
ENV FLASK_APP=app/views.py

# Use the start.sh script as the entry point to start your application
CMD ["./start.sh"]
