# Use a base image with necessary dependencies
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the test code into the container
COPY . /app

# Install test dependencies if needed
RUN pip install pytest

# Command to run tests when the container starts
CMD ["pytest"]
