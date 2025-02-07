# Use an official Python runtime as a parent image.
FROM python:3.11-slim

# Set the working directory in the container.
WORKDIR /app

# Copy only requirements.txt first to take advantage of Docker cache.
COPY requirements.txt .

# Install the dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Expose port 7860 (required for Hugging Face Spaces)
EXPOSE 7860

# Run the Gradio application.
CMD ["python", "app.py"]
