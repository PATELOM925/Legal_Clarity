FROM python:3.12.2

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/tmp/transformers_cache \
    FLASK_ENV=production \
    FLASK_APP=app.py \
    PORT=5003

# Set working directory in container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Preload NLTK data
RUN python -m nltk.downloader punkt punkt_tab -d /tmp/nltk_data

# Expose the port the app runs on
EXPOSE 5003

# Run the Flask app
# CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
CMD ["python", "app.py"]


## For hugging Face deployment
# FROM python:3.12.2

# ENV PYTHONUNBUFFERED=1 \
#     HF_HOME=/tmp/huggingface_cache \  # Update to HF_HOME instead of TRANSFORMERS_CACHE
#     FLASK_ENV=production \
#     FLASK_APP=app.py \
#     PORT=5003

# # Set working directory in container
# WORKDIR /app

# # Copy the current directory contents into the container
# COPY . .

# # Install dependencies
# RUN pip install -r requirements.txt

# # Preload NLTK data
# RUN python -m nltk.downloader punkt punkt_tab -d /tmp/nltk_data


# # Expose the port the app runs on
# EXPOSE 5003

# # Run the Flask app
# # CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
# CMD ["python", "app.py"]
