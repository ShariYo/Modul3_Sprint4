# Start from base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the required packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY ["model.pkl", "app.py", "./"] .

# Expose the app port
EXPOSE 8000

# Run command
CMD ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"]