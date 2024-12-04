
# Base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port
EXPOSE 5000

# Command to run the app
CMD ["python", "parkinsons_diagnosis.py"]
