# Use the official Python base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Copy the RagTextFiles directory to the working directory
COPY RagTextFiles RagTextFiles

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the port that the FastAPI app runs on
EXPOSE 8000

# Run the FastAPI application using uvicorn
CMD ["uvicorn", "fastrun1:app", "--host", "0.0.0.0", "--port", "8000"]
