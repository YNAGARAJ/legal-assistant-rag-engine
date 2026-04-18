# Use a slim version of Python 3.12 for a smaller image footprint
FROM python:3.12-slim

# Set environment variables to prevent Python from writing .pyc files 
# and to ensure logs are sent straight to the terminal
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system-level dependencies for FAISS (CPU) and GPT4All
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements to leverage Docker's layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create the data and vectorstore directories if they don't exist
RUN mkdir -p data vectorstore

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the application using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]