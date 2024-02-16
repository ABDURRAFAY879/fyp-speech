# Use an official Python runtime as a parent image
FROM python:3.7

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN apt-get update && \
    apt-get install -y ffmpeg

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run Uvicorn when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]