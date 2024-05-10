# Use an official Python runtime as a base image
FROM python:3.9.17

# Define environment variable
#ENV APP_DIR="/Soft Tissue Tumor"

# Set the working directory in the container
WORKDIR /app

# Copy the local code to the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 8080

# Run the Python script for your project
CMD ["python", "manage.py", "runserver", "127.0.0.1:8080"]