# Use the official Python image from the Docker Hub
FROM python:3.12.5-slim

# Set environment variables
# This prevents Python from writing .pyc files to disc and allows output to be shown in the logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your Django project files into the container
COPY . /app/

# Collect static files (optional, if you have static files)
RUN python manage.py collectstatic --noinput

# Expose the port your app runs on
EXPOSE 8000

# Start the Django development server (or you can change it to a production server like gunicorn)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
