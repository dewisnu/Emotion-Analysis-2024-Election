# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

ENV MONGO_URI="mongodb://mongoadmin:A2HCSPFcz9yaXIWMLpkv8uZ4ht@34.143.131.197:27017/?directConnection=true"

# Expose the port Streamlit runs on
EXPOSE 8080

# Command to run the Streamlit app on port 8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]