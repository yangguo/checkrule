# Use a multi-stage build
# First stage: Builder
FROM python:3.10-slim as builder

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies to /root/.local to avoid conflicts with system packages
RUN pip install --user --no-cache-dir -r requirements.txt

# Second stage: Final image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the dependencies from the builder stage
COPY --from=builder /root/.local /root/.local

# Update the PATH environment variable to include /root/.local/bin
ENV PATH=/root/.local/bin:$PATH

# Copy the rest of the application code into the container
COPY . .

# Expose the port your FastAPI app will run on
EXPOSE 8000

# Start the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
