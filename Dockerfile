# Dockerfile

# ----------------------------------------------------
# --- Stage 1: Builder Stage (Install Dependencies) ---
# ----------------------------------------------------
# Use a specific slim Python image for stability and smaller size
FROM python:3.10-slim as builder

# Set working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
# This ensures that if only the code changes, dependencies aren't re-installed.
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces the image size
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------
# --- Stage 2: Production Stage (Lean Runtime Image) ---
# -----------------------------------------------------
# Use the same base image for compatibility
FROM python:3.10-slim as production

# Set working directory
WORKDIR /app

# --- FIX for "uvicorn: executable file not found" ---
# Copy the executables (like uvicorn, fastapi) into the container's PATH
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy the installed packages (libraries like torch, transformers)
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy the application code
COPY app/. /app

# Expose the port Uvicorn will run on
EXPOSE 8000

# Set environment variables with defaults (these are overridden by docker-compose)
ENV LLM_MODEL_NAME=distilgpt2
ENV API_KEY=default_api_key

# The command to run the application using Uvicorn
# We use --workers 1 to ensure only one instance of the large LLM is loaded
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]