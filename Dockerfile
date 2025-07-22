FROM python:3.9-slim

# 1. Install build tools and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Set working directory
WORKDIR /app

# 3. Copy all files into container
COPY . .

# 4. Upgrade pip and install dependencies
# Install NumPy first with a compatible version
RUN pip install numpy==1.24.4 && \
    pip install --no-cache-dir -r api/requirements.txt


# 5. Expose the port
EXPOSE 8000

# 6. Run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
