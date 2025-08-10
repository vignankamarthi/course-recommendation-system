FROM python:3.11-slim

# Set build arguments for user ID matching
ARG USER_ID=1000
ARG GROUP_ID=1000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create user with matching ID for seamless file permissions
# Handle existing group/user gracefully
RUN (getent group $GROUP_ID || groupadd -g $GROUP_ID appgroup) && \
    (getent passwd $USER_ID || useradd -u $USER_ID -g $GROUP_ID -m -s /bin/bash appuser)

# Create necessary directories and set ownership using numeric IDs
RUN mkdir -p /app/agents /app/core /app/database /app/tools /app/utils /app/app /app/data && \
    chown -R $USER_ID:$GROUP_ID /app

# Switch to non-root user  
USER $USER_ID:$GROUP_ID

# Set Python path to include project root (now a proper Python package)
ENV PYTHONPATH="/app:$PYTHONPATH"

# Default command to run the application
CMD ["python3", "app.py"]
