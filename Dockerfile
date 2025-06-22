# Python Dockerfile - Optimized for production
FROM python:3.13-slim-bookworm

# Set working directory
WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user early for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements first for better Docker layer caching
COPY deploy-requirements-minimal.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --root-user-action=ignore --upgrade pip && \
    pip install --no-cache-dir --root-user-action=ignore -r deploy-requirements-minimal.txt

# Copy only essential application files
COPY app/ ./app/
COPY server/ ./server/
COPY gunicorn.conf.py .

# Create uploads directory and set ownership for required directories only
RUN mkdir -p app/uploads && \
    chown -R appuser:appuser app/uploads server/ gunicorn.conf.py && \
    chmod -R 775 app/uploads

# Create entrypoint script to fix permissions at runtime
RUN echo '#!/bin/bash\n\
    # Fix permissions for mounted volumes\n\
    chown -R appuser:appuser /app/app/uploads\n\
    chmod -R 775 /app/app/uploads\n\
    # Switch to appuser and run the application\n\
    exec su-exec appuser gunicorn --config gunicorn.conf.py server.wsgi:application\n\
    ' > /entrypoint.sh && chmod +x /entrypoint.sh

# Install su-exec for user switching
RUN apt-get update && apt-get install -y su-exec && rm -rf /var/lib/apt/lists/*

# Don't switch to non-root user yet - we'll do it in entrypoint
# USER appuser

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Run the application with entrypoint script
CMD ["/entrypoint.sh"]