# Python Dockerfile - Optimized for production
FROM python:3.13-slim-bookworm

# Set working directory
WORKDIR /app

# Install only essential system dependencies including openssl
RUN apt-get update && apt-get install -y \
    curl \
    openssl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY deploy-requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --root-user-action=ignore --upgrade pip && \
    pip install --no-cache-dir --root-user-action=ignore -r deploy-requirements.txt

# Copy only essential application files
COPY app/ ./app/
COPY server/ ./server/
COPY gunicorn.conf.py .

# Create uploads directory with proper permissions
RUN mkdir -p app/uploads && \
    chmod -R 777 app/uploads

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Run the application directly
CMD ["gunicorn", "--config", "gunicorn.conf.py", "server.wsgi:application"]