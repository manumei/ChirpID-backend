# ChirpID Backend Deployment Guide

## Quick Summary of Recent Changes

The deployment issues have been addressed with the following improvements:

### 1. **Production WSGI Server**

- Added Gunicorn for production deployments instead of Flask development server
- Added `gunicorn.conf.py` configuration file
- Updated Dockerfile to use Gunicorn

### 2. **Enhanced Health Checks**

- Added `curl` to the Docker image for health checks
- Increased health check timeout from 30 to 60 attempts
- Added better error reporting and debugging information
- Created Python-based health check script (`scripts/health_check.py`)

### 3. **Better Error Handling**

- Added container startup testing before deployment
- Enhanced logging and debugging output
- Added internal health check testing

### 4. **Testing Infrastructure**

- Added startup test script (`scripts/test_startup.py`)
- Added development Docker compose configuration
- Enhanced existing docker-compose.yml

## Deployment Options

### Option 1: GitHub Actions (Recommended)

The `.github/workflows/deploy.yml` file handles automatic deployment when code is pushed to main/master branch.

### Option 2: Manual Docker Deployment

1. **Build and run with Docker Compose (Production):**

   ```bash
   docker-compose up -d --build
   ```

2. **Build and run with Docker Compose (Development):**

   ```bash
   docker-compose -f docker-compose.dev.yml up --build
   ```

3. **Manual Docker commands:**

   ```bash
   # Build the image
   docker build -t chirpid-backend .

   # Run the container
   docker run -d \
     --name chirpid-backend \
     --restart unless-stopped \
     -p 5000:5001 \
     -v ~/chirpid/uploads:/app/app/uploads \
     -v ~/chirpid/database:/app/database \
     -e FLASK_ENV=production \
     -e FLASK_DEBUG=False \
     -e PORT=5001 \
     chirpid-backend
   ```

## Testing the Deployment

### Health Check

```bash
curl http://localhost:5000/health
```

Expected response:

```json
{ "status": "healthy", "service": "ChirpID Backend" }
```

### Using the Python Health Check Script

```bash
python scripts/health_check.py http://localhost:5000
```

### Testing App Startup

```bash
python scripts/test_startup.py
```

## Environment Variables

- `FLASK_ENV`: Set to "production" for production deployments
- `FLASK_DEBUG`: Set to "False" for production
- `PORT`: Port number (default: 5001)
- `HOST`: Host address (default: 0.0.0.0)

## File Structure

```
ChirpID-backend/
├── app/                          # Flask application
│   ├── __init__.py              # App factory
│   ├── routes/                  # API routes
│   └── services/                # Business logic
├── server/                      # Server configuration
│   ├── run.py                   # Development server
│   ├── wsgi.py                  # Production WSGI entry point
│   └── start.py                 # Universal startup script
├── scripts/                     # Utility scripts
│   ├── health_check.py          # Health check utility
│   └── test_startup.py          # Startup testing
├── gunicorn.conf.py             # Gunicorn configuration
├── Dockerfile                   # Production container
├── Dockerfile.dev               # Development container
├── docker-compose.yml           # Production compose
├── docker-compose.dev.yml       # Development compose
└── requirements.txt             # Python dependencies
```

## Troubleshooting

### Common Issues

1. **Health check fails:**

   - Check if the container is running: `docker ps`
   - Check container logs: `docker logs chirpid-backend`
   - Test internal health: `docker exec chirpid-backend curl http://localhost:5001/health`

2. **Port conflicts:**

   - Change the host port in docker-compose.yml or Docker run command
   - Default mapping is `5000:5001` (host:container)

3. **Permission issues:**
   - Ensure upload directories have proper permissions
   - Check if the non-root user in container can write to mounted volumes

### Debugging Commands

```bash
# Check container status
docker ps -a

# View container logs
docker logs chirpid-backend

# Execute commands inside container
docker exec -it chirpid-backend bash

# Test health endpoint from inside container
docker exec chirpid-backend curl http://localhost:5001/health

# Run startup tests
docker exec chirpid-backend python scripts/test_startup.py
```

## Performance Tuning

### Gunicorn Workers

Adjust in `gunicorn.conf.py`:

- `workers`: Number of worker processes (default: 2)
- `worker_class`: Worker type (default: "sync")
- `timeout`: Request timeout (default: 30s)

### Resource Limits

Add to docker-compose.yml:

```yaml
deploy:
  resources:
    limits:
      memory: 1G
      cpus: "0.5"
```

## Security Considerations

1. **Environment Variables**: Store sensitive data in environment variables or secrets
2. **HTTPS**: Use nginx or a reverse proxy for HTTPS termination
3. **Firewall**: Restrict access to necessary ports only
4. **Updates**: Keep base images and dependencies updated

## Monitoring

### Health Checks

- Docker health checks are configured with 60s start period
- Health endpoint: `/health`
- Ping endpoint: `/ping`

### Logs

- Application logs are sent to stdout/stderr
- View with: `docker logs chirpid-backend`
- For persistent logging, mount a log volume

## Next Steps

1. **SSL/HTTPS**: Configure nginx with SSL certificates
2. **Database**: Add persistent database if needed
3. **Monitoring**: Add application performance monitoring
4. **Backup**: Implement backup strategy for uploads and data
5. **Scaling**: Consider horizontal scaling with load balancer
