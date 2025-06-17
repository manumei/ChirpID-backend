# ChirpID Backend

A Flask-based backend service for the ChirpID bird sound identification application. This service processes audio files and uses machine learning models to identify bird species from audio recordings.

## Features

- Audio file upload and processing
- Bird species identification using ML models
- RESTful API for mobile app integration
- Health monitoring and status endpoints
- Docker containerization for easy deployment

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- Git

### Local Development

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd ChirpID-backend
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python run.py
   ```

The server will start on `http://localhost:5000`

## Deployment

For complete deployment instructions including SSH deployment to Ubuntu servers, see **[DEPLOYMENT.md](DEPLOYMENT.md)**.

### Quick Deployment Options

#### Option 1: Using Docker directly

1. **Build the image:**

   ```bash
   docker build -t chirpid-backend .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name chirpid-backend \
     -p 5000:5000 \
     -v $(pwd)/app/uploads:/app/app/uploads \
     -v $(pwd)/database:/app/database \
     chirpid-backend
   ```

#### Option 2: Using docker-compose (Development)

```bash
docker-compose up -d
```

#### Option 3: Using docker-compose (Production with Nginx)

```bash
docker-compose -f docker-compose.prod.yml up -d --profile production
```

#### Option 4: Using the deployment script

Make the script executable and run it:

```bash
chmod +x deploy.sh
./deploy.sh build          # For local build and deploy
./deploy.sh deploy         # For deploying from registry
./deploy.sh deploy-compose # For production deployment with nginx
```

## Automated Deployment with GitHub Actions

This repository includes automated CI/CD pipeline that:

1. **Tests** the application on every push and pull request
2. **Builds** Docker images and pushes them to GitHub Container Registry
3. **Deploys** to production on pushes to main/master branch

### Setting up GitHub Actions

1. **Enable GitHub Container Registry:**

   - Go to your repository settings
   - Navigate to "Actions" → "General"
   - Enable "Read and write permissions" for GITHUB_TOKEN

2. **Configure deployment target:**

   - Edit `.github/workflows/deploy.yml`
   - Uncomment and configure the deployment section for your hosting platform

3. **Set up secrets (if deploying to VPS):**
   - `HOST`: Your server IP address
   - `USERNAME`: SSH username
   - `KEY`: SSH private key

### Supported Deployment Platforms

The automated deployment supports various platforms:

- **AWS ECS/Fargate**
- **Google Cloud Run**
- **Azure Container Instances**
- **DigitalOcean App Platform**
- **Heroku**
- **Self-hosted VPS with Docker**

## API Endpoints

- `GET /health` - Health check endpoint
- `GET /ping` - Simple ping endpoint
- `POST /api/audio/upload` - Upload audio file for bird identification
- `GET /api/audio/status` - Check processing status

## Configuration

### Environment Variables

- `FLASK_ENV`: Set to 'production' for production deployment
- `FLASK_DEBUG`: Set to 'False' for production

### File Uploads

Audio files are stored in the `app/uploads` directory. Make sure this directory is persistent in production deployments.

## Monitoring and Logs

### Check container status:

```bash
docker ps -f name=chirpid-backend
```

### View logs:

```bash
docker logs -f chirpid-backend
```

### Health check:

```bash
curl http://localhost:5000/health
```

## Development

##### For instructions on how to run the connection between the Flask server and the React Native front-end, see [the App instructions](app/README.md).

---

##### For instructions on how to run the Neural Network model for bird prediction, see [the Notebooks Instructions](notebooks/README.md).

## Troubleshooting

### Common Issues

1. **Port already in use:**

   ```bash
   docker stop chirpid-backend
   docker rm chirpid-backend
   ```

2. **Permission issues with volumes:**

   ```bash
   sudo chown -R $USER:$USER app/uploads database/
   ```

3. **Image not found:**
   Make sure to build the image first or check the registry URL

### Performance Tuning

For production deployments:

1. **Increase worker processes** in the Flask configuration
2. **Use a reverse proxy** like Nginx (included in docker-compose.prod.yml)
3. **Configure proper logging** and monitoring
4. **Set up SSL certificates** for HTTPS

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if needed
5. Submit a pull request

## License

[Add your license information here]
