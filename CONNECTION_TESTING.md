# ChirpID Backend - Connection Testing Guide

This guide helps you validate the connection between the React Native frontend and Flask backend.

## Prerequisites

1. **Flask Backend Setup**

   ```bash
   cd "\ChirpID-backend"
   pip install -r requirements.txt
   ```

2. **React Native Frontend Setup**
   ```bash
   cd "\chirpid"
   npm install
   ```

## Step 1: Get Your Local IP Address

Run this script to get your computer's local IP address:

```bash
cd "\ChirpID-backend"
python get_ip.py
```

Update the `.env` file in your React Native project:

```
EXPO_PUBLIC_API_URL=http://YOUR_IP_ADDRESS:5000
```

## Step 2: Start the Flask Backend

```bash
cd "\ChirpID-backend"
python run.py
```

You should see:

```
Starting ChirpID Backend Server...
Server will be available at: http://0.0.0.0:5000
Local network access at: http://<your-ip>:5000
```

## Step 3: Test Backend Endpoints

Run the test script to verify all endpoints work:

```bash
cd "\ChirpID-backend"
python test_server.py
```

Expected output:

```
Testing ChirpID Backend Server...
========================================
Ping test: 200
Response: {'status': 'ok', 'message': 'ChirpID backend is running'}

Health test: 200
Response: {'status': 'healthy', 'service': 'ChirpID Backend'}

Upload test: 200
Response: {'success': True, 'message': 'Audio uploaded and processed successfully', ...}

========================================
Test Results:
Ping: ✓
Health: ✓
Upload: ✓

All tests passed! 🎉
```

## Step 4: Start the React Native App

```bash
cd "\chirpid"
npx expo start
```

## Step 5: Test Frontend Connection

1. **Upload Test**:

   - Record an audio clip using the record button
   - Tap "Send" to upload to the backend
   - Check for success/error messages

2. **File Upload Test**:
   - Go to the "Upload" tab
   - Select an audio file
   - Verify it uploads successfully

## Troubleshooting

### Backend Connection Issues

**Solutions**:

1. Verify the Flask server is running on port 5000
2. Check that your IP address in `.env` is correct
3. Ensure both devices are on the same network
4. Try disabling firewall temporarily
5. Check if Windows Defender is blocking the connection

### Upload Failures

**Problem**: "Upload Failed" errors

**Solutions**:

1. Check the Flask server logs for error messages
2. Verify CORS is properly configured
3. Ensure the audio file format is supported (wav, mp3, ogg, flac, m4a)
4. Check network connectivity

### Network Configuration

**For LAN Access**:

- Backend binds to `0.0.0.0:5000` (all interfaces)
- CORS allows all origins for development
- React Native uses absolute URI with local IP

**Firewall Settings**:

- Allow Python through Windows Firewall
- Allow inbound connections on port 5000

## API Endpoints

- `GET /ping` - Connection test
- `GET /health` - Health check
- `POST /api/audio/upload` - Audio file upload

## Features Implemented

✅ Flask server binds to 0.0.0.0:5000  
✅ CORS enabled for all origins  
✅ React Native uses correct local IP via EXPO_PUBLIC_API_URL  
✅ /ping route implemented in Flask  
✅ Network error logging and alerts  
✅ FormData POST requests with absolute URIs  
✅ Correct MIME type (audio/wav)  
✅ Proper error handling on fetch failure  
✅ Retry mechanisms for failed requests

## Development Notes

- Backend runs in debug mode for development
- Both upload methods (record + file picker) use the same API
- Comprehensive error handling with user-friendly messages
- All requests use absolute URIs with the configured API base URL
