"""Gunicorn configuration for ChirpID Backend"""
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', 5001)}"
backlog = 2048

# Worker processes
workers = os.getenv('WEB_WORKERS', 2)
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 100

# Logging
loglevel = os.getenv('LOG_LEVEL', 'info')
accesslog = '-'  # Log to stdout
errorlog = '-'   # Log to stderr
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'chirpid-backend'

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# SSL (if needed in future)
keyfile = None
certfile = None

# Application
wsgi_module = 'server.wsgi:application'
