#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the Pi3 venv
source ~/pi3venv/bin/activate

# Generate self-signed SSL certs if missing
if [ ! -f webserver/server.cert ] || [ ! -f webserver/server.key ]; then
    echo "Generating self-signed SSL certificates..."
    openssl req -x509 -newkey rsa:2048 -keyout webserver/server.key -out webserver/server.cert \
        -days 365 -nodes -subj "/CN=localhost"
fi

# Install Python deps if needed
pip install -q fastapi uvicorn[standard] python-multipart websockets 2>/dev/null

# Install frontend deps and build if needed
if [ ! -d webserver/node_modules ]; then
    echo "Installing frontend dependencies..."
    cd webserver
    npm install
    cd ..
fi

# Build frontend
echo "Building frontend..."
cd webserver
npx vite build 2>/dev/null || echo "Frontend build skipped (dev mode available)"
cd ..

echo ""
echo "Starting Pi3X Real-Time Server..."
echo ""

# Pass through all arguments  
python server.py "$@"
