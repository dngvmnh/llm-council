#!/bin/bash

# ... (keep your existing trap and cleanup logic) ...

echo "Starting Multiple LLMs Council..."

# 1. Start Backend
echo "Initializing Backend..."
cd backend

# Use the absolute path to the venv python to be 100% sure
VENV_PYTHON="./.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment not found at backend/.venv"
    exit 1
fi

# Ensure uvicorn is installed in the venv before running
$VENV_PYTHON -m pip install -q uvicorn fastapi python-dotenv 

# Run the server using the venv python
$VENV_PYTHON server.py &
BACKEND_PID=$!
cd ..

# 2. Start Frontend
echo "Initializing Frontend..."
cd frontend
# Optional: npm install if node_modules is missing
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi
npm run dev &
FRONTEND_PID=$!
cd ..

echo "Both servers are running!"
echo "Frontend: http://localhost:5173"
echo "Backend:  http://localhost:8080"
echo "Press Ctrl+C to stop both."

# Keep the script running to catch the Ctrl+C
wait