#!/usr/bin/env bash
# Stops the LeRobot GUI backend if it is running on port 8000

echo "Checking for LeRobot GUI process on port 8000..."

if command -v fuser >/dev/null 2>&1; then
    if fuser 8000/tcp >/dev/null 2>&1; then
        echo "Found process on port 8000. Killing it..."
        fuser -k 8000/tcp
        echo "LeRobot GUI stopped."
    else
        echo "No process found on port 8000."
    fi
elif command -v lsof >/dev/null 2>&1; then
    PID=$(lsof -t -i:8000)
    if [ -n "$PID" ]; then
        echo "Found process $PID on port 8000. Killing it..."
        kill "$PID"
        echo "LeRobot GUI stopped."
    else
        echo "No process found on port 8000."
    fi
else
    echo "Error: Neither 'fuser' nor 'lsof' found. Please install one to use this script."
    exit 1
fi
