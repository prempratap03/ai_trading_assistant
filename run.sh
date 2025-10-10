#!/bin/bash

# AI Trading Assistant - Quick Start Script

echo "=================================="
echo "AI Trading Assistant"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update requirements
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

# Run the application
echo ""
echo "Starting AI Trading Assistant..."
echo "Access at: http://localhost:8501"
echo ""
streamlit run src/main.py

# Deactivate on exit
deactivate
