#!/bin/bash
# Activate the virtual environment located in the parent directory
source ../.venv/bin/activate
echo "Virtual environment activated!"
echo "Python path: $(which python)"
echo "Python version: $(python --version)" 