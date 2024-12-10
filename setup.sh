#!/bin/bash

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel

# Install Python dependencies from requirements.txt
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_trf
