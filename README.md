# Neuro

# NeuroVoice — BCI Communication System

Deep Learning Based Brain-Computer Interface for Recognition of 
Covert Speech to Augment Patient Care Communication.

## Features
- EEGNet deep learning model for EEG signal classification
- 8-class imagined speech recognition (Yes, No, Help, Pain, Water, Toilet, Doctor, Family)
- Real-time WebSocket streaming
- Doctor dashboard with live decoding
- Patient AAC communication board with large tap buttons
- Text-to-speech in English, Hindi, and Marathi
- Emergency alert system
- Patient session management with SQLite database

## Tech Stack
Python · TensorFlow · FastAPI · SQLite · SQLAlchemy · MNE · HTML/CSS/JS

## Setup
pip install -r requirements.txt
cd backend && uvicorn main:app --reload --port 8000

## Dataset
PhysioNet EEG Motor Movement/Imagery Dataset
