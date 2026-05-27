# Φίλτρο Φωνής — Voice Filter

A web application for speaker separation and isolation from audio files.
Built in MATLAB with an HTML/JS frontend.

## Features
- Automatic speaker detection and separation
- Three modes: Conversation, Simultaneous, Mixed
- Web interface for upload, preview, and download
- Supports WAV, MP3, M4A

## Requirements
- MATLAB R2025b
- Audio Toolbox
- Signal Processing Toolbox
- Deep Learning Toolbox
- seperateSpeakers Library (For Sepformer)
- vggish Library

## Important
The "seperateSpeakers" and "vggish" libraries can be downloaded as zip files from the MATLAB command window.

## How to Run
1. Open MATLAB and navigate to the project folder
2. Run: startServer
3. Open Chrome at http://localhost:8080
4. Upload an audio file, select mode, click Analyse Speakers

## Pipeline
- **Conversation mode**: VGGish embeddings + K-means clustering + VAD
- **Simultaneous mode**: SepFormer (deep learning source separation)
- **Mixed mode**: SepFormer + VGGish diarization mask

## Files
- `startServer.m` - MATLAB TCP server and HTTP handler
- `runSeparation.m` - Audio processing pipeline
- `index.html` - HTML Web interface
- `Catching Up With Friends Audio 2` - audio file for testing *alternating* speakers
- `2SpeakersSimult` - audio file for testing *simultaneous* speakers

