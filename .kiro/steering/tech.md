# Technology Stack

## Core Technologies
- **Python 3.x** - Primary programming language
- **MediaPipe 0.10.21** - Google's framework for building multimodal applied ML pipelines
- **OpenCV 4.11.0** - Computer vision library for image and video processing
- **NumPy 1.26.4** - Numerical computing library

## Key Dependencies
- **matplotlib** - Plotting and visualization
- **scipy** - Scientific computing
- **pillow** - Image processing
- **sounddevice** - Audio I/O (if audio processing is needed)

## Development Environment
- **Virtual Environment** - Uses `venv/` for dependency isolation
- **Requirements Management** - Dependencies managed via `requirements.txt`

## Common Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Run face detection
python face_detect.py
```

### Development Workflow
```bash
# Update dependencies
pip freeze > requirements.txt

# Deactivate virtual environment
deactivate
```

## Code Style Guidelines
- Follow PEP 8 Python style guidelines
- Use meaningful variable names for computer vision operations
- Import MediaPipe as `mp` (standard convention)
- Import OpenCV as `cv2` (standard convention)