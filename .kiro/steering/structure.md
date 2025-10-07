# Project Structure

## Root Directory Layout
```
├── .git/                 # Git version control
├── .kiro/                # Kiro AI assistant configuration
│   └── steering/         # AI guidance documents
├── venv/                 # Python virtual environment (excluded from git)
├── face_detect.py        # Main face detection implementation
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── .gitignore           # Git ignore patterns
```

## File Organization Conventions

### Python Files
- **Main scripts** - Place executable scripts in the root directory
- **Modules** - If the project grows, create a `src/` or package directory
- **Tests** - Use `tests/` directory for unit tests (when added)
- **Utilities** - Use `utils/` for helper functions (when needed)

### Configuration Files
- **Dependencies** - `requirements.txt` for pip dependencies
- **Environment** - Use `.env` files for environment variables (git-ignored)
- **Config** - JSON/YAML config files in root or `config/` directory

### Data and Assets
- **Input data** - Use `data/` directory for sample images/videos
- **Output** - Use `output/` or `results/` for processed files
- **Models** - Use `models/` for any custom trained models

## Naming Conventions
- **Files** - Use snake_case for Python files (e.g., `face_detect.py`)
- **Functions** - Use snake_case for function names
- **Classes** - Use PascalCase for class names
- **Constants** - Use UPPER_SNAKE_CASE for constants

## Import Organization
```python
# Standard library imports
import os
import sys

# Third-party imports
import cv2
import mediapipe as mp
import numpy as np

# Local imports (when applicable)
from utils import helper_functions
```

## Git Workflow
- Keep the virtual environment (`venv/`) out of version control
- Include `requirements.txt` for dependency management
- Use meaningful commit messages for computer vision features
- Tag releases when adding new detection capabilities