# ============================================================
# CONFIGURATION FILE
# ============================================================

import os
import warnings
warnings.filterwarnings('ignore')

# Get current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# Figure settings
FIGURE_DPI = 300
ALPHA = 0.05

# Color palettes
COLORS = {
    'primary': '#3498db',
    'secondary': '#e74c3c',
    'success': '#2ecc71',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'info': '#3498db',
    'dark': '#2c3e50',
    'light': '#ecf0f1'
}

# Survival colors
SURVIVAL_COLORS = ['#e74c3c', '#2ecc71']  # Red = Not Survived, Green = Survived
CLASS_COLORS = ['#2ecc71', '#3498db', '#e74c3c']  # Class 1, 2, 3

print(f"✅ Configuration loaded")
print(f"   Working Directory: {CURRENT_DIR}")