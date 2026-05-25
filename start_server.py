import sys
import os

# Add the project directory to the path
sys.path.insert(0, r'e:\Machine Learing 2026')

# Set the working directory
os.chdir(r'e:\Machine Learing 2026')

# Import and run the app
from application import app

if __name__ == '__main__':
    print("=" * 50)
    print("Flask server starting...")
    print("=" * 50)
    print("Access the app at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)
