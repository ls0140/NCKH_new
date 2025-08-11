#!/usr/bin/env python3
"""
Simple HTTP Server for AI Fake News Detection Website
Run this file to start the local server
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

# Configuration
PORT = 8000
DIRECTORY = Path(__file__).parent

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)
    
    def end_headers(self):
        # Add CORS headers for development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    """Start the local server"""
    print("🚀 Starting AI Fake News Detection Website Server...")
    print(f"📁 Serving files from: {DIRECTORY}")
    print(f"🌐 Server will be available at: http://localhost:{PORT}")
    print("=" * 50)
    
    # Check if required files exist
    required_files = ['index.html', 'styles.css', 'script.js']
    missing_files = [f for f in required_files if not (DIRECTORY / f).exists()]
    
    if missing_files:
        print("❌ Error: Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease make sure all files are in the same directory as server.py")
        return
    
    print("✅ All required files found!")
    print("\n📋 Available files:")
    for file in DIRECTORY.iterdir():
        if file.is_file() and file.suffix in ['.html', '.css', '.js', '.py']:
            print(f"   - {file.name}")
    
    try:
        with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
            print(f"\n🎯 Server started successfully!")
            print(f"📱 Open your browser and go to: http://localhost:{PORT}")
            print("🔄 Press Ctrl+C to stop the server")
            print("=" * 50)
            
            # Try to open browser automatically
            try:
                webbrowser.open(f'http://localhost:{PORT}')
                print("🌐 Browser opened automatically!")
            except:
                print("⚠️  Could not open browser automatically. Please open it manually.")
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        print("👋 Goodbye!")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"\n❌ Error: Port {PORT} is already in use!")
            print("💡 Try using a different port or stop the existing server")
        else:
            print(f"\n❌ Error starting server: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 