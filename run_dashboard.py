#!/usr/bin/env python3
"""
Script to run the Streamlit dashboard
"""

import os
import sys
import subprocess
import webbrowser
import time

def main():
    """Run the Streamlit dashboard"""
    
    print("=" * 60)
    print("🚀 Starting Bati Bank Credit Risk Dashboard")
    print("=" * 60)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Get the path to the dashboard app
    dashboard_path = os.path.join("src", "dashboard", "app.py")
    
    if not os.path.exists(dashboard_path):
        print(f"❌ Dashboard not found at {dashboard_path}")
        return
    
    print(f"📊 Dashboard path: {dashboard_path}")
    print("\n" + "=" * 60)
    print("🌐 Opening browser in 3 seconds...")
    print("⏳ Press Ctrl+C to stop the dashboard")
    print("=" * 60 + "\n")
    
    # Open browser after a short delay
    time.sleep(3)
    webbrowser.open("http://localhost:8501")
    
    # Run the dashboard
    cmd = ["streamlit", "run", dashboard_path]
    subprocess.run(cmd)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped. Goodbye!")