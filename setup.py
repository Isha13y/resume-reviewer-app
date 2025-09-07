#!/usr/bin/env python3
"""
Setup script for LLM Resume Reviewer
This script helps users set up the application quickly
"""

import os
import subprocess
import sys

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    try:
        print("📦 Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def check_api_key():
    """Check if API key is configured"""
    if os.path.exists("config.env"):
        with open("config.env", "r") as f:
            content = f.read()
            if "your_openai_api_key_here" in content:
                print("⚠️  Please configure your OpenAI API key in config.env")
                return False
            elif "OPENAI_API_KEY=" in content and len(content.split("OPENAI_API_KEY=")[1].split("\n")[0]) > 10:
                print("✅ API key appears to be configured")
                return True
    print("⚠️  config.env file not found or API key not configured")
    return False

def main():
    """Main setup process"""
    print("🚀 Setting up LLM Resume Reviewer")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Check API key
    api_configured = check_api_key()
    
    print("\n" + "=" * 50)
    print("📋 Setup Summary:")
    print("✅ Python version compatible")
    print("✅ Required packages installed")
    
    if api_configured:
        print("✅ API key configured")
        print("\n🎉 Setup complete! Run the application with:")
        print("   streamlit run app.py")
    else:
        print("⚠️  API key needs configuration")
        print("\n📝 Next steps:")
        print("1. Get an OpenAI API key from: https://platform.openai.com/")
        print("2. Edit config.env and replace 'your_openai_api_key_here' with your actual key")
        print("3. Run: streamlit run app.py")
    
    return True

if __name__ == "__main__":
    main()
