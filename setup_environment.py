#!/usr/bin/env python3
"""
PORTAL VII - Environment Setup
Create virtual environment and install dependencies for PORTAL VII DIBLE Vault
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("🚀 PORTAL VII - Environment Setup")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  No virtual environment detected")
        print("💡 Creating virtual environment...")
        
        venv_path = Path("portal7_env")
        if not run_command(f"{sys.executable} -m venv {venv_path}", "Creating virtual environment"):
            print("❌ Failed to create virtual environment")
            print("💡 Please install python3-venv:")
            print("   sudo apt install python3-venv")
            return 1
        
        # Activate virtual environment
        if os.name == 'nt':  # Windows
            activate_script = venv_path / "Scripts" / "activate"
            python_path = venv_path / "Scripts" / "python.exe"
            pip_path = venv_path / "Scripts" / "pip.exe"
        else:  # Unix/Linux
            activate_script = venv_path / "bin" / "activate"
            python_path = venv_path / "bin" / "python"
            pip_path = venv_path / "bin" / "pip"
        
        print(f"✅ Virtual environment created at: {venv_path}")
        print(f"💡 To activate: source {activate_script}")
        print(f"💡 Python: {python_path}")
        print(f"💡 Pip: {pip_path}")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        print("⚠️  Failed to upgrade pip, continuing...")
    
    # Install required packages
    required_packages = [
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "cryptography>=3.4.8",
        "pycryptodome>=3.15.0",
        "sympy>=1.9",
        "networkx>=2.6",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
        "psutil>=5.8.0",
        "click>=8.0.0",
        "rich>=13.0.0"
    ]
    
    for package in required_packages:
        if not run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}"):
            print(f"❌ Failed to install {package}")
            return 1
    
    # Install optional packages
    print("\n📦 Installing optional dependencies...")
    optional_packages = [
        "py-cpuinfo>=8.0.0",
        "GPUtil>=1.4.0",
        "platform-info>=1.0.0",
        "colorama>=0.4.4",
        "pyfiglet>=0.8.0"
    ]
    
    for package in optional_packages:
        if not run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}"):
            print(f"⚠️  Failed to install optional package {package}")
    
    # Test installation
    print("\n🧪 Testing installation...")
    test_script = """
import sys
try:
    import numpy
    import scipy
    import cryptography
    import Crypto
    import sympy
    import networkx
    import matplotlib
    import seaborn
    import psutil
    import click
    import rich
    print("✅ All required packages imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
"""
    
    if run_command(f"{sys.executable} -c \"{test_script}\"", "Testing package imports"):
        print("\n🎉 PORTAL VII environment setup completed successfully!")
        print("\n🚀 You can now run PORTAL VII:")
        print("   python3 portal7_launcher.py")
        print("   python3 dible_vault_cli.py")
        print("   python3 portal7_custom_terminal.py")
        return 0
    else:
        print("\n❌ Environment setup failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())