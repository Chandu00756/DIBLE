#!/usr/bin/env python3
"""
PORTAL VII - DIBLE Professional Package Builder
Universal installer and integration system

This creates proper packages, libraries, and plugins for different environments
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
import platform
import zipfile

class DIBLEPackageBuilder:
    """Professional package builder for DIBLE integration"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.build_dir = self.project_root / "dist"
        self.build_dir.mkdir(exist_ok=True)
        
    def create_python_package(self):
        """Create installable Python package"""
        print("📦 Creating Python package...")
        
        # Create package structure
        package_dir = self.build_dir / "portal7-dible-package"
        package_dir.mkdir(exist_ok=True)
        
        # Copy core files
        core_files = [
            "dible_vault_cli.py",
            "portal7_integration_advanced.py", 
            "portal7_custom_terminal.py",
            "portal7_launcher.py",
            "src/"
        ]
        
        for file in core_files:
            src = self.project_root / file
            dst = package_dir / file
            if src.exists():
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
        
        # Create setup.py for pip installation
        setup_content = '''from setuptools import setup, find_packages

setup(
    name="portal7-dible-vault",
    version="2.0.0",
    description="PORTAL VII DIBLE Vault - Professional Cryptographic System",
    long_description="PORTAL VII Professional DIBLE Vault System",
    author="PORTAL VII Systems",
    packages=find_packages(),
    py_modules=["dible_vault_cli", "portal7_integration_advanced", "portal7_custom_terminal"],
    install_requires=[
        "rich>=13.0.0",
        "click>=8.0.0", 
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "cryptography>=3.4.0",
        "pycryptodome>=3.15.0",
        "sympy>=1.9.0"
    ],
    entry_points={
        "console_scripts": [
            "portal7-vault=dible_vault_cli:main",
            "portal7-terminal=portal7_launcher:main"
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security :: Cryptography"
    ],
    python_requires=">=3.8"
)
'''
        
        with open(package_dir / "setup.py", 'w') as f:
            f.write(setup_content)
        
        print(f"✅ Python package structure created in {package_dir}")
        
    def create_vscode_extension_package(self):
        """Create VS Code extension package"""
        print("🔌 Creating VS Code extension package...")
        
        extension_dir = self.project_root / "vscode-extension"
        if not extension_dir.exists():
            print("❌ VS Code extension source not found")
            return
        
        # Copy to build directory
        build_extension_dir = self.build_dir / "portal7-dible-vscode-extension"
        if build_extension_dir.exists():
            shutil.rmtree(build_extension_dir)
        shutil.copytree(extension_dir, build_extension_dir)
        
        print(f"✅ VS Code extension copied to {build_extension_dir}")
        
    def create_standalone_executable(self):
        """Create standalone executable"""
        print("📱 Creating standalone executable...")
        
        try:
            import PyInstaller
        except ImportError:
            print("⚠️  PyInstaller not installed. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
        
        # Create executable
        try:
            os.chdir(self.project_root)
            
            cmd = [
                sys.executable, "-m", "PyInstaller",
                "--onefile",
                "--name", "portal7-vault",
                "--distpath", str(self.build_dir),
                "dible_vault_cli.py"
            ]
            
            subprocess.run(cmd, check=True)
            print("✅ Standalone executable created")
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Executable creation failed: {e}")
        
    def create_docker_container(self):
        """Create Docker container"""
        print("🐳 Creating Docker container...")
        
        dockerfile_content = '''FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create vault directory
RUN mkdir -p /root/.portal7_dible

# Expose port
EXPOSE 8080

# Set entrypoint
ENTRYPOINT ["python", "dible_vault_cli.py"]
CMD ["--help"]
'''
        
        with open(self.build_dir / "Dockerfile", 'w') as f:
            f.write(dockerfile_content)
        
        # Copy necessary files
        files_to_copy = [
            "dible_vault_cli.py",
            "portal7_integration_advanced.py",
            "requirements.txt",
            "src/"
        ]
        
        for file in files_to_copy:
            src = self.project_root / file
            dst = self.build_dir / file
            if src.exists():
                if src.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
        
        print("✅ Docker container files created")
        
    def create_library_examples(self):
        """Create integration examples"""
        print("📚 Creating library examples...")
        
        examples_dir = self.build_dir / "examples"
        examples_dir.mkdir(exist_ok=True)
        
        # Basic usage example
        basic_example = '''#!/usr/bin/env python3
"""
PORTAL VII DIBLE - Basic Usage Example
Simple encryption/decryption with professional integration
"""

try:
    from portal7_integration_advanced import PortalVIIIntegrator, quick_encrypt, quick_decrypt
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

def basic_usage():
    """Basic DIBLE usage example"""
    print("🔐 PORTAL VII DIBLE - Basic Usage Example")
    print("=" * 50)
    
    if not INTEGRATION_AVAILABLE:
        print("❌ Integration library not available")
        return
    
    # Method 1: Quick functions
    print("\\n1. Quick Encryption/Decryption:")
    data = "Hello, PORTAL VII DIBLE!"
    encrypted = quick_encrypt(data)
    decrypted = quick_decrypt(encrypted)
    
    print(f"Original:  {data}")
    print(f"Encrypted: ✓")
    print(f"Decrypted: {decrypted}")
    
    # Method 2: Full integrator
    print("\\n2. Full Integrator:")
    integrator = PortalVIIIntegrator(security_level=256)
    
    data2 = "Professional cryptographic data"
    encrypted2 = integrator.professional_encrypt(data2)
    decrypted2 = integrator.professional_decrypt(encrypted2)
    
    print(f"Original:  {data2}")
    print(f"Encrypted: ✓")
    print(f"Decrypted: {decrypted2['decrypted_data']}")
    
    # System status
    print("\\n3. System Status:")
    status = integrator.get_professional_status()
    print(f"  Session ID: {status['session_id']}")
    print(f"  Algorithm: {status['config']['algorithm']}")
    print(f"  Security: {status['config']['security_level']}-bit")

if __name__ == "__main__":
    basic_usage()
'''
        
        with open(examples_dir / "basic_usage.py", 'w') as f:
            f.write(basic_example)
        
        # File encryption example
        file_example = '''#!/usr/bin/env python3
"""
PORTAL VII DIBLE - File Encryption Example
Encrypt and decrypt files with DIBLE
"""

try:
    from portal7_integration_advanced import PortalVIIIntegrator
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

import tempfile
import os

def file_encryption_example():
    """File encryption example"""
    print("📁 PORTAL VII DIBLE - File Encryption Example")
    print("=" * 50)
    
    if not INTEGRATION_AVAILABLE:
        print("❌ Integration library not available")
        return
    
    # Initialize DIBLE
    integrator = PortalVIIIntegrator(security_level=256)
    
    # Create test file
    test_content = """PORTAL VII Professional Test Document
=====================================

This document contains sensitive information that will be
encrypted using the advanced HC-DIBLE-VAULT algorithm.

Features:
- Quantum-resistant encryption
- Device identity binding  
- Chaos-enhanced entropy
- Professional security
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        test_file = f.name
    
    try:
        # Encrypt file
        encrypted_file = test_file + '.portal7'
        print(f"\\n📄 Original file: {os.path.basename(test_file)}")
        
        result = integrator.professional_file_encrypt(test_file, encrypted_file)
        print(f"🔐 Encrypted to: {os.path.basename(encrypted_file)}")
        print(f"📊 Operation ID: {result['operation_id']}")
        
        # Decrypt file
        decrypted_file = test_file + '.decrypted'
        decrypt_result = integrator.professional_file_decrypt(encrypted_file, decrypted_file)
        print(f"🔓 Decrypted to: {os.path.basename(decrypted_file)}")
        
        # Verify integrity
        with open(decrypted_file, 'r') as f:
            decrypted_content = f.read()
        
        if decrypted_content == test_content:
            print("✅ File integrity verified - encryption/decryption successful!")
        else:
            print("❌ File integrity check failed!")
        
    finally:
        # Cleanup
        for file_path in [test_file, encrypted_file, decrypted_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)

if __name__ == "__main__":
    file_encryption_example()
'''
        
        with open(examples_dir / "file_encryption.py", 'w') as f:
            f.write(file_example)
        
        print("✅ Library examples created")
        
    def create_installation_guide(self):
        """Create comprehensive installation guide"""
        print("📖 Creating installation guide...")
        
        guide_content = """# PORTAL VII DIBLE - Installation & Integration Guide

## Quick Installation

### Python Package
```
pip install portal7-dible-vault
```

### From Source
```
git clone <repository-url>
cd DIBLE
python portal7_professional_setup.py
```

## Usage Examples

### Python Library Integration
```python
from portal7_integration_advanced import PortalVIIIntegrator

# Initialize
integrator = PortalVIIIntegrator(security_level=256)

# Encrypt/Decrypt
encrypted = integrator.professional_encrypt("Secret data")
decrypted = integrator.professional_decrypt(encrypted)
```

### Command Line Interface
```
python portal7_launcher.py
```

### Custom Terminal
```
python portal7_custom_terminal.py
```

## System Requirements

- Python 3.8+
- NumPy, SciPy (mathematical operations)
- Cryptography libraries
- Rich, Click (CLI interface)

## Security Features

- Quantum Resistant: Lattice-based cryptography
- Device Binding: Hardware-specific encryption  
- Chaos Enhanced: Multiple entropy sources
- Professional Grade: Enterprise security standards

## Support

For professional support:
- Documentation: Professional usage guides included
- Issues: Check error logs and status reports
- Professional Support: Contact system administrator

---

**PORTAL VII DIBLE Vault** - Professional cryptographic system.
"""
        
        with open(self.build_dir / "INSTALLATION_GUIDE.md", 'w') as f:
            f.write(guide_content)
        
        print("✅ Installation guide created")
    
    def build_all_packages(self):
        """Build all packages and integrations"""
        print("🏗️  PORTAL VII DIBLE - Professional Package Builder")
        print("=" * 60)
        
        try:
            self.create_python_package()
            self.create_vscode_extension_package()
            self.create_docker_container()
            self.create_library_examples()
            self.create_installation_guide()
            
            # Try to create standalone (optional)
            try:
                self.create_standalone_executable()
            except Exception as e:
                print(f"⚠️  Standalone executable creation failed: {e}")
            
            print("\n🎉 Package building complete!")
            print(f"📦 All packages available in: {self.build_dir}")
            print("\n📋 Created packages:")
            
            for item in self.build_dir.iterdir():
                if item.is_file():
                    print(f"   📄 {item.name}")
                elif item.is_dir():
                    print(f"   📁 {item.name}/")
            
            print("\n🚀 Ready for distribution and integration!")
            
        except Exception as e:
            print(f"❌ Package building failed: {e}")
            raise

if __name__ == "__main__":
    builder = DIBLEPackageBuilder()
    builder.build_all_packages()
