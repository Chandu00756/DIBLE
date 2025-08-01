#!/usr/bin/env python3
"""
PORTAL VII - Dependency Checker
Check and report missing dependencies for PORTAL VII DIBLE Vault
"""

import sys
import subprocess
from pathlib import Path

# Required dependencies
REQUIRED_PACKAGES = {
    'numpy': 'numpy>=1.21.0',
    'scipy': 'scipy>=1.7.0',
    'cryptography': 'cryptography>=3.4.8',
    'pycryptodome': 'pycryptodome>=3.15.0',
    'sympy': 'sympy>=1.9',
    'networkx': 'networkx>=2.6',
    'matplotlib': 'matplotlib>=3.4.3',
    'seaborn': 'seaborn>=0.11.2',
    'psutil': 'psutil>=5.8.0',
    'click': 'click>=8.0.0',
    'rich': 'rich>=13.0.0'
}

# Optional dependencies
OPTIONAL_PACKAGES = {
    'cpuinfo': 'py-cpuinfo>=8.0.0',
    'GPUtil': 'GPUtil>=1.4.0',
    'platform-info': 'platform-info>=1.0.0',
    'colorama': 'colorama>=0.4.4',
    'pyfiglet': 'pyfiglet>=0.8.0'
}

def check_package(package_name):
    """Check if a package is available"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_package(package_spec):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_spec])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Main dependency checker"""
    print("🔍 PORTAL VII - Dependency Checker")
    print("=" * 50)
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    print("\n📋 Checking Required Dependencies:")
    for package, spec in REQUIRED_PACKAGES.items():
        if check_package(package):
            print(f"✅ {package}")
        else:
            print(f"❌ {package} - MISSING")
            missing_required.append(spec)
    
    # Check optional packages
    print("\n📋 Checking Optional Dependencies:")
    for package, spec in OPTIONAL_PACKAGES.items():
        if check_package(package):
            print(f"✅ {package}")
        else:
            print(f"⚠️  {package} - MISSING (optional)")
            missing_optional.append(spec)
    
    # Report results
    print("\n" + "=" * 50)
    
    if not missing_required and not missing_optional:
        print("🎉 All dependencies are available!")
        print("🚀 PORTAL VII is ready to run!")
        return 0
    
    if missing_required:
        print(f"❌ Missing {len(missing_required)} required dependencies:")
        for spec in missing_required:
            print(f"   {spec}")
        
        print("\n🔧 Installing missing required dependencies...")
        failed_installations = []
        
        for spec in missing_required:
            print(f"Installing {spec}...")
            if install_package(spec):
                print(f"✅ {spec} installed successfully")
            else:
                print(f"❌ Failed to install {spec}")
                failed_installations.append(spec)
        
        if failed_installations:
            print(f"\n❌ Failed to install {len(failed_installations)} packages:")
            for spec in failed_installations:
                print(f"   {spec}")
            print("\n💡 Try installing manually:")
            print(f"   pip install {' '.join(failed_installations)}")
            return 1
    
    if missing_optional:
        print(f"\n⚠️  Missing {len(missing_optional)} optional dependencies:")
        for spec in missing_optional:
            print(f"   {spec}")
        
        print("\n💡 Optional dependencies enhance functionality but are not required.")
        print("   To install optional dependencies:")
        print(f"   pip install {' '.join(missing_optional)}")
    
    print("\n✅ Dependency check complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())