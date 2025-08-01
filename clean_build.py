#!/usr/bin/env python3
"""
PORTAL VII - Build Artifact Cleaner
Clean Python build artifacts and temporary files
"""

import os
import shutil
import glob
from pathlib import Path

def clean_python_artifacts():
    """Clean Python build artifacts"""
    print("🧹 Cleaning Python build artifacts...")
    
    # Patterns to clean
    patterns = [
        "__pycache__",
        "*.pyc",
        "*.pyo", 
        "*.pyd",
        "*.so",
        "build",
        "dist",
        "*.egg-info",
        "*.egg",
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        ".mypy_cache",
        ".tox",
        "*.tmp",
        "*.temp"
    ]
    
    cleaned_count = 0
    
    for pattern in patterns:
        if pattern == "__pycache__":
            # Handle directories
            for root, dirs, files in os.walk("."):
                if "__pycache__" in dirs:
                    cache_dir = os.path.join(root, "__pycache__")
                    try:
                        shutil.rmtree(cache_dir)
                        print(f"✅ Removed: {cache_dir}")
                        cleaned_count += 1
                    except Exception as e:
                        print(f"⚠️  Could not remove {cache_dir}: {e}")
        else:
            # Handle files
            for file_path in glob.glob(pattern, recursive=True):
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"✅ Removed: {file_path}")
                        cleaned_count += 1
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        print(f"✅ Removed: {file_path}")
                        cleaned_count += 1
                except Exception as e:
                    print(f"⚠️  Could not remove {file_path}: {e}")
    
    return cleaned_count

def clean_virtual_environments():
    """Clean virtual environment directories"""
    print("\n🧹 Cleaning virtual environments...")
    
    venv_patterns = [
        "portal7_env",
        "venv",
        "env",
        "ENV",
        "env.bak",
        "venv.bak",
        ".venv"
    ]
    
    cleaned_count = 0
    
    for pattern in venv_patterns:
        if os.path.exists(pattern):
            try:
                shutil.rmtree(pattern)
                print(f"✅ Removed: {pattern}")
                cleaned_count += 1
            except Exception as e:
                print(f"⚠️  Could not remove {pattern}: {e}")
    
    return cleaned_count

def clean_ide_files():
    """Clean IDE and editor files"""
    print("\n🧹 Cleaning IDE files...")
    
    ide_patterns = [
        ".vscode",
        ".idea",
        "*.swp",
        "*.swo",
        "*~",
        ".DS_Store",
        "Thumbs.db"
    ]
    
    cleaned_count = 0
    
    for pattern in ide_patterns:
        if pattern.startswith(".") and os.path.exists(pattern):
            try:
                if os.path.isfile(pattern):
                    os.remove(pattern)
                else:
                    shutil.rmtree(pattern)
                print(f"✅ Removed: {pattern}")
                cleaned_count += 1
            except Exception as e:
                print(f"⚠️  Could not remove {pattern}: {e}")
        else:
            for file_path in glob.glob(pattern, recursive=True):
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"✅ Removed: {file_path}")
                        cleaned_count += 1
                except Exception as e:
                    print(f"⚠️  Could not remove {file_path}: {e}")
    
    return cleaned_count

def main():
    """Main cleaning function"""
    print("🚀 PORTAL VII - Build Artifact Cleaner")
    print("=" * 50)
    
    # Clean Python artifacts
    python_cleaned = clean_python_artifacts()
    
    # Clean virtual environments
    venv_cleaned = clean_virtual_environments()
    
    # Clean IDE files
    ide_cleaned = clean_ide_files()
    
    # Summary
    total_cleaned = python_cleaned + venv_cleaned + ide_cleaned
    
    print("\n" + "=" * 50)
    print(f"🎉 Cleaning completed!")
    print(f"📊 Files/directories cleaned: {total_cleaned}")
    print(f"   - Python artifacts: {python_cleaned}")
    print(f"   - Virtual environments: {venv_cleaned}")
    print(f"   - IDE files: {ide_cleaned}")
    
    if total_cleaned == 0:
        print("✨ Repository is already clean!")
    
    print("\n💡 Tip: Run this script regularly to keep your repository clean.")
    print("   You can also add it to your pre-commit hooks.")

if __name__ == "__main__":
    main()