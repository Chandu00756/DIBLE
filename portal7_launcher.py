#!/usr/bin/env python3
"""
PORTAL VII - Professional CLI Launcher
Launch custom terminal application

Independent terminal launcher - No system dependencies
"""

import sys
import subprocess
import platform
from pathlib import Path


def launch_custom_terminal():
    """Launch PORTAL VII custom terminal"""
    print("🚀 PORTAL VII - Launching Custom Professional Terminal")
    print("=" * 60)
    
    # Get current directory
    current_dir = Path(__file__).parent
    terminal_script = current_dir / "portal7_custom_terminal.py"
    
    if not terminal_script.exists():
        print(f"❌ Custom terminal not found: {terminal_script}")
        return False
    
    try:
        # Check if GUI is available
        try:
            import tkinter
            gui_available = True
        except ImportError:
            gui_available = False
            
        if not gui_available:
            print("❌ GUI libraries not available")
            print("   Install tkinter: python -m pip install tk")
            return False
        
        print("✅ GUI libraries available")
        print("✅ Custom terminal found")
        print("🚀 Launching PORTAL VII Custom Terminal...")
        print()
        
        # Launch custom terminal
        result = subprocess.run([
            sys.executable, 
            str(terminal_script)
        ], cwd=str(current_dir))
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        return False


def show_banner():
    """Show PORTAL VII banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║  ██████╗  ██████╗ ██████╗ ████████╗ █████╗ ██╗    ██╗   ██╗██╗██╗   ║
║  ██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔══██╗██║    ██║   ██║██║██║   ║
║  ██████╔╝██║   ██║██████╔╝   ██║   ███████║██║    ██║   ██║██║██║   ║
║  ██╔═══╝ ██║   ██║██╔══██╗   ██║   ██╔══██║██║    ╚██╗ ██╔╝██║██║   ║
║  ██║     ╚██████╔╝██║  ██║   ██║   ██║  ██║███████╗╚████╔╝ ██║██║   ║
║  ╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝ ╚═══╝  ╚═╝╚═╝   ║
║                                                                      ║
║                    DIBLE VAULT PROFESSIONAL SYSTEM                   ║
║                     Custom Terminal Application                      ║
║                                                                      ║
║  🚀 Independent custom-built terminal                                ║
║  🔧 No system terminal dependencies                                   ║
║  🏢 Professional enterprise grade                                     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def main():
    """Main launcher function"""
    show_banner()
    
    # System information
    print(f"🖥️  System: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {platform.python_version()}")
    print(f"📁 Location: {Path(__file__).parent}")
    print()
    
    # Launch custom terminal
    success = launch_custom_terminal()
    
    if success:
        print("✅ Custom terminal launched successfully!")
    else:
        print("❌ Failed to launch custom terminal")
        
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
