#!/usr/bin/env python3
"""
PORTAL VII - Main Professional Launcher
Launch PORTAL VII DIBLE Vault system

Choose between CLI and Custom Terminal
"""

import sys
import subprocess


def main():
    """Main launcher menu"""
    print("🚀 PORTAL VII - Professional DIBLE Vault System")
    print("=" * 50)
    print()
    print("Available Options:")
    print("1. Launch Custom Terminal (GUI)")
    print("2. Launch CLI Interface")
    print("3. Test Integration Library")
    print("4. Exit")
    print()
    
    while True:
        try:
            choice = input("Select option (1-4): ").strip()
            
            if choice == "1":
                print("🚀 Launching Custom Terminal...")
                subprocess.run([sys.executable, "portal7_launcher.py"])
                break
            elif choice == "2":
                print("🚀 Launching CLI Interface...")
                subprocess.run([sys.executable, "dible_vault_cli.py"])
                break
            elif choice == "3":
                print("🧪 Testing Integration Library...")
                subprocess.run([sys.executable, "portal7_integration_advanced.py"])
                break
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break


if __name__ == "__main__":
    main()
