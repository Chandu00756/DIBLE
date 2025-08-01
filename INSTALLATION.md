# PORTAL VII DIBLE Vault - Installation Guide

## Overview

PORTAL VII DIBLE Vault is a professional-grade cryptographic system that implements advanced lattice-based cryptography with quantum resistance, chaos theory integration, and device binding capabilities.

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 1GB free space
- **Dependencies**: See requirements.txt for full list

## Quick Installation

### Option 1: Automated Setup (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd portal7-dible-vault
   ```

2. **Run the automated setup**:
   ```bash
   python3 setup_environment.py
   ```

3. **Activate the virtual environment** (if created):
   ```bash
   source portal7_env/bin/activate  # Linux/macOS
   # or
   portal7_env\Scripts\activate     # Windows
   ```

4. **Verify installation**:
   ```bash
   python3 check_dependencies.py
   ```

### Option 2: Manual Installation

1. **Create a virtual environment**:
   ```bash
   python3 -m venv portal7_env
   source portal7_env/bin/activate  # Linux/macOS
   # or
   portal7_env\Scripts\activate     # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python3 check_dependencies.py
   ```

## Running PORTAL VII

### Main Launcher
```bash
python3 launch_portal7.py
```

### Custom Terminal (GUI)
```bash
python3 portal7_launcher.py
```

### CLI Interface
```bash
python3 dible_vault_cli.py
```

### Direct Terminal
```bash
python3 portal7_custom_terminal.py
```

## Troubleshooting

### Common Issues

#### 1. Missing Dependencies
**Error**: `ModuleNotFoundError: No module named 'psutil'`

**Solution**:
```bash
# Check missing dependencies
python3 check_dependencies.py

# Install missing packages
pip install psutil numpy scipy cryptography pycryptodome
```

#### 2. Permission Errors
**Error**: `PermissionError: [Errno 13] Permission denied`

**Solution**:
- Use a virtual environment
- Check file permissions
- Run with appropriate user privileges

#### 3. GUI Not Available
**Error**: `ImportError: No module named 'tkinter'`

**Solution**:
```bash
# Ubuntu/Debian
sudo apt install python3-tk

# CentOS/RHEL
sudo yum install python3-tkinter

# macOS
brew install python-tk

# Windows
# Usually included with Python installation
```

#### 4. Externally Managed Environment
**Error**: `error: externally-managed-environment`

**Solution**:
- Use a virtual environment
- Or use `--break-system-packages` (not recommended)

#### 5. Import Errors in Core Modules
**Error**: `ImportError` in src/core modules

**Solution**:
```bash
# Ensure src directory is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or run from project root
python3 -c "import sys; sys.path.insert(0, 'src'); from core.device_id import generate_device_identity"
```

### Advanced Troubleshooting

#### Debug Mode
Run with verbose output:
```bash
python3 -v dible_vault_cli.py
```

#### Check System Information
```bash
python3 -c "
import platform
import sys
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Architecture: {platform.machine()}')
"
```

#### Test Individual Components
```bash
# Test core modules
python3 final_system_test.py

# Test specific functionality
python3 -c "
import sys
sys.path.insert(0, 'src')
from core.device_id import DeviceIDGenerator
gen = DeviceIDGenerator()
print('Device ID generation works')
"
```

## Configuration

### Vault Configuration
The system creates a vault configuration at `~/.portal7_dible/vault.config`:

```json
{
  "vault_version": "2.0.0",
  "algorithm": "HC-DIBLE-VAULT",
  "security_level": 256,
  "quantum_resistant": true,
  "homomorphic_enabled": true,
  "chaos_enhanced": true,
  "device_binding": true,
  "professional_mode": true
}
```

### Environment Variables
- `PORTAL7_DEBUG`: Enable debug mode
- `PORTAL7_VAULT_PATH`: Custom vault path
- `PORTAL7_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Security Considerations

1. **Virtual Environment**: Always use a virtual environment to isolate dependencies
2. **Permissions**: Ensure proper file permissions for vault directories
3. **Updates**: Keep dependencies updated for security patches
4. **Backup**: Regularly backup vault configuration and keys
5. **Network**: Be cautious when using over network connections

## Support

### Getting Help
1. Check this installation guide
2. Run `python3 check_dependencies.py` for dependency issues
3. Check the main README.md for usage instructions
4. Review error messages and logs

### Reporting Issues
When reporting issues, please include:
- Operating system and version
- Python version
- Error messages
- Steps to reproduce
- Output from `python3 check_dependencies.py`

## Uninstallation

To remove PORTAL VII:

1. **Remove virtual environment**:
   ```bash
   rm -rf portal7_env
   ```

2. **Remove vault data** (optional):
   ```bash
   rm -rf ~/.portal7_dible
   ```

3. **Remove source code**:
   ```bash
   cd ..
   rm -rf portal7-dible-vault
   ```

## License

See LICENSE file for details.

---

**Note**: PORTAL VII is a professional cryptographic system. Ensure you understand the security implications before use in production environments.