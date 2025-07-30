"""
PORTAL VII - DIBLE Vault Professional System
Advanced Device Identity-Based Lattice Encryption

Professional cryptographic vault for enterprise applications.
No simplified versions - fully advanced implementation.
"""

__version__ = "2.0.0"
__author__ = "PORTAL VII Systems"
__description__ = "Professional DIBLE Vault - Advanced Cryptographic Architecture"
__algorithm__ = "HC-DIBLE-VAULT"
__security_level__ = 256

# Professional system imports
from .dible_vault_cli import DIBLEVaultManager, DIBLECryptographicCore, PortalVIITheme

__all__ = [
    'DIBLEVaultManager',
    'DIBLECryptographicCore', 
    'PortalVIITheme'
]
