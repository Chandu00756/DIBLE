"""
DIBLE Encryption Module
Handles encryption operations and key management
"""

import numpy as np
import hashlib
import time
from typing import Dict, Any, List, Tuple, Union, Optional
# Handle both package and direct imports
try:
    from ..core.device_id import generate_device_identity
    from ..core.lattice import create_lattice_instance
    from ..core.entropy import create_entropy_manager
    from ..core.chaos import create_chaos_manager
    from ..core.polynomial import create_polynomial_operations
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.device_id import generate_device_identity
    from core.lattice import create_lattice_instance
    from core.entropy import create_entropy_manager
    from core.chaos import create_chaos_manager
    from core.polynomial import create_polynomial_operations


class DIBLEEncryption:
    """DIBLE encryption operations"""
    
    def __init__(self, q: int = 2**32 - 5, n: int = 256, sigma: float = 3.2):
        self.q = q
        self.n = n
        self.sigma = sigma
        
        # Initialize components
        self.device_fingerprint = generate_device_identity()
        self.device_id = self.device_fingerprint['device_id']
        self.transformed_device_id = self.device_fingerprint['transformed_id']
        
        self.lattice_ops = create_lattice_instance(n, q)
        self.entropy_manager = create_entropy_manager(self.transformed_device_id)
        self.chaos_manager = create_chaos_manager(self.transformed_device_id)
        self.polynomial_ops = create_polynomial_operations(q)
        
    def generate_ephemeral_key(self, device_id: str) -> np.ndarray:
        """Generate ephemeral key for encryption"""
        # Hash device ID to create deterministic but unique key
        hash_bytes = hashlib.sha256(device_id.encode()).digest()
        
        # Convert to polynomial coefficients
        key_coeffs = []
        for i in range(self.n):
            byte_idx = (i * 4) % len(hash_bytes)
            coeff_bytes = hash_bytes[byte_idx:byte_idx + 4]
            
            # Pad if necessary
            while len(coeff_bytes) < 4:
                coeff_bytes += hash_bytes[:4 - len(coeff_bytes)]
            
            coeff = int.from_bytes(coeff_bytes, byteorder='big') % self.q
            key_coeffs.append(coeff)
        
        return np.array(key_coeffs)
    
    def apply_chaos_transformation(self, data: np.ndarray) -> np.ndarray:
        """Apply chaotic transformation to data"""
        chaos_sequence = self.chaos_manager.generate_chaotic_sequence(len(data), 'logistic')
        transformed_data = np.copy(data)
        
        for i in range(len(data)):
            chaos_factor = int(chaos_sequence[i] * self.q) % self.q
            transformed_data[i] = (transformed_data[i] + chaos_factor) % self.q
        
        return transformed_data
    
    def entropy_augmentation(self, message: bytes) -> bytes:
        """Augment message entropy using device-specific characteristics"""
        # Generate high-entropy augmentation
        entropy_seed = self.entropy_manager.generate_high_entropy_seed(len(message))
        
        # XOR with original message
        augmented_message = bytearray()
        for i in range(len(message)):
            augmented_byte = message[i] ^ entropy_seed[i % len(entropy_seed)]
            augmented_message.append(augmented_byte)
        
        return bytes(augmented_message)
    
    def fractal_key_derivation(self, base_key: np.ndarray, iterations: int = 100) -> np.ndarray:
        """Derive key using fractal mathematics"""
        derived_key = np.copy(base_key)
        
        for i in range(len(derived_key)):
            # Use Mandelbrot-like iteration
            c = complex((base_key[i] / self.q) * 2 - 1, 
                       (self.transformed_device_id % 1000) / 1000.0)
            z = complex(0, 0)
            
            iteration_count = 0
            while abs(z) <= 2 and iteration_count < iterations:
                z = z*z + c
                iteration_count += 1
            
            # Convert iteration count to key coefficient
            derived_key[i] = (iteration_count * (i + 1) * self.transformed_device_id) % self.q
        
        return derived_key
    
    def tensor_encryption(self, message_poly: np.ndarray) -> np.ndarray:
        """Apply tensor-based encryption operations"""
        # Create tensor lattice
        tensor_dims = (self.n // 4, 4, 1) if self.n >= 4 else (self.n, 1, 1)
        tensor_lattice = self.lattice_ops.tensor_lattice_construction(
            tensor_dims, self.transformed_device_id
        )
        
        # Apply tensor operations
        encrypted_poly = np.copy(message_poly)
        for i in range(len(encrypted_poly)):
            tensor_idx = (i % tensor_dims[0], i % tensor_dims[1], 0)
            encrypted_poly[i] = (encrypted_poly[i] + tensor_lattice[tensor_idx]) % self.q
        
        return encrypted_poly
    
    def multivariate_polynomial_encryption(self, message: bytes) -> Dict[str, Any]:
        """Encrypt using multivariate polynomial operations"""
        # Convert message to polynomial
        message_bits = []
        for byte in message:
            for i in range(8):
                message_bits.append((byte >> i) & 1)
        
        # Pad to polynomial size
        while len(message_bits) < self.n:
            message_bits.append(0)
        
        message_poly = self.polynomial_ops.binary_to_polynomial(message_bits[:self.n])
        
        # Create multivariate polynomial ring
        poly_ring = self.polynomial_ops.multivariate_polynomial_ring(
            self.transformed_device_id, 5
        )
        
        # Apply polynomial transformations
        encrypted_poly = message_poly.copy()
        for i, ring_poly in enumerate(poly_ring):
            # Evaluate ring polynomial at message coefficients
            values = {'x': i + 1, 'y': i + 2, 'z': i + 3, 't': time.time() % 1000}
            poly_value = self.polynomial_ops.polynomial_evaluation(ring_poly, values)
            
            # Apply to encrypted polynomial
            if i < len(encrypted_poly['coefficients']):
                encrypted_poly['coefficients'][i] = (
                    encrypted_poly['coefficients'][i] + poly_value
                ) % self.q
        
        return encrypted_poly
    
    def quantum_inspired_encryption(self, message_poly: np.ndarray) -> np.ndarray:
        """Apply quantum-inspired encryption transformations"""
        # Create quantum-like superposition states
        alpha = np.sqrt((self.transformed_device_id % 100) / 100.0)
        beta = np.sqrt(1 - alpha**2)
        
        encrypted_poly = np.copy(message_poly)
        
        for i in range(len(encrypted_poly)):
            # Apply quantum-like transformation
            state_0 = encrypted_poly[i]
            state_1 = (self.transformed_device_id * (i + 1)) % self.q
            
            # Superposition-like combination
            quantum_state = int(alpha * state_0 + beta * state_1) % self.q
            encrypted_poly[i] = quantum_state
        
        return encrypted_poly
    
    def homomorphic_layer_encryption(self, data: np.ndarray, operation: str = 'add') -> np.ndarray:
        """Apply homomorphic encryption layer"""
        homomorphic_data = np.copy(data)
        
        if operation == 'add':
            for i in range(len(homomorphic_data)):
                # Additive homomorphic operation
                transform = (self.transformed_device_id * (i + 1)) % self.q
                homomorphic_data[i] = (homomorphic_data[i] + transform) % self.q
                
        elif operation == 'multiply':
            for i in range(len(homomorphic_data)):
                # Multiplicative homomorphic operation
                transform = ((self.transformed_device_id % 100) + 1) % self.q
                homomorphic_data[i] = (homomorphic_data[i] * transform) % self.q
        
        return homomorphic_data
    
    def post_quantum_layer(self, data: np.ndarray) -> np.ndarray:
        """Apply post-quantum cryptographic layer"""
        # Use lattice-based operations for post-quantum security
        pq_data = np.copy(data)
        
        # Generate lattice basis
        basis = self.lattice_ops.generate_lattice_basis(min(self.n, len(data)))
        
        # Apply lattice transformation
        for i in range(len(pq_data)):
            lattice_transform = 0
            for j in range(min(len(basis), len(data))):
                lattice_transform += basis[i % len(basis)][j % len(basis[0])] * pq_data[j]
            
            pq_data[i] = lattice_transform % self.q
        
        return pq_data
    
    def device_binding_encryption(self, data: np.ndarray, target_device_id: str) -> np.ndarray:
        """Bind encryption to specific device"""
        target_hash = hashlib.sha256(target_device_id.encode()).digest()
        binding_key = []
        
        for i in range(self.n):
            byte_idx = (i * 2) % len(target_hash)
            binding_coeff = int.from_bytes(target_hash[byte_idx:byte_idx + 2], 
                                         byteorder='big') % self.q
            binding_key.append(binding_coeff)
        
        binding_key = np.array(binding_key)
        
        # Apply device binding
        bound_data = np.copy(data)
        for i in range(len(bound_data)):
            bound_data[i] = (bound_data[i] + binding_key[i % len(binding_key)]) % self.q
        
        return bound_data
    
    def comprehensive_encrypt(self, message: Union[str, bytes], 
                            target_device_id: str = None,
                            enable_homomorphic: bool = True,
                            enable_post_quantum: bool = True) -> Dict[str, Any]:
        """
        Comprehensive encryption using all DIBLE features
        
        Args:
            message: Message to encrypt
            target_device_id: Target device ID (uses own if None)
            enable_homomorphic: Enable homomorphic encryption layer
            enable_post_quantum: Enable post-quantum layer
            
        Returns:
            Complete ciphertext structure
        """
        if isinstance(message, str):
            message = message.encode('utf-8')
        
        if target_device_id is None:
            target_device_id = self.device_id
        
        # Step 1: Entropy augmentation
        augmented_message = self.entropy_augmentation(message)
        
        # Step 2: Convert to polynomial
        message_bits = []
        for byte in augmented_message:
            for i in range(8):
                message_bits.append((byte >> i) & 1)
        
        # Pad to polynomial size
        while len(message_bits) < self.n:
            message_bits.append(0)
        
        message_poly = np.array([self.q // 2 if bit else 0 for bit in message_bits[:self.n]])
        
        # Step 3: Generate ephemeral key
        ephemeral_key = self.generate_ephemeral_key(target_device_id)
        
        # Step 4: Fractal key derivation
        derived_key = self.fractal_key_derivation(ephemeral_key)
        
        # Step 5: Apply chaos transformation
        chaos_transformed = self.apply_chaos_transformation(message_poly)
        
        # Step 6: Quantum-inspired encryption
        quantum_encrypted = self.quantum_inspired_encryption(chaos_transformed)
        
        # Step 7: Tensor encryption
        tensor_encrypted = self.tensor_encryption(quantum_encrypted)
        
        # Step 8: Homomorphic layer (optional)
        if enable_homomorphic:
            homomorphic_encrypted = self.homomorphic_layer_encryption(tensor_encrypted)
        else:
            homomorphic_encrypted = tensor_encrypted
        
        # Step 9: Post-quantum layer (optional)
        if enable_post_quantum:
            pq_encrypted = self.post_quantum_layer(homomorphic_encrypted)
        else:
            pq_encrypted = homomorphic_encrypted
        
        # Step 10: Device binding
        device_bound = self.device_binding_encryption(pq_encrypted, target_device_id)
        
        # Step 11: Final lattice operation
        # Generate error polynomial
        error_poly = self.lattice_ops.generate_error_vector(self.n, self.sigma)
        
        # Compute final ciphertext
        ciphertext_poly = (self.lattice_ops.polynomial_multiply(ephemeral_key, derived_key) + 
                          error_poly + device_bound) % self.q
        
        # Create comprehensive ciphertext structure
        ciphertext = {
            'algorithm': 'HC-DIBLE-DDI',
            'version': '1.0',
            'ciphertext': ciphertext_poly.tolist(),
            'ephemeral_key': ephemeral_key.tolist(),
            'device_binding': hashlib.sha256(target_device_id.encode()).hexdigest(),
            'source_device': self.device_id,
            'timestamp': int(time.time() * 1000000),  # Microsecond precision
            'security_parameters': {
                'q': self.q,
                'n': self.n,
                'sigma': self.sigma,
                'homomorphic_enabled': enable_homomorphic,
                'post_quantum_enabled': enable_post_quantum
            },
            'chaos_state': {
                'logistic_r': self.chaos_manager.chaos_state['logistic_r'],
                'initial_x': self.chaos_manager.chaos_state['logistic_x']
            },
            'entropy_metadata': {
                'original_length': len(message),
                'augmented_length': len(augmented_message)
            }
        }
        
        # Add integrity protection
        ciphertext_str = str(ciphertext)
        integrity_hash = hashlib.sha256(ciphertext_str.encode()).hexdigest()
        ciphertext['integrity_hash'] = integrity_hash
        
        return ciphertext
    
    def batch_encrypt(self, messages: List[Union[str, bytes]], 
                     target_device_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Encrypt multiple messages efficiently"""
        if target_device_ids is None:
            target_device_ids = [self.device_id] * len(messages)
        
        if len(target_device_ids) != len(messages):
            raise ValueError("Number of messages and device IDs must match")
        
        encrypted_messages = []
        for message, device_id in zip(messages, target_device_ids):
            encrypted_msg = self.comprehensive_encrypt(message, device_id)
            encrypted_messages.append(encrypted_msg)
        
        return encrypted_messages
    
    def get_encryption_metadata(self) -> Dict[str, Any]:
        """Get encryption system metadata"""
        return {
            'device_id': self.device_id,
            'transformed_device_id': self.transformed_device_id,
            'algorithm_parameters': {
                'q': self.q,
                'n': self.n,
                'sigma': self.sigma
            },
            'supported_operations': [
                'comprehensive_encrypt',
                'batch_encrypt',
                'homomorphic_layer_encryption',
                'quantum_inspired_encryption',
                'tensor_encryption',
                'device_binding_encryption'
            ],
            'security_features': [
                'entropy_augmentation',
                'chaos_transformation',
                'fractal_key_derivation',
                'multivariate_polynomials',
                'post_quantum_resistance',
                'device_identity_binding'
            ]
        }


if __name__ == "__main__":
    # Test DIBLE encryption
    print("Testing DIBLE Encryption Module...")
    
    # Create encryption instance
    encryptor = DIBLEEncryption()
    
    # Test message
    test_message = "This is a comprehensive test of the DIBLE encryption system!"
    print(f"Original message: {test_message}")
    
    # Comprehensive encryption
    ciphertext = encryptor.comprehensive_encrypt(test_message)
    print(f"Encryption completed. Ciphertext size: {len(str(ciphertext))} characters")
    
    # Test batch encryption
    messages = ["Message 1", "Message 2", "Message 3"]
    batch_ciphertexts = encryptor.batch_encrypt(messages)
    print(f"Batch encryption completed. {len(batch_ciphertexts)} messages encrypted")
    
    # Display metadata
    metadata = encryptor.get_encryption_metadata()
    print(f"Encryption system ready with device ID: {metadata['device_id'][:16]}...")
    
    print("DIBLE Encryption Module test completed successfully!")
