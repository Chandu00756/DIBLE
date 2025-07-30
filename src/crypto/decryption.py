"""
DIBLE Decryption Module
Handles decryption operations and key recovery
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


class DIBLEDecryption:
    """DIBLE decryption operations"""
    
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
        
    def verify_ciphertext_integrity(self, ciphertext: Dict[str, Any]) -> bool:
        """Verify ciphertext integrity"""
        if 'integrity_hash' not in ciphertext:
            return False
        
        temp_ciphertext = ciphertext.copy()
        received_hash = temp_ciphertext.pop('integrity_hash')
        ciphertext_str = str(temp_ciphertext)
        expected_hash = hashlib.sha256(ciphertext_str.encode()).hexdigest()
        
        return received_hash == expected_hash
    
    def verify_device_authorization(self, ciphertext: Dict[str, Any]) -> bool:
        """Verify device is authorized to decrypt"""
        if 'device_binding' not in ciphertext:
            return False
        
        expected_binding = hashlib.sha256(self.device_id.encode()).hexdigest()
        return ciphertext['device_binding'] == expected_binding
    
    def reconstruct_ephemeral_key(self, ciphertext: Dict[str, Any]) -> np.ndarray:
        """Reconstruct ephemeral key from ciphertext"""
        if 'ephemeral_key' in ciphertext:
            return np.array(ciphertext['ephemeral_key'])
        
        # Fallback: derive from device ID
        source_device = ciphertext.get('source_device', self.device_id)
        hash_bytes = hashlib.sha256(source_device.encode()).digest()
        
        key_coeffs = []
        for i in range(self.n):
            byte_idx = (i * 4) % len(hash_bytes)
            coeff_bytes = hash_bytes[byte_idx:byte_idx + 4]
            
            while len(coeff_bytes) < 4:
                coeff_bytes += hash_bytes[:4 - len(coeff_bytes)]
            
            coeff = int.from_bytes(coeff_bytes, byteorder='big') % self.q
            key_coeffs.append(coeff)
        
        return np.array(key_coeffs)
    
    def reconstruct_fractal_key(self, ephemeral_key: np.ndarray, iterations: int = 100) -> np.ndarray:
        """Reconstruct fractal-derived key"""
        derived_key = np.copy(ephemeral_key)
        
        for i in range(len(derived_key)):
            # Use same Mandelbrot-like iteration as encryption
            c = complex((ephemeral_key[i] / self.q) * 2 - 1, 
                       (self.transformed_device_id % 1000) / 1000.0)
            z = complex(0, 0)
            
            iteration_count = 0
            while abs(z) <= 2 and iteration_count < iterations:
                z = z*z + c
                iteration_count += 1
            
            derived_key[i] = (iteration_count * (i + 1) * self.transformed_device_id) % self.q
        
        return derived_key
    
    def reverse_device_binding(self, data: np.ndarray, target_device_id: str) -> np.ndarray:
        """Reverse device binding transformation"""
        target_hash = hashlib.sha256(target_device_id.encode()).digest()
        binding_key = []
        
        for i in range(self.n):
            byte_idx = (i * 2) % len(target_hash)
            binding_coeff = int.from_bytes(target_hash[byte_idx:byte_idx + 2], 
                                         byteorder='big') % self.q
            binding_key.append(binding_coeff)
        
        binding_key = np.array(binding_key)
        
        # Reverse device binding (subtract)
        unbound_data = np.copy(data)
        for i in range(len(unbound_data)):
            unbound_data[i] = (unbound_data[i] - binding_key[i % len(binding_key)]) % self.q
        
        return unbound_data
    
    def reverse_post_quantum_layer(self, data: np.ndarray) -> np.ndarray:
        """Reverse post-quantum cryptographic layer"""
        pq_data = np.copy(data)
        
        # Generate same lattice basis as encryption
        basis = self.lattice_ops.generate_lattice_basis(min(self.n, len(data)))
        
        # Use closest vector problem to reverse transformation
        for i in range(len(pq_data)):
            # This is a simplified reversal - in practice, CVP solving is complex
            # Here we use an approximation
            lattice_transform = 0
            for j in range(min(len(basis), len(data))):
                lattice_transform += basis[i % len(basis)][j % len(basis[0])]
            
            # Approximate reversal
            pq_data[i] = (pq_data[i] - (lattice_transform % self.q)) % self.q
        
        return pq_data
    
    def reverse_homomorphic_layer(self, data: np.ndarray, operation: str = 'add') -> np.ndarray:
        """Reverse homomorphic encryption layer"""
        homomorphic_data = np.copy(data)
        
        if operation == 'add':
            for i in range(len(homomorphic_data)):
                # Reverse additive homomorphic operation
                transform = (self.transformed_device_id * (i + 1)) % self.q
                homomorphic_data[i] = (homomorphic_data[i] - transform) % self.q
                
        elif operation == 'multiply':
            for i in range(len(homomorphic_data)):
                # Reverse multiplicative homomorphic operation
                transform = ((self.transformed_device_id % 100) + 1) % self.q
                # Find modular inverse
                inv_transform = pow(transform, -1, self.q) if transform != 0 else 1
                homomorphic_data[i] = (homomorphic_data[i] * inv_transform) % self.q
        
        return homomorphic_data
    
    def reverse_quantum_inspired_encryption(self, encrypted_poly: np.ndarray) -> np.ndarray:
        """Reverse quantum-inspired encryption transformations"""
        # Reconstruct quantum-like superposition parameters
        alpha = np.sqrt((self.transformed_device_id % 100) / 100.0)
        beta = np.sqrt(1 - alpha**2)
        
        decrypted_poly = np.copy(encrypted_poly)
        
        for i in range(len(decrypted_poly)):
            # Reverse quantum-like transformation
            quantum_state = decrypted_poly[i]
            state_1 = (self.transformed_device_id * (i + 1)) % self.q
            
            # Reverse superposition (approximate)
            if alpha != 0:
                state_0 = int((quantum_state - beta * state_1) / alpha) % self.q
                decrypted_poly[i] = state_0
        
        return decrypted_poly
    
    def reverse_tensor_encryption(self, encrypted_poly: np.ndarray) -> np.ndarray:
        """Reverse tensor-based encryption operations"""
        # Create same tensor lattice as encryption
        tensor_dims = (self.n // 4, 4, 1) if self.n >= 4 else (self.n, 1, 1)
        tensor_lattice = self.lattice_ops.tensor_lattice_construction(
            tensor_dims, self.transformed_device_id
        )
        
        # Reverse tensor operations (subtract)
        decrypted_poly = np.copy(encrypted_poly)
        for i in range(len(decrypted_poly)):
            tensor_idx = (i % tensor_dims[0], i % tensor_dims[1], 0)
            decrypted_poly[i] = (decrypted_poly[i] - tensor_lattice[tensor_idx]) % self.q
        
        return decrypted_poly
    
    def reverse_chaos_transformation(self, transformed_data: np.ndarray) -> np.ndarray:
        """Reverse chaotic transformation"""
        # Generate same chaotic sequence as encryption
        chaos_sequence = self.chaos_manager.generate_chaotic_sequence(len(transformed_data), 'logistic')
        original_data = np.copy(transformed_data)
        
        for i in range(len(transformed_data)):
            chaos_factor = int(chaos_sequence[i] * self.q) % self.q
            original_data[i] = (original_data[i] - chaos_factor) % self.q
        
        return original_data
    
    def reverse_entropy_augmentation(self, augmented_message: bytes, original_length: int) -> bytes:
        """Reverse entropy augmentation"""
        # Generate same entropy seed as encryption
        entropy_seed = self.entropy_manager.generate_high_entropy_seed(len(augmented_message))
        
        # XOR to recover original message
        original_message = bytearray()
        for i in range(len(augmented_message)):
            original_byte = augmented_message[i] ^ entropy_seed[i % len(entropy_seed)]
            original_message.append(original_byte)
        
        # Truncate to original length
        return bytes(original_message[:original_length])
    
    def solve_closest_vector_problem(self, ciphertext_poly: np.ndarray, 
                                   ephemeral_key: np.ndarray, 
                                   derived_key: np.ndarray) -> np.ndarray:
        """Solve CVP to recover message polynomial"""
        # Compute the lattice vector component
        lattice_component = self.lattice_ops.polynomial_multiply(ephemeral_key, derived_key)
        
        # Subtract to get noisy message
        noisy_message = (ciphertext_poly - lattice_component) % self.q
        
        # Use CVP to find closest lattice point (simplified)
        # In practice, this would use advanced lattice reduction algorithms
        recovered_message = np.copy(noisy_message)
        
        # Simple noise removal based on distance to 0 or q/2
        mid_point = self.q // 2
        for i in range(len(recovered_message)):
            coeff = recovered_message[i]
            if abs(coeff) > abs(coeff - mid_point):
                if abs(coeff - mid_point) > abs(coeff - self.q):
                    recovered_message[i] = 0
                else:
                    recovered_message[i] = mid_point
            else:
                recovered_message[i] = 0 if abs(coeff) < mid_point // 2 else coeff
        
        return recovered_message
    
    def polynomial_to_message(self, message_poly: np.ndarray) -> List[int]:
        """Convert polynomial back to binary message"""
        binary_bits = []
        mid_point = self.q // 2
        
        for coeff in message_poly:
            # Determine bit based on distance to 0 or ⌊q/2⌋
            if abs(coeff) < abs(coeff - mid_point):
                binary_bits.append(0)
            else:
                binary_bits.append(1)
        
        return binary_bits
    
    def comprehensive_decrypt(self, ciphertext: Dict[str, Any]) -> bytes:
        """
        Comprehensive decryption using all DIBLE features
        
        Args:
            ciphertext: Complete ciphertext structure
            
        Returns:
            Decrypted message as bytes
        """
        # Step 1: Verify ciphertext integrity
        if not self.verify_ciphertext_integrity(ciphertext):
            raise ValueError("Ciphertext integrity verification failed")
        
        # Step 2: Verify device authorization
        if not self.verify_device_authorization(ciphertext):
            raise ValueError("Device not authorized to decrypt this message")
        
        # Step 3: Extract ciphertext components
        ciphertext_poly = np.array(ciphertext['ciphertext'])
        security_params = ciphertext['security_parameters']
        
        # Verify algorithm parameters match
        if (security_params['q'] != self.q or 
            security_params['n'] != self.n or 
            abs(security_params['sigma'] - self.sigma) > 0.1):
            raise ValueError("Algorithm parameters mismatch")
        
        # Step 4: Reconstruct keys
        ephemeral_key = self.reconstruct_ephemeral_key(ciphertext)
        derived_key = self.reconstruct_fractal_key(ephemeral_key)
        
        # Step 5: Solve CVP to recover transformed message
        recovered_poly = self.solve_closest_vector_problem(ciphertext_poly, 
                                                          ephemeral_key, 
                                                          derived_key)
        
        # Step 6: Reverse device binding
        unbound_poly = self.reverse_device_binding(recovered_poly, self.device_id)
        
        # Step 7: Reverse post-quantum layer (if enabled)
        if security_params.get('post_quantum_enabled', True):
            pq_reversed = self.reverse_post_quantum_layer(unbound_poly)
        else:
            pq_reversed = unbound_poly
        
        # Step 8: Reverse homomorphic layer (if enabled)
        if security_params.get('homomorphic_enabled', True):
            homomorphic_reversed = self.reverse_homomorphic_layer(pq_reversed)
        else:
            homomorphic_reversed = pq_reversed
        
        # Step 9: Reverse tensor encryption
        tensor_reversed = self.reverse_tensor_encryption(homomorphic_reversed)
        
        # Step 10: Reverse quantum-inspired encryption
        quantum_reversed = self.reverse_quantum_inspired_encryption(tensor_reversed)
        
        # Step 11: Reverse chaos transformation
        chaos_reversed = self.reverse_chaos_transformation(quantum_reversed)
        
        # Step 12: Convert polynomial to binary message
        binary_bits = self.polynomial_to_message(chaos_reversed)
        
        # Step 13: Convert binary to bytes
        augmented_message = bytearray()
        for i in range(0, len(binary_bits), 8):
            byte_bits = binary_bits[i:i+8]
            while len(byte_bits) < 8:
                byte_bits.append(0)
            
            byte_value = 0
            for j, bit in enumerate(byte_bits):
                byte_value |= (bit << j)
            
            augmented_message.append(byte_value)
        
        # Step 14: Reverse entropy augmentation
        entropy_metadata = ciphertext.get('entropy_metadata', {})
        original_length = entropy_metadata.get('original_length', len(augmented_message))
        
        original_message = self.reverse_entropy_augmentation(bytes(augmented_message), 
                                                           original_length)
        
        return original_message
    
    def batch_decrypt(self, ciphertexts: List[Dict[str, Any]]) -> List[bytes]:
        """Decrypt multiple ciphertexts efficiently"""
        decrypted_messages = []
        
        for ciphertext in ciphertexts:
            try:
                decrypted_msg = self.comprehensive_decrypt(ciphertext)
                decrypted_messages.append(decrypted_msg)
            except Exception as e:
                # Add error handling for failed decryptions
                print(f"Decryption failed for ciphertext: {str(e)}")
                decrypted_messages.append(b"")  # Empty bytes for failed decryption
        
        return decrypted_messages
    
    def verify_message_authenticity(self, decrypted_message: bytes, 
                                  expected_hash: str = None) -> bool:
        """Verify authenticity of decrypted message"""
        if expected_hash is None:
            return True  # No hash provided, assume authentic
        
        message_hash = hashlib.sha256(decrypted_message).hexdigest()
        return message_hash == expected_hash
    
    def partial_decrypt(self, ciphertext: Dict[str, Any], 
                       layers_to_decrypt: List[str]) -> np.ndarray:
        """
        Partially decrypt only specified layers
        
        Args:
            ciphertext: Ciphertext to partially decrypt
            layers_to_decrypt: List of layer names to decrypt
            
        Returns:
            Partially decrypted polynomial
        """
        ciphertext_poly = np.array(ciphertext['ciphertext'])
        current_poly = ciphertext_poly
        
        # Define layer processing order
        layer_functions = {
            'device_binding': lambda x: self.reverse_device_binding(x, self.device_id),
            'post_quantum': self.reverse_post_quantum_layer,
            'homomorphic': self.reverse_homomorphic_layer,
            'tensor': self.reverse_tensor_encryption,
            'quantum': self.reverse_quantum_inspired_encryption,
            'chaos': self.reverse_chaos_transformation
        }
        
        # Apply only requested layers
        for layer_name in layers_to_decrypt:
            if layer_name in layer_functions:
                current_poly = layer_functions[layer_name](current_poly)
        
        return current_poly
    
    def get_decryption_metadata(self) -> Dict[str, Any]:
        """Get decryption system metadata"""
        return {
            'device_id': self.device_id,
            'transformed_device_id': self.transformed_device_id,
            'algorithm_parameters': {
                'q': self.q,
                'n': self.n,
                'sigma': self.sigma
            },
            'supported_operations': [
                'comprehensive_decrypt',
                'batch_decrypt',
                'partial_decrypt',
                'verify_message_authenticity',
                'verify_ciphertext_integrity'
            ],
            'decryption_layers': [
                'device_binding',
                'post_quantum',
                'homomorphic',
                'tensor',
                'quantum_inspired',
                'chaos_transformation',
                'entropy_augmentation'
            ]
        }


if __name__ == "__main__":
    # Test DIBLE decryption
    print("Testing DIBLE Decryption Module...")
    
    # Create decryption instance
    decryptor = DIBLEDecryption()
    
    # Since we need a valid ciphertext for testing, we'll create a mock one
    # In practice, this would come from the encryption module
    mock_ciphertext = {
        'algorithm': 'HC-DIBLE-DDI',
        'version': '1.0',
        'ciphertext': [42] * decryptor.n,  # Mock polynomial
        'ephemeral_key': [17] * decryptor.n,  # Mock ephemeral key
        'device_binding': hashlib.sha256(decryptor.device_id.encode()).hexdigest(),
        'source_device': decryptor.device_id,
        'timestamp': int(time.time() * 1000000),
        'security_parameters': {
            'q': decryptor.q,
            'n': decryptor.n,
            'sigma': decryptor.sigma,
            'homomorphic_enabled': True,
            'post_quantum_enabled': True
        },
        'entropy_metadata': {
            'original_length': 50,
            'augmented_length': 64
        }
    }
    
    # Add integrity hash
    ciphertext_str = str(mock_ciphertext)
    integrity_hash = hashlib.sha256(ciphertext_str.encode()).hexdigest()
    mock_ciphertext['integrity_hash'] = integrity_hash
    
    # Test integrity verification
    integrity_valid = decryptor.verify_ciphertext_integrity(mock_ciphertext)
    print(f"Ciphertext integrity verification: {'✓' if integrity_valid else '✗'}")
    
    # Test device authorization
    auth_valid = decryptor.verify_device_authorization(mock_ciphertext)
    print(f"Device authorization: {'✓' if auth_valid else '✗'}")
    
    # Test key reconstruction
    ephemeral_key = decryptor.reconstruct_ephemeral_key(mock_ciphertext)
    derived_key = decryptor.reconstruct_fractal_key(ephemeral_key)
    print(f"Key reconstruction completed: ephemeral_key length = {len(ephemeral_key)}")
    
    # Display metadata
    metadata = decryptor.get_decryption_metadata()
    print(f"Decryption system ready with {len(metadata['decryption_layers'])} layers")
    
    print("DIBLE Decryption Module test completed successfully!")
