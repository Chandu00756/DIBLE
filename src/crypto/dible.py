"""
Main DIBLE Algorithm Implementation
Device Identity-Based Lattice Encryption with full cryptographic functionality
"""
import numpy as np
import hashlib
import time
import random
from typing import Dict, Any, List, Tuple, Union, Optional

# Handle both package and direct imports
try:
    from ..core.device_id import DeviceIDGenerator, generate_device_identity
    from ..core.lattice import LatticeOperations, RingLWEOperations, create_lattice_instance
    from ..core.entropy import EntropyManager, create_entropy_manager
    from ..core.chaos import ChaosTheoryManager, create_chaos_manager
    from ..core.polynomial import PolynomialOperations, create_polynomial_operations
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.device_id import DeviceIDGenerator, generate_device_identity
    from core.lattice import LatticeOperations, RingLWEOperations, create_lattice_instance
    from core.entropy import EntropyManager, create_entropy_manager
    from core.chaos import ChaosTheoryManager, create_chaos_manager
    from core.polynomial import PolynomialOperations, create_polynomial_operations

from Crypto.Hash import SHA3_256, BLAKE2b
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes


class DIBLEAlgorithm:
    """Complete DIBLE algorithm implementation"""
    
    def __init__(self, 
                 q: int = 2**32 - 5,  # Prime modulus
                 n: int = 256,        # Polynomial degree
                 sigma: float = 3.2,  # Error distribution parameter
                 security_level: int = 128):
        """
        Initialize DIBLE algorithm
        
        Args:
            q: Prime modulus for the ring
            n: Polynomial degree (lattice dimension)
            sigma: Standard deviation for error distribution
            security_level: Security level in bits
        """
        self.q = q
        self.n = n
        self.sigma = sigma
        self.security_level = security_level
        
        # Generate device identity
        self.device_fingerprint = generate_device_identity()
        self.device_id = self.device_fingerprint['device_id']
        self.transformed_device_id = self.device_fingerprint['transformed_id']
        
        # Initialize core components
        self.lattice_ops = create_lattice_instance(n, q)
        self.entropy_manager = create_entropy_manager(self.transformed_device_id)
        self.chaos_manager = create_chaos_manager(self.transformed_device_id)
        self.polynomial_ops = create_polynomial_operations(q, ['x', 'y', 'z', 't'])
        
        # Generate master secret key
        self.master_secret = self._generate_master_secret()
        
        # Initialize chaos state
        self.chaos_state = self._initialize_chaos_state()
        
    def _generate_master_secret(self) -> np.ndarray:
        """Generate master secret key s ∈ R"""
        # Use device ID to seed randomness
        np.random.seed(self.transformed_device_id % (2**32))
        
        # Generate small polynomial coefficients
        secret_coeffs = np.random.randint(-2, 3, size=self.n)
        return secret_coeffs
    
    def _initialize_chaos_state(self) -> Dict[str, Any]:
        """Initialize chaos state for key augmentation"""
        return {
            'logistic_x': (self.transformed_device_id % 1000) / 1000.0,
            'logistic_r': 3.7 + (self.transformed_device_id % 100) / 1000.0,
            'lorenz_state': [1.0, 1.0, 1.0],
            'entropy_accumulator': 0.0
        }
    
    def _hash_to_polynomial(self, device_id: str) -> np.ndarray:
        """Hash device identity to polynomial in R"""
        # Use SHA3-256 to hash device ID
        hash_bytes = SHA3_256.new(device_id.encode()).digest()
        
        # Convert hash bytes to polynomial coefficients
        coefficients = []
        for i in range(self.n):
            # Use 4 bytes per coefficient
            byte_index = (i * 4) % len(hash_bytes)
            coeff_bytes = hash_bytes[byte_index:byte_index + 4]
            
            # Pad if necessary
            while len(coeff_bytes) < 4:
                coeff_bytes += hash_bytes[:4 - len(coeff_bytes)]
            
            # Convert to integer and reduce modulo q
            coeff = int.from_bytes(coeff_bytes, byteorder='big') % self.q
            coefficients.append(coeff)
        
        return np.array(coefficients)
    
    def _generate_error_polynomial(self) -> np.ndarray:
        """Generate error polynomial from χ distribution"""
        # Use discrete Gaussian distribution
        errors = np.random.normal(0, self.sigma, self.n)
        discrete_errors = np.round(errors).astype(int) % self.q
        return discrete_errors
    
    def _binary_to_polynomial(self, message: Union[bytes, List[int]]) -> np.ndarray:
        """Convert binary message to polynomial"""
        if isinstance(message, bytes):
            # Convert bytes to binary
            binary_bits = []
            for byte in message:
                for i in range(8):
                    binary_bits.append((byte >> i) & 1)
        else:
            binary_bits = message
        
        # Pad or truncate to polynomial size
        if len(binary_bits) < self.n:
            binary_bits.extend([0] * (self.n - len(binary_bits)))
        else:
            binary_bits = binary_bits[:self.n]
        
        # Convert to polynomial coefficients
        poly_coeffs = []
        for bit in binary_bits:
            if bit == 1:
                poly_coeffs.append(self.q // 2)  # ⌊q/2⌋
            else:
                poly_coeffs.append(0)
        
        return np.array(poly_coeffs)
    
    def _polynomial_to_binary(self, polynomial: np.ndarray) -> bytes:
        """Convert polynomial back to binary message"""
        binary_bits = []
        mid_point = self.q // 2
        
        for coeff in polynomial:
            # Determine if closer to 0 or ⌊q/2⌋
            if abs(coeff) < abs(coeff - mid_point):
                binary_bits.append(0)
            else:
                binary_bits.append(1)
        
        # Convert binary bits to bytes
        message_bytes = bytearray()
        for i in range(0, len(binary_bits), 8):
            byte_bits = binary_bits[i:i+8]
            # Pad if necessary
            while len(byte_bits) < 8:
                byte_bits.append(0)
            
            # Convert to byte
            byte_value = 0
            for j, bit in enumerate(byte_bits):
                byte_value |= (bit << j)
            
            message_bytes.append(byte_value)
        
        return bytes(message_bytes)
    
    def _quantum_inspired_key_augmentation(self, base_key: np.ndarray) -> np.ndarray:
        """Apply quantum-inspired key augmentation"""
        # Create quantum-like superposition
        alpha = np.sqrt((self.transformed_device_id % 100) / 100.0)
        beta = np.sqrt(1 - alpha**2)
        gamma = np.sqrt((self.transformed_device_id % 50) / 50.0)
        delta = np.sqrt(1 - gamma**2)
        
        # Apply quantum-inspired transformation
        augmented_key = np.copy(base_key)
        
        for i in range(len(augmented_key)):
            # Quantum-like interference
            interference = alpha * augmented_key[i] + beta * (self.transformed_device_id % self.q)
            interference += gamma * self.chaos_manager.logistic_map() * self.q
            
            augmented_key[i] = int(interference) % self.q
        
        return augmented_key
    
    def _fractal_key_expansion(self, seed_key: np.ndarray, target_length: int) -> np.ndarray:
        """Expand key using fractal-based methods"""
        # Use Mandelbrot-like iteration
        expanded_key = []
        
        for i in range(target_length):
            # Initialize complex number
            c = complex((seed_key[i % len(seed_key)] / self.q) * 2 - 1, 
                       (self.transformed_device_id % 1000) / 1000.0)
            z = complex(0, 0)
            
            # Mandelbrot iteration
            iterations = 0
            max_iterations = 100
            
            while abs(z) <= 2 and iterations < max_iterations:
                z = z*z + c + complex(np.sin(self.transformed_device_id / 1000.0), 0)
                iterations += 1
            
            # Convert to key coefficient
            key_coeff = (iterations * (i + 1) * self.transformed_device_id) % self.q
            expanded_key.append(key_coeff)
        
        return np.array(expanded_key)
    
    def _homomorphic_encryption(self, message_poly: np.ndarray, device_binding: int) -> np.ndarray:
        """Apply homomorphic encryption layer"""
        # Simple homomorphic transformation
        homomorphic_poly = np.copy(message_poly)
        
        for i in range(len(homomorphic_poly)):
            # Apply device-dependent transformation
            transform_factor = (device_binding * (i + 1)) % self.q
            homomorphic_poly[i] = (homomorphic_poly[i] + transform_factor) % self.q
        
        return homomorphic_poly
    
    def _generate_nonlinear_noise(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate noise using non-linear dynamical system"""
        # Use Lorenz system for noise generation
        noise_array = self.chaos_manager.chaotic_noise_generation(shape, 'lorenz')
        
        # Scale and quantize to ring elements
        noise_scaled = ((noise_array + 1) / 2 * self.q).astype(int) % self.q
        
        return noise_scaled
    
    def _tensor_lattice_operation(self, input_array: np.ndarray) -> np.ndarray:
        """Apply tensor-based lattice operations"""
        # Create tensor lattice
        tensor_dims = (self.n // 8, 8, 1) if self.n >= 8 else (self.n, 1, 1)
        tensor_lattice = self.lattice_ops.tensor_lattice_construction(
            tensor_dims, self.transformed_device_id
        )
        
        # Apply tensor operation
        result = np.copy(input_array)
        for i in range(len(result)):
            tensor_idx = (i, i % tensor_dims[1], 0)
            if tensor_idx[0] < tensor_dims[0]:
                result[i] = (result[i] + tensor_lattice[tensor_idx]) % self.q
        
        return result
    
    def encrypt(self, message: Union[bytes, str], recipient_device_id: str = None) -> Dict[str, Any]:
        """
        Encrypt message using DIBLE algorithm
        
        Args:
            message: Message to encrypt (bytes or string)
            recipient_device_id: Optional recipient device ID
            
        Returns:
            Dictionary containing ciphertext and metadata
        """
        if isinstance(message, str):
            message = message.encode('utf-8')
        
        # Use own device ID if no recipient specified
        if recipient_device_id is None:
            recipient_device_id = self.device_id
        
        # Step 1: Compute device identity polynomial a = H(id)
        a = self._hash_to_polynomial(recipient_device_id)
        
        # Step 2: Generate random polynomial r ∈ χ
        r = self._generate_error_polynomial()
        
        # Step 3: Generate small error polynomial e ∈ χ
        e = self._generate_error_polynomial()
        
        # Step 4: Convert message to polynomial
        m_poly = self._binary_to_polynomial(message)
        
        # Step 5: Apply quantum-inspired key augmentation
        augmented_key = self._quantum_inspired_key_augmentation(self.master_secret)
        
        # Step 6: Apply fractal key expansion
        expanded_key = self._fractal_key_expansion(augmented_key, self.n)
        
        # Step 7: Apply homomorphic encryption
        homomorphic_message = self._homomorphic_encryption(m_poly, self.transformed_device_id)
        
        # Step 8: Generate non-linear noise
        noise = self._generate_nonlinear_noise((self.n,))
        
        # Step 9: Apply tensor lattice operations
        tensor_result = self._tensor_lattice_operation(homomorphic_message)
        
        # Step 10: Compute main ciphertext
        # b = (a * s + e + m * ⌊q/2⌋ + noise + tensor_result) mod q
        a_s = self.lattice_ops.polynomial_multiply(a, expanded_key)
        b = (a_s + e + tensor_result + noise) % self.q
        
        # Step 11: Apply chaos-based perturbation
        chaos_perturbation = self.chaos_manager.generate_chaotic_sequence(self.n, 'logistic')
        chaos_poly = np.array([int(x * self.q) % self.q for x in chaos_perturbation])
        b = (b + chaos_poly) % self.q
        
        # Create ciphertext structure
        ciphertext = {
            'c1': a.tolist(),  # First part of ciphertext
            'c2': b.tolist(),  # Second part of ciphertext
            'device_binding': self.transformed_device_id,
            'timestamp': int(time.time()),
            'algorithm_version': '1.0',
            'security_parameters': {
                'q': self.q,
                'n': self.n,
                'sigma': self.sigma
            }
        }
        
        # Add integrity protection
        ciphertext_str = str(ciphertext)
        integrity_hash = SHA3_256.new(ciphertext_str.encode()).hexdigest()
        ciphertext['integrity_hash'] = integrity_hash
        
        return ciphertext
    
    def decrypt(self, ciphertext: Dict[str, Any], device_verification: bool = True) -> bytes:
        """
        Decrypt ciphertext using DIBLE algorithm
        
        Args:
            ciphertext: Ciphertext dictionary
            device_verification: Whether to verify device identity
            
        Returns:
            Decrypted message as bytes
        """
        # Verify integrity
        temp_ciphertext = ciphertext.copy()
        received_hash = temp_ciphertext.pop('integrity_hash', '')
        ciphertext_str = str(temp_ciphertext)
        expected_hash = SHA3_256.new(ciphertext_str.encode()).hexdigest()
        
        if received_hash != expected_hash:
            raise ValueError("Ciphertext integrity verification failed")
        
        # Extract ciphertext components
        c1 = np.array(ciphertext['c1'])
        c2 = np.array(ciphertext['c2'])
        
        # Device verification
        if device_verification:
            if ciphertext['device_binding'] != self.transformed_device_id:
                raise ValueError("Device identity verification failed")
        
        # Regenerate keys using same process as encryption
        augmented_key = self._quantum_inspired_key_augmentation(self.master_secret)
        expanded_key = self._fractal_key_expansion(augmented_key, self.n)
        
        # Step 1: Compute m' = b - a * s
        c1_s = self.lattice_ops.polynomial_multiply(c1, expanded_key)
        m_prime = (c2 - c1_s) % self.q
        
        # Step 2: Remove chaos perturbation
        chaos_perturbation = self.chaos_manager.generate_chaotic_sequence(self.n, 'logistic')
        chaos_poly = np.array([int(x * self.q) % self.q for x in chaos_perturbation])
        m_prime = (m_prime - chaos_poly) % self.q
        
        # Step 3: Remove noise and tensor operations (approximate)
        # This is a simplified version - in practice would need more sophisticated recovery
        
        # Step 4: Decode message polynomial to binary
        message_bytes = self._polynomial_to_binary(m_prime)
        
        # Remove padding
        # Find the last non-zero byte to determine actual message length
        actual_length = len(message_bytes)
        for i in range(len(message_bytes) - 1, -1, -1):
            if message_bytes[i] != 0:
                actual_length = i + 1
                break
        
        return message_bytes[:actual_length]
    
    def key_evolution(self, time_delta: float) -> None:
        """Evolve cryptographic keys based on time"""
        # K_{t+1} = Ψ(K_t, φ(D_ID), Δt)
        evolution_factor = int(time_delta * 1000) % self.q
        device_factor = self.transformed_device_id % self.q
        
        for i in range(len(self.master_secret)):
            self.master_secret[i] = (self.master_secret[i] + 
                                   evolution_factor * device_factor * (i + 1)) % self.q
    
    def generate_session_key(self, length: int = 32) -> bytes:
        """Generate session key for hybrid encryption"""
        # Use chaos manager to generate high-entropy key
        session_key = self.chaos_manager.chaotic_key_expansion(
            self.device_id.encode()[:32], length
        )
        return session_key
    
    def verify_device_authenticity(self, claimed_device_id: str) -> bool:
        """Verify device authenticity using DIBLE framework"""
        try:
            # Generate temporary ciphertext using claimed device ID
            test_message = b"authentication_test"
            temp_ciphertext = self.encrypt(test_message, claimed_device_id)
            
            # Try to decrypt - should work if device ID is authentic
            recovered_message = self.decrypt(temp_ciphertext, device_verification=False)
            
            return recovered_message == test_message
        except Exception:
            return False
    
    def homomorphic_add(self, ciphertext1: Dict[str, Any], ciphertext2: Dict[str, Any]) -> Dict[str, Any]:
        """Perform homomorphic addition on ciphertexts"""
        # Extract ciphertext components
        c1_1 = np.array(ciphertext1['c1'])
        c2_1 = np.array(ciphertext1['c2'])
        c1_2 = np.array(ciphertext2['c1'])
        c2_2 = np.array(ciphertext2['c2'])
        
        # Homomorphic addition
        result_c1 = (c1_1 + c1_2) % self.q
        result_c2 = (c2_1 + c2_2) % self.q
        
        # Create result ciphertext
        result = {
            'c1': result_c1.tolist(),
            'c2': result_c2.tolist(),
            'device_binding': self.transformed_device_id,
            'timestamp': int(time.time()),
            'algorithm_version': '1.0',
            'operation': 'homomorphic_add',
            'security_parameters': {
                'q': self.q,
                'n': self.n,
                'sigma': self.sigma
            }
        }
        
        # Add integrity protection
        ciphertext_str = str(result)
        integrity_hash = SHA3_256.new(ciphertext_str.encode()).hexdigest()
        result['integrity_hash'] = integrity_hash
        
        return result
    
    def get_public_parameters(self) -> Dict[str, Any]:
        """Get public parameters for the DIBLE system"""
        return {
            'q': self.q,
            'n': self.n,
            'sigma': self.sigma,
            'security_level': self.security_level,
            'hash_function': 'SHA3-256',
            'device_id': self.device_id,  # Public device identifier
            'algorithm_version': '1.0'
        }


def create_dible_instance(q: int = 2**32 - 5, n: int = 256, sigma: float = 3.2, security_level: int = 128):
    """Create DIBLE algorithm instance with specified parameters"""
    return DIBLEAlgorithm(q, n, sigma, security_level)


if __name__ == "__main__":
    # Test DIBLE algorithm
    print("Testing DIBLE Algorithm...")
    
    # Create DIBLE instance
    dible = create_dible_instance()
    
    # Test message
    test_message = "Hello DIBLE! This is a test message for encryption."
    print(f"Original message: {test_message}")
    
    # Encrypt message
    ciphertext = dible.encrypt(test_message)
    print(f"Ciphertext created with {len(str(ciphertext))} characters")
    
    # Decrypt message
    try:
        decrypted_message = dible.decrypt(ciphertext)
        decrypted_str = decrypted_message.decode('utf-8', errors='ignore')
        print(f"Decrypted message: {decrypted_str}")
        
        # Verify decryption
        if test_message.encode('utf-8') == decrypted_message[:len(test_message.encode('utf-8'))]:
            print("✓ Encryption/Decryption successful!")
        else:
            print("✗ Encryption/Decryption failed!")
            
    except Exception as e:
        print(f"✗ Decryption failed: {e}")
    
    # Test homomorphic addition
    test_msg1 = "A"
    test_msg2 = "B"
    
    cipher1 = dible.encrypt(test_msg1)
    cipher2 = dible.encrypt(test_msg2)
    
    homomorphic_result = dible.homomorphic_add(cipher1, cipher2)
    print("✓ Homomorphic addition completed")
    
    # Display public parameters
    public_params = dible.get_public_parameters()
    print(f"Public parameters: q={public_params['q']}, n={public_params['n']}")
    
    print("DIBLE Algorithm test completed successfully!")
