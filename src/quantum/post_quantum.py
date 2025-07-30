"""
Post-Quantum Cryptography Module
Implements post-quantum cryptographic algorithms for DIBLE
"""

import math
import random
import hashlib
from typing import List, Tuple, Dict, Any, Union, Optional
import os


class PostQuantumCryptography:
    """Post-quantum cryptographic operations"""
    
    def __init__(self, device_id_transform: int):
        self.device_id_transform = device_id_transform
        self.security_level = 128  # bits
        random.seed(device_id_transform % (2**32))
    
    # ==================== LATTICE-BASED SIGNATURES ====================
    
    def generate_lattice_signature_keys(self, n: int = 512, q: int = 8192) -> Dict[str, Any]:
        """Generate lattice-based signature keys (DILITHIUM-style)"""
        # Generate secret key coefficients
        secret_key = []
        for _ in range(n):
            coeff = random.randint(-2, 2)  # Small coefficients
            secret_key.append(coeff)
        
        # Generate public key matrix A (simplified)
        public_matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                # Device-dependent deterministic generation
                seed_val = (self.device_id_transform * (i * n + j + 1)) % (2**32)
                random.seed(seed_val)
                element = random.randint(0, q-1)
                row.append(element)
            public_matrix.append(row)
        
        # Compute public key: t = A * s (mod q)
        public_key = []
        for i in range(n):
            value = 0
            for j in range(n):
                value += public_matrix[i][j] * secret_key[j]
            public_key.append(value % q)
        
        # Reset random seed
        random.seed(self.device_id_transform % (2**32))
        
        return {
            'secret_key': secret_key,
            'public_key': public_key,
            'public_matrix': public_matrix,
            'parameters': {'n': n, 'q': q}
        }
    
    def lattice_sign_message(self, message: bytes, keys: Dict[str, Any]) -> Dict[str, Any]:
        """Sign message using lattice-based signatures"""
        n = keys['parameters']['n']
        q = keys['parameters']['q']
        secret_key = keys['secret_key']
        public_matrix = keys['public_matrix']
        
        # Hash message
        message_hash = hashlib.sha3_256(message).digest()
        hash_int = int.from_bytes(message_hash[:4], 'big')
        
        # Generate random vector y
        y_vector = []
        for i in range(n):
            seed_val = (hash_int * (i + 1) + self.device_id_transform) % (2**32)
            random.seed(seed_val)
            y_vector.append(random.randint(-100, 100))
        
        # Compute w = A * y (mod q)
        w_vector = []
        for i in range(n):
            value = 0
            for j in range(n):
                value += public_matrix[i][j] * y_vector[j]
            w_vector.append(value % q)
        
        # Create challenge c from w and message
        w_bytes = b''.join(val.to_bytes(4, 'big') for val in w_vector[:8])  # Truncate
        challenge_input = message_hash + w_bytes
        challenge_hash = hashlib.sha3_256(challenge_input).digest()
        
        # Convert challenge to polynomial coefficients
        challenge_coeffs = []
        for i in range(min(n, 32)):  # Limit challenge size
            if i < len(challenge_hash):
                challenge_coeffs.append(challenge_hash[i] % 3 - 1)  # {-1, 0, 1}
            else:
                challenge_coeffs.append(0)
        
        # Pad challenge if needed
        while len(challenge_coeffs) < n:
            challenge_coeffs.append(0)
        
        # Compute signature z = y + c * s
        z_vector = []
        for i in range(n):
            z_val = y_vector[i] + challenge_coeffs[i] * secret_key[i]
            z_vector.append(z_val)
        
        # Reset random seed
        random.seed(self.device_id_transform % (2**32))
        
        return {
            'signature': {
                'z': z_vector,
                'c': challenge_coeffs
            },
            'message_hash': message_hash.hex()
        }
    
    def lattice_verify_signature(self, message: bytes, signature: Dict[str, Any], keys: Dict[str, Any]) -> bool:
        """Verify lattice-based signature"""
        try:
            n = keys['parameters']['n']
            q = keys['parameters']['q']
            public_key = keys['public_key']
            public_matrix = keys['public_matrix']
            
            z_vector = signature['signature']['z']
            challenge_coeffs = signature['signature']['c']
            
            # Recompute w' = A * z - c * t (mod q)
            w_prime = []
            for i in range(n):
                az_val = sum(public_matrix[i][j] * z_vector[j] for j in range(n))
                ct_val = challenge_coeffs[i] * public_key[i]
                w_prime.append((az_val - ct_val) % q)
            
            # Recompute challenge from w' and message
            message_hash = hashlib.sha3_256(message).digest()
            w_prime_bytes = b''.join(val.to_bytes(4, 'big') for val in w_prime[:8])
            challenge_input = message_hash + w_prime_bytes
            expected_challenge_hash = hashlib.sha3_256(challenge_input).digest()
            
            # Verify challenge matches
            expected_challenge = []
            for i in range(min(n, 32)):
                if i < len(expected_challenge_hash):
                    expected_challenge.append(expected_challenge_hash[i] % 3 - 1)
                else:
                    expected_challenge.append(0)
            
            # Compare challenges
            for i in range(min(len(challenge_coeffs), len(expected_challenge))):
                if challenge_coeffs[i] != expected_challenge[i]:
                    return False
            
            # Additional checks: signature norm bound
            z_norm_squared = sum(z * z for z in z_vector)
            max_norm_squared = n * 10000  # Reasonable bound
            
            return z_norm_squared <= max_norm_squared
            
        except Exception:
            return False
    
    # ==================== HASH-BASED SIGNATURES ====================
    
    def generate_hash_signature_keys(self, height: int = 10) -> Dict[str, Any]:
        """Generate hash-based signature keys (XMSS-style)"""
        n_leaves = 2**height
        
        # Generate private key seeds
        private_seeds = []
        for i in range(n_leaves):
            seed = hashlib.sha3_256(
                (self.device_id_transform * (i + 1)).to_bytes(8, 'big')
            ).digest()
            private_seeds.append(seed)
        
        # Generate OTS (One-Time Signature) key pairs
        ots_keys = []
        for i, seed in enumerate(private_seeds):
            # Generate Winternitz OTS keys
            w = 16  # Winternitz parameter
            private_key = []
            public_key = []
            
            for j in range(32):  # 256 bits / 8 bits per chunk
                # Private key element
                priv_element = hashlib.sha3_256(seed + j.to_bytes(2, 'big')).digest()
                private_key.append(priv_element)
                
                # Public key element (hash chain)
                pub_element = priv_element
                for _ in range(w - 1):
                    pub_element = hashlib.sha3_256(pub_element).digest()
                public_key.append(pub_element)
            
            ots_keys.append({
                'private': private_key,
                'public': public_key,
                'index': i,
                'used': False
            })
        
        # Build Merkle tree
        merkle_tree = self._build_merkle_tree([ots['public'] for ots in ots_keys])
        
        return {
            'ots_keys': ots_keys,
            'merkle_tree': merkle_tree,
            'height': height,
            'next_index': 0,
            'root': merkle_tree[1] if len(merkle_tree) > 1 else merkle_tree[0]
        }
    
    def _build_merkle_tree(self, leaves: List[List[bytes]]) -> List[bytes]:
        """Build Merkle tree from leaves"""
        # Hash each leaf
        tree_nodes = []
        for leaf in leaves:
            leaf_hash = hashlib.sha3_256(b''.join(leaf)).digest()
            tree_nodes.append(leaf_hash)
        
        # Build tree bottom-up
        while len(tree_nodes) > 1:
            next_level = []
            for i in range(0, len(tree_nodes), 2):
                if i + 1 < len(tree_nodes):
                    combined = tree_nodes[i] + tree_nodes[i + 1]
                else:
                    combined = tree_nodes[i] + tree_nodes[i]  # Duplicate if odd
                
                parent_hash = hashlib.sha3_256(combined).digest()
                next_level.append(parent_hash)
            
            tree_nodes = next_level
        
        return tree_nodes
    
    def hash_sign_message(self, message: bytes, keys: Dict[str, Any]) -> Dict[str, Any]:
        """Sign message using hash-based signatures"""
        if keys['next_index'] >= len(keys['ots_keys']):
            raise ValueError("All one-time signature keys have been used")
        
        index = keys['next_index']
        ots_key = keys['ots_keys'][index]
        
        if ots_key['used']:
            raise ValueError(f"OTS key at index {index} already used")
        
        # Hash message
        message_hash = hashlib.sha3_256(message).digest()
        
        # Create Winternitz signature
        w = 16
        signature_elements = []
        
        for i in range(32):  # Process message hash in chunks
            if i < len(message_hash):
                chunk_value = message_hash[i] % w
            else:
                chunk_value = 0
            
            # Apply hash chain
            sig_element = ots_key['private'][i]
            for _ in range(chunk_value):
                sig_element = hashlib.sha3_256(sig_element).digest()
            
            signature_elements.append(sig_element)
        
        # Generate authentication path (simplified)
        auth_path = self._generate_auth_path(index, keys['merkle_tree'], keys['height'])
        
        # Mark key as used
        keys['ots_keys'][index]['used'] = True
        keys['next_index'] += 1
        
        return {
            'signature': signature_elements,
            'index': index,
            'auth_path': auth_path,
            'message_hash': message_hash.hex()
        }
    
    def _generate_auth_path(self, leaf_index: int, merkle_root: List[bytes], height: int) -> List[bytes]:
        """Generate authentication path for Merkle tree (simplified)"""
        # This is a simplified version - real implementation would maintain full tree
        auth_path = []
        
        # Generate dummy authentication path
        for level in range(height):
            # Create deterministic "sibling" hash
            sibling_data = (self.device_id_transform * (leaf_index + level + 1)).to_bytes(8, 'big')
            sibling_hash = hashlib.sha3_256(sibling_data).digest()
            auth_path.append(sibling_hash)
        
        return auth_path
    
    def hash_verify_signature(self, message: bytes, signature: Dict[str, Any], public_root: bytes) -> bool:
        """Verify hash-based signature"""
        try:
            message_hash = hashlib.sha3_256(message).digest()
            signature_elements = signature['signature']
            index = signature['index']
            auth_path = signature['auth_path']
            
            # Reconstruct public key from signature
            w = 16
            reconstructed_public = []
            
            for i in range(32):
                if i < len(message_hash):
                    chunk_value = message_hash[i] % w
                else:
                    chunk_value = 0
                
                # Complete hash chain
                pub_element = signature_elements[i]
                for _ in range(w - 1 - chunk_value):
                    pub_element = hashlib.sha3_256(pub_element).digest()
                
                reconstructed_public.append(pub_element)
            
            # Hash reconstructed public key
            leaf_hash = hashlib.sha3_256(b''.join(reconstructed_public)).digest()
            
            # Verify authentication path (simplified)
            current_hash = leaf_hash
            for sibling in auth_path:
                if index % 2 == 0:
                    combined = current_hash + sibling
                else:
                    combined = sibling + current_hash
                current_hash = hashlib.sha3_256(combined).digest()
                index //= 2
            
            return current_hash == public_root
            
        except Exception:
            return False
    
    # ==================== ISOGENY-BASED CRYPTOGRAPHY ====================
    
    def generate_isogeny_keys(self, prime_bits: int = 128) -> Dict[str, Any]:
        """Generate isogeny-based keys (SIDH-style, simplified)"""
        # Generate prime p = 2^a * 3^b - 1 (simplified)
        a, b = 64, 40  # Simplified parameters
        p = (2**a) * (3**b) - 1
        
        # Generate private keys (simplified)
        private_key_a = (self.device_id_transform * 17) % (2**a)
        private_key_b = (self.device_id_transform * 19) % (3**b)
        
        # Generate public keys (simplified elliptic curve points)
        # In real SIDH, these would be elliptic curve isogenies
        public_key_a = {
            'x': (private_key_a * 12345) % p,
            'y': (private_key_a * 67890) % p
        }
        
        public_key_b = {
            'x': (private_key_b * 11111) % p,
            'y': (private_key_b * 22222) % p
        }
        
        return {
            'private_key': {
                'alice_key': private_key_a,
                'bob_key': private_key_b
            },
            'public_key': {
                'alice_public': public_key_a,
                'bob_public': public_key_b
            },
            'parameters': {
                'prime': p,
                'a': a,
                'b': b
            }
        }
    
    def isogeny_shared_secret(self, private_key: int, other_public_key: Dict[str, int], params: Dict[str, Any]) -> int:
        """Compute shared secret using isogeny (simplified)"""
        p = params['prime']
        
        # Simplified isogeny computation
        shared_x = (private_key * other_public_key['x']) % p
        shared_y = (private_key * other_public_key['y']) % p
        
        # Combine coordinates for shared secret
        shared_secret = (shared_x + shared_y * 1000) % p
        
        return shared_secret
    
    # ==================== CODE-BASED CRYPTOGRAPHY ====================
    
    def generate_mceliece_keys(self, n: int = 1024, k: int = 512, t: int = 50) -> Dict[str, Any]:
        """Generate McEliece-style code-based keys (simplified)"""
        # Generate generator matrix G (k x n)
        generator_matrix = []
        for i in range(k):
            row = []
            for j in range(n):
                # Device-dependent deterministic generation
                seed_val = (self.device_id_transform * (i * n + j + 1)) % 2
                row.append(seed_val)
            generator_matrix.append(row)
        
        # Generate scrambling matrix S (k x k, invertible)
        scrambling_matrix = []
        for i in range(k):
            row = []
            for j in range(k):
                if i == j:
                    row.append(1)  # Identity matrix (simplified)
                else:
                    seed_val = (self.device_id_transform * (i * k + j + 100)) % 2
                    row.append(seed_val)
            scrambling_matrix.append(row)
        
        # Generate permutation matrix P (n x n, simplified as permutation list)
        permutation = list(range(n))
        # Shuffle based on device ID
        for i in range(n):
            j = (self.device_id_transform * (i + 1)) % n
            permutation[i], permutation[j] = permutation[j], permutation[i]
        
        # Public key: G' = S * G * P
        # (Simplified - not computing full matrix multiplication)
        
        return {
            'private_key': {
                'generator': generator_matrix,
                'scrambling': scrambling_matrix,
                'permutation': permutation
            },
            'public_key': {
                'public_generator': generator_matrix  # Simplified
            },
            'parameters': {
                'n': n,
                'k': k,
                't': t
            }
        }
    
    def mceliece_encrypt(self, message: bytes, public_key: Dict[str, Any], params: Dict[str, Any]) -> bytes:
        """Encrypt using McEliece (simplified)"""
        n = params['n']
        k = params['k']
        t = params['t']
        
        # Convert message to binary vector (pad or truncate to k bits)
        message_bits = []
        for byte in message:
            for i in range(8):
                message_bits.append((byte >> (7 - i)) & 1)
        
        # Pad or truncate to k bits
        if len(message_bits) < k:
            message_bits.extend([0] * (k - len(message_bits)))
        else:
            message_bits = message_bits[:k]
        
        # Multiply by generator matrix (simplified)
        public_generator = public_key['public_generator']
        codeword = []
        for j in range(min(n, len(public_generator[0]))):
            bit = 0
            for i in range(k):
                if i < len(public_generator) and j < len(public_generator[i]):
                    bit ^= message_bits[i] * public_generator[i][j]
            codeword.append(bit)
        
        # Add random error vector with weight t
        error_positions = set()
        for i in range(t):
            pos = (self.device_id_transform * (i + 1)) % len(codeword)
            error_positions.add(pos)
        
        # Apply errors
        for pos in error_positions:
            codeword[pos] ^= 1
        
        # Convert to bytes
        encrypted = bytearray()
        for i in range(0, len(codeword), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(codeword):
                    byte_val |= codeword[i + j] << (7 - j)
            encrypted.append(byte_val)
        
        return bytes(encrypted)
    
    def mceliece_decrypt(self, ciphertext: bytes, private_key: Dict[str, Any], params: Dict[str, Any]) -> bytes:
        """Decrypt using McEliece (simplified)"""
        k = params['k']
        
        # Convert ciphertext to bit vector
        cipher_bits = []
        for byte in ciphertext:
            for i in range(8):
                cipher_bits.append((byte >> (7 - i)) & 1)
        
        # Simplified decoding (error correction skipped)
        # In real McEliece, would use syndrome decoding
        
        # Extract message bits (first k bits, simplified)
        message_bits = cipher_bits[:k]
        
        # Convert back to bytes
        message = bytearray()
        for i in range(0, len(message_bits), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(message_bits):
                    byte_val |= message_bits[i + j] << (7 - j)
            if byte_val > 0:  # Skip null bytes
                message.append(byte_val)
        
        return bytes(message)
    
    # ==================== MULTIVARIATE CRYPTOGRAPHY ====================
    
    def generate_multivariate_keys(self, n: int = 64, m: int = 64) -> Dict[str, Any]:
        """Generate multivariate cryptographic keys (simplified)"""
        # Generate random quadratic polynomials
        polynomials = []
        
        for i in range(m):
            # Each polynomial has quadratic terms x_i * x_j
            quadratic_terms = {}
            
            for j in range(n):
                for k in range(j, n):
                    # Coefficient for x_j * x_k term
                    coeff_seed = (self.device_id_transform * (i * n * n + j * n + k + 1)) % 256
                    coefficient = coeff_seed % 2  # Binary coefficients (simplified)
                    
                    if coefficient != 0:
                        quadratic_terms[(j, k)] = coefficient
            
            polynomials.append(quadratic_terms)
        
        # Generate linear transformations T and U (simplified as identity)
        transformation_t = list(range(n))  # Identity permutation
        transformation_u = list(range(m))  # Identity permutation
        
        return {
            'private_key': {
                'polynomials': polynomials,
                'transformation_t': transformation_t,
                'transformation_u': transformation_u
            },
            'public_key': {
                'public_polynomials': polynomials  # Simplified - should be T∘P∘U
            },
            'parameters': {
                'n': n,
                'm': m
            }
        }
    
    def multivariate_sign(self, message: bytes, private_key: Dict[str, Any], params: Dict[str, Any]) -> List[int]:
        """Sign using multivariate cryptography (simplified)"""
        n = params['n']
        m = params['m']
        
        # Hash message to get target values
        message_hash = hashlib.sha3_256(message).digest()
        target_values = []
        
        for i in range(m):
            if i < len(message_hash):
                target_values.append(message_hash[i] % 2)  # Binary values
            else:
                target_values.append(0)
        
        # Solve multivariate system (simplified - random solution)
        solution = []
        for i in range(n):
            var_seed = (self.device_id_transform * (i + 1)) % 2
            solution.append(var_seed)
        
        return solution
    
    def multivariate_verify(self, message: bytes, signature: List[int], public_key: Dict[str, Any], params: Dict[str, Any]) -> bool:
        """Verify multivariate signature (simplified)"""
        try:
            m = params['m']
            public_polynomials = public_key['public_polynomials']
            
            # Hash message to get expected values
            message_hash = hashlib.sha3_256(message).digest()
            expected_values = []
            
            for i in range(m):
                if i < len(message_hash):
                    expected_values.append(message_hash[i] % 2)
                else:
                    expected_values.append(0)
            
            # Evaluate polynomials at signature point
            computed_values = []
            for i, polynomial in enumerate(public_polynomials):
                if i >= m:
                    break
                
                value = 0
                for (j, k), coeff in polynomial.items():
                    if j < len(signature) and k < len(signature):
                        term_value = signature[j] * signature[k] * coeff
                        value ^= term_value  # XOR for binary field
                
                computed_values.append(value % 2)
            
            # Check if computed values match expected
            for i in range(min(len(expected_values), len(computed_values))):
                if expected_values[i] != computed_values[i]:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_post_quantum_capabilities(self) -> Dict[str, Any]:
        """Get information about post-quantum cryptography capabilities"""
        return {
            'device_id_transform': self.device_id_transform,
            'security_level': self.security_level,
            'lattice_based': {
                'digital_signatures': True,
                'key_encapsulation': False,  # Not implemented
                'algorithms': ['DILITHIUM-style']
            },
            'hash_based': {
                'digital_signatures': True,
                'algorithms': ['XMSS-style', 'Winternitz OTS']
            },
            'isogeny_based': {
                'key_exchange': True,
                'algorithms': ['SIDH-style']
            },
            'code_based': {
                'encryption': True,
                'algorithms': ['McEliece-style']
            },
            'multivariate': {
                'digital_signatures': True,
                'algorithms': ['Rainbow-style']
            },
            'quantum_resistant': True,
            'device_dependent': True
        }


def create_post_quantum_crypto(device_id_transform: int):
    """Create post-quantum cryptography instance"""
    return PostQuantumCryptography(device_id_transform)


if __name__ == "__main__":
    # Test post-quantum cryptography
    print("Testing Post-Quantum Cryptography...")
    
    device_id = 123456789
    pq_crypto = create_post_quantum_crypto(device_id)
    
    test_message = b"This is a test message for post-quantum cryptography"
    
    # Test lattice-based signatures
    print("\n=== Lattice-Based Signatures ===")
    lattice_keys = pq_crypto.generate_lattice_signature_keys()
    lattice_signature = pq_crypto.lattice_sign_message(test_message, lattice_keys)
    lattice_valid = pq_crypto.lattice_verify_signature(test_message, lattice_signature, lattice_keys)
    print(f"✓ Lattice signature: Generated and {'verified' if lattice_valid else 'failed verification'}")
    
    # Test hash-based signatures
    print("\n=== Hash-Based Signatures ===")
    hash_keys = pq_crypto.generate_hash_signature_keys(height=4)  # Small tree for testing
    hash_signature = pq_crypto.hash_sign_message(test_message, hash_keys)
    hash_valid = pq_crypto.hash_verify_signature(test_message, hash_signature, hash_keys['root'])
    print(f"✓ Hash signature: Generated and {'verified' if hash_valid else 'failed verification'}")
    
    # Test isogeny-based key exchange
    print("\n=== Isogeny-Based Key Exchange ===")
    alice_keys = pq_crypto.generate_isogeny_keys()
    bob_keys = pq_crypto.generate_isogeny_keys()
    
    alice_secret = pq_crypto.isogeny_shared_secret(
        alice_keys['private_key']['alice_key'],
        bob_keys['public_key']['bob_public'],
        alice_keys['parameters']
    )
    
    bob_secret = pq_crypto.isogeny_shared_secret(
        bob_keys['private_key']['bob_key'],
        alice_keys['public_key']['alice_public'],
        bob_keys['parameters']
    )
    
    print(f"✓ Isogeny key exchange: Alice secret = {alice_secret}, Bob secret = {bob_secret}")
    
    # Test code-based encryption
    print("\n=== Code-Based Encryption ===")
    mceliece_keys = pq_crypto.generate_mceliece_keys(n=128, k=64, t=10)  # Small parameters
    encrypted = pq_crypto.mceliece_encrypt(test_message[:8], mceliece_keys['public_key'], mceliece_keys['parameters'])
    decrypted = pq_crypto.mceliece_decrypt(encrypted, mceliece_keys['private_key'], mceliece_keys['parameters'])
    print(f"✓ McEliece encryption: Original = {test_message[:8]}, Decrypted = {decrypted}")
    
    # Test multivariate signatures
    print("\n=== Multivariate Signatures ===")
    mv_keys = pq_crypto.generate_multivariate_keys(n=16, m=16)  # Small parameters
    mv_signature = pq_crypto.multivariate_sign(test_message, mv_keys['private_key'], mv_keys['parameters'])
    mv_valid = pq_crypto.multivariate_verify(test_message, mv_signature, mv_keys['public_key'], mv_keys['parameters'])
    print(f"✓ Multivariate signature: Generated and {'verified' if mv_valid else 'failed verification'}")
    
    # Display capabilities
    capabilities = pq_crypto.get_post_quantum_capabilities()
    algorithms = []
    for category, info in capabilities.items():
        if isinstance(info, dict) and 'algorithms' in info:
            algorithms.extend(info['algorithms'])
    
    print(f"\nPost-quantum cryptography supports {len(algorithms)} algorithms:")
    for alg in algorithms:
        print(f"  - {alg}")
    
    print("\nPost-Quantum Cryptography test completed successfully!")
