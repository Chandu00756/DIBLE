"""
Hash Functions Utilities
Comprehensive hash functions for DIBLE algorithm
"""

import hashlib
import hmac
import time
from typing import Dict, Any, List, Union, Optional
from Crypto.Hash import SHA3_256, SHA3_512, BLAKE2b, BLAKE2s


class HashUtilities:
    """Comprehensive hash function utilities for DIBLE"""
    
    def __init__(self, device_id_transform: int):
        self.device_id_transform = device_id_transform
        
    def sha3_256(self, data: Union[str, bytes]) -> str:
        """SHA3-256 hash function"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return SHA3_256.new(data).hexdigest()
    
    def sha3_512(self, data: Union[str, bytes]) -> str:
        """SHA3-512 hash function"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return SHA3_512.new(data).hexdigest()
    
    def blake2b_hash(self, data: Union[str, bytes], digest_size: int = 32) -> str:
        """BLAKE2b hash function"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return BLAKE2b.new(data, digest_bits=digest_size * 8).hexdigest()
    
    def blake2s_hash(self, data: Union[str, bytes], digest_size: int = 32) -> str:
        """BLAKE2s hash function"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return BLAKE2s.new(data, digest_bits=digest_size * 8).hexdigest()
    
    def device_dependent_hash(self, data: Union[str, bytes]) -> str:
        """Device-dependent hash function using device ID"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Combine data with device ID
        device_bytes = self.device_id_transform.to_bytes(32, byteorder='big')
        combined_data = data + device_bytes
        
        # Use SHA3-256 with device-specific salt
        return self.sha3_256(combined_data)
    
    def cascade_hash(self, data: Union[str, bytes], rounds: int = 5) -> str:
        """Cascade multiple hash functions for enhanced security"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Start with device-dependent hash
        current_hash = self.device_dependent_hash(data)
        
        for i in range(rounds):
            if i % 4 == 0:
                current_hash = self.sha3_256(current_hash)
            elif i % 4 == 1:
                current_hash = self.blake2b_hash(current_hash)
            elif i % 4 == 2:
                current_hash = self.sha3_512(current_hash)
            else:
                current_hash = self.blake2s_hash(current_hash)
        
        return current_hash
    
    def hmac_hash(self, data: Union[str, bytes], key: Union[str, bytes] = None) -> str:
        """HMAC hash with device-specific key"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if key is None:
            key = self.device_id_transform.to_bytes(32, byteorder='big')
        elif isinstance(key, str):
            key = key.encode('utf-8')
        
        return hmac.new(key, data, hashlib.sha256).hexdigest()
    
    def time_dependent_hash(self, data: Union[str, bytes], precision: int = 1000) -> str:
        """Time-dependent hash function"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Get current time rounded to specified precision
        current_time = int(time.time() / precision) * precision
        time_bytes = current_time.to_bytes(8, byteorder='big')
        
        # Combine data with time
        time_data = data + time_bytes
        
        return self.device_dependent_hash(time_data)
    
    def polynomial_hash(self, coefficients: List[int], modulus: int) -> int:
        """Polynomial rolling hash function"""
        base = 31
        hash_value = 0
        
        for i, coeff in enumerate(coefficients):
            hash_value = (hash_value + coeff * pow(base, i, modulus)) % modulus
        
        # Add device-specific component
        device_component = (self.device_id_transform * len(coefficients)) % modulus
        hash_value = (hash_value + device_component) % modulus
        
        return hash_value
    
    def merkle_tree_hash(self, data_list: List[Union[str, bytes]]) -> Dict[str, Any]:
        """Merkle tree hash construction"""
        if not data_list:
            return {'root': '', 'tree': []}
        
        # Convert all data to hash values
        leaf_hashes = []
        for data in data_list:
            leaf_hash = self.device_dependent_hash(data)
            leaf_hashes.append(leaf_hash)
        
        # Build Merkle tree
        tree_levels = [leaf_hashes]
        current_level = leaf_hashes
        
        while len(current_level) > 1:
            next_level = []
            
            # Process pairs
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    # Hash pair
                    combined = current_level[i] + current_level[i + 1]
                    pair_hash = self.sha3_256(combined)
                else:
                    # Odd number of nodes - duplicate last node
                    combined = current_level[i] + current_level[i]
                    pair_hash = self.sha3_256(combined)
                
                next_level.append(pair_hash)
            
            tree_levels.append(next_level)
            current_level = next_level
        
        return {
            'root': current_level[0] if current_level else '',
            'tree': tree_levels,
            'leaf_count': len(data_list)
        }
    
    def verify_merkle_proof(self, data: Union[str, bytes], 
                          proof_path: List[str], 
                          root_hash: str, 
                          leaf_index: int) -> bool:
        """Verify Merkle tree proof"""
        current_hash = self.device_dependent_hash(data)
        
        for i, sibling_hash in enumerate(proof_path):
            if (leaf_index >> i) & 1:
                # Current node is right child
                combined = sibling_hash + current_hash
            else:
                # Current node is left child
                combined = current_hash + sibling_hash
            
            current_hash = self.sha3_256(combined)
        
        return current_hash == root_hash
    
    def commitment_scheme(self, value: Union[str, bytes], 
                         nonce: Union[str, bytes] = None) -> Dict[str, str]:
        """Cryptographic commitment scheme"""
        if isinstance(value, str):
            value = value.encode('utf-8')
        
        if nonce is None:
            nonce = hashlib.sha256(str(time.time()).encode()).digest()
        elif isinstance(nonce, str):
            nonce = nonce.encode('utf-8')
        
        # Create commitment
        commitment_data = value + nonce
        commitment = self.device_dependent_hash(commitment_data)
        
        return {
            'commitment': commitment,
            'nonce': nonce.hex(),
            'algorithm': 'device_dependent_hash'
        }
    
    def verify_commitment(self, value: Union[str, bytes], 
                         nonce: str, 
                         commitment: str) -> bool:
        """Verify cryptographic commitment"""
        if isinstance(value, str):
            value = value.encode('utf-8')
        
        nonce_bytes = bytes.fromhex(nonce)
        commitment_data = value + nonce_bytes
        expected_commitment = self.device_dependent_hash(commitment_data)
        
        return expected_commitment == commitment
    
    def hash_chain(self, seed: Union[str, bytes], length: int) -> List[str]:
        """Generate hash chain for one-time passwords"""
        if isinstance(seed, str):
            seed = seed.encode('utf-8')
        
        chain = []
        current = self.device_dependent_hash(seed)
        
        for i in range(length):
            chain.append(current)
            current = self.sha3_256(current)
        
        return chain
    
    def puzzle_hash(self, data: Union[str, bytes], difficulty: int) -> Dict[str, Any]:
        """Proof-of-work puzzle hash"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        nonce = 0
        target = "0" * difficulty
        
        while True:
            nonce_bytes = nonce.to_bytes(8, byteorder='big')
            puzzle_data = data + nonce_bytes
            hash_result = self.sha3_256(puzzle_data)
            
            if hash_result.startswith(target):
                return {
                    'hash': hash_result,
                    'nonce': nonce,
                    'difficulty': difficulty,
                    'data': data.hex()
                }
            
            nonce += 1
            
            # Prevent infinite loops in testing
            if nonce > 1000000:
                break
        
        return {'error': 'Puzzle solution not found within reasonable iterations'}
    
    def distributed_hash(self, data: Union[str, bytes], 
                        node_count: int) -> Dict[str, List[str]]:
        """Distributed hash across multiple nodes"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        node_hashes = []
        
        for node_id in range(node_count):
            # Create node-specific data
            node_data = data + node_id.to_bytes(4, byteorder='big')
            node_hash = self.device_dependent_hash(node_data)
            node_hashes.append(node_hash)
        
        # Combine all node hashes
        combined_hash = self.sha3_256(''.join(node_hashes))
        
        return {
            'node_hashes': node_hashes,
            'combined_hash': combined_hash,
            'node_count': node_count
        }
    
    def quantum_resistant_hash(self, data: Union[str, bytes]) -> str:
        """Quantum-resistant hash using multiple algorithms"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Use multiple hash algorithms for quantum resistance
        sha3_hash = self.sha3_512(data)
        blake2b_hash = self.blake2b_hash(data, 64)
        
        # Combine with device-specific component
        device_component = self.device_id_transform.to_bytes(32, byteorder='big')
        combined_data = sha3_hash.encode() + blake2b_hash.encode() + device_component
        
        # Final hash
        return self.sha3_256(combined_data)
    
    def get_hash_info(self) -> Dict[str, Any]:
        """Get information about available hash functions"""
        return {
            'device_id_transform': self.device_id_transform,
            'available_algorithms': [
                'sha3_256', 'sha3_512', 'blake2b', 'blake2s',
                'device_dependent', 'cascade', 'hmac',
                'time_dependent', 'polynomial', 'quantum_resistant'
            ],
            'special_functions': [
                'merkle_tree', 'commitment_scheme', 'hash_chain',
                'puzzle_hash', 'distributed_hash'
            ],
            'quantum_resistant': True,
            'device_binding': True
        }


def create_hash_utility(device_id_transform: int):
    """Create hash utilities instance"""
    return HashUtilities(device_id_transform)


if __name__ == "__main__":
    # Test hash utilities
    print("Testing Hash Utilities...")
    
    device_id = 987654321
    hash_util = create_hash_utility(device_id)
    
    test_data = "Hello DIBLE Hash Functions!"
    
    # Test basic hash functions
    sha3_result = hash_util.sha3_256(test_data)
    blake2b_result = hash_util.blake2b_hash(test_data)
    device_hash = hash_util.device_dependent_hash(test_data)
    
    print(f"SHA3-256: {sha3_result[:16]}...")
    print(f"BLAKE2b: {blake2b_result[:16]}...")
    print(f"Device hash: {device_hash[:16]}...")
    
    # Test cascade hash
    cascade_result = hash_util.cascade_hash(test_data, 3)
    print(f"Cascade hash: {cascade_result[:16]}...")
    
    # Test Merkle tree
    data_list = ["data1", "data2", "data3", "data4"]
    merkle_result = hash_util.merkle_tree_hash(data_list)
    print(f"Merkle root: {merkle_result['root'][:16]}...")
    
    # Test commitment scheme
    commitment_result = hash_util.commitment_scheme("secret_value")
    print(f"Commitment: {commitment_result['commitment'][:16]}...")
    
    # Verify commitment
    verification = hash_util.verify_commitment(
        "secret_value", 
        commitment_result['nonce'], 
        commitment_result['commitment']
    )
    print(f"Commitment verification: {'✓' if verification else '✗'}")
    
    # Test quantum-resistant hash
    qr_hash = hash_util.quantum_resistant_hash(test_data)
    print(f"Quantum-resistant hash: {qr_hash[:16]}...")
    
    # Display capabilities
    info = hash_util.get_hash_info()
    print(f"Hash utilities support {len(info['available_algorithms'])} algorithms")
    
    print("Hash Utilities test completed successfully!")
