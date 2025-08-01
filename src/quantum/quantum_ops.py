"""
Quantum-Inspired Operations Module
Implements quantum-inspired operations for DIBLE algorithm
"""

import numpy as np
import cmath
import random
from typing import List, Tuple, Dict, Any, Union, Optional, Complex
import math


class QuantumOperations:
    """Quantum-inspired operations for DIBLE"""
    
    def __init__(self, device_id_transform: int):
        self.device_id_transform = device_id_transform
        self.num_qubits = 8  # Default number of qubits
        random.seed(device_id_transform % (2**32))
        np.random.seed(device_id_transform % (2**32))
    
    def create_qubit_state(self, alpha: complex, beta: complex) -> np.ndarray:
        """Create a qubit state |ψ⟩ = α|0⟩ + β|1⟩"""
        # Normalize the state
        norm = math.sqrt(abs(alpha)**2 + abs(beta)**2)
        if norm == 0:
            raise ValueError("Invalid qubit state: both amplitudes are zero")
        
        alpha_normalized = alpha / norm
        beta_normalized = beta / norm
        
        return np.array([alpha_normalized, beta_normalized], dtype=complex)
    
    def create_bell_state(self, state_type: str = 'phi_plus') -> np.ndarray:
        """Create Bell states (EPR pairs)"""
        sqrt_half = 1 / math.sqrt(2)
        
        if state_type == 'phi_plus':
            # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
            return np.array([sqrt_half, 0, 0, sqrt_half], dtype=complex)
        elif state_type == 'phi_minus':
            # |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
            return np.array([sqrt_half, 0, 0, -sqrt_half], dtype=complex)
        elif state_type == 'psi_plus':
            # |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
            return np.array([0, sqrt_half, sqrt_half, 0], dtype=complex)
        elif state_type == 'psi_minus':
            # |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
            return np.array([0, sqrt_half, -sqrt_half, 0], dtype=complex)
        else:
            raise ValueError(f"Unknown Bell state type: {state_type}")
    
    def quantum_superposition(self, n_qubits: int, device_dependent: bool = True) -> np.ndarray:
        """Create quantum superposition state"""
        n_states = 2**n_qubits
        
        if device_dependent:
            # Create device-dependent amplitudes
            amplitudes = []
            for i in range(n_states):
                # Use device ID to create deterministic but unique amplitudes
                phase_seed = (self.device_id_transform * (i + 1)) % (2**32)
                amplitude = complex(
                    math.cos(phase_seed / 1000.0),
                    math.sin(phase_seed / 1000.0)
                )
                amplitudes.append(amplitude)
        else:
            # Equal superposition (Hadamard on all qubits)
            amplitude = 1 / math.sqrt(n_states)
            amplitudes = [amplitude] * n_states
        
        # Normalize
        amplitudes = np.array(amplitudes)
        norm = np.sqrt(np.sum(np.abs(amplitudes)**2))
        return amplitudes / norm
    
    def apply_pauli_x(self, state: np.ndarray) -> np.ndarray:
        """Apply Pauli-X (NOT) gate"""
        if len(state) != 2:
            raise ValueError("Pauli-X gate requires single qubit state")
        
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        return pauli_x @ state
    
    def apply_pauli_y(self, state: np.ndarray) -> np.ndarray:
        """Apply Pauli-Y gate"""
        if len(state) != 2:
            raise ValueError("Pauli-Y gate requires single qubit state")
        
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        return pauli_y @ state
    
    def apply_pauli_z(self, state: np.ndarray) -> np.ndarray:
        """Apply Pauli-Z gate"""
        if len(state) != 2:
            raise ValueError("Pauli-Z gate requires single qubit state")
        
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        return pauli_z @ state
    
    def apply_hadamard(self, state: np.ndarray) -> np.ndarray:
        """Apply Hadamard gate"""
        if len(state) != 2:
            raise ValueError("Hadamard gate requires single qubit state")
        
        sqrt_half = 1 / math.sqrt(2)
        hadamard = np.array([[sqrt_half, sqrt_half], [sqrt_half, -sqrt_half]], dtype=complex)
        return hadamard @ state
    
    def apply_phase_gate(self, state: np.ndarray, phase: float) -> np.ndarray:
        """Apply phase gate with given phase"""
        if len(state) != 2:
            raise ValueError("Phase gate requires single qubit state")
        
        phase_gate = np.array([[1, 0], [0, cmath.exp(1j * phase)]], dtype=complex)
        return phase_gate @ state
    
    def apply_rotation_x(self, state: np.ndarray, theta: float) -> np.ndarray:
        """Apply rotation around X-axis"""
        if len(state) != 2:
            raise ValueError("Rotation-X gate requires single qubit state")
        
        cos_half = math.cos(theta / 2)
        sin_half = math.sin(theta / 2)
        rx = np.array([[cos_half, -1j * sin_half], [-1j * sin_half, cos_half]], dtype=complex)
        return rx @ state
    
    def apply_rotation_y(self, state: np.ndarray, theta: float) -> np.ndarray:
        """Apply rotation around Y-axis"""
        if len(state) != 2:
            raise ValueError("Rotation-Y gate requires single qubit state")
        
        cos_half = math.cos(theta / 2)
        sin_half = math.sin(theta / 2)
        ry = np.array([[cos_half, -sin_half], [sin_half, cos_half]], dtype=complex)
        return ry @ state
    
    def apply_rotation_z(self, state: np.ndarray, theta: float) -> np.ndarray:
        """Apply rotation around Z-axis"""
        if len(state) != 2:
            raise ValueError("Rotation-Z gate requires single qubit state")
        
        exp_neg = cmath.exp(-1j * theta / 2)
        exp_pos = cmath.exp(1j * theta / 2)
        rz = np.array([[exp_neg, 0], [0, exp_pos]], dtype=complex)
        return rz @ state
    
    def apply_cnot(self, state: np.ndarray, control: int = 0, target: int = 1) -> np.ndarray:
        """Apply CNOT gate to two-qubit state"""
        if len(state) != 4:
            raise ValueError("CNOT gate requires two-qubit state")
        
        if control == 0 and target == 1:
            cnot = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ], dtype=complex)
        elif control == 1 and target == 0:
            cnot = np.array([
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0]
            ], dtype=complex)
        else:
            raise ValueError("Invalid control/target qubit indices")
        
        return cnot @ state
    
    def measure_qubit(self, state: np.ndarray, qubit_index: int = 0) -> Tuple[int, np.ndarray]:
        """Measure a qubit and return result and post-measurement state"""
        if len(state) == 2:
            # Single qubit measurement
            prob_0 = abs(state[0])**2
            prob_1 = abs(state[1])**2
            
            # Random measurement based on probabilities
            measurement = 0 if random.random() < prob_0 else 1
            
            # Post-measurement state
            if measurement == 0:
                post_state = np.array([1, 0], dtype=complex)
            else:
                post_state = np.array([0, 1], dtype=complex)
            
            return measurement, post_state
        
        else:
            # Multi-qubit measurement (simplified)
            n_qubits = int(math.log2(len(state)))
            probabilities = np.abs(state)**2
            
            # Choose measurement outcome based on probabilities
            outcome = np.random.choice(len(state), p=probabilities)
            
            # Create post-measurement state
            post_state = np.zeros_like(state)
            post_state[outcome] = 1
            
            return outcome, post_state
    
    def quantum_fidelity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """Calculate quantum fidelity between two states"""
        if len(state1) != len(state2):
            raise ValueError("States must have the same dimension")
        
        # Fidelity = |⟨ψ₁|ψ₂⟩|²
        inner_product = np.dot(np.conj(state1), state2)
        return abs(inner_product)**2
    
    def quantum_entropy(self, state: np.ndarray) -> float:
        """Calculate von Neumann entropy of quantum state"""
        # For pure states, entropy is 0
        # For mixed states, we'd need density matrix
        
        # Calculate entropy based on amplitudes
        probabilities = np.abs(state)**2
        entropy = 0
        
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def quantum_teleportation_protocol(self, state_to_teleport: np.ndarray) -> Dict[str, Any]:
        """Simulate quantum teleportation protocol"""
        if len(state_to_teleport) != 2:
            raise ValueError("Can only teleport single qubit states")
        
        # Create Bell pair (entangled qubits)
        bell_state = self.create_bell_state('phi_plus')
        
        # Alice has the state to teleport and one qubit of the Bell pair
        # Bob has the other qubit of the Bell pair
        
        # Alice performs Bell measurement (simplified simulation)
        measurement_results = []
        
        # First measurement (X basis)
        x_result = random.choice([0, 1])
        measurement_results.append(x_result)
        
        # Second measurement (Z basis)
        z_result = random.choice([0, 1])
        measurement_results.append(z_result)
        
        # Bob applies corrections based on Alice's measurements
        teleported_state = state_to_teleport.copy()
        
        if x_result == 1:
            teleported_state = self.apply_pauli_z(teleported_state)
        if z_result == 1:
            teleported_state = self.apply_pauli_x(teleported_state)
        
        return {
            'original_state': state_to_teleport,
            'teleported_state': teleported_state,
            'measurement_results': measurement_results,
            'fidelity': self.quantum_fidelity(state_to_teleport, teleported_state)
        }
    
    def quantum_key_distribution_bb84(self, key_length: int) -> Dict[str, Any]:
        """Simulate BB84 quantum key distribution protocol"""
        # Alice's random bits and bases
        alice_bits = [random.randint(0, 1) for _ in range(key_length * 2)]
        alice_bases = [random.randint(0, 1) for _ in range(key_length * 2)]  # 0: Z, 1: X
        
        # Alice prepares qubits
        alice_qubits = []
        for bit, basis in zip(alice_bits, alice_bases):
            if basis == 0:  # Z basis
                if bit == 0:
                    qubit = np.array([1, 0], dtype=complex)  # |0⟩
                else:
                    qubit = np.array([0, 1], dtype=complex)  # |1⟩
            else:  # X basis
                if bit == 0:
                    qubit = np.array([1, 1], dtype=complex) / math.sqrt(2)  # |+⟩
                else:
                    qubit = np.array([1, -1], dtype=complex) / math.sqrt(2)  # |-⟩
            alice_qubits.append(qubit)
        
        # Bob's random measurement bases
        bob_bases = [random.randint(0, 1) for _ in range(key_length * 2)]
        
        # Bob measures qubits
        bob_bits = []
        for qubit, basis in zip(alice_qubits, bob_bases):
            if basis == 0:  # Z basis measurement
                prob_0 = abs(qubit[0])**2
                bit = 0 if random.random() < prob_0 else 1
            else:  # X basis measurement
                # Transform to X basis first
                h_qubit = self.apply_hadamard(qubit)
                prob_0 = abs(h_qubit[0])**2
                bit = 0 if random.random() < prob_0 else 1
            bob_bits.append(bit)
        
        # Sifting: keep bits where bases match
        shared_key = []
        for i in range(len(alice_bits)):
            if alice_bases[i] == bob_bases[i]:
                shared_key.append(alice_bits[i])
                if len(shared_key) >= key_length:
                    break
        
        return {
            'alice_bits': alice_bits[:len(shared_key)],
            'bob_bits': [bob_bits[i] for i in range(len(alice_bits)) if alice_bases[i] == bob_bases[i]][:len(shared_key)],
            'shared_key': shared_key,
            'key_length': len(shared_key),
            'efficiency': len(shared_key) / len(alice_bits)
        }
    
    def quantum_random_walk(self, steps: int, dimensions: int = 1) -> List[int]:
        """Simulate quantum random walk in multiple dimensions"""
        if dimensions == 1:
            # 1D quantum random walk
            position = 0
            positions = [position]
            
            # Quantum coin (Hadamard)
            sqrt_half = 1 / math.sqrt(2)
            
            for _ in range(steps):
                # Flip quantum coin
                coin_result = random.choice([0, 1])  # Simplified measurement
                
                # Move based on coin result
                if coin_result == 0:
                    position -= 1
                else:
                    position += 1
                
                positions.append(position)
            
            return positions
        elif dimensions == 2:
            # 2D quantum random walk
            position = [0, 0]
            positions = [position.copy()]
            
            for _ in range(steps):
                # Choose direction (up, down, left, right)
                direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                position[0] += direction[0]
                position[1] += direction[1]
                positions.append(position.copy())
            
            return positions
        elif dimensions == 3:
            # 3D quantum random walk
            position = [0, 0, 0]
            positions = [position.copy()]
            
            for _ in range(steps):
                # Choose direction in 3D space
                direction = random.choice([
                    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
                ])
                position[0] += direction[0]
                position[1] += direction[1]
                position[2] += direction[2]
                positions.append(position.copy())
            
            return positions
        else:
            # Higher dimensional quantum random walk
            position = [0] * dimensions
            positions = [position.copy()]
            
            for _ in range(steps):
                # Choose random dimension and direction
                dim = random.randint(0, dimensions - 1)
                direction = random.choice([-1, 1])
                position[dim] += direction
                positions.append(position.copy())
            
            return positions
    
    def quantum_fourier_transform(self, state: np.ndarray) -> np.ndarray:
        """Apply Quantum Fourier Transform"""
        n = len(state)
        if n & (n - 1) != 0:  # Check if n is power of 2
            raise ValueError("State length must be a power of 2")
        
        n_qubits = int(math.log2(n))
        qft_matrix = np.zeros((n, n), dtype=complex)
        
        # Construct QFT matrix
        omega = cmath.exp(2j * math.pi / n)
        for i in range(n):
            for j in range(n):
                qft_matrix[i, j] = omega**(i * j) / math.sqrt(n)
        
        return qft_matrix @ state
    
    def grover_oracle(self, state: np.ndarray, marked_items: List[int]) -> np.ndarray:
        """Apply Grover's oracle (phase flip marked items)"""
        oracle_state = state.copy()
        for item in marked_items:
            if 0 <= item < len(oracle_state):
                oracle_state[item] *= -1
        return oracle_state
    
    def grover_diffuser(self, state: np.ndarray) -> np.ndarray:
        """Apply Grover's diffusion operator"""
        n = len(state)
        uniform_state = np.ones(n, dtype=complex) / math.sqrt(n)
        
        # 2|s⟩⟨s| - I where |s⟩ is uniform superposition
        diffuser = 2 * np.outer(uniform_state, np.conj(uniform_state)) - np.eye(n)
        
        return diffuser @ state
    
    def grover_algorithm(self, n_items: int, marked_items: List[int], iterations: int = None) -> Dict[str, Any]:
        """Simulate Grover's search algorithm"""
        if iterations is None:
            # Optimal number of iterations
            iterations = int(math.pi * math.sqrt(n_items) / 4)
        
        # Initialize uniform superposition
        state = np.ones(n_items, dtype=complex) / math.sqrt(n_items)
        
        # Apply Grover iterations
        for _ in range(iterations):
            # Apply oracle
            state = self.grover_oracle(state, marked_items)
            
            # Apply diffuser
            state = self.grover_diffuser(state)
        
        # Measure to find most likely outcomes
        probabilities = np.abs(state)**2
        max_prob_index = np.argmax(probabilities)
        
        return {
            'final_state': state,
            'probabilities': probabilities,
            'most_likely_item': max_prob_index,
            'success_probability': probabilities[max_prob_index] if max_prob_index in marked_items else 0,
            'iterations_used': iterations
        }
    
    def device_dependent_quantum_state(self, n_qubits: int) -> np.ndarray:
        """Create device-dependent quantum state"""
        n_states = 2**n_qubits
        amplitudes = []
        
        for i in range(n_states):
            # Use device ID to create unique phase and amplitude
            phase_seed = (self.device_id_transform * (i + 1)) % 1000
            amplitude_seed = (self.device_id_transform * (i + 1) * 17) % 1000
            
            phase = 2 * math.pi * phase_seed / 1000.0
            amplitude = math.sqrt(amplitude_seed / 1000.0)
            
            complex_amplitude = amplitude * cmath.exp(1j * phase)
            amplitudes.append(complex_amplitude)
        
        # Normalize
        amplitudes = np.array(amplitudes)
        norm = np.sqrt(np.sum(np.abs(amplitudes)**2))
        return amplitudes / norm
    
    def quantum_entanglement_measure(self, state: np.ndarray) -> float:
        """Measure quantum entanglement (simplified)"""
        if len(state) != 4:  # Two-qubit state
            return 0.0
        
        # Reshape to 2x2 matrix (partial trace approach)
        state_matrix = state.reshape(2, 2)
        
        # Calculate reduced density matrix for first qubit
        rho_a = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                rho_a[i, j] = np.sum([state_matrix[i, k] * np.conj(state_matrix[j, k]) for k in range(2)])
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(rho_a)
        
        # Von Neumann entropy as entanglement measure
        entropy = 0
        for lam in eigenvalues:
            if lam.real > 1e-10:
                entropy -= lam.real * math.log2(lam.real)
        
        return entropy
    
    def get_quantum_capabilities(self) -> Dict[str, Any]:
        """Get information about quantum operation capabilities"""
        return {
            'device_id_transform': self.device_id_transform,
            'single_qubit_gates': [
                'pauli_x', 'pauli_y', 'pauli_z', 'hadamard',
                'phase_gate', 'rotation_x', 'rotation_y', 'rotation_z'
            ],
            'multi_qubit_gates': [
                'cnot'
            ],
            'quantum_states': [
                'qubit_state', 'bell_states', 'quantum_superposition',
                'device_dependent_quantum_state'
            ],
            'quantum_algorithms': [
                'quantum_teleportation', 'bb84_qkd', 'quantum_random_walk',
                'quantum_fourier_transform', 'grover_algorithm'
            ],
            'quantum_measures': [
                'quantum_fidelity', 'quantum_entropy', 'quantum_entanglement_measure'
            ],
            'max_qubits': 16,  # Practical limit for simulation
            'device_dependent': True
        }


def create_quantum_operations(device_id_transform: int):
    """Create quantum operations instance"""
    return QuantumOperations(device_id_transform)


if __name__ == "__main__":
    # Test quantum operations
    print("Testing Quantum Operations...")
    
    device_id = 987654321
    quantum_ops = create_quantum_operations(device_id)
    
    # Test qubit creation
    qubit = quantum_ops.create_qubit_state(1+0j, 0+0j)  # |0⟩ state
    print(f"Created qubit state: {qubit}")
    
    # Test Hadamard gate
    superposition = quantum_ops.apply_hadamard(qubit)
    print(f"After Hadamard: {superposition}")
    
    # Test Bell state
    bell_state = quantum_ops.create_bell_state('phi_plus')
    print(f"Bell state |Φ⁺⟩: {bell_state}")
    
    # Test measurement
    measurement, post_state = quantum_ops.measure_qubit(superposition)
    print(f"Measurement result: {measurement}, post-state: {post_state}")
    
    # Test quantum fidelity
    fidelity = quantum_ops.quantum_fidelity(qubit, post_state)
    print(f"Quantum fidelity: {fidelity:.3f}")
    
    # Test quantum entropy
    entropy = quantum_ops.quantum_entropy(superposition)
    print(f"Quantum entropy: {entropy:.3f}")
    
    # Test BB84 protocol
    bb84_result = quantum_ops.quantum_key_distribution_bb84(10)
    print(f"✓ BB84 QKD completed: shared key length = {bb84_result['key_length']}")
    
    # Test Grover's algorithm
    grover_result = quantum_ops.grover_algorithm(8, [3, 5])
    print(f"✓ Grover's algorithm: most likely item = {grover_result['most_likely_item']}")
    
    # Test device-dependent quantum state
    device_state = quantum_ops.device_dependent_quantum_state(3)
    print(f"Device-dependent state (8 amplitudes): norm = {np.linalg.norm(device_state):.3f}")
    
    # Test quantum random walk
    walk_positions = quantum_ops.quantum_random_walk(10)
    print(f"Quantum walk final position: {walk_positions[-1]}")
    
    # Display capabilities
    capabilities = quantum_ops.get_quantum_capabilities()
    total_features = (len(capabilities['single_qubit_gates']) + 
                     len(capabilities['quantum_algorithms']) + 
                     len(capabilities['quantum_measures']))
    print(f"Quantum operations support {total_features} features")
    
    print("Quantum Operations test completed successfully!")
