"""
Chaos Theory Integration Module
Implements chaotic functions and non-linear dynamical systems
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Any
import random


class ChaosTheoryManager:
    """Manage chaos theory operations for DIBLE algorithm"""
    
    def __init__(self, device_id_transform: int):
        self.device_id_transform = device_id_transform
        self.chaos_state = self._initialize_chaos_state()
        
    def _initialize_chaos_state(self) -> Dict[str, Any]:
        """Initialize chaos state based on device ID"""
        # Use device ID to seed chaos parameters
        seed = self.device_id_transform % (2**32)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        
        return {
            'logistic_x': (self.device_id_transform % 1000) / 1000.0,
            'logistic_r': 3.7 + (self.device_id_transform % 100) / 1000.0,
            'henon_state': [0.1, 0.1],
            'henon_a': 1.4,
            'henon_b': 0.3,
            'lorenz_state': [1.0, 1.0, 1.0],
            'lorenz_sigma': 10.0,
            'lorenz_rho': 28.0,
            'lorenz_beta': 8.0/3.0,
            'chua_state': [0.1, 0.1, 0.1],
            'chua_alpha': 15.6,
            'chua_beta': 28.0,
            'chua_m0': -1.143,
            'chua_m1': -0.714,
            'rossler_state': [1.0, 1.0, 1.0],
            'rossler_a': 0.2,
            'rossler_b': 0.2,
            'rossler_c': 5.7
        }
    
    def logistic_map(self, x: float = None, r: float = None) -> float:
        """Chaotic logistic map: x_{n+1} = r * x_n * (1 - x_n)"""
        if x is None:
            x = self.chaos_state['logistic_x']
        if r is None:
            r = self.chaos_state['logistic_r']
            
        next_x = r * x * (1 - x)
        self.chaos_state['logistic_x'] = next_x
        return next_x
    
    def tent_map(self, x: float, mu: float = 2.0) -> float:
        """Tent map chaotic function"""
        if x < 0.5:
            return mu * x
        else:
            return mu * (1 - x)
    
    def henon_map(self, state: List[float] = None) -> List[float]:
        """Hénon map chaotic system"""
        if state is None:
            state = self.chaos_state['henon_state']
            
        x, y = state
        a = self.chaos_state['henon_a']
        b = self.chaos_state['henon_b']
        
        x_next = 1 - a * x**2 + y
        y_next = b * x
        
        next_state = [x_next, y_next]
        self.chaos_state['henon_state'] = next_state
        return next_state
    
    def lorenz_system(self, state: List[float] = None, dt: float = 0.01) -> List[float]:
        """Lorenz attractor chaotic system"""
        if state is None:
            state = self.chaos_state['lorenz_state']
            
        x, y, z = state
        sigma = self.chaos_state['lorenz_sigma']
        rho = self.chaos_state['lorenz_rho']
        beta = self.chaos_state['lorenz_beta']
        
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        
        next_state = [x + dx * dt, y + dy * dt, z + dz * dt]
        self.chaos_state['lorenz_state'] = next_state
        return next_state
    
    def chua_circuit(self, state: List[float] = None, dt: float = 0.01) -> List[float]:
        """Chua's circuit chaotic system"""
        if state is None:
            state = self.chaos_state['chua_state']
            
        x, y, z = state
        alpha = self.chaos_state['chua_alpha']
        beta = self.chaos_state['chua_beta']
        m0 = self.chaos_state['chua_m0']
        m1 = self.chaos_state['chua_m1']
        
        # Chua's diode function
        if abs(x) <= 1:
            f_x = m1 * x
        else:
            f_x = m0 * x + (m1 - m0) * (abs(x) - 1) * (1 if x > 0 else -1)
        
        dx = alpha * (y - x - f_x)
        dy = x - y + z
        dz = -beta * y
        
        next_state = [x + dx * dt, y + dy * dt, z + dz * dt]
        self.chaos_state['chua_state'] = next_state
        return next_state
    
    def rossler_attractor(self, state: List[float] = None, dt: float = 0.01) -> List[float]:
        """Rössler attractor chaotic system"""
        if state is None:
            state = self.chaos_state['rossler_state']
            
        x, y, z = state
        a = self.chaos_state['rossler_a']
        b = self.chaos_state['rossler_b']
        c = self.chaos_state['rossler_c']
        
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        
        next_state = [x + dx * dt, y + dy * dt, z + dz * dt]
        self.chaos_state['rossler_state'] = next_state
        return next_state
    
    def ikeda_map(self, state: List[float], u: float = 0.9) -> List[float]:
        """Ikeda map chaotic system"""
        x, y = state
        
        t = 0.4 - 6 / (1 + x**2 + y**2)
        x_next = 1 + u * (x * math.cos(t) - y * math.sin(t))
        y_next = u * (x * math.sin(t) + y * math.cos(t))
        
        return [x_next, y_next]
    
    def arnold_cat_map(self, state: List[float]) -> List[float]:
        """Arnold's cat map"""
        x, y = state
        
        x_next = (2 * x + y) % 1
        y_next = (x + y) % 1
        
        return [x_next, y_next]
    
    def generate_chaotic_sequence(self, length: int, system: str = 'logistic') -> List[float]:
        """Generate chaotic sequence using specified system"""
        sequence = []
        
        if system == 'logistic':
            for _ in range(length):
                sequence.append(self.logistic_map())
                
        elif system == 'henon':
            for _ in range(length):
                state = self.henon_map()
                sequence.append(state[0])  # Use x component
                
        elif system == 'lorenz':
            for _ in range(length):
                state = self.lorenz_system()
                sequence.append(state[0])  # Use x component
                
        elif system == 'chua':
            for _ in range(length):
                state = self.chua_circuit()
                sequence.append(state[0])  # Use x component
                
        elif system == 'rossler':
            for _ in range(length):
                state = self.rossler_attractor()
                sequence.append(state[0])  # Use x component
                
        elif system == 'tent':
            x = (self.device_id_transform % 1000) / 1000.0
            for _ in range(length):
                x = self.tent_map(x)
                sequence.append(x)
                
        elif system == 'ikeda':
            state = [0.1, 0.1]
            for _ in range(length):
                state = self.ikeda_map(state)
                sequence.append(state[0])
                
        elif system == 'arnold':
            state = [0.1, 0.2]
            for _ in range(length):
                state = self.arnold_cat_map(state)
                sequence.append(state[0])
                
        return sequence
    
    def chaotic_pseudo_random_generator(self, seed: int, length: int) -> List[int]:
        """Generate pseudo-random sequence using chaos"""
        # Use multiple chaotic systems
        systems = ['logistic', 'henon', 'lorenz', 'tent']
        random.seed(seed)
        
        all_sequences = []
        for system in systems:
            seq = self.generate_chaotic_sequence(length // len(systems), system)
            all_sequences.extend(seq)
        
        # Convert to integers
        int_sequence = []
        for value in all_sequences[:length]:
            int_value = int(abs(value) * 2**32) % 256
            int_sequence.append(int_value)
            
        return int_sequence
    
    def chaotic_noise_generation(self, shape: Tuple[int, ...], system: str = 'lorenz') -> np.ndarray:
        """Generate chaotic noise array"""
        total_elements = np.prod(shape)
        noise_sequence = self.generate_chaotic_sequence(total_elements, system)
        
        # Reshape to desired shape
        noise_array = np.array(noise_sequence).reshape(shape)
        
        # Normalize to [-1, 1]
        noise_array = 2 * (noise_array - np.min(noise_array)) / (np.max(noise_array) - np.min(noise_array)) - 1
        
        return noise_array
    
    def lyapunov_exponent(self, system: str, n_iterations: int = 1000) -> float:
        """Calculate largest Lyapunov exponent"""
        if system == 'logistic':
            x = self.chaos_state['logistic_x']
            r = self.chaos_state['logistic_r']
            
            sum_log = 0
            for _ in range(n_iterations):
                x = self.logistic_map(x, r)
                derivative = abs(r * (1 - 2 * x))
                if derivative > 0:
                    sum_log += math.log(derivative)
                    
            return sum_log / n_iterations
        
        # For other systems, use numerical approximation
        return self._numerical_lyapunov(system, n_iterations)
    
    def _numerical_lyapunov(self, system: str, n_iterations: int) -> float:
        """Numerical approximation of Lyapunov exponent"""
        epsilon = 1e-8
        
        # Get initial state
        if system == 'henon':
            state1 = self.chaos_state['henon_state'].copy()
            state2 = [state1[0] + epsilon, state1[1]]
        elif system == 'lorenz':
            state1 = self.chaos_state['lorenz_state'].copy()
            state2 = [state1[0] + epsilon, state1[1], state1[2]]
        else:
            return 0.0  # Default for unsupported systems
        
        sum_log = 0
        
        for _ in range(n_iterations):
            # Evolve both states
            if system == 'henon':
                state1 = self.henon_map(state1)
                state2 = self.henon_map(state2)
            elif system == 'lorenz':
                state1 = self.lorenz_system(state1)
                state2 = self.lorenz_system(state2)
            
            # Calculate separation
            separation = math.sqrt(sum((s1 - s2)**2 for s1, s2 in zip(state1, state2)))
            
            if separation > 0:
                sum_log += math.log(separation / epsilon)
                
                # Renormalize
                factor = epsilon / separation
                state2 = [state1[i] + factor * (state2[i] - state1[i]) for i in range(len(state1))]
        
        return sum_log / n_iterations
    
    def fractal_dimension(self, sequence: List[float], max_r: float = 1.0, n_points: int = 50) -> float:
        """Calculate fractal dimension using box-counting method"""
        if len(sequence) < 2:
            return 1.0
            
        # Embed sequence in 2D space
        embedded = [(sequence[i], sequence[i+1]) for i in range(len(sequence)-1)]
        
        r_values = np.logspace(-3, math.log10(max_r), n_points)
        counts = []
        
        for r in r_values:
            # Count number of boxes needed
            boxes = set()
            for x, y in embedded:
                box_x = int(x / r)
                box_y = int(y / r)
                boxes.add((box_x, box_y))
            counts.append(len(boxes))
        
        # Calculate fractal dimension as slope of log-log plot
        log_r = np.log(r_values)
        log_counts = np.log(counts)
        
        # Linear regression
        slope = np.polyfit(log_r, log_counts, 1)[0]
        return -slope
    
    def correlation_dimension(self, sequence: List[float], max_r: float = 1.0) -> float:
        """Calculate correlation dimension"""
        if len(sequence) < 10:
            return 1.0
            
        # Embed in higher dimension
        embedding_dim = 3
        embedded = []
        for i in range(len(sequence) - embedding_dim + 1):
            embedded.append([sequence[i + j] for j in range(embedding_dim)])
        
        n = len(embedded)
        if n < 2:
            return 1.0
        
        # Calculate pairwise distances
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = math.sqrt(sum((embedded[i][k] - embedded[j][k])**2 for k in range(embedding_dim)))
                distances.append(dist)
        
        distances.sort()
        
        # Calculate correlation sum for different radii
        r_values = np.linspace(0.1 * max_r, max_r, 20)
        correlations = []
        
        for r in r_values:
            count = sum(1 for d in distances if d < r)
            correlation = count / len(distances)
            correlations.append(correlation + 1e-10)  # Avoid log(0)
        
        # Calculate correlation dimension
        log_r = np.log(r_values)
        log_corr = np.log(correlations)
        
        slope = np.polyfit(log_r, log_corr, 1)[0]
        return slope
    
    def chaotic_key_expansion(self, seed_key: bytes, target_length: int) -> bytes:
        """Expand key using chaotic systems"""
        # Initialize chaos with seed
        seed_int = int.from_bytes(seed_key, byteorder='big') % (2**32)
        
        # Generate chaotic sequences
        logistic_seq = self.generate_chaotic_sequence(target_length // 4, 'logistic')
        henon_seq = self.generate_chaotic_sequence(target_length // 4, 'henon')
        lorenz_seq = self.generate_chaotic_sequence(target_length // 4, 'lorenz')
        tent_seq = self.generate_chaotic_sequence(target_length // 4, 'tent')
        
        # Combine sequences
        combined_seq = logistic_seq + henon_seq + lorenz_seq + tent_seq
        
        # Pad if necessary
        while len(combined_seq) < target_length:
            combined_seq.extend(self.generate_chaotic_sequence(target_length - len(combined_seq), 'logistic'))
        
        # Convert to bytes
        expanded_key = bytearray()
        for i in range(target_length):
            byte_value = int(abs(combined_seq[i]) * 255) % 256
            expanded_key.append(byte_value)
        
        return bytes(expanded_key)
    
    def device_dependent_chaos(self, input_data: bytes) -> bytes:
        """Apply device-dependent chaotic transformation"""
        # Use device ID to modify chaos parameters
        device_factor = (self.device_id_transform % 1000) / 1000.0
        
        # Modify chaos parameters based on device
        self.chaos_state['logistic_r'] += device_factor * 0.1
        self.chaos_state['henon_a'] += device_factor * 0.1
        self.chaos_state['lorenz_sigma'] += device_factor
        
        # Generate chaotic transformation
        chaos_seq = self.generate_chaotic_sequence(len(input_data), 'logistic')
        
        # Apply transformation
        transformed_data = bytearray()
        for i, byte in enumerate(input_data):
            chaos_byte = int(abs(chaos_seq[i]) * 255) % 256
            transformed_byte = byte ^ chaos_byte
            transformed_data.append(transformed_byte)
        
        return bytes(transformed_data)


def create_chaos_manager(device_id_transform: int):
    """Create chaos theory manager instance"""
    return ChaosTheoryManager(device_id_transform)


if __name__ == "__main__":
    # Test chaos theory manager
    device_id = 98765
    chaos_manager = create_chaos_manager(device_id)
    
    # Test chaotic sequence generation
    logistic_seq = chaos_manager.generate_chaotic_sequence(100, 'logistic')
    henon_seq = chaos_manager.generate_chaotic_sequence(100, 'henon')
    lorenz_seq = chaos_manager.generate_chaotic_sequence(100, 'lorenz')
    
    print(f"Logistic sequence sample: {logistic_seq[:5]}")
    print(f"Henon sequence sample: {henon_seq[:5]}")
    print(f"Lorenz sequence sample: {lorenz_seq[:5]}")
    
    # Test Lyapunov exponent
    lyap_exp = chaos_manager.lyapunov_exponent('logistic')
    print(f"Logistic map Lyapunov exponent: {lyap_exp}")
    
    # Test fractal dimension
    fractal_dim = chaos_manager.fractal_dimension(logistic_seq)
    print(f"Logistic sequence fractal dimension: {fractal_dim}")
    
    # Test chaotic key expansion
    seed_key = b"test_seed_key"
    expanded_key = chaos_manager.chaotic_key_expansion(seed_key, 256)
    print(f"Expanded key length: {len(expanded_key)}")
