"""
Mathematical Utilities
Advanced mathematical functions for DIBLE algorithm
"""

import math
import random
from typing import List, Tuple, Dict, Any, Union, Optional
import numpy as np
from scipy.special import gamma, factorial
from scipy.stats import norm
import sympy as sp


class MathematicalUtilities:
    """Advanced mathematical utilities for DIBLE"""
    
    def __init__(self, device_id_transform: int):
        self.device_id_transform = device_id_transform
        random.seed(device_id_transform % (2**32))
        np.random.seed(device_id_transform % (2**32))
    
    def modular_exponentiation(self, base: int, exponent: int, modulus: int) -> int:
        """Fast modular exponentiation using binary method"""
        if modulus == 1:
            return 0
        
        result = 1
        base = base % modulus
        
        while exponent > 0:
            if exponent % 2 == 1:
                result = (result * base) % modulus
            exponent = exponent >> 1
            base = (base * base) % modulus
        
        return result
    
    def extended_euclidean(self, a: int, b: int) -> Tuple[int, int, int]:
        """Extended Euclidean algorithm"""
        if a == 0:
            return b, 0, 1
        
        gcd, x1, y1 = self.extended_euclidean(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        
        return gcd, x, y
    
    def modular_inverse(self, a: int, m: int) -> Optional[int]:
        """Compute modular inverse using extended Euclidean algorithm"""
        gcd, x, _ = self.extended_euclidean(a, m)
        
        if gcd != 1:
            return None  # Modular inverse doesn't exist
        
        return (x % m + m) % m
    
    def chinese_remainder_theorem(self, remainders: List[int], moduli: List[int]) -> int:
        """Solve system of congruences using Chinese Remainder Theorem"""
        if len(remainders) != len(moduli):
            raise ValueError("Number of remainders and moduli must be equal")
        
        total = 0
        prod = 1
        
        for m in moduli:
            prod *= m
        
        for r, m in zip(remainders, moduli):
            p = prod // m
            total += r * self.modular_inverse(p, m) * p
        
        return total % prod
    
    def miller_rabin_primality(self, n: int, k: int = 5) -> bool:
        """Miller-Rabin primality test"""
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        
        # Write n-1 as d * 2^r
        r = 0
        d = n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        
        # Perform k rounds of testing
        for _ in range(k):
            a = random.randrange(2, n - 1)
            x = self.modular_exponentiation(a, d, n)
            
            if x == 1 or x == n - 1:
                continue
            
            for _ in range(r - 1):
                x = self.modular_exponentiation(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        
        return True
    
    def generate_prime(self, bits: int) -> int:
        """Generate a prime number with specified bit length"""
        while True:
            candidate = random.getrandbits(bits)
            candidate |= (1 << bits - 1) | 1  # Set MSB and LSB
            
            if self.miller_rabin_primality(candidate):
                return candidate
    
    def discrete_gaussian_sample(self, sigma: float, center: float = 0.0) -> int:
        """Sample from discrete Gaussian distribution"""
        # Use Box-Muller transform for continuous Gaussian
        u1 = random.random()
        u2 = random.random()
        
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        continuous_sample = center + sigma * z
        
        # Round to nearest integer
        discrete_sample = round(continuous_sample)
        
        return discrete_sample
    
    def lattice_gram_schmidt(self, basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Gram-Schmidt orthogonalization for lattice basis"""
        n, m = basis.shape
        orthogonal_basis = np.zeros_like(basis, dtype=float)
        mu = np.zeros((n, n))
        
        for i in range(n):
            orthogonal_basis[i] = basis[i].astype(float)
            
            for j in range(i):
                if np.dot(orthogonal_basis[j], orthogonal_basis[j]) > 0:
                    mu[i, j] = (np.dot(basis[i], orthogonal_basis[j]) / 
                               np.dot(orthogonal_basis[j], orthogonal_basis[j]))
                    orthogonal_basis[i] -= mu[i, j] * orthogonal_basis[j]
        
        return orthogonal_basis, mu
    
    def babai_nearest_plane(self, target: np.ndarray, basis: np.ndarray) -> np.ndarray:
        """Babai's nearest plane algorithm for CVP"""
        orthogonal_basis, mu = self.lattice_gram_schmidt(basis)
        n = len(basis)
        
        coefficients = np.zeros(n)
        target_float = target.astype(float)
        
        for i in range(n - 1, -1, -1):
            if np.dot(orthogonal_basis[i], orthogonal_basis[i]) > 0:
                projection = (np.dot(target_float, orthogonal_basis[i]) / 
                            np.dot(orthogonal_basis[i], orthogonal_basis[i]))
                
                for j in range(i + 1, n):
                    projection -= coefficients[j] * mu[j, i]
                
                coefficients[i] = round(projection)
        
        # Compute closest vector
        closest_vector = np.zeros_like(target)
        for i in range(n):
            closest_vector += coefficients[i] * basis[i]
        
        return closest_vector.astype(int)
    
    def polynomial_gcd(self, poly1: List[int], poly2: List[int], modulus: int) -> List[int]:
        """Compute GCD of two polynomials over finite field"""
        def poly_mod(poly, mod):
            return [coeff % mod for coeff in poly]
        
        def poly_degree(poly):
            for i in range(len(poly) - 1, -1, -1):
                if poly[i] != 0:
                    return i
            return -1
        
        def poly_divide(dividend, divisor, mod):
            if poly_degree(divisor) == -1:
                raise ValueError("Division by zero polynomial")
            
            quotient = [0] * max(1, len(dividend) - len(divisor) + 1)
            remainder = dividend[:]
            
            while poly_degree(remainder) >= poly_degree(divisor):
                lead_coeff = remainder[poly_degree(remainder)]
                divisor_lead = divisor[poly_degree(divisor)]
                
                # Find modular inverse
                inv = self.modular_inverse(divisor_lead, mod)
                if inv is None:
                    break
                
                coeff = (lead_coeff * inv) % mod
                degree_diff = poly_degree(remainder) - poly_degree(divisor)
                
                quotient[degree_diff] = coeff
                
                # Subtract divisor * coeff * x^degree_diff from remainder
                for i in range(len(divisor)):
                    if i + degree_diff < len(remainder):
                        remainder[i + degree_diff] = (remainder[i + degree_diff] - 
                                                    coeff * divisor[i]) % mod
            
            return quotient, remainder
        
        # Euclidean algorithm for polynomials
        a = poly_mod(poly1, modulus)
        b = poly_mod(poly2, modulus)
        
        while poly_degree(b) >= 0:
            _, remainder = poly_divide(a, b, modulus)
            a, b = b, remainder
        
        # Normalize the result
        if poly_degree(a) >= 0:
            lead_coeff = a[poly_degree(a)]
            inv = self.modular_inverse(lead_coeff, modulus)
            if inv is not None:
                a = [(coeff * inv) % modulus for coeff in a]
        
        return a
    
    def matrix_determinant(self, matrix: np.ndarray, modulus: int) -> int:
        """Compute determinant of matrix over finite field"""
        n = matrix.shape[0]
        if matrix.shape[1] != n:
            raise ValueError("Matrix must be square")
        
        # Make a copy to avoid modifying original
        mat = matrix.copy() % modulus
        det = 1
        
        for i in range(n):
            # Find pivot
            pivot_row = i
            for j in range(i + 1, n):
                if mat[j, i] != 0:
                    pivot_row = j
                    break
            
            if mat[pivot_row, i] == 0:
                return 0  # Singular matrix
            
            if pivot_row != i:
                # Swap rows
                mat[[i, pivot_row]] = mat[[pivot_row, i]]
                det = (-det) % modulus
            
            # Update determinant
            det = (det * mat[i, i]) % modulus
            
            # Eliminate column
            inv_pivot = self.modular_inverse(mat[i, i], modulus)
            if inv_pivot is None:
                return 0
            
            for j in range(i + 1, n):
                if mat[j, i] != 0:
                    factor = (mat[j, i] * inv_pivot) % modulus
                    for k in range(i, n):
                        mat[j, k] = (mat[j, k] - factor * mat[i, k]) % modulus
        
        return det % modulus
    
    def matrix_inverse(self, matrix: np.ndarray, modulus: int) -> Optional[np.ndarray]:
        """Compute matrix inverse over finite field"""
        n = matrix.shape[0]
        if matrix.shape[1] != n:
            raise ValueError("Matrix must be square")
        
        # Create augmented matrix [A | I]
        augmented = np.zeros((n, 2 * n), dtype=int)
        augmented[:, :n] = matrix % modulus
        augmented[:, n:] = np.eye(n, dtype=int)
        
        # Gaussian elimination
        for i in range(n):
            # Find pivot
            pivot_row = i
            for j in range(i + 1, n):
                if augmented[j, i] != 0:
                    pivot_row = j
                    break
            
            if augmented[pivot_row, i] == 0:
                return None  # Singular matrix
            
            if pivot_row != i:
                # Swap rows
                augmented[[i, pivot_row]] = augmented[[pivot_row, i]]
            
            # Make diagonal element 1
            inv_pivot = self.modular_inverse(augmented[i, i], modulus)
            if inv_pivot is None:
                return None
            
            for j in range(2 * n):
                augmented[i, j] = (augmented[i, j] * inv_pivot) % modulus
            
            # Eliminate column
            for j in range(n):
                if j != i and augmented[j, i] != 0:
                    factor = augmented[j, i]
                    for k in range(2 * n):
                        augmented[j, k] = (augmented[j, k] - factor * augmented[i, k]) % modulus
        
        return augmented[:, n:] % modulus
    
    def quadratic_residue(self, a: int, p: int) -> bool:
        """Check if a is a quadratic residue modulo p"""
        if a % p == 0:
            return True
        
        # Use Euler's criterion: a^((p-1)/2) ≡ 1 (mod p)
        return self.modular_exponentiation(a, (p - 1) // 2, p) == 1
    
    def tonelli_shanks(self, a: int, p: int) -> Optional[int]:
        """Find square root modulo prime using Tonelli-Shanks algorithm"""
        if not self.quadratic_residue(a, p):
            return None
        
        if a % p == 0:
            return 0
        
        if p % 4 == 3:
            # Simple case
            return self.modular_exponentiation(a, (p + 1) // 4, p)
        
        # Find Q and S such that p - 1 = Q * 2^S with Q odd
        q = p - 1
        s = 0
        while q % 2 == 0:
            q //= 2
            s += 1
        
        # Find quadratic non-residue
        z = 2
        while self.quadratic_residue(z, p):
            z += 1
        
        # Initialize variables
        m = s
        c = self.modular_exponentiation(z, q, p)
        t = self.modular_exponentiation(a, q, p)
        r = self.modular_exponentiation(a, (q + 1) // 2, p)
        
        while t != 1:
            # Find smallest i such that t^(2^i) = 1
            i = 1
            temp = (t * t) % p
            while temp != 1:
                temp = (temp * temp) % p
                i += 1
            
            # Update variables
            b = self.modular_exponentiation(c, 1 << (m - i - 1), p)
            m = i
            c = (b * b) % p
            t = (t * c) % p
            r = (r * b) % p
        
        return r
    
    def legendre_symbol(self, a: int, p: int) -> int:
        """Compute Legendre symbol (a/p)"""
        if a % p == 0:
            return 0
        
        result = self.modular_exponentiation(a, (p - 1) // 2, p)
        return -1 if result == p - 1 else result
    
    def jacobi_symbol(self, a: int, n: int) -> int:
        """Compute Jacobi symbol (a/n)"""
        if math.gcd(a, n) != 1:
            return 0
        
        result = 1
        a = a % n
        
        while a != 0:
            while a % 2 == 0:
                a //= 2
                if n % 8 in [3, 5]:
                    result = -result
            
            a, n = n, a
            if a % 4 == 3 and n % 4 == 3:
                result = -result
            
            a = a % n
        
        return result if n == 1 else 0
    
    def continued_fraction(self, x: float, max_terms: int = 100) -> List[int]:
        """Compute continued fraction representation"""
        result = []
        
        for _ in range(max_terms):
            if abs(x - round(x)) < 1e-10:
                result.append(round(x))
                break
            
            integer_part = int(x)
            result.append(integer_part)
            
            x = x - integer_part
            if abs(x) < 1e-10:
                break
            
            x = 1.0 / x
        
        return result
    
    def device_dependent_random(self, n: int, range_max: int) -> List[int]:
        """Generate device-dependent pseudo-random sequence"""
        # Use device ID as seed
        random.seed(self.device_id_transform)
        
        sequence = []
        for i in range(n):
            # Combine device ID with counter for deterministic randomness
            seed_value = (self.device_id_transform * (i + 1)) % (2**32)
            random.seed(seed_value)
            value = random.randint(0, range_max - 1)
            sequence.append(value)
        
        return sequence
    
    def get_math_capabilities(self) -> Dict[str, Any]:
        """Get information about mathematical capabilities"""
        return {
            'device_id_transform': self.device_id_transform,
            'number_theory': [
                'modular_exponentiation', 'extended_euclidean', 'modular_inverse',
                'chinese_remainder_theorem', 'miller_rabin_primality', 'generate_prime'
            ],
            'lattice_operations': [
                'discrete_gaussian_sample', 'lattice_gram_schmidt', 'babai_nearest_plane'
            ],
            'polynomial_operations': [
                'polynomial_gcd'
            ],
            'linear_algebra': [
                'matrix_determinant', 'matrix_inverse'
            ],
            'quadratic_residues': [
                'quadratic_residue', 'tonelli_shanks', 'legendre_symbol', 'jacobi_symbol'
            ],
            'other_functions': [
                'continued_fraction', 'device_dependent_random'
            ]
        }


def create_math_utility(device_id_transform: int):
    """Create mathematical utilities instance"""
    return MathematicalUtilities(device_id_transform)


if __name__ == "__main__":
    # Test mathematical utilities
    print("Testing Mathematical Utilities...")
    
    device_id = 123456789
    math_util = create_math_utility(device_id)
    
    # Test modular arithmetic
    base, exp, mod = 123, 456, 1009
    mod_exp_result = math_util.modular_exponentiation(base, exp, mod)
    print(f"Modular exponentiation: {base}^{exp} mod {mod} = {mod_exp_result}")
    
    # Test modular inverse
    a, m = 17, 97
    inv = math_util.modular_inverse(a, m)
    print(f"Modular inverse: {a}^(-1) mod {m} = {inv}")
    
    # Test primality
    test_number = 97
    is_prime = math_util.miller_rabin_primality(test_number)
    print(f"Primality test: {test_number} is {'prime' if is_prime else 'composite'}")
    
    # Test discrete Gaussian sampling
    gaussian_samples = [math_util.discrete_gaussian_sample(3.2) for _ in range(5)]
    print(f"Discrete Gaussian samples: {gaussian_samples}")
    
    # Test Chinese Remainder Theorem
    remainders = [2, 3, 2]
    moduli = [3, 5, 7]
    crt_result = math_util.chinese_remainder_theorem(remainders, moduli)
    print(f"Chinese Remainder Theorem result: {crt_result}")
    
    # Test matrix operations
    test_matrix = np.array([[1, 2], [3, 4]]) % 97
    det = math_util.matrix_determinant(test_matrix, 97)
    print(f"Matrix determinant mod 97: {det}")
    
    # Test quadratic residue
    qr_test = math_util.quadratic_residue(4, 97)
    print(f"Quadratic residue test: 4 is {'a quadratic residue' if qr_test else 'not a quadratic residue'} mod 97")
    
    # Test device-dependent randomness
    random_seq = math_util.device_dependent_random(5, 100)
    print(f"Device-dependent random sequence: {random_seq}")
    
    # Display capabilities
    capabilities = math_util.get_math_capabilities()
    print(f"Mathematical utilities support {len(capabilities['number_theory'])} number theory functions")
    
    print("Mathematical Utilities test completed successfully!")
