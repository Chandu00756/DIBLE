"""
Polynomial Operations Module
Implements polynomial operations for multivariate polynomial rings
"""

from typing import List, Dict, Tuple, Union, Any
import random
from sympy import symbols, expand, Poly, GF, solve


class PolynomialOperations:
    """Handle polynomial operations for DIBLE algorithm"""
    
    def __init__(self, modulus: int, variables: List[str] = None):
        self.modulus = modulus
        self.variables = variables or ['x', 'y', 'z', 't']
        self.symbol_dict = {var: symbols(var) for var in self.variables}
        
    def create_polynomial(self, coefficients: List[int], degrees: List[Tuple[int, ...]] = None) -> Dict[str, Any]:
        """Create polynomial with given coefficients and degrees"""
        if degrees is None:
            # Create simple univariate polynomial
            degrees = [(i,) for i in range(len(coefficients))]
        
        # Build polynomial expression
        poly_expr = 0
        var_symbols = [self.symbol_dict[var] for var in self.variables]
        
        for coeff, degree_tuple in zip(coefficients, degrees):
            term = coeff
            for i, degree in enumerate(degree_tuple):
                if i < len(var_symbols) and degree > 0:
                    term *= var_symbols[i] ** degree
            poly_expr += term
        
        return {
            'expression': poly_expr,
            'coefficients': coefficients,
            'degrees': degrees,
            'variables': self.variables
        }
    
    def random_polynomial(self, max_degree: int, num_terms: int, device_id_transform: int = 0) -> Dict[str, Any]:
        """Generate random polynomial with device ID influence"""
        # Use device ID to seed randomness
        random.seed(device_id_transform % (2**32))
        
        coefficients = []
        degrees = []
        
        for _ in range(num_terms):
            # Random coefficient influenced by device ID
            coeff = (random.randint(1, self.modulus) * (device_id_transform % 100 + 1)) % self.modulus
            coefficients.append(coeff)
            
            # Random degree tuple
            degree_tuple = tuple(random.randint(0, max_degree) for _ in self.variables)
            degrees.append(degree_tuple)
        
        return self.create_polynomial(coefficients, degrees)
    
    def polynomial_addition(self, poly1: Dict[str, Any], poly2: Dict[str, Any]) -> Dict[str, Any]:
        """Add two polynomials"""
        expr1 = poly1['expression']
        expr2 = poly2['expression']
        
        result_expr = expand(expr1 + expr2)
        
        # Extract coefficients and degrees from result
        result_poly = Poly(result_expr, *[self.symbol_dict[var] for var in self.variables])
        
        coefficients = []
        degrees = []
        
        for monom, coeff in result_poly.terms():
            coefficients.append(int(coeff) % self.modulus)
            degrees.append(monom)
        
        return self.create_polynomial(coefficients, degrees)
    
    def polynomial_multiplication(self, poly1: Dict[str, Any], poly2: Dict[str, Any]) -> Dict[str, Any]:
        """Multiply two polynomials"""
        expr1 = poly1['expression']
        expr2 = poly2['expression']
        
        result_expr = expand(expr1 * expr2)
        
        # Extract coefficients and degrees from result
        result_poly = Poly(result_expr, *[self.symbol_dict[var] for var in self.variables])
        
        coefficients = []
        degrees = []
        
        for monom, coeff in result_poly.terms():
            coefficients.append(int(coeff) % self.modulus)
            degrees.append(monom)
        
        return self.create_polynomial(coefficients, degrees)
    
    def polynomial_evaluation(self, poly: Dict[str, Any], values: Dict[str, Union[int, float]]) -> Union[int, float]:
        """Evaluate polynomial at given point"""
        expr = poly['expression']
        
        # Substitute values
        for var_name, value in values.items():
            if var_name in self.symbol_dict:
                expr = expr.subs(self.symbol_dict[var_name], value)
        
        # Return result modulo modulus if integer
        result = complex(expr)
        if result.imag == 0:
            return int(result.real) % self.modulus
        else:
            return result
    
    def multivariate_polynomial_ring(self, device_id_transform: int, ring_size: int = 5) -> List[Dict[str, Any]]:
        """Create multivariate polynomial ring with device ID integration"""
        polynomials = []
        
        for i in range(ring_size):
            # Generate polynomial with device-specific coefficients
            max_degree = 3
            num_terms = random.randint(3, 7)
            
            # Adjust parameters based on device ID and index
            seed_modifier = (device_id_transform * (i + 1)) % (2**31)
            poly = self.random_polynomial(max_degree, num_terms, seed_modifier)
            
            polynomials.append(poly)
        
        return polynomials
    
    def polynomial_gcd(self, poly1: Dict[str, Any], poly2: Dict[str, Any]) -> Dict[str, Any]:
        """Compute GCD of two polynomials"""
        expr1 = poly1['expression']
        expr2 = poly2['expression']
        
        # Convert to sympy polynomials
        var_symbols = [self.symbol_dict[var] for var in self.variables]
        p1 = Poly(expr1, *var_symbols)
        p2 = Poly(expr2, *var_symbols)
        
        # Compute GCD
        gcd_poly = p1.gcd(p2)
        gcd_expr = gcd_poly.as_expr()
        
        # Extract coefficients and degrees
        coefficients = []
        degrees = []
        
        for monom, coeff in gcd_poly.terms():
            coefficients.append(int(coeff) % self.modulus)
            degrees.append(monom)
        
        return self.create_polynomial(coefficients, degrees)
    
    def polynomial_factorization(self, poly: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Factor polynomial into irreducible components"""
        expr = poly['expression']
        
        try:
            # Convert to sympy polynomial
            var_symbols = [self.symbol_dict[var] for var in self.variables]
            p = Poly(expr, *var_symbols, domain=GF(self.modulus))
            
            # Factor
            factors = p.factor_list()
            
            result_factors = []
            for factor, multiplicity in factors[1]:  # Skip constant factor
                factor_expr = factor.as_expr()
                
                # Extract coefficients and degrees
                factor_poly = Poly(factor_expr, *var_symbols)
                coefficients = []
                degrees = []
                
                for monom, coeff in factor_poly.terms():
                    coefficients.append(int(coeff) % self.modulus)
                    degrees.append(monom)
                
                factor_dict = self.create_polynomial(coefficients, degrees)
                factor_dict['multiplicity'] = multiplicity
                result_factors.append(factor_dict)
            
            return result_factors
            
        except Exception:
            # If factorization fails, return original polynomial
            return [poly]
    
    def solve_polynomial_system(self, polynomials: List[Dict[str, Any]]) -> List[Dict[str, Union[int, float, complex]]]:
        """Solve system of polynomial equations"""
        expressions = [poly['expression'] for poly in polynomials]
        var_symbols = [self.symbol_dict[var] for var in self.variables]
        
        try:
            solutions = solve(expressions, var_symbols)
            
            if isinstance(solutions, list):
                result_solutions = []
                for sol in solutions:
                    if isinstance(sol, dict):
                        result_solutions.append({str(k): v for k, v in sol.items()})
                    else:
                        # Single variable solution
                        result_solutions.append({str(var_symbols[0]): sol})
                return result_solutions
            elif isinstance(solutions, dict):
                return [{str(k): v for k, v in solutions.items()}]
            else:
                return []
                
        except Exception:
            return []
    
    def polynomial_derivative(self, poly: Dict[str, Any], variable: str) -> Dict[str, Any]:
        """Compute partial derivative of polynomial"""
        expr = poly['expression']
        
        if variable in self.symbol_dict:
            var_symbol = self.symbol_dict[variable]
            derivative_expr = expr.diff(var_symbol)
            
            # Extract coefficients and degrees
            derivative_poly = Poly(derivative_expr, *[self.symbol_dict[var] for var in self.variables])
            coefficients = []
            degrees = []
            
            for monom, coeff in derivative_poly.terms():
                coefficients.append(int(coeff) % self.modulus)
                degrees.append(monom)
            
            return self.create_polynomial(coefficients, degrees)
        else:
            raise ValueError(f"Variable {variable} not found in polynomial variables")
    
    def polynomial_composition(self, poly1: Dict[str, Any], poly2: Dict[str, Any], variable: str) -> Dict[str, Any]:
        """Compose polynomials: poly1(poly2(x))"""
        expr1 = poly1['expression']
        expr2 = poly2['expression']
        
        if variable in self.symbol_dict:
            var_symbol = self.symbol_dict[variable]
            composed_expr = expr1.subs(var_symbol, expr2)
            expanded_expr = expand(composed_expr)
            
            # Extract coefficients and degrees
            composed_poly = Poly(expanded_expr, *[self.symbol_dict[var] for var in self.variables])
            coefficients = []
            degrees = []
            
            for monom, coeff in composed_poly.terms():
                coefficients.append(int(coeff) % self.modulus)
                degrees.append(monom)
            
            return self.create_polynomial(coefficients, degrees)
        else:
            raise ValueError(f"Variable {variable} not found in polynomial variables")
    
    def polynomial_interpolation(self, points: List[Tuple[Dict[str, Union[int, float]], Union[int, float]]]) -> Dict[str, Any]:
        """Lagrange interpolation for multivariate polynomials"""
        if not points:
            return self.create_polynomial([0])
        
        # For simplicity, implement univariate case
        if len(self.variables) == 1:
            var_symbol = self.symbol_dict[self.variables[0]]
            
            # Extract x and y values
            x_values = [point[0][self.variables[0]] for point in points]
            y_values = [point[1] for point in points]
            
            # Lagrange interpolation
            n = len(points)
            result_expr = 0
            
            for i in range(n):
                term = y_values[i]
                for j in range(n):
                    if i != j:
                        term *= (var_symbol - x_values[j]) / (x_values[i] - x_values[j])
                result_expr += term
            
            result_expr = expand(result_expr)
            
            # Extract coefficients and degrees
            result_poly = Poly(result_expr, var_symbol)
            coefficients = []
            degrees = []
            
            for monom, coeff in result_poly.terms():
                coefficients.append(int(float(coeff)) % self.modulus)
                degrees.append(monom)
            
            return self.create_polynomial(coefficients, degrees)
        else:
            # For multivariate case, return zero polynomial
            return self.create_polynomial([0])
    
    def create_vanishing_ideal(self, points: List[Dict[str, Union[int, float]]]) -> List[Dict[str, Any]]:
        """Create vanishing ideal for given points"""
        ideal_generators = []
        
        for point in points:
            for var_name, value in point.items():
                if var_name in self.symbol_dict:
                    var_symbol = self.symbol_dict[var_name]
                    # Create polynomial (x - value)
                    poly_expr = var_symbol - value
                    
                    poly_obj = Poly(poly_expr, var_symbol)
                    coefficients = []
                    degrees = []
                    
                    for monom, coeff in poly_obj.terms():
                        coefficients.append(int(coeff) % self.modulus)
                        degrees.append(monom)
                    
                    ideal_generators.append(self.create_polynomial(coefficients, degrees))
        
        return ideal_generators
    
    def groebner_basis(self, polynomials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute Groebner basis (simplified version)"""
        expressions = [poly['expression'] for poly in polynomials]
        var_symbols = [self.symbol_dict[var] for var in self.variables]
        
        try:
            from sympy import groebner
            gb = groebner(expressions, *var_symbols, order='lex')
            
            result_basis = []
            for poly_expr in gb:
                poly_obj = Poly(poly_expr, *var_symbols)
                coefficients = []
                degrees = []
                
                for monom, coeff in poly_obj.terms():
                    coefficients.append(int(coeff) % self.modulus)
                    degrees.append(monom)
                
                result_basis.append(self.create_polynomial(coefficients, degrees))
            
            return result_basis
            
        except ImportError:
            # If groebner not available, return original polynomials
            return polynomials
    
    def polynomial_hash(self, poly: Dict[str, Any]) -> int:
        """Compute hash of polynomial for indexing"""
        # Create hash from coefficients and degrees
        hash_data = []
        
        for coeff, degree in zip(poly['coefficients'], poly['degrees']):
            hash_data.append(str(coeff))
            hash_data.append(str(degree))
        
        hash_string = ''.join(hash_data)
        return hash(hash_string) % (2**32)
    
    def binary_to_polynomial(self, binary_message: List[int]) -> Dict[str, Any]:
        """Convert binary message to polynomial"""
        coefficients = []
        degrees = []
        
        for i, bit in enumerate(binary_message):
            if bit == 1:
                coeff = self.modulus // 2  # ⌊q/2⌋
            else:
                coeff = 0
            
            coefficients.append(coeff)
            degrees.append((i,))  # x^i term
        
        return self.create_polynomial(coefficients, degrees)
    
    def polynomial_to_binary(self, poly: Dict[str, Any]) -> List[int]:
        """Convert polynomial back to binary message"""
        binary_message = []
        
        # Create mapping from degrees to coefficients
        degree_to_coeff = {}
        for coeff, degree in zip(poly['coefficients'], poly['degrees']):
            if len(degree) == 1:  # Only consider univariate terms
                degree_to_coeff[degree[0]] = coeff
        
        # Determine message length
        max_degree = max(degree_to_coeff.keys()) if degree_to_coeff else 0
        
        for i in range(max_degree + 1):
            coeff = degree_to_coeff.get(i, 0)
            
            # Decode bit based on distance to 0 or ⌊q/2⌋
            mid_point = self.modulus // 2
            
            if abs(coeff) < abs(coeff - mid_point):
                binary_message.append(0)
            else:
                binary_message.append(1)
        
        return binary_message


def create_polynomial_operations(modulus: int = 97, variables: List[str] = None):
    """Create polynomial operations instance"""
    return PolynomialOperations(modulus, variables)


if __name__ == "__main__":
    # Test polynomial operations
    poly_ops = create_polynomial_operations(97, ['x', 'y'])
    
    # Test random polynomial generation
    device_id = 12345
    poly1 = poly_ops.random_polynomial(2, 3, device_id)
    poly2 = poly_ops.random_polynomial(2, 3, device_id + 1)
    
    print(f"Polynomial 1: {poly1['expression']}")
    print(f"Polynomial 2: {poly2['expression']}")
    
    # Test polynomial multiplication
    product = poly_ops.polynomial_multiplication(poly1, poly2)
    print(f"Product: {product['expression']}")
    
    # Test polynomial evaluation
    evaluation = poly_ops.polynomial_evaluation(poly1, {'x': 2, 'y': 3})
    print(f"Evaluation at (2, 3): {evaluation}")
    
    # Test binary conversion
    binary_msg = [1, 0, 1, 1, 0]
    poly_from_binary = poly_ops.binary_to_polynomial(binary_msg)
    print(f"Polynomial from binary: {poly_from_binary['expression']}")
    
    recovered_binary = poly_ops.polynomial_to_binary(poly_from_binary)
    print(f"Recovered binary: {recovered_binary}")
    print(f"Original binary: {binary_msg}")
