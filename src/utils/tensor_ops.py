"""
Tensor Operations Module
Tensor operations for high-dimensional lattice construction
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Union, Optional
import math


class TensorOperations:
    """Tensor operations for DIBLE algorithm"""
    
    def __init__(self, device_id_transform: int):
        self.device_id_transform = device_id_transform
        np.random.seed(device_id_transform % (2**32))
    
    def create_tensor(self, dimensions: Tuple[int, ...], 
                     fill_type: str = 'random', 
                     modulus: int = 2**32 - 5) -> np.ndarray:
        """Create tensor with specified dimensions and fill type"""
        if fill_type == 'random':
            tensor = np.random.randint(0, modulus, size=dimensions)
        elif fill_type == 'zeros':
            tensor = np.zeros(dimensions, dtype=int)
        elif fill_type == 'ones':
            tensor = np.ones(dimensions, dtype=int)
        elif fill_type == 'identity':
            if len(dimensions) == 2 and dimensions[0] == dimensions[1]:
                tensor = np.eye(dimensions[0], dtype=int)
            else:
                raise ValueError("Identity tensor requires square 2D dimensions")
        elif fill_type == 'device_dependent':
            tensor = self._create_device_dependent_tensor(dimensions, modulus)
        else:
            raise ValueError(f"Unsupported fill type: {fill_type}")
        
        return tensor
    
    def _create_device_dependent_tensor(self, dimensions: Tuple[int, ...], 
                                      modulus: int) -> np.ndarray:
        """Create tensor with device-dependent values"""
        tensor = np.zeros(dimensions, dtype=int)
        
        # Use device ID to generate deterministic values
        for idx in np.ndindex(dimensions):
            # Create unique seed for each tensor element
            element_seed = self.device_id_transform
            for i, dim_idx in enumerate(idx):
                element_seed = (element_seed * 31 + dim_idx * (i + 1)) % (2**32)
            
            tensor[idx] = element_seed % modulus
        
        return tensor
    
    def tensor_addition(self, tensor1: np.ndarray, tensor2: np.ndarray, 
                       modulus: int) -> np.ndarray:
        """Element-wise tensor addition with modular arithmetic"""
        if tensor1.shape != tensor2.shape:
            raise ValueError("Tensor dimensions must match for addition")
        
        return (tensor1 + tensor2) % modulus
    
    def tensor_subtraction(self, tensor1: np.ndarray, tensor2: np.ndarray, 
                         modulus: int) -> np.ndarray:
        """Element-wise tensor subtraction with modular arithmetic"""
        if tensor1.shape != tensor2.shape:
            raise ValueError("Tensor dimensions must match for subtraction")
        
        return (tensor1 - tensor2) % modulus
    
    def tensor_multiplication(self, tensor1: np.ndarray, tensor2: np.ndarray, 
                            modulus: int) -> np.ndarray:
        """Element-wise tensor multiplication with modular arithmetic"""
        if tensor1.shape != tensor2.shape:
            raise ValueError("Tensor dimensions must match for multiplication")
        
        return (tensor1 * tensor2) % modulus
    
    def tensor_contraction(self, tensor: np.ndarray, axes: Tuple[int, int], 
                         modulus: int) -> np.ndarray:
        """Tensor contraction along specified axes"""
        if len(tensor.shape) < 2:
            raise ValueError("Tensor must have at least 2 dimensions for contraction")
        
        axis1, axis2 = axes
        if tensor.shape[axis1] != tensor.shape[axis2]:
            raise ValueError("Contracted axes must have the same dimension")
        
        # Perform contraction (generalized trace operation)
        result_shape = list(tensor.shape)
        result_shape.pop(max(axis1, axis2))
        result_shape.pop(min(axis1, axis2))
        
        if not result_shape:
            # Scalar result
            result = 0
            for i in range(tensor.shape[axis1]):
                indices = [slice(None)] * len(tensor.shape)
                indices[axis1] = i
                indices[axis2] = i
                result = (result + tensor[tuple(indices)].sum()) % modulus
            return np.array(result)
        
        result = np.zeros(result_shape, dtype=int)
        
        # Complex contraction for higher-order tensors
        for result_idx in np.ndindex(result.shape):
            value = 0
            for i in range(tensor.shape[axis1]):
                # Build full tensor index
                full_idx = list(result_idx)
                full_idx.insert(min(axis1, axis2), i)
                full_idx.insert(max(axis1, axis2), i)
                value = (value + tensor[tuple(full_idx)]) % modulus
            
            result[result_idx] = value
        
        return result
    
    def tensor_product(self, tensor1: np.ndarray, tensor2: np.ndarray, 
                      modulus: int) -> np.ndarray:
        """Tensor (outer) product"""
        # Flatten tensors to vectors for outer product
        vec1 = tensor1.flatten()
        vec2 = tensor2.flatten()
        
        # Compute outer product
        outer = np.outer(vec1, vec2) % modulus
        
        # Reshape to combined tensor shape
        new_shape = tensor1.shape + tensor2.shape
        return outer.reshape(new_shape)
    
    def tensor_transpose(self, tensor: np.ndarray, axes: Optional[List[int]] = None) -> np.ndarray:
        """Transpose tensor along specified axes"""
        if axes is None:
            # Default: reverse all axes
            axes = list(range(len(tensor.shape)))[::-1]
        
        return np.transpose(tensor, axes)
    
    def tensor_reshape(self, tensor: np.ndarray, new_shape: Tuple[int, ...]) -> np.ndarray:
        """Reshape tensor to new dimensions"""
        if np.prod(tensor.shape) != np.prod(new_shape):
            raise ValueError("Total number of elements must remain the same")
        
        return tensor.reshape(new_shape)
    
    def tensor_slice(self, tensor: np.ndarray, slice_indices: Dict[int, Union[int, slice]]) -> np.ndarray:
        """Slice tensor along specified dimensions"""
        full_slice = [slice(None)] * len(tensor.shape)
        
        for axis, index in slice_indices.items():
            if 0 <= axis < len(tensor.shape):
                full_slice[axis] = index
        
        return tensor[tuple(full_slice)]
    
    def tensor_concatenate(self, tensors: List[np.ndarray], axis: int) -> np.ndarray:
        """Concatenate tensors along specified axis"""
        if not tensors:
            raise ValueError("No tensors provided for concatenation")
        
        # Verify compatible shapes
        base_shape = list(tensors[0].shape)
        for tensor in tensors[1:]:
            shape = list(tensor.shape)
            for i, (dim1, dim2) in enumerate(zip(base_shape, shape)):
                if i != axis and dim1 != dim2:
                    raise ValueError(f"Incompatible shapes for concatenation at axis {axis}")
        
        return np.concatenate(tensors, axis=axis)
    
    def tensor_split(self, tensor: np.ndarray, sections: int, axis: int) -> List[np.ndarray]:
        """Split tensor into sections along specified axis"""
        return np.array_split(tensor, sections, axis=axis)
    
    def tensor_norm(self, tensor: np.ndarray, ord: Union[int, float, str] = 'fro') -> float:
        """Compute tensor norm"""
        if ord == 'fro':
            # Frobenius norm
            return np.sqrt(np.sum(tensor**2))
        elif ord == 1:
            # L1 norm
            return np.sum(np.abs(tensor))
        elif ord == 2:
            # L2 norm
            return np.sqrt(np.sum(tensor**2))
        elif ord == np.inf:
            # Infinity norm
            return np.max(np.abs(tensor))
        else:
            raise ValueError(f"Unsupported norm type: {ord}")
    
    def tensor_svd(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tensor SVD (for 2D tensors/matrices)"""
        if len(tensor.shape) != 2:
            raise ValueError("SVD is only supported for 2D tensors (matrices)")
        
        U, s, Vt = np.linalg.svd(tensor.astype(float))
        return U, s, Vt
    
    def tensor_eigenvalues(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and eigenvectors (for square 2D tensors)"""
        if len(tensor.shape) != 2 or tensor.shape[0] != tensor.shape[1]:
            raise ValueError("Eigenvalues require square 2D tensor")
        
        eigenvalues, eigenvectors = np.linalg.eig(tensor.astype(float))
        return eigenvalues, eigenvectors
    
    def tensor_determinant(self, tensor: np.ndarray, modulus: int) -> int:
        """Compute determinant (for square 2D tensors) with modular arithmetic"""
        if len(tensor.shape) != 2 or tensor.shape[0] != tensor.shape[1]:
            raise ValueError("Determinant requires square 2D tensor")
        
        n = tensor.shape[0]
        mat = tensor.copy() % modulus
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
            inv_pivot = self._modular_inverse(mat[i, i], modulus)
            if inv_pivot is None:
                return 0
            
            for j in range(i + 1, n):
                if mat[j, i] != 0:
                    factor = (mat[j, i] * inv_pivot) % modulus
                    for k in range(i, n):
                        mat[j, k] = (mat[j, k] - factor * mat[i, k]) % modulus
        
        return det % modulus
    
    def _modular_inverse(self, a: int, m: int) -> Optional[int]:
        """Compute modular inverse"""
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        gcd, x, _ = extended_gcd(a, m)
        if gcd != 1:
            return None
        return (x % m + m) % m
    
    def tensor_convolution(self, tensor1: np.ndarray, kernel: np.ndarray, 
                         modulus: int, mode: str = 'valid') -> np.ndarray:
        """Tensor convolution operation"""
        if len(tensor1.shape) != len(kernel.shape):
            raise ValueError("Tensor1 and kernel must have same number of dimensions")
        
        # For simplicity, implement 1D and 2D convolution
        if len(tensor1.shape) == 1:
            # 1D convolution
            result_size = len(tensor1) - len(kernel) + 1
            if result_size <= 0:
                raise ValueError("Kernel is larger than input tensor")
            
            result = np.zeros(result_size, dtype=int)
            for i in range(result_size):
                value = 0
                for j in range(len(kernel)):
                    value = (value + tensor1[i + j] * kernel[j]) % modulus
                result[i] = value
            
            return result
        
        elif len(tensor1.shape) == 2:
            # 2D convolution
            h_out = tensor1.shape[0] - kernel.shape[0] + 1
            w_out = tensor1.shape[1] - kernel.shape[1] + 1
            
            if h_out <= 0 or w_out <= 0:
                raise ValueError("Kernel is larger than input tensor")
            
            result = np.zeros((h_out, w_out), dtype=int)
            
            for i in range(h_out):
                for j in range(w_out):
                    value = 0
                    for ki in range(kernel.shape[0]):
                        for kj in range(kernel.shape[1]):
                            value = (value + tensor1[i + ki, j + kj] * kernel[ki, kj]) % modulus
                    result[i, j] = value
            
            return result
        
        else:
            raise ValueError("Convolution not implemented for tensors with > 2 dimensions")
    
    def tensor_fft(self, tensor: np.ndarray) -> np.ndarray:
        """Fast Fourier Transform of tensor (for 1D tensors)"""
        if len(tensor.shape) != 1:
            raise ValueError("FFT only implemented for 1D tensors")
        
        return np.fft.fft(tensor.astype(complex))
    
    def tensor_compress(self, tensor: np.ndarray, rank: int) -> Tuple[List[np.ndarray], float]:
        """Tensor compression using rank approximation"""
        if len(tensor.shape) == 2:
            # Matrix case - use SVD
            U, s, Vt = self.tensor_svd(tensor)
            
            # Keep only top 'rank' components
            U_compressed = U[:, :rank]
            s_compressed = s[:rank]
            Vt_compressed = Vt[:rank, :]
            
            # Compute compression ratio
            original_size = tensor.size
            compressed_size = U_compressed.size + s_compressed.size + Vt_compressed.size
            compression_ratio = compressed_size / original_size
            
            return [U_compressed, np.diag(s_compressed), Vt_compressed], compression_ratio
        
        else:
            # Higher-order tensor - use simple mode-wise compression
            factors = []
            current_tensor = tensor.copy()
            
            for mode in range(len(tensor.shape)):
                # Matricize along current mode
                matricized = self._tensorize_mode(current_tensor, mode)
                
                # SVD compression
                U, s, Vt = np.linalg.svd(matricized, full_matrices=False)
                
                # Keep top components
                rank_mode = min(rank, len(s))
                U_compressed = U[:, :rank_mode]
                factors.append(U_compressed)
                
                # Update tensor for next mode
                current_tensor = self._update_tensor_compression(current_tensor, U_compressed, mode)
            
            # Compute compression ratio
            original_size = tensor.size
            compressed_size = sum(factor.size for factor in factors)
            compression_ratio = compressed_size / original_size
            
            return factors, compression_ratio
    
    def _tensorize_mode(self, tensor: np.ndarray, mode: int) -> np.ndarray:
        """Matricize tensor along specified mode"""
        # Move the specified mode to the front
        axes = [mode] + [i for i in range(len(tensor.shape)) if i != mode]
        reordered_tensor = np.transpose(tensor, axes)
        
        # Reshape to matrix
        mode_size = tensor.shape[mode]
        other_size = tensor.size // mode_size
        
        return reordered_tensor.reshape(mode_size, other_size)
    
    def _update_tensor_compression(self, tensor: np.ndarray, factor: np.ndarray, mode: int) -> np.ndarray:
        """Update tensor for compression iteration"""
        # This is a simplified version - full tensor decomposition is more complex
        return tensor  # Placeholder
    
    def device_dependent_tensor_transform(self, tensor: np.ndarray, modulus: int) -> np.ndarray:
        """Apply device-dependent transformation to tensor"""
        transformed = tensor.copy()
        
        for idx in np.ndindex(tensor.shape):
            # Create device-dependent transformation value
            transform_seed = self.device_id_transform
            for i, dim_idx in enumerate(idx):
                transform_seed = (transform_seed * 17 + dim_idx * (i + 1)) % modulus
            
            # Apply transformation
            transformed[idx] = (transformed[idx] + transform_seed) % modulus
        
        return transformed
    
    def get_tensor_capabilities(self) -> Dict[str, Any]:
        """Get information about tensor operation capabilities"""
        return {
            'device_id_transform': self.device_id_transform,
            'creation_operations': [
                'create_tensor', 'device_dependent_tensor'
            ],
            'arithmetic_operations': [
                'tensor_addition', 'tensor_subtraction', 'tensor_multiplication',
                'tensor_contraction', 'tensor_product'
            ],
            'structural_operations': [
                'tensor_transpose', 'tensor_reshape', 'tensor_slice',
                'tensor_concatenate', 'tensor_split'
            ],
            'analysis_operations': [
                'tensor_norm', 'tensor_svd', 'tensor_eigenvalues', 'tensor_determinant'
            ],
            'advanced_operations': [
                'tensor_convolution', 'tensor_fft', 'tensor_compress'
            ],
            'device_operations': [
                'device_dependent_tensor_transform'
            ],
            'supported_dimensions': 'arbitrary',
            'modular_arithmetic': True
        }


def create_tensor_operations(device_id_transform: int):
    """Create tensor operations instance"""
    return TensorOperations(device_id_transform)


if __name__ == "__main__":
    # Test tensor operations
    print("Testing Tensor Operations...")
    
    device_id = 555666777
    tensor_ops = create_tensor_operations(device_id)
    
    # Create test tensors
    tensor_2d = tensor_ops.create_tensor((3, 3), 'device_dependent', 97)
    tensor_1d = tensor_ops.create_tensor((5,), 'random', 97)
    
    print(f"Created 2D tensor (3x3): shape = {tensor_2d.shape}")
    print(f"Created 1D tensor (5,): shape = {tensor_1d.shape}")
    
    # Test tensor arithmetic
    tensor_2d_copy = tensor_ops.create_tensor((3, 3), 'random', 97)
    sum_tensor = tensor_ops.tensor_addition(tensor_2d, tensor_2d_copy, 97)
    print(f"✓ Tensor addition completed: result shape = {sum_tensor.shape}")
    
    # Test tensor contraction
    try:
        contracted = tensor_ops.tensor_contraction(tensor_2d, (0, 1), 97)
        print(f"✓ Tensor contraction completed: result = {contracted}")
    except Exception as e:
        print(f"Tensor contraction test: {e}")
    
    # Test tensor product
    product_tensor = tensor_ops.tensor_product(tensor_1d[:2], tensor_1d[2:4], 97)
    print(f"✓ Tensor product completed: result shape = {product_tensor.shape}")
    
    # Test tensor transpose
    transposed = tensor_ops.tensor_transpose(tensor_2d)
    print(f"✓ Tensor transpose completed: original {tensor_2d.shape} -> transposed {transposed.shape}")
    
    # Test tensor norm
    norm_value = tensor_ops.tensor_norm(tensor_2d)
    print(f"✓ Tensor norm computed: {norm_value:.2f}")
    
    # Test tensor determinant
    det_value = tensor_ops.tensor_determinant(tensor_2d, 97)
    print(f"✓ Tensor determinant computed: {det_value}")
    
    # Test device-dependent transformation
    transformed = tensor_ops.device_dependent_tensor_transform(tensor_2d, 97)
    print(f"✓ Device-dependent transformation completed")
    
    # Test convolution
    kernel = tensor_ops.create_tensor((2,), 'ones')
    try:
        conv_result = tensor_ops.tensor_convolution(tensor_1d, kernel, 97)
        print(f"✓ Tensor convolution completed: result shape = {conv_result.shape}")
    except Exception as e:
        print(f"Convolution test: {e}")
    
    # Display capabilities
    capabilities = tensor_ops.get_tensor_capabilities()
    total_ops = sum(len(ops) for ops in capabilities.values() if isinstance(ops, list))
    print(f"Tensor operations support {total_ops} different operations")
    
    print("Tensor Operations test completed successfully!")
