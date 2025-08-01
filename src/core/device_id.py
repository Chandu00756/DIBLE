"""
Device ID Generation Module
Generates unique device identifiers based on comprehensive system information
"""

import hashlib
import time
import platform
import os
import socket
import uuid
import json
from typing import Dict, Any
from Crypto.Hash import SHA3_256, BLAKE2b

# Optional dependencies with proper fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import cpuinfo
    CPUINFO_AVAILABLE = True
except ImportError:
    CPUINFO_AVAILABLE = False
    cpuinfo = None

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    GPUtil = None


class DeviceIDGenerator:
    """Generate unique device identifiers using comprehensive system information"""
    
    def __init__(self):
        self.device_data = {}
        
    def collect_hardware_info(self) -> Dict[str, Any]:
        """Collect comprehensive hardware information"""
        hardware_info = {}
        
        # CPU Information
        try:
            if CPUINFO_AVAILABLE and cpuinfo:
                cpu_info = cpuinfo.get_cpu_info()
                hardware_info['cpu'] = {
                    'brand': cpu_info.get('brand_raw', ''),
                    'arch': cpu_info.get('arch', ''),
                    'bits': cpu_info.get('bits', 0),
                    'count': cpu_info.get('count', 0),
                    'frequency': cpu_info.get('hz_advertised_friendly', ''),
                    'flags': cpu_info.get('flags', [])
                }
            else:
                hardware_info['cpu'] = {
                    'brand': platform.processor() or 'Unknown',
                    'arch': platform.machine(),
                    'count': os.cpu_count() or 1
                }
        except Exception as e:
            hardware_info['cpu'] = {
                'brand': platform.processor() or 'Unknown',
                'arch': platform.machine(),
                'count': os.cpu_count() or 1,
                'error': str(e)
            }
            
        # Memory Information
        try:
            if PSUTIL_AVAILABLE and psutil:
                memory = psutil.virtual_memory()
                hardware_info['memory'] = {
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used,
                    'free': memory.free
                }
            else:
                # Fallback memory info
                hardware_info['memory'] = {
                    'total': 0,
                    'available': 0,
                    'used': 0,
                    'free': 0,
                    'note': 'psutil not available'
                }
        except Exception as e:
            hardware_info['memory'] = {
                'total': 0,
                'available': 0,
                'used': 0,
                'free': 0,
                'error': str(e)
            }
            
        # Disk Information
        try:
            if PSUTIL_AVAILABLE and psutil:
                disk_usage = psutil.disk_usage('/')
                hardware_info['disk'] = {
                    'total': disk_usage.total,
                    'used': disk_usage.used,
                    'free': disk_usage.free
                }
            else:
                # Fallback disk info
                hardware_info['disk'] = {
                    'total': 0,
                    'used': 0,
                    'free': 0,
                    'note': 'psutil not available'
                }
        except Exception as e:
            hardware_info['disk'] = {
                'total': 0,
                'used': 0,
                'free': 0,
                'error': str(e)
            }
            
        # GPU Information
        try:
            if GPUTIL_AVAILABLE and GPUtil:
                gpus = GPUtil.getGPUs()
                hardware_info['gpu'] = []
                for gpu in gpus:
                    hardware_info['gpu'].append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'memory_total': gpu.memoryTotal,
                        'memory_used': gpu.memoryUsed,
                        'memory_free': gpu.memoryFree,
                        'temperature': gpu.temperature,
                        'uuid': gpu.uuid
                    })
            else:
                hardware_info['gpu'] = {'note': 'GPUtil not available'}
        except Exception as e:
            hardware_info['gpu'] = {'error': str(e)}
            
        return hardware_info
    
    def collect_system_info(self) -> Dict[str, Any]:
        """Collect system-level information"""
        system_info = {}
        
        # Platform Information
        system_info['platform'] = {
            'system': platform.system(),
            'node': platform.node(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation()
        }
        
        # Network Information
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            system_info['network'] = {
                'hostname': hostname,
                'local_ip': local_ip,
                'mac_address': ':'.join(['{:02x}'.format((uuid.getnode() >> ele) & 0xff) 
                                       for ele in range(0, 8*6, 8)][::-1])
            }
        except Exception as e:
            system_info['network'] = {'error': str(e)}
            
        # Boot Time
        try:
            if PSUTIL_AVAILABLE and psutil:
                boot_time = psutil.boot_time()
                system_info['boot_time'] = boot_time
            else:
                system_info['boot_time'] = {'error': 'psutil not available'}
        except Exception as e:
            system_info['boot_time'] = {'error': str(e)}
            
        return system_info
    
    def collect_process_info(self) -> Dict[str, Any]:
        """Collect current process information"""
        process_info = {}
        
        try:
            if PSUTIL_AVAILABLE and psutil:
                current_process = psutil.Process()
                process_info = {
                    'pid': current_process.pid,
                    'ppid': current_process.ppid(),
                    'name': current_process.name(),
                    'create_time': current_process.create_time(),
                    'num_threads': current_process.num_threads(),
                    'memory_info': current_process.memory_info()._asdict(),
                    'cpu_percent': current_process.cpu_percent()
                }
            else:
                process_info = {'error': 'psutil not available'}
        except Exception as e:
            process_info = {'error': str(e)}
            
        return process_info
    
    def collect_entropy_sources(self) -> Dict[str, Any]:
        """Collect entropy sources for randomness"""
        entropy_info = {}
        
        # Current time with high precision
        entropy_info['time'] = {
            'current_time': time.time(),
            'perf_counter': time.perf_counter(),
            'monotonic': time.monotonic(),
            'process_time': time.process_time()
        }
        
        # Random bytes from OS
        try:
            if PSUTIL_AVAILABLE and psutil:
                entropy_info['os_random'] = psutil.urandom(64).hex()
            else:
                entropy_info['os_random'] = {'error': 'psutil not available'}
        except Exception as e:
            entropy_info['os_random'] = {'error': str(e)}
            
        # Environment variables hash
        try:
            env_str = json.dumps(dict(os.environ), sort_keys=True)
            entropy_info['env_hash'] = hashlib.sha256(env_str.encode()).hexdigest()
        except Exception as e:
            entropy_info['env_hash'] = {'error': str(e)}
            
        return entropy_info
    
    def collect_filesystem_info(self) -> Dict[str, Any]:
        """Collect filesystem information"""
        fs_info = {}
        
        try:
            # Current working directory
            fs_info['cwd'] = os.getcwd()
            
            # Home directory
            fs_info['home'] = os.path.expanduser('~')
            
            # Temporary directory
            fs_info['temp'] = os.path.dirname(os.path.abspath(__file__))
            
            # File system stats
            statvfs = os.statvfs('/')
            fs_info['filesystem'] = {
                'f_bavail': statvfs.f_bavail,
                'f_bfree': statvfs.f_bfree,
                'f_blocks': statvfs.f_blocks,
                'f_bsize': statvfs.f_bsize,
                'f_files': statvfs.f_files,
                'f_ffree': statvfs.f_ffree
            }
        except Exception as e:
            fs_info = {'error': str(e)}
            
        return fs_info
    
    def augment_entropy(self, data: Dict[str, Any]) -> bytes:
        """Augment entropy using chaos theory and random elements"""
        # Convert data to string
        data_str = json.dumps(data, sort_keys=True, default=str)
        data_bytes = data_str.encode('utf-8')
        
        # Generate random elements
        random_elements = []
        for i in range(10):
            random_elements.append(os.urandom(32))
            
        # Current time in microseconds
        current_time = int(time.time() * 1000000)
        time_bytes = current_time.to_bytes(8, byteorder='big')
        
        # XOR operation with entropy augmentation
        augmented_data = bytearray(data_bytes)
        for i, rand_elem in enumerate(random_elements):
            for j, byte in enumerate(rand_elem):
                if j + i * 32 < len(augmented_data):
                    augmented_data[j + i * 32] ^= byte
                    
        # Add time component
        for i, byte in enumerate(time_bytes):
            if i < len(augmented_data):
                augmented_data[i] ^= byte
                
        return bytes(augmented_data)
    
    def generate_device_id(self) -> str:
        """Generate comprehensive device ID"""
        # Collect all device information
        device_data = {
            'hardware': self.collect_hardware_info(),
            'system': self.collect_system_info(),
            'process': self.collect_process_info(),
            'entropy': self.collect_entropy_sources(),
            'filesystem': self.collect_filesystem_info()
        }
        
        # Store for later use
        self.device_data = device_data
        
        # Augment entropy
        augmented_data = self.augment_entropy(device_data)
        
        # Generate device ID using multiple hash functions
        sha3_hash = SHA3_256.new(augmented_data).hexdigest()
        blake2b_hash = BLAKE2b.new(digest_bits=256, data=augmented_data).hexdigest()
        
        # Combine hashes
        combined_hash = sha3_hash + blake2b_hash
        final_hash = hashlib.sha256(combined_hash.encode()).hexdigest()
        
        return final_hash
    
    def transform_device_id(self, device_id: str, phi_1: int, phi_2: int, p: int) -> int:
        """Transform device ID using mathematical transformation"""
        # Convert device ID to integer
        device_id_int = int(device_id, 16)
        
        # Apply transformation: φ(D_ID) = Φ₁ · D_ID + Φ₂ mod p
        transformed_id = (phi_1 * device_id_int + phi_2) % p
        
        return transformed_id
    
    def get_device_fingerprint(self) -> Dict[str, Any]:
        """Get complete device fingerprint"""
        device_id = self.generate_device_id()
        
        # Generate transformation parameters
        phi_1 = int.from_bytes(os.urandom(32), byteorder='big') % (2**256 - 1)
        phi_2 = int.from_bytes(os.urandom(32), byteorder='big') % (2**256 - 1)
        p = 2**256 - 189  # Large prime
        
        transformed_id = self.transform_device_id(device_id, phi_1, phi_2, p)
        
        return {
            'device_id': device_id,
            'transformed_id': transformed_id,
            'phi_1': phi_1,
            'phi_2': phi_2,
            'p': p,
            'device_data': self.device_data
        }


def generate_device_identity():
    """Convenience function to generate device identity"""
    generator = DeviceIDGenerator()
    return generator.get_device_fingerprint()


if __name__ == "__main__":
    # Test device ID generation
    device_identity = generate_device_identity()
    print(f"Device ID: {device_identity['device_id']}")
    print(f"Transformed ID: {device_identity['transformed_id']}")
