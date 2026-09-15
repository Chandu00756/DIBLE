import pytest
from research.dible_v2 import Parameters, device_commitment, keygen, encapsulate, decapsulate


def test_device_bound_kem_round_trip():
    p = Parameters()
    commitment = device_commitment("test-device", b"x" * 32)
    pk, sk = keygen(p, commitment)
    for _ in range(20):
        ct, shared_sender = encapsulate(p, pk, commitment)
        assert decapsulate(p, sk, ct, commitment) == shared_sender


def test_rejects_other_device():
    p = Parameters()
    a = device_commitment("device-a", b"a" * 32)
    b = device_commitment("device-b", b"b" * 32)
    pk, sk = keygen(p, a)
    with pytest.raises(ValueError):
        encapsulate(p, pk, b)
    ct, _ = encapsulate(p, pk, a)
    with pytest.raises(ValueError):
        decapsulate(p, sk, ct, b)
