import tempfile
import unittest
from pathlib import Path
from research.dible_v2 import Parameters, device_commitment, encapsulate, keygen
from diblelab.codec import (ciphertext_from_dict, ciphertext_to_dict, public_key_from_dict, public_key_to_dict, read_artifact, secret_key_from_dict, secret_key_to_dict, write_artifact)
from research.dible_v2 import decapsulate

class DibleLabTests(unittest.TestCase):
    def setUp(self):
        self.p = Parameters(); self.c = device_commitment("device", b"d" * 32); self.pk, self.sk = keygen(self.p, self.c)
    def test_artifact_roundtrip(self):
        ct, sent = encapsulate(self.p, self.pk, self.c)
        pk = public_key_from_dict(public_key_to_dict(self.pk)); sk = secret_key_from_dict(secret_key_to_dict(self.sk)); restored = ciphertext_from_dict(ciphertext_to_dict(ct))
        self.assertEqual(decapsulate(self.p, sk, restored, self.c), sent); self.assertEqual(pk, self.pk)
    def test_private_artifact_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "secret.json"; write_artifact(path, secret_key_to_dict(self.sk))
            self.assertEqual(read_artifact(path)["kind"], "diblelab-secret-key")
            self.assertEqual(path.stat().st_mode & 0o077, 0)
    def test_rejects_invalid_artifact(self):
        with self.assertRaises(ValueError): secret_key_from_dict({"kind": "wrong", "version": 1})

if __name__ == "__main__": unittest.main()
