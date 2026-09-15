from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
from research.dible_v2 import Parameters, decapsulate, device_commitment, encapsulate, keygen, security_notes
from .codec import (ciphertext_from_dict, ciphertext_to_dict, public_key_from_dict, public_key_to_dict, read_artifact, secret_key_from_dict, secret_key_to_dict, write_artifact)

BANNER = "DIBLE Lab v0.2 — RESEARCH ONLY: never use for real secrets"


def _commitment(claim: str, salt_hex: str | None) -> bytes:
    return device_commitment(claim, bytes.fromhex(salt_hex) if salt_hex else None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dible-lab", description=BANNER)
    sub = parser.add_subparsers(dest="command", required=True)
    kg = sub.add_parser("keygen", help="create research key artifacts")
    kg.add_argument("--device-claim", required=True)
    kg.add_argument("--salt-hex", required=True, help="32-byte device commitment salt in hex")
    kg.add_argument("--out", required=True)
    enc = sub.add_parser("encapsulate", help="create a research ciphertext and shared-secret digest")
    enc.add_argument("--public-key", required=True); enc.add_argument("--device-claim", required=True); enc.add_argument("--salt-hex", required=True); enc.add_argument("--out", required=True)
    dec = sub.add_parser("decapsulate", help="recover and print shared-secret digest")
    dec.add_argument("--secret-key", required=True); dec.add_argument("--ciphertext", required=True); dec.add_argument("--device-claim", required=True); dec.add_argument("--salt-hex", required=True)
    verify = sub.add_parser("verify", help="run randomized round-trip and wrong-device rejection checks")
    verify.add_argument("--rounds", type=int, default=200)
    info = sub.add_parser("info", help="print explicit research security status")
    args = parser.parse_args(argv)
    print(BANNER)
    p = Parameters()
    if args.command == "info":
        print(json.dumps(security_notes(p), indent=2)); return 0
    if args.command == "keygen":
        c = _commitment(args.device_claim, args.salt_hex); pk, sk = keygen(p, c); root = Path(args.out)
        write_artifact(root / "public-key.json", public_key_to_dict(pk)); write_artifact(root / "secret-key.json", secret_key_to_dict(sk))
        print(f"created research key artifacts in {root}"); return 0
    if args.command == "encapsulate":
        c = _commitment(args.device_claim, args.salt_hex); pk = public_key_from_dict(read_artifact(args.public_key)); ct, shared = encapsulate(p, pk, c)
        write_artifact(args.out, ciphertext_to_dict(ct)); print(hashlib.sha3_256(shared).hexdigest()); return 0
    if args.command == "decapsulate":
        c = _commitment(args.device_claim, args.salt_hex); sk = secret_key_from_dict(read_artifact(args.secret_key)); ct = ciphertext_from_dict(read_artifact(args.ciphertext)); shared = decapsulate(p, sk, ct, c)
        print(hashlib.sha3_256(shared).hexdigest()); return 0
    if args.command == "verify":
        if args.rounds < 1 or args.rounds > 10000: parser.error("rounds must be in [1, 10000]")
        c = device_commitment("verification-device", b"v" * 32); pk, sk = keygen(p, c); started = time.perf_counter()
        for _ in range(args.rounds):
            ct, sender = encapsulate(p, pk, c)
            if decapsulate(p, sk, ct, c) != sender: raise RuntimeError("round-trip mismatch")
        wrong = device_commitment("wrong-device", b"w" * 32)
        try: decapsulate(p, sk, ct, wrong)
        except ValueError: pass
        else: raise RuntimeError("wrong-device rejection failed")
        print(json.dumps({"rounds": args.rounds, "seconds": round(time.perf_counter() - started, 4), "wrong_device_rejected": True, "status": security_notes(p)["status"]}, indent=2)); return 0
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
