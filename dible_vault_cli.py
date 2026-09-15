#!/usr/bin/env python3
"""
PORTAL VII DIBLE – Production CLI
AES-256-GCM vault with device binding, password-KDF, and non-interactive commands.

Usage: python dible_vault_cli.py <command> [OPTIONS]
       (or: dible-vault <command> after pip install -e .)
"""
from __future__ import annotations

import json
import os
import sys
import secrets
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

sys.path.insert(0, str(Path(__file__).parent))

from src.core.device_id import DeviceIDGenerator
from src.crypto.dible import DIBLECore

# ---------------------------------------------------------------------------
VAULT_HOME = Path.home() / ".portal7_dible"
KEYS_DIR   = VAULT_HOME / "keys"
DATA_DIR   = VAULT_HOME / "data"
LOGS_DIR   = VAULT_HOME / "logs"

for _d in (VAULT_HOME, KEYS_DIR, DATA_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

console = Console()

BANNER = """
[bold bright_blue]
 ╔════════════════════════════════════════════════════════════╗
 ║       PORTAL VII DIBLE VAULT  v2.0.0         ║
 ║   AES-256-GCM • Device-Bound • PBKDF2-SHA256  ║
 ╚════════════════════════════════════════════════════════════╝
[/bold bright_blue]"""


# ---------------------------------------------------------------------------
def _engine() -> DIBLECore:
    device_id = DeviceIDGenerator().generate_device_id()
    return DIBLECore(device_id)


def _audit(action: str, detail: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    line = json.dumps({"ts": ts, "action": action, "detail": detail}) + "\n"
    (LOGS_DIR / "audit.log").open("a").write(line)


def _load_key(key_id: str) -> bytes:
    kf = KEYS_DIR / f"{key_id}.key"
    if not kf.exists():
        raise click.ClickException(f"Key not found: {key_id}")
    data = json.loads(kf.read_text())
    return bytes.fromhex(data["raw_key_hex"])


def _newest_key_id() -> str:
    keys = sorted(KEYS_DIR.glob("*.key"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not keys:
        raise click.ClickException("No keys found. Run: dible-vault keygen")
    return json.loads(keys[0].read_text())["key_id"]


# ---------------------------------------------------------------------------
@click.group()
@click.version_option("2.0.0", prog_name="dible-vault")
def vault():
    """PORTAL VII DIBLE – encrypted vault CLI."""


# ---- status -----------------------------------------------------------------
@vault.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def status(as_json):
    """Show vault system status."""
    import platform
    import psutil
    device_id = DeviceIDGenerator().generate_device_id()
    info = {
        "vault_version": "2.0.0",
        "algorithm": "AES-256-GCM",
        "kdf": "PBKDF2-SHA256 / 600k iters",
        "device_id": device_id[:16] + "...",
        "keys_stored": len(list(KEYS_DIR.glob("*.key"))),
        "encrypted_files": len(list(DATA_DIR.glob("*.dible"))),
        "vault_home": str(VAULT_HOME),
        "python": platform.python_version(),
        "os": platform.system(),
    }
    if as_json:
        click.echo(json.dumps(info, indent=2))
        return
    console.print(BANNER)
    t = Table(title="Vault Status", box=box.ROUNDED, show_header=True,
              header_style="bold bright_blue")
    t.add_column("Property", style="bright_cyan", no_wrap=True)
    t.add_column("Value", style="white")
    for k, v in info.items():
        t.add_row(k, str(v))
    console.print(t)
    _audit("status", "OK")


# ---- keygen -----------------------------------------------------------------
@vault.command()
@click.option("--name", default="", help="Optional label for this key")
def keygen(name):
    """Generate and store a new 256-bit vault key."""
    engine = _engine()
    key_obj = engine.generate_key()
    meta = key_obj.to_dict()
    meta["label"] = name or key_obj.key_id
    meta["raw_key_hex"] = key_obj.raw_key.hex()
    kf = KEYS_DIR / f"{key_obj.key_id}.key"
    kf.write_text(json.dumps(meta, indent=2))
    kf.chmod(0o600)
    _audit("keygen", key_obj.key_id)
    console.print(Panel(
        f"[bright_green]Key generated[/bright_green]\n"
        f"[bright_cyan]key_id:[/bright_cyan] {key_obj.key_id}\n"
        f"[bright_cyan]file  :[/bright_cyan] {kf}",
        title="[bold]Key Generation", border_style="bright_blue",
    ))


# ---- list-keys --------------------------------------------------------------
@vault.command("list-keys")
def list_keys():
    """List all stored vault keys."""
    keys = list(KEYS_DIR.glob("*.key"))
    if not keys:
        console.print("[bright_yellow]No keys found. Run: dible-vault keygen[/bright_yellow]")
        return
    t = Table(title="Stored Keys", box=box.ROUNDED, header_style="bold bright_blue")
    t.add_column("key_id", style="bright_cyan")
    t.add_column("label", style="white")
    t.add_column("created", style="bright_black")
    for kf in keys:
        d = json.loads(kf.read_text())
        t.add_row(d.get("key_id", "?"), d.get("label", ""), d.get("created", "")[:19])
    console.print(t)


# ---- encrypt ----------------------------------------------------------------
@vault.command()
@click.option("-i", "--input",   "src",      required=True, help="Input file path")
@click.option("-o", "--output",  "dst",      default="",    help="Output file (default: ~/.portal7_dible/data/<name>.dible)")
@click.option("-k", "--key-id",  "kid",      default="",    help="Key ID (default: newest stored key)")
@click.option("-p", "--password",            default="",    help="Derive key from password instead of stored key")
def encrypt(src, dst, kid, password):
    """Encrypt a file."""
    engine = _engine()
    src_path = Path(src)
    if not src_path.exists():
        raise click.ClickException(f"Input file not found: {src}")
    dst_path = Path(dst) if dst else DATA_DIR / (src_path.name + ".dible")

    if password:
        key, salt = engine.derive_key(password)
        kid_used = hashlib.sha256(key).hexdigest()[:16]
        salt_hex = salt.hex()
    else:
        kid = kid or _newest_key_id()
        key = _load_key(kid)
        kid_used = kid
        salt_hex = ""

    blob = engine.encrypt(src_path.read_bytes(), key)
    meta = json.dumps({"key_id": kid_used, "salt": salt_hex,
                       "src_name": src_path.name}).encode()
    meta_len = len(meta).to_bytes(4, "big")
    dst_path.write_bytes(meta_len + meta + blob)
    dst_path.chmod(0o600)
    _audit("encrypt", f"{src} -> {dst_path}")
    console.print(
        f"[bright_green]Encrypted[/bright_green] {src_path.name} "
        f"→ [bright_cyan]{dst_path}[/bright_cyan]"
    )


# ---- decrypt ----------------------------------------------------------------
@vault.command()
@click.option("-i", "--input",   "src", required=True, help="Encrypted .dible file")
@click.option("-o", "--output",  "dst", default="",    help="Output plaintext file")
@click.option("-p", "--password",       default="",    help="Password (if file was encrypted with --password)")
def decrypt(src, dst, password):
    """Decrypt a .dible file."""
    engine = _engine()
    src_path = Path(src)
    if not src_path.exists():
        raise click.ClickException(f"File not found: {src}")
    raw = src_path.read_bytes()
    meta_len = int.from_bytes(raw[:4], "big")
    meta = json.loads(raw[4: 4 + meta_len])
    blob = raw[4 + meta_len:]
    if password:
        salt = bytes.fromhex(meta.get("salt") or secrets.token_bytes(32).hex())
        key, _ = engine.derive_key(password, salt)
    else:
        key = _load_key(meta["key_id"])
    pt = engine.decrypt(blob, key)
    out_name = meta.get("src_name", src_path.stem)
    dst_path = Path(dst) if dst else src_path.parent / out_name
    dst_path.write_bytes(pt)
    _audit("decrypt", f"{src} -> {dst_path}")
    console.print(
        f"[bright_green]Decrypted[/bright_green] {src_path.name} "
        f"→ [bright_cyan]{dst_path}[/bright_cyan]"
    )


# ---- encrypt-text -----------------------------------------------------------
@vault.command("encrypt-text")
@click.argument("text")
@click.option("-k", "--key-id", "kid", default="", help="Key ID (default: newest)")
@click.option("-p", "--password",      default="", help="Derive key from password")
def encrypt_text(text, kid, password):
    """Encrypt a text string and print JSON with base64 ciphertext."""
    import base64 as _b64
    engine = _engine()
    if password:
        key, salt = engine.derive_key(password)
        salt_hex = salt.hex()
        kid_used = hashlib.sha256(key).hexdigest()[:16]
    else:
        kid = kid or _newest_key_id()
        key = _load_key(kid)
        kid_used = kid
        salt_hex = ""
    blob = engine.encrypt(text.encode("utf-8"), key)
    out = {"ciphertext": _b64.b64encode(blob).decode(),
           "key_id": kid_used, "salt": salt_hex}
    console.print_json(json.dumps(out))
    _audit("encrypt-text", "OK")


# ---- decrypt-text -----------------------------------------------------------
@vault.command("decrypt-text")
@click.argument("ciphertext_b64")
@click.option("-k", "--key-id",  "kid",      default="", help="Key ID")
@click.option("-p", "--password",            default="", help="Password")
@click.option("-s", "--salt",    "salt_hex", default="", help="Salt hex (with --password)")
def decrypt_text(ciphertext_b64, kid, password, salt_hex):
    """Decrypt a base64 ciphertext string and print plaintext."""
    import base64 as _b64
    engine = _engine()
    blob = _b64.b64decode(ciphertext_b64)
    if password:
        salt = bytes.fromhex(salt_hex) if salt_hex else b"\x00" * 32
        key, _ = engine.derive_key(password, salt)
    else:
        key = _load_key(kid)
    pt = engine.decrypt(blob, key)
    console.print(f"[bright_green]{pt.decode(errors='replace')}[/bright_green]")
    _audit("decrypt-text", "OK")


# ---- delete-key -------------------------------------------------------------
@vault.command("delete-key")
@click.argument("key_id")
@click.option("--confirm", is_flag=True, required=True,
              help="Must pass --confirm to delete (irreversible)")
def delete_key(key_id, confirm):
    """Permanently delete a vault key."""
    kf = KEYS_DIR / f"{key_id}.key"
    if not kf.exists():
        raise click.ClickException(f"Key not found: {key_id}")
    kf.unlink()
    _audit("delete-key", key_id)
    console.print(f"[bright_red]Deleted key:[/bright_red] {key_id}")


# ---- audit ------------------------------------------------------------------
@vault.command()
@click.option("-n", "--lines", default=20, help="Last N audit entries (default: 20)")
def audit(lines):
    """Show vault audit log."""
    log = LOGS_DIR / "audit.log"
    if not log.exists():
        console.print("[bright_yellow]No audit log yet.[/bright_yellow]")
        return
    entries = log.read_text().strip().splitlines()[-lines:]
    t = Table(title="Audit Log", box=box.ROUNDED, header_style="bold bright_blue")
    t.add_column("timestamp",  style="bright_black", no_wrap=True)
    t.add_column("action",     style="bright_cyan")
    t.add_column("detail",     style="white")
    for line in entries:
        try:
            d = json.loads(line)
            t.add_row(d["ts"][:19], d["action"], d["detail"])
        except Exception:
            t.add_row("", "raw", line)
    console.print(t)


# ---- test -------------------------------------------------------------------
@vault.command()
def test():
    """Run built-in round-trip self-tests."""
    import tempfile
    engine = _engine()
    errors = []

    # 1. raw-key round-trip
    key_obj = engine.generate_key()
    pt = b"PORTAL VII DIBLE self-test payload 2026-09-15"
    ct = engine.encrypt(pt, key_obj.raw_key)
    if engine.decrypt(ct, key_obj.raw_key) != pt:
        errors.append("raw-key round-trip FAILED")
    else:
        console.print("[bright_green]\u2713[/bright_green] Raw-key AES-256-GCM round-trip")

    # 2. PBKDF2 round-trip
    passwd = secrets.token_urlsafe(16)
    k2, salt2 = engine.derive_key(passwd)
    ct2 = engine.encrypt(pt, k2)
    k2b, _ = engine.derive_key(passwd, salt2)
    if engine.decrypt(ct2, k2b) != pt:
        errors.append("PBKDF2 round-trip FAILED")
    else:
        console.print("[bright_green]\u2713[/bright_green] PBKDF2 key-derivation round-trip")

    # 3. tamper detection
    ct_bad = bytearray(ct)
    ct_bad[-1] ^= 0xFF
    try:
        engine.decrypt(bytes(ct_bad), key_obj.raw_key)
        errors.append("tamper detection FAILED")
    except Exception:
        console.print("[bright_green]\u2713[/bright_green] Tamper detection (GCM authentication)")

    # 4. file round-trip
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "test.txt"
        enc = Path(td) / "test.enc"
        dec = Path(td) / "test_out.txt"
        src.write_bytes(pt)
        engine.encrypt_file(str(src), str(enc), key_obj.raw_key)
        engine.decrypt_file(str(enc), str(dec), key_obj.raw_key)
        if dec.read_bytes() != pt:
            errors.append("file round-trip FAILED")
        else:
            console.print("[bright_green]\u2713[/bright_green] File encrypt/decrypt round-trip")

    if errors:
        for e in errors:
            console.print(f"[bright_red]\u2717 {e}[/bright_red]")
        sys.exit(1)
    console.print("\n[bold bright_green]All self-tests passed.[/bold bright_green]")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    vault()
