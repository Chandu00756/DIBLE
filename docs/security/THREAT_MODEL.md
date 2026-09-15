# DIBLE Threat Model (Draft)

## Assets
Device commitments, research keys, ciphertext artifacts, audit records, and policy decisions.

## Adversaries
Remote callers, unauthorized local users, artifact tamperers, device-clone attempts, and malicious insiders. Physical compromise, kernel compromise, side channels, and cryptanalytic attacks are out of scope for the Python reference implementation.

## Current claim
This code provides no production cryptographic claim. It is a research simulator; device commitments are identifiers, not proof of device possession.
