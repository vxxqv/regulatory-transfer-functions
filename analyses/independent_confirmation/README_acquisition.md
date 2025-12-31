# Genetic input acquisition

`acquire_genetic_inputs.py` transfers the inputs locked by `config/genetic_confirmation_protocol.yaml`. It verifies pinned protocol and resource-table hashes and the frozen CD4 program files before any network request. It then enforces the reference denominator and all 16 outcome roles, checks live capacity, and handles one physical file at a time. Scientific payloads remain opaque bytes. The script does not decompress, decode, inspect, or analyze them.

Run a metadata preflight without writing data:

```powershell
python analyses/independent_confirmation/acquire_genetic_inputs.py --dry-run --max-files 1 --verbose
```

Acquire a bounded prefix of the locked sequence:

```powershell
python analyses/independent_confirmation/acquire_genetic_inputs.py --max-files 1 --verbose
```

The metadata pass writes a hashed, outcome-blind roster containing the complete physical file denominator. Each payload is written to `.part`, checked for its locked byte count and provider checksum when available, assigned a local SHA256, and atomically renamed. A hashed state record binds each resumable prefix to its URL, expected size, frozen identities, prefix hash, and strong ETag or Last-Modified value. Full requests use the matching conditional header. A range is used only when the stored state, current metadata, and partial response agree. Otherwise the current file restarts from byte zero. A complete validated partial file can be finalized without network access.

Only one process can operate on a data root. Remaining source bytes are added to the frozen summary, temporary, cache, and safety reserves before metadata access and before every transfer. The JSON checksum manifest and SHA256 sidecar use a recoverable journaled replacement. Repeated runs recover interrupted manifest transactions, verify completed files, and continue in locked order.

`--manifest-only` verifies the full saved roster and reconstructs every manifest row without network access. It returns 17 while the locked sequence is incomplete. `--remove-stale-parts` is the only general cleanup action and removes only regular `.part` files and their state records inside the selected data root. Completed source files are never deleted or replaced.

Exit codes are 0 for success, 2 for invalid command use, 10 for protocol failure, 11 for role or order failure, 12 for insufficient space, 13 for network failure, 14 for byte-count failure, 15 for checksum failure, 16 for unsafe paths, and 17 for an incomplete manifest-only check. A failed file or role is not replaced with another resource.
