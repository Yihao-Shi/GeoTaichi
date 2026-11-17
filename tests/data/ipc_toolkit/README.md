# IPC Toolkit reference data

This directory contains the exact `friction/cube_cube` subset used by
GeoTaichi's frozen-friction parity test.  The 446 upstream JSON files are kept
inside one gzip-compressed tar archive so filesystems with large allocation
units do not turn a 6.5 MiB fixture into hundreds of MiB of small files.
Tests read the archive directly and never extract it to a machine-local
directory.

`SOURCE.json` records the upstream data and generator commits, the upstream
Git tree, the member count, and two SHA-256 checksums:

- `archive_sha256` hashes the checked-in archive bytes;
- `contents_sha256` hashes every original filename and payload in numeric
  order, separated by NUL bytes.

The upstream data is distributed under the MIT license reproduced in
`LICENSE`.  Only this required subset is vendored; no external checkout is
needed to run the test suite.
