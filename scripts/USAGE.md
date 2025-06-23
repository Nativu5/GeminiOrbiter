# Scripts Usage

This directory contains maintenance utilities for the project.

## Table of Contents

- [dump_lmdb.py](#dump_lmdbpy)
- [rotate_lmdb.py](#rotate_lmdbpy)

## dump_lmdb.py

Dump records from an LMDB database as a JSON array. If no keys are provided, the script outputs every record. When keys are supplied, only the specified records are returned.

### Usage

Dump all entries:

```bash
python scripts/dump_lmdb.py /path/to/lmdb
```

Dump specific keys:

```bash
python scripts/dump_lmdb.py /path/to/lmdb key1 key2
```

## rotate_lmdb.py

Delete LMDB records older than a given duration or remove all records.

### Usage

Delete entries older than 14 days:

```bash
python scripts/rotate_lmdb.py /path/to/lmdb 14d
```

Delete all entries:

```bash
python scripts/rotate_lmdb.py /path/to/lmdb all
```
