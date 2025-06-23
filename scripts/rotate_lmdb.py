import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import lmdb
import orjson


def _parse_duration(value: str) -> timedelta:
    """Parse duration in the format '14d' or '24h'."""
    if value.endswith("d"):
        return timedelta(days=int(value[:-1]))
    if value.endswith("h"):
        return timedelta(hours=int(value[:-1]))
    raise ValueError("Invalid duration format. Use Nd or Nh")


def _should_delete(record: dict[str, Any], threshold: datetime) -> bool:
    """Check if the record is older than the threshold."""
    timestamp = record.get("updated_at") or record.get("created_at")
    if not timestamp:
        return False
    try:
        ts = datetime.fromisoformat(timestamp)
    except ValueError:
        return False
    return ts < threshold


def rotate_lmdb(path: Path, keep: str) -> None:
    """Remove records older than the specified duration."""
    env = lmdb.open(str(path), writemap=True, readahead=False, meminit=False)
    if keep == "all":
        with env.begin(write=True) as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                txn.delete(key)
        env.close()
        return

    delta = _parse_duration(keep)
    threshold = datetime.now() - delta

    with env.begin(write=True) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            try:
                record = orjson.loads(value)
            except orjson.JSONDecodeError:
                continue
            if _should_delete(record, threshold):
                txn.delete(key)
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove outdated LMDB records")
    parser.add_argument("path", type=Path, help="Path to LMDB directory")
    parser.add_argument(
        "keep",
        help="Retention period, e.g. 14d or 24h. Use 'all' to delete every record",
    )
    args = parser.parse_args()

    rotate_lmdb(args.path, args.keep)


if __name__ == "__main__":
    main()
