"""Persistent evidence storage and integrity verification.

This module preserves the exact uploaded bytes, records their SHA-256
identity, and maintains a hash-chained audit log for the prototype.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"
ZERO_HASH = "0" * 64


class EvidenceStoreError(RuntimeError):
    """Raised when evidence storage or audit operations fail."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: dict[str, Any]) -> str:
    """Serialize an object deterministically for audit hashing."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_filename(filename: str) -> str:
    """Remove path components and unsafe characters from a filename."""
    basename = str(filename).replace("\\", "/").split("/")[-1].strip()
    basename = re.sub(r"[^A-Za-z0-9._-]+", "_", basename)
    basename = basename.strip("._")
    return (basename or "evidence.bin")[:120]


def _append_audit_event(
    storage_root: Path,
    *,
    event_type: str,
    evidence_id: str,
    expected_sha256: str,
    actual_sha256: str | None,
    status: str,
) -> dict[str, Any]:
    """Append an event linked to the preceding audit event."""
    audit_path = storage_root / "audit.jsonl"

    previous_hash = ZERO_HASH

    if audit_path.exists() and audit_path.stat().st_size:
        try:
            with audit_path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                position = stream.tell() - 1

                # Ignore trailing newline characters.
                while position >= 0:
                    stream.seek(position)
                    if stream.read(1) not in (b"\n", b"\r"):
                        break
                    position -= 1

                if position >= 0:
                    start = position
                    while start >= 0:
                        stream.seek(start)
                        if stream.read(1) in (b"\n", b"\r"):
                            break
                        start -= 1

                    stream.seek(start + 1)
                    last_line = stream.read(position - start).decode("utf-8")
                    previous_hash = json.loads(last_line)["event_hash"]

        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise EvidenceStoreError(
                "The last audit entry cannot be read. "
                "Preserve the audit file and investigate before continuing."
            ) from exc

    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp_utc": _utc_now(),
        "event_type": event_type,
        "evidence_id": evidence_id,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "status": status,
        "previous_event_hash": previous_hash,
    }

    event["event_hash"] = _sha256(
        _canonical_json(event).encode("utf-8")
    )

    try:
        with audit_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(_canonical_json(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise EvidenceStoreError("Could not append to the audit log.") from exc

    return event


def store_evidence(
    data: bytes,
    filename: str,
    storage_root: str | Path = "evidence_store",
) -> dict[str, Any]:
    """Store the exact uploaded bytes and create an evidence manifest."""
    if not isinstance(data, bytes):
        raise TypeError("Evidence data must be bytes.")

    if not data:
        raise ValueError("Cannot store an empty evidence file.")

    root = Path(storage_root).expanduser().resolve()
    originals_dir = root / "originals"
    records_dir = root / "records"

    originals_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)

    evidence_id = f"EV-{uuid.uuid4().hex.upper()}"
    original_filename = _safe_filename(filename)
    expected_hash = _sha256(data)

    stored_path = originals_dir / f"{evidence_id}__{original_filename}"
    manifest_path = records_dir / f"{evidence_id}.json"

    # Exclusive creation prevents accidentally overwriting an existing file.
    try:
        with stored_path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

        actual_hash = hashlib.sha256(stored_path.read_bytes()).hexdigest()

        if actual_hash != expected_hash:
            stored_path.unlink(missing_ok=True)
            raise EvidenceStoreError(
                "Stored bytes did not match the uploaded evidence hash."
            )

        record = {
            "schema_version": SCHEMA_VERSION,
            "evidence_id": evidence_id,
            "original_filename": original_filename,
            "ingested_at_utc": _utc_now(),
            "hash_algorithm": "SHA-256",
            "sha256": expected_hash,
            "file_size_bytes": len(data),
            "stored_relative_path": stored_path.relative_to(root).as_posix(),
            "storage_mode": "local_copy",
            "integrity_status_at_ingestion": "verified",
        }

        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)

        _append_audit_event(
            root,
            event_type="evidence_ingested",
            evidence_id=evidence_id,
            expected_sha256=expected_hash,
            actual_sha256=actual_hash,
            status="verified",
        )

        return record

    except Exception:
        # Do not silently leave a partial original when ingestion fails.
        # A manifest, if successfully written, is preserved for investigation.
        if not manifest_path.exists():
            stored_path.unlink(missing_ok=True)
        raise


def verify_evidence(
    evidence_id: str,
    storage_root: str | Path = "evidence_store",
) -> dict[str, Any]:
    """Compare the stored file against the hash in its evidence manifest."""
    if not re.fullmatch(r"EV-[0-9A-F]{32}", evidence_id):
        raise ValueError("Invalid evidence ID.")

    root = Path(storage_root).expanduser().resolve()
    manifest_path = root / "records" / f"{evidence_id}.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No evidence manifest exists for {evidence_id}."
        )

    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_hash = str(record["sha256"])
        expected_size = int(record["file_size_bytes"])
        relative_path = Path(record["stored_relative_path"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EvidenceStoreError(
            f"The manifest for {evidence_id} is unreadable or incomplete."
        ) from exc

    # Ensure a manifest cannot redirect verification outside the storage root.
    stored_path = (root / relative_path).resolve()
    try:
        stored_path.relative_to(root)
    except ValueError as exc:
        raise EvidenceStoreError(
            "The manifest contains a storage path outside the evidence root."
        ) from exc

    if not stored_path.is_file():
        result = {
            "evidence_id": evidence_id,
            "status": "missing",
            "expected_sha256": expected_hash,
            "actual_sha256": None,
            "expected_size_bytes": expected_size,
            "actual_size_bytes": None,
            "message": "The stored original file is missing.",
        }
    else:
        actual_hash = hashlib.sha256(stored_path.read_bytes()).hexdigest()
        actual_size = stored_path.stat().st_size
        verified = (
            actual_hash == expected_hash
            and actual_size == expected_size
        )

        result = {
            "evidence_id": evidence_id,
            "status": "verified" if verified else "mismatch",
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "expected_size_bytes": expected_size,
            "actual_size_bytes": actual_size,
            "message": (
                "Stored bytes match the manifest."
                if verified
                else "Stored bytes differ from the recorded evidence identity."
            ),
        }

    _append_audit_event(
        root,
        event_type="evidence_integrity_check",
        evidence_id=evidence_id,
        expected_sha256=expected_hash,
        actual_sha256=result["actual_sha256"],
        status=result["status"],
    )

    return result


def verify_audit_log(
    storage_root: str | Path = "evidence_store",
) -> dict[str, Any]:
    """Check every event hash and its link to the preceding event."""
    audit_path = Path(storage_root).expanduser().resolve() / "audit.jsonl"

    if not audit_path.exists():
        return {
            "valid": True,
            "event_count": 0,
            "message": "No audit events exist yet.",
        }

    previous_hash = ZERO_HASH
    event_count = 0

    try:
        with audit_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue

                event = json.loads(line)
                recorded_hash = event.pop("event_hash")
                linked_hash = event.get("previous_event_hash")

                if linked_hash != previous_hash:
                    return {
                        "valid": False,
                        "event_count": event_count,
                        "line": line_number,
                        "message": "Audit-chain link mismatch.",
                    }

                calculated_hash = _sha256(
                    _canonical_json(event).encode("utf-8")
                )

                if calculated_hash != recorded_hash:
                    return {
                        "valid": False,
                        "event_count": event_count,
                        "line": line_number,
                        "message": "Audit event hash mismatch.",
                    }

                previous_hash = recorded_hash
                event_count += 1

    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return {
            "valid": False,
            "event_count": event_count,
            "message": f"Could not validate the audit log: {exc}",
        }

    return {
        "valid": True,
        "event_count": event_count,
        "message": "All audit event hashes and links are valid.",
    }