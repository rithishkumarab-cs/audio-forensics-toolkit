import json

from audio_forensics.evidence_store import (
    store_evidence,
    verify_audit_log,
    verify_evidence,
)


def test_storage_preserves_exact_original_bytes(tmp_path):
    original = b"\x00audio-original-data\xff"

    record = store_evidence(
        original,
        "recording.wav",
        storage_root=tmp_path,
    )

    stored_path = tmp_path / record["stored_relative_path"]

    assert stored_path.read_bytes() == original
    assert record["file_size_bytes"] == len(original)
    assert record["integrity_status_at_ingestion"] == "verified"


def test_verification_passes_for_unchanged_file(tmp_path):
    record = store_evidence(
        b"original recording",
        "recording.wav",
        storage_root=tmp_path,
    )

    result = verify_evidence(
        record["evidence_id"],
        storage_root=tmp_path,
    )

    assert result["status"] == "verified"
    assert result["expected_sha256"] == result["actual_sha256"]


def test_verification_detects_modified_file(tmp_path):
    record = store_evidence(
        b"original recording",
        "recording.wav",
        storage_root=tmp_path,
    )

    stored_path = tmp_path / record["stored_relative_path"]
    stored_path.write_bytes(b"modified recording")

    result = verify_evidence(
        record["evidence_id"],
        storage_root=tmp_path,
    )

    assert result["status"] == "mismatch"
    assert result["expected_sha256"] != result["actual_sha256"]


def test_verification_detects_missing_file(tmp_path):
    record = store_evidence(
        b"original recording",
        "recording.wav",
        storage_root=tmp_path,
    )

    stored_path = tmp_path / record["stored_relative_path"]
    stored_path.unlink()

    result = verify_evidence(
        record["evidence_id"],
        storage_root=tmp_path,
    )

    assert result["status"] == "missing"


def test_audit_log_verifies_and_detects_tampering(tmp_path):
    record = store_evidence(
        b"original recording",
        "recording.wav",
        storage_root=tmp_path,
    )

    verify_evidence(record["evidence_id"], storage_root=tmp_path)

    audit_path = tmp_path / "audit.jsonl"
    audit_result = verify_audit_log(storage_root=tmp_path)

    assert audit_result["valid"] is True
    assert audit_result["event_count"] == 2

    events = audit_path.read_text(encoding="utf-8").splitlines()
    event = json.loads(events[0])
    event["status"] = "tampered"
    events[0] = json.dumps(event)

    audit_path.write_text("\n".join(events) + "\n", encoding="utf-8")

    assert verify_audit_log(storage_root=tmp_path)["valid"] is False