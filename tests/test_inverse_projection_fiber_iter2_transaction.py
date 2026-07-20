"""Iteration 2 receipt-domain and owned-transition fault tests."""

from __future__ import annotations

import copy
import os

import pytest
from benchmarks import inverse_projection_fiber_iter2_transaction as i2tx
from benchmarks import inverse_projection_fiber_transaction as tx


def _receipt(
    kind: str,
    *,
    status: str,
    serial: int,
    phase: str,
    commit_state: str | None = None,
) -> dict[str, object]:
    return i2tx.official_domain().make_receipt(
        kind,
        {"serial": serial, "transaction_id": "a" * 32},
        root_consumption_status=status,
        roots=i2tx.OFFICIAL_ROOTS,
        official_phase=phase,
        commit_state=commit_state,
    )


def _intent(receipt: dict[str, object]) -> dict[str, object]:
    return {
        "expected_target_name": "RESULT.json",
        "expected_receipt_kind": "result",
        "expected_transaction_id": "a" * 32,
        "expected_root_consumption_status": receipt["root_consumption_status"],
        "expected_roots": i2tx.OFFICIAL_ROOTS,
        "expected_official_phase": "FINAL",
        "expected_commit_state": None,
    }


def test_iter2_domain_is_exact_and_rejects_out_of_domain_roots():
    domain = i2tx.official_domain()
    value = _receipt(
        "root_state",
        status=i2tx.RESERVED_UNCONSUMED,
        serial=0,
        phase="RESERVED",
    )
    domain.validate_receipt("root_state", value)
    assert tuple(value["roots"]) == i2tx.OFFICIAL_ROOTS
    changed = dict(value)
    changed["roots"] = [*i2tx.OFFICIAL_ROOTS[:-1], 999]
    with pytest.raises(tx.ReceiptDomainError, match="outside the domain"):
        domain.validate_receipt("root_state", changed)


def test_iter2_publish_exchange_and_capture_preserve_displaced_reservation(tmp_path):
    reservation = _receipt(
        "result_reservation",
        status=i2tx.RESERVED_UNCONSUMED,
        serial=1,
        phase="RESERVED",
    )
    initial = i2tx.publish_receipt(
        tmp_path,
        "RESULT.json",
        "result_reservation",
        reservation,
        nonce="1" * 32,
    )
    result = _receipt(
        "result",
        status=i2tx.CONSUMED,
        serial=2,
        phase="FINAL",
    )
    final = i2tx.exchange_receipt(
        tmp_path,
        "RESULT.json",
        "result",
        result,
        expected_public=initial["public"],
        nonce="2" * 32,
    )
    observed, descriptor = i2tx.capture_receipt(
        tmp_path,
        "RESULT.json",
        "result",
        expected_public=final["public"],
    )
    assert observed == result
    assert descriptor == final["public"]
    displaced = final["mutation"]["displaced_entry"]
    assert displaced["ownership"]["sha256"] == initial["public"]["ownership"]["sha256"]
    assert (tmp_path / final["prepared"]["recovery_name"]).exists()


def test_iter2_root_state_transition_is_monotonic_and_owned(tmp_path):
    states = []
    for serial, status in enumerate(
        (
            i2tx.RESERVED_UNCONSUMED,
            i2tx.CONSUMPTION_STARTED,
            i2tx.PARTIALLY_CONSUMED,
            i2tx.CONSUMED,
        )
    ):
        states.append(
            _receipt(
                "root_state",
                status=status,
                serial=serial,
                phase=f"ROOT_STATE_{serial}",
            )
        )
    publication = i2tx.publish_receipt(
        tmp_path,
        "ROOT_STATE.json",
        "root_state",
        states[0],
        nonce="3" * 32,
    )
    for serial, state in enumerate(states[1:], start=4):
        publication = i2tx.exchange_receipt(
            tmp_path,
            "ROOT_STATE.json",
            "root_state",
            state,
            expected_public=publication["public"],
            nonce=f"{serial:032x}",
        )
    observed, _descriptor = i2tx.capture_receipt(
        tmp_path,
        "ROOT_STATE.json",
        "root_state",
        expected_public=publication["public"],
    )
    assert observed["root_consumption_status"] == i2tx.CONSUMED


def test_iter2_exchange_fault_never_stale_accepts_public_replacement(tmp_path):
    reservation = _receipt(
        "result_reservation",
        status=i2tx.RESERVED_UNCONSUMED,
        serial=10,
        phase="RESERVED",
    )
    initial = i2tx.publish_receipt(
        tmp_path,
        "RESULT.json",
        "result_reservation",
        reservation,
        nonce="a" * 32,
    )
    collider = _receipt(
        "fault_artifact",
        status=i2tx.RESERVED_UNCONSUMED,
        serial=11,
        phase="FAULT",
    )
    i2tx.publish_receipt(
        tmp_path,
        "COLLIDER.json",
        "fault_artifact",
        collider,
        nonce="b" * 32,
    )
    fired = False

    def replace_after_exchange(event, context):
        nonlocal fired
        if event == "after_exchange" and not fired:
            fired = True
            os.replace(
                "COLLIDER.json",
                "RESULT.json",
                src_dir_fd=context["dir_fd"],
                dst_dir_fd=context["dir_fd"],
            )

    result = _receipt(
        "result",
        status=i2tx.CONSUMED,
        serial=12,
        phase="FINAL",
    )
    with pytest.raises(tx.OwnedMutationError) as caught:
        i2tx.exchange_receipt(
            tmp_path,
            "RESULT.json",
            "result",
            result,
            expected_public=initial["public"],
            nonce="c" * 32,
            event_hook=replace_after_exchange,
        )
    assert fired
    assert not caught.value.report.accepted
    assert caught.value.report.recovery_uncertainty


def test_iter2_prepared_handoff_is_canonical_and_does_not_publish_target(tmp_path):
    receipt = _receipt(
        "result",
        status=i2tx.CONSUMED,
        serial=20,
        phase="FINAL",
    )
    descriptor = i2tx.prepare_receipt(
        tmp_path,
        "RESULT.json",
        "result",
        receipt,
        nonce="d" * 32,
    )
    assert not (tmp_path / "RESULT.json").exists()
    assert (tmp_path / descriptor["recovery_name"]).is_file()

    observed, prepared, recovery = i2tx.capture_prepared(
        tmp_path,
        descriptor,
        **_intent(receipt),
    )
    assert observed == receipt
    assert i2tx.prepared_descriptor(prepared, directory=recovery["directory"]) == descriptor
    assert recovery["ownership"] == descriptor["ownership"]


def test_iter2_parent_exchanges_intent_bound_worker_prepared_result(tmp_path):
    reservation = _receipt(
        "result_reservation",
        status=i2tx.RESERVED_UNCONSUMED,
        serial=31,
        phase="RESERVED",
    )
    publication = i2tx.publish_receipt(
        tmp_path,
        "RESULT.json",
        "result_reservation",
        reservation,
        nonce="a" * 32,
    )
    result = _receipt(
        "result",
        status=i2tx.CONSUMED,
        serial=32,
        phase="FINAL",
    )
    prepared = i2tx.prepare_receipt(
        tmp_path,
        "RESULT.json",
        "result",
        result,
        nonce="b" * 32,
    )
    exchanged = i2tx.exchange_prepared_receipt(
        tmp_path,
        prepared,
        expected_public=publication["public"],
        **_intent(result),
    )
    observed, public = i2tx.capture_receipt(
        tmp_path,
        "RESULT.json",
        "result",
        expected_public=exchanged["public"],
    )
    assert observed == result
    assert public == exchanged["public"]
    assert exchanged["mutation"]["accepted"]


def test_iter2_prepared_descriptor_rejects_json_coercion_and_extra_keys(tmp_path):
    receipt = _receipt(
        "result",
        status=i2tx.CONSUMED,
        serial=21,
        phase="FINAL",
    )
    descriptor = i2tx.prepare_receipt(
        tmp_path,
        "RESULT.json",
        "result",
        receipt,
        nonce="e" * 32,
    )
    cases = []
    changed = copy.deepcopy(descriptor)
    changed["payload_size"] = True
    cases.append(changed)
    changed = copy.deepcopy(descriptor)
    changed["ownership"]["device"] = False
    cases.append(changed)
    changed = copy.deepcopy(descriptor)
    changed["unexpected"] = "field"
    cases.append(changed)
    changed = copy.deepcopy(descriptor)
    changed["recovery_name"] = ".OTHER.json.recovery." + "e" * 32 + ".prepared"
    cases.append(changed)
    for value in cases:
        with pytest.raises(ValueError):
            i2tx.prepared_from_descriptor(value)


def test_iter2_held_directory_fd_survives_path_replacement(tmp_path):
    original = tmp_path / "official"
    moved = tmp_path / "held_identity"
    original.mkdir()
    held_fd = tx.open_directory(original)
    try:
        original.rename(moved)
        original.mkdir()
        receipt = _receipt(
            "result",
            status=i2tx.CONSUMED,
            serial=22,
            phase="FINAL",
        )
        descriptor = i2tx.prepare_receipt(
            original,
            "RESULT.json",
            "result",
            receipt,
            nonce="f" * 32,
            directory_fd=held_fd,
        )
        assert not (original / descriptor["recovery_name"]).exists()
        assert (moved / descriptor["recovery_name"]).is_file()
        observed, _prepared, _snapshot = i2tx.capture_prepared(
            original,
            descriptor,
            **_intent(receipt),
            directory_fd=held_fd,
        )
        assert observed == receipt
    finally:
        os.close(held_fd)


def test_iter2_capture_prepared_rejects_recovery_replacement(tmp_path):
    receipt = _receipt(
        "result",
        status=i2tx.CONSUMED,
        serial=23,
        phase="FINAL",
    )
    descriptor = i2tx.prepare_receipt(
        tmp_path,
        "RESULT.json",
        "result",
        receipt,
        nonce="0" * 32,
    )
    replacement = tmp_path / "replacement"
    replacement.write_text("{}\n", encoding="utf-8")
    replacement.replace(tmp_path / descriptor["recovery_name"])
    with pytest.raises(tx.OwnershipCaptureError):
        i2tx.capture_prepared(tmp_path, descriptor, **_intent(receipt))


def test_iter2_capture_prepared_rejects_mode_change(tmp_path):
    receipt = _receipt(
        "result",
        status=i2tx.CONSUMED,
        serial=24,
        phase="FINAL",
    )
    descriptor = i2tx.prepare_receipt(
        tmp_path,
        "RESULT.json",
        "result",
        receipt,
        nonce="1" * 32,
    )
    os.chmod(tmp_path / descriptor["recovery_name"], 0o644)
    with pytest.raises(tx.OwnershipCaptureError):
        i2tx.capture_prepared(tmp_path, descriptor, **_intent(receipt))
    directory_fd = tx.open_directory(tmp_path)
    try:
        refreshed = tx.capture_entry(directory_fd, descriptor["recovery_name"])
    finally:
        os.close(directory_fd)
    assert refreshed.ownership is not None
    forged = copy.deepcopy(descriptor)
    forged["ownership"] = i2tx.ownership_dict(refreshed.ownership)
    forged["payload_sha256"] = refreshed.ownership.sha256
    forged["payload_size"] = refreshed.ownership.size
    with pytest.raises(ValueError, match="private 0600"):
        i2tx.capture_prepared(tmp_path, forged, **_intent(receipt))


def test_iter2_prepare_enforces_private_mode_despite_umask(tmp_path):
    receipt = _receipt(
        "result",
        status=i2tx.CONSUMED,
        serial=33,
        phase="FINAL",
    )
    previous = os.umask(0o200)
    try:
        descriptor = i2tx.prepare_receipt(
            tmp_path,
            "RESULT.json",
            "result",
            receipt,
            nonce="c" * 32,
        )
    finally:
        os.umask(previous)
    assert descriptor["ownership"]["mode"] & 0o777 == 0o600
    observed, _prepared, _recovery = i2tx.capture_prepared(
        tmp_path,
        descriptor,
        **_intent(receipt),
    )
    assert observed == receipt


def test_iter2_capture_prepared_rejects_different_held_directory(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    first_fd = tx.open_directory(first)
    second_fd = tx.open_directory(second)
    try:
        receipt = _receipt(
            "result",
            status=i2tx.CONSUMED,
            serial=25,
            phase="FINAL",
        )
        descriptor = i2tx.prepare_receipt(
            first,
            "RESULT.json",
            "result",
            receipt,
            nonce="2" * 32,
            directory_fd=first_fd,
        )
        with pytest.raises(tx.OwnershipCaptureError, match="directory differs"):
            i2tx.capture_prepared(
                first,
                descriptor,
                **_intent(receipt),
                directory_fd=second_fd,
            )
    finally:
        os.close(first_fd)
        os.close(second_fd)


def test_iter2_capture_prepared_binds_launcher_target_and_kind(tmp_path):
    domain = i2tx.official_domain()
    receipt = domain.make_receipt(
        "unplanned_kind",
        {"serial": 26, "transaction_id": "a" * 32},
        root_consumption_status=i2tx.CONSUMED,
        roots=i2tx.OFFICIAL_ROOTS,
        official_phase="FINAL",
    )
    descriptor = i2tx.prepare_receipt(
        tmp_path,
        "UNPLANNED_TARGET.json",
        "unplanned_kind",
        receipt,
        nonce="3" * 32,
    )
    with pytest.raises(tx.OwnershipCaptureError, match="target differs"):
        i2tx.capture_prepared(
            tmp_path,
            descriptor,
            **_intent(receipt),
        )


def test_iter2_prepared_descriptor_rejects_long_and_non_mapping_values(tmp_path):
    receipt = _receipt(
        "result",
        status=i2tx.CONSUMED,
        serial=27,
        phase="FINAL",
    )
    descriptor = i2tx.prepare_receipt(
        tmp_path,
        "RESULT.json",
        "result",
        receipt,
        nonce="4" * 32,
    )
    too_long = copy.deepcopy(descriptor)
    too_long["target_name"] = "x" * 204
    too_long["recovery_name"] = f".{too_long['target_name']}.recovery.{'4' * 32}.prepared"
    malformed = [None, {1: "value", "mixed": "keys"}]
    for value in [too_long, *malformed]:
        with pytest.raises(ValueError):
            i2tx.prepared_from_descriptor(value)

    bad_ownership = copy.deepcopy(descriptor)
    bad_ownership["ownership"] = None
    with pytest.raises(ValueError):
        i2tx.prepared_from_descriptor(bad_ownership)

    bad_directory = copy.deepcopy(descriptor)
    bad_directory["directory"] = {1: 2, "mixed": 3}
    with pytest.raises(ValueError):
        i2tx.prepared_from_descriptor(bad_directory)


@pytest.mark.parametrize(
    ("phase", "commit_state", "match"),
    (
        ("UNPLANNED", None, "official_phase differs"),
        (None, None, "official_phase differs"),
        ("FINAL", "COMMITTED", "commit_state presence differs"),
    ),
)
def test_iter2_capture_prepared_binds_phase_and_commit_presence(
    tmp_path,
    phase,
    commit_state,
    match,
):
    case_directory = tmp_path / f"case_{phase}_{commit_state}"
    case_directory.mkdir()
    receipt = i2tx.official_domain().make_receipt(
        "result",
        {"serial": 28, "transaction_id": "a" * 32},
        root_consumption_status=i2tx.CONSUMED,
        roots=i2tx.OFFICIAL_ROOTS,
        official_phase=phase,
        commit_state=commit_state,
    )
    descriptor = i2tx.prepare_receipt(
        case_directory,
        "RESULT.json",
        "result",
        receipt,
        nonce="5" * 32,
    )
    with pytest.raises(tx.OwnershipCaptureError, match=match):
        i2tx.capture_prepared(
            case_directory,
            descriptor,
            **_intent(receipt),
        )


def test_iter2_size_limit_rejects_before_prepare_publish_or_exchange(tmp_path, monkeypatch):
    receipt = _receipt(
        "result",
        status=i2tx.CONSUMED,
        serial=29,
        phase="FINAL",
    )
    prepare_directory = tmp_path / "prepare"
    publish_directory = tmp_path / "publish"
    exchange_directory = tmp_path / "exchange"
    prepare_directory.mkdir()
    publish_directory.mkdir()
    exchange_directory.mkdir()

    reservation = _receipt(
        "result_reservation",
        status=i2tx.RESERVED_UNCONSUMED,
        serial=30,
        phase="RESERVED",
    )
    initial = i2tx.publish_receipt(
        exchange_directory,
        "RESULT.json",
        "result_reservation",
        reservation,
        nonce="6" * 32,
    )
    before_names = sorted(path.name for path in exchange_directory.iterdir())
    before_bytes = (exchange_directory / "RESULT.json").read_bytes()
    monkeypatch.setattr(i2tx, "MAX_RECEIPT_BYTES", 8)

    with pytest.raises(ValueError, match="receipt-size bound"):
        i2tx.prepare_receipt(
            prepare_directory,
            "RESULT.json",
            "result",
            receipt,
            nonce="7" * 32,
        )
    with pytest.raises(ValueError, match="receipt-size bound"):
        i2tx.publish_receipt(
            publish_directory,
            "RESULT.json",
            "result",
            receipt,
            nonce="8" * 32,
        )
    with pytest.raises(ValueError, match="receipt-size bound"):
        i2tx.exchange_receipt(
            exchange_directory,
            "RESULT.json",
            "result",
            receipt,
            expected_public=initial["public"],
            nonce="9" * 32,
        )
    assert not list(prepare_directory.iterdir())
    assert not list(publish_directory.iterdir())
    assert sorted(path.name for path in exchange_directory.iterdir()) == before_names
    assert (exchange_directory / "RESULT.json").read_bytes() == before_bytes
