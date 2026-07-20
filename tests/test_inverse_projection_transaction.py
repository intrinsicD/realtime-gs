"""Root-free fault sentinels for the Iteration 1e JSON transaction protocol.

Only a synthetic development receipt domain is ever serialized.  No frozen official namespace
or root is constructed by these tests.
"""

from __future__ import annotations

import errno
import hashlib
import os
from dataclasses import FrozenInstanceError, replace

import pytest
from benchmarks import inverse_projection_fiber_transaction as tx

SYNTHETIC_FORBIDDEN_ROOTS = (8_001_001, 8_001_002)
SYNTHETIC_OFFICIAL_LITERALS = (
    "SYNTHETIC_ROOT_TRANSITION",
    "SYNTHETIC_GENERATORS_CONSUMED",
)


@pytest.fixture
def tmp_path(request, tmp_path_factory):
    """Module-local temp path without pytest's persistent ``*current`` symlink."""

    digest = hashlib.sha256(request.node.nodeid.encode("utf-8")).hexdigest()[:20]
    path = tmp_path_factory.getbasetemp() / f"case_{digest}"
    path.mkdir()
    return path


@pytest.fixture
def domain() -> tx.ReceiptDomain:
    return tx.ReceiptDomain(
        protocol_label="synthetic-transaction-protocol",
        label="development",
        namespace="synthetic.inverse-projection.development.iter1e.v1",
        schema_family="synthetic_inverse_projection_development_iter1e",
        permitted_root_consumption_statuses=(tx.DEVELOPMENT_ONLY,),
        forbidden_roots=SYNTHETIC_FORBIDDEN_ROOTS,
        forbidden_literals=SYNTHETIC_OFFICIAL_LITERALS,
    )


@pytest.fixture
def directory(tmp_path):
    descriptor = tx.open_directory(tmp_path)
    try:
        yield descriptor, tmp_path
    finally:
        os.close(descriptor)


def _receipt(domain: tx.ReceiptDomain, serial: int) -> dict[str, object]:
    return domain.make_receipt(
        "fault_artifact",
        {"serial": serial, "status": "DEVELOPMENT_FIXTURE"},
        root_consumption_status=tx.DEVELOPMENT_ONLY,
    )


def _prepare(
    descriptor: int,
    domain: tx.ReceiptDomain,
    target_name: str,
    serial: int,
    nonce: int,
    *,
    event_hook=None,
) -> tx.PreparedJSON:
    return tx.prepare_json(
        descriptor,
        target_name,
        domain,
        "fault_artifact",
        _receipt(domain, serial),
        nonce=f"{nonce:032x}",
        event_hook=event_hook,
    )


def _create_public(
    descriptor: int,
    domain: tx.ReceiptDomain,
    target_name: str,
    serial: int,
    nonce: int,
) -> tuple[tx.PreparedJSON, tx.Ownership]:
    prepared = _prepare(descriptor, domain, target_name, serial, nonce)
    report = tx.publish_exclusive(descriptor, prepared)
    assert report.public_entry is not None
    assert report.public_entry.ownership is not None
    return prepared, report.public_entry.ownership


def _one_shot(event_name, action):
    fired = False

    def hook(event, context):
        nonlocal fired
        if event == event_name and not fired:
            fired = True
            action(context)

    return hook


def test_receipt_domain_is_frozen_and_drives_all_reserved_metadata(domain):
    with pytest.raises(FrozenInstanceError):
        domain.namespace = "changed"  # type: ignore[misc]
    receipt = _receipt(domain, 1)
    assert receipt["schema"] == domain.schema("fault_artifact")
    assert receipt["namespace"] == domain.namespace
    assert receipt["root_consumption_status"] == tx.DEVELOPMENT_ONLY
    assert receipt["roots"] == []
    domain.validate_receipt("fault_artifact", receipt)

    with pytest.raises(tx.ReceiptDomainError, match="independently supply"):
        domain.make_receipt(
            "fault_artifact",
            {"schema": "caller-chosen"},
            root_consumption_status=tx.DEVELOPMENT_ONLY,
        )
    with pytest.raises(tx.ReceiptDomainError, match="development domains cannot permit roots"):
        replace(domain, permitted_roots=(1,))
    with pytest.raises(tx.ReceiptDomainError, match="DEVELOPMENT_ONLY"):
        replace(domain, permitted_root_consumption_statuses=("SYNTHETIC_ROOT_STARTED",))


def test_receipt_domain_narrowly_supports_iter2_without_changing_iter1e_default():
    domain = tx.ReceiptDomain(
        protocol_label="synthetic-iter2-transaction-protocol",
        label="official",
        namespace="synthetic.inverse-projection.iter2.v1",
        schema_family="synthetic_inverse_projection_iter2",
        permitted_root_consumption_statuses=("RESERVED_UNCONSUMED", "CONSUMED"),
        permitted_roots=(101, 102, 103),
        official_phases_permitted=True,
        commit_states_permitted=True,
        protocol_generation="iter2",
    )
    receipt = domain.make_receipt(
        "result",
        {"status": "FAIL"},
        root_consumption_status="CONSUMED",
        roots=(101, 102, 103),
        official_phase="FINAL",
        commit_state="COMMITTED",
    )
    assert receipt["schema"] == "synthetic_inverse_projection_iter2_result_v1"
    domain.validate_receipt("result", receipt)
    with pytest.raises(tx.ReceiptDomainError, match="protocol_generation"):
        replace(domain, protocol_generation="iter3")


def test_development_domain_rejects_official_shaped_values_before_disk(domain, directory):
    descriptor, path = directory
    official_shaped = dict(_receipt(domain, 2))
    official_shaped["namespace"] = "synthetic.inverse-projection.official.iter1e.v1"
    with pytest.raises(tx.ReceiptDomainError, match="namespace"):
        tx.prepare_json(
            descriptor,
            "rejected.json",
            domain,
            "fault_artifact",
            official_shaped,
            nonce="1" * 32,
        )
    forbidden = dict(_receipt(domain, 3))
    forbidden["diagnostic"] = f"root={SYNTHETIC_FORBIDDEN_ROOTS[0]}"
    with pytest.raises(tx.ReceiptDomainError, match="forbidden root"):
        domain.validate_receipt("fault_artifact", forbidden)
    assert not list(path.iterdir())


def test_prepare_is_canonical_durable_and_recovery_qualified(domain, directory):
    descriptor, path = directory
    events = []

    def observe(event, context):
        events.append((event, context.get("reason")))

    value = _receipt(domain, 4)
    prepared = tx.prepare_json(
        descriptor,
        "result.json",
        domain,
        "fault_artifact",
        value,
        nonce="2" * 32,
        event_hook=observe,
    )
    expected = tx.canonical_json_bytes(value)
    recovery = path / prepared.recovery_name
    assert prepared.recovery_name == f".result.json.recovery.{'2' * 32}.prepared"
    assert recovery.read_bytes() == expected
    assert prepared.payload_sha256 == prepared.ownership.sha256
    assert prepared.payload_size == len(expected)
    assert events.index(("before_prepared_file_fsync", None)) < events.index(
        ("after_prepared_file_fsync", None)
    )
    assert events.index(("after_prepared_file_fsync", None)) < events.index(
        ("before_directory_fsync", "prepared_json_created")
    )
    assert events.index(("before_directory_fsync", "prepared_json_created")) < events.index(
        ("after_directory_fsync", "prepared_json_created")
    )


def test_prepare_rejects_permission_drift_before_return(domain, directory):
    descriptor, path = directory

    def broaden_permissions(context):
        os.chmod(
            context["recovery_name"],
            0o666,
            dir_fd=context["dir_fd"],
            follow_symlinks=False,
        )

    with pytest.raises(tx.OwnershipCaptureError, match="private 0600"):
        _prepare(
            descriptor,
            domain,
            "result.json",
            401,
            401,
            event_hook=_one_shot("after_prepared_create", broaden_permissions),
        )
    assert not (path / "result.json").exists()


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_prepare_rejects_nonfinite_json_before_disk(domain, directory, nonfinite):
    descriptor, path = directory
    receipt = domain.make_receipt(
        "fault_artifact",
        {"status": "DEVELOPMENT_FIXTURE", "nonfinite": nonfinite},
        root_consumption_status=tx.DEVELOPMENT_ONLY,
    )
    with pytest.raises(ValueError, match="JSON compliant"):
        tx.prepare_json(
            descriptor,
            "nonfinite.json",
            domain,
            "fault_artifact",
            receipt,
            nonce="3" * 32,
        )
    assert not list(path.iterdir())


def test_exclusive_link_retains_recovery_and_preserves_collider(domain, directory):
    descriptor, path = directory
    first, first_ownership = _create_public(descriptor, domain, "result.json", 5, 5)
    assert (path / first.recovery_name).exists()
    assert (
        os.stat("result.json", dir_fd=descriptor).st_ino
        == os.stat(first.recovery_name, dir_fd=descriptor).st_ino
    )

    second = _prepare(descriptor, domain, "result.json", 6, 6)
    with pytest.raises(tx.OwnedMutationError) as caught:
        tx.publish_exclusive(descriptor, second)
    report = caught.value.report
    assert report.last_observed == tx.LAST_OBSERVED_NO_BEFORE_MUTATION
    assert report.public_entry is not None
    assert report.public_entry.ownership is not None
    assert report.public_entry.ownership.same_identity_and_hash(first_ownership)
    assert (path / second.recovery_name).exists()
    assert (path / "result.json").read_bytes() == tx.canonical_json_bytes(_receipt(domain, 5))


@pytest.mark.parametrize(
    "event_name",
    ("before_prepared_verification", "before_exclusive_link"),
)
def test_exclusive_link_rejects_pre_link_permission_drift(
    domain,
    directory,
    event_name,
):
    descriptor, path = directory
    prepared = _prepare(descriptor, domain, "result.json", 501, 501)

    def broaden_permissions(context):
        os.chmod(
            prepared.recovery_name,
            0o666,
            dir_fd=context["dir_fd"],
            follow_symlinks=False,
        )

    with pytest.raises(tx.OwnedMutationError) as caught:
        tx.publish_exclusive(
            descriptor,
            prepared,
            event_hook=_one_shot(event_name, broaden_permissions),
        )
    assert not caught.value.report.accepted
    assert caught.value.report.last_observed == tx.LAST_OBSERVED_NO_BEFORE_MUTATION
    assert not (path / "result.json").exists()


def test_exclusive_link_rejects_post_link_permission_drift(domain, directory):
    descriptor, _ = directory
    prepared = _prepare(descriptor, domain, "result.json", 502, 502)

    def broaden_permissions(context):
        os.chmod(
            context["target_name"],
            0o666,
            dir_fd=context["dir_fd"],
            follow_symlinks=False,
        )

    with pytest.raises(tx.OwnedMutationError) as caught:
        tx.publish_exclusive(
            descriptor,
            prepared,
            event_hook=_one_shot("after_exclusive_link", broaden_permissions),
        )
    assert not caught.value.report.accepted
    assert caught.value.report.recovery_uncertainty


@pytest.mark.parametrize("event_name", ("before_prepared_verification", "before_exchange"))
def test_owned_exchange_rejects_pre_exchange_permission_drift(
    domain,
    directory,
    event_name,
):
    descriptor, path = directory
    _old_prepared, old_ownership = _create_public(
        descriptor,
        domain,
        "result.json",
        503,
        503,
    )
    prepared = _prepare(descriptor, domain, "result.json", 504, 504)

    def broaden_permissions(context):
        os.chmod(
            prepared.recovery_name,
            0o666,
            dir_fd=context["dir_fd"],
            follow_symlinks=False,
        )

    with pytest.raises(tx.OwnedMutationError) as caught:
        tx.exchange_owned(
            descriptor,
            prepared,
            old_ownership,
            event_hook=_one_shot(event_name, broaden_permissions),
        )
    assert not caught.value.report.accepted
    assert caught.value.report.last_observed == tx.LAST_OBSERVED_NO_BEFORE_MUTATION
    assert (path / "result.json").read_bytes() == tx.canonical_json_bytes(_receipt(domain, 503))


def test_owned_exchange_rejects_post_exchange_permission_drift(domain, directory):
    descriptor, _ = directory
    _old_prepared, old_ownership = _create_public(
        descriptor,
        domain,
        "result.json",
        505,
        505,
    )
    prepared = _prepare(descriptor, domain, "result.json", 506, 506)

    def broaden_permissions(context):
        os.chmod(
            context["target_name"],
            0o666,
            dir_fd=context["dir_fd"],
            follow_symlinks=False,
        )

    with pytest.raises(tx.OwnedMutationError) as caught:
        tx.exchange_owned(
            descriptor,
            prepared,
            old_ownership,
            event_hook=_one_shot("after_exchange", broaden_permissions),
        )
    assert not caught.value.report.accepted
    assert caught.value.report.recovery_uncertainty


@pytest.mark.parametrize("mutated_name", ["public", "recovery"])
def test_exclusive_link_cannot_stale_accept_capture_complete_mutation(
    domain,
    directory,
    mutated_name,
):
    descriptor, _ = directory
    prepared = _prepare(descriptor, domain, "result.json", 601, 601)
    _create_public(descriptor, domain, "collider.json", 602, 602)
    after_link = False
    fired = False

    def mutate_after_capture(event, context):
        nonlocal after_link, fired
        if event == "after_exclusive_link":
            after_link = True
        target_name = "result.json" if mutated_name == "public" else prepared.recovery_name
        if (
            after_link
            and not fired
            and event == "capture_complete"
            and context.get("name") == target_name
        ):
            fired = True
            os.replace(
                "collider.json",
                target_name,
                src_dir_fd=descriptor,
                dst_dir_fd=descriptor,
            )

    with pytest.raises(tx.OwnedMutationError) as caught:
        tx.publish_exclusive(descriptor, prepared, event_hook=mutate_after_capture)
    report = caught.value.report
    assert fired
    assert not report.accepted
    assert report.recovery_uncertainty
    assert report.last_observed == tx.LAST_OBSERVED_UNKNOWN_AFTER_DISRUPTION


def test_capture_uses_same_fd_pread_and_rejects_path_replacement(domain, directory):
    descriptor, _ = directory
    _, original = _create_public(descriptor, domain, "owned.json", 7, 7)
    _, collider = _create_public(descriptor, domain, "collider.json", 8, 8)

    def replace_public(_context):
        os.replace(
            "collider.json",
            "owned.json",
            src_dir_fd=descriptor,
            dst_dir_fd=descriptor,
        )

    hook = _one_shot("after_capture_first_fstat", replace_public)
    with pytest.raises(
        tx.OwnershipCaptureError,
        match="metadata changed|path identity changed",
    ):
        tx.capture_entry(descriptor, "owned.json", event_hook=hook)
    observed = tx.capture_entry(descriptor, "owned.json")
    assert observed.ownership is not None
    assert observed.ownership.same_identity_and_hash(collider)
    assert not observed.ownership.same_identity_and_hash(original)


def test_capture_entry_bytes_returns_the_same_validated_inode_payload(domain, directory):
    descriptor, _ = directory
    prepared, ownership = _create_public(descriptor, domain, "owned.json", 701, 701)
    payload, snapshot = tx.capture_entry_bytes(
        descriptor,
        "owned.json",
        expected_sha256=ownership.sha256,
    )
    assert payload == tx.canonical_json_bytes(_receipt(domain, 701))
    assert snapshot.ownership is not None
    assert snapshot.ownership.same_identity_and_hash(prepared.ownership)


def test_capture_entry_bytes_enforces_exact_size_and_maximum(directory):
    descriptor, _ = directory
    raw = os.open(
        "bounded.bin",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
        dir_fd=descriptor,
    )
    try:
        os.write(raw, b"abcdef")
        os.fsync(raw)
    finally:
        os.close(raw)
    payload, _snapshot = tx.capture_entry_bytes(
        descriptor,
        "bounded.bin",
        expected_size=6,
        max_bytes=6,
    )
    assert payload == b"abcdef"
    with pytest.raises(tx.OwnershipCaptureError, match="size mismatch"):
        tx.capture_entry_bytes(
            descriptor,
            "bounded.bin",
            expected_size=5,
            max_bytes=6,
        )
    with pytest.raises(tx.OwnershipCaptureError, match="exceeds capture bound"):
        tx.capture_entry_bytes(descriptor, "bounded.bin", max_bytes=5)
    for kwargs in (
        {"max_bytes": True},
        {"max_bytes": 0},
        {"expected_size": False, "max_bytes": 6},
        {"expected_size": -1, "max_bytes": 6},
        {"expected_size": 7, "max_bytes": 6},
    ):
        with pytest.raises(ValueError):
            tx.capture_entry_bytes(descriptor, "bounded.bin", **kwargs)


def test_capture_entry_bytes_rejects_growth_after_initial_size(directory):
    descriptor, _ = directory
    raw = os.open(
        "growing.bin",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
        dir_fd=descriptor,
    )
    try:
        os.write(raw, b"start")
        os.fsync(raw)
    finally:
        os.close(raw)

    fired = False

    def append_after_size(event, context):
        nonlocal fired
        if event == "after_capture_first_fstat" and not fired:
            fired = True
            writer = os.open(
                context["name"],
                os.O_WRONLY | os.O_APPEND | os.O_CLOEXEC,
                dir_fd=context["dir_fd"],
            )
            try:
                os.write(writer, b"-growth")
                os.fsync(writer)
            finally:
                os.close(writer)

    with pytest.raises(tx.OwnershipCaptureError, match="grew beyond"):
        tx.capture_entry_bytes(
            descriptor,
            "growing.bin",
            max_bytes=1024,
            event_hook=append_after_size,
        )
    assert fired


def test_capture_rejects_symlink_directory_unreadable_and_checks_hash_last(
    domain,
    directory,
    monkeypatch,
):
    descriptor, _ = directory
    _create_public(descriptor, domain, "owned.json", 9, 9)
    os.symlink("owned.json", "link.json", dir_fd=descriptor)
    os.mkdir("directory.json", dir_fd=descriptor)
    os.mkfifo("fifo.json", dir_fd=descriptor)
    with pytest.raises(OSError):
        tx.capture_entry(descriptor, "link.json")
    with pytest.raises(tx.OwnershipCaptureError, match="not a regular"):
        tx.capture_entry(descriptor, "directory.json")
    with pytest.raises(tx.OwnershipCaptureError, match="not a regular"):
        tx.capture_entry(descriptor, "fifo.json")
    assert tx.observe_entry(descriptor, "link.json").state == "SYMLINK"
    assert tx.observe_entry(descriptor, "directory.json").state == "DIRECTORY"
    assert tx.observe_entry(descriptor, "fifo.json").state == "NON_REGULAR"

    original_open = tx.os.open

    def deny_owned(name, flags, *args, **kwargs):
        if name == "owned.json":
            raise PermissionError(errno.EACCES, "injected unreadable entry", name)
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(tx.os, "open", deny_owned)
    assert tx.observe_entry(descriptor, "owned.json").state == "UNREADABLE"
    monkeypatch.setattr(tx.os, "open", original_open)

    events = []
    with pytest.raises(tx.OwnershipCaptureError, match="hash mismatch"):
        tx.capture_entry(
            descriptor,
            "owned.json",
            expected_sha256="0" * 64,
            event_hook=lambda event, _context: events.append(event),
        )
    assert "before_capture_path_stat" in events
    assert events.index("before_capture_path_stat") > events.index("after_capture_second_fstat")
    os.unlink("link.json", dir_fd=descriptor)
    os.rmdir("directory.json", dir_fd=descriptor)
    os.unlink("fifo.json", dir_fd=descriptor)


def test_observe_entry_bounds_same_fd_capture_instability(domain, directory, monkeypatch):
    descriptor, _ = directory
    _create_public(descriptor, domain, "owned.json", 901, 901)
    original_capture = tx.capture_entry

    def fail_owned(dir_fd, name, **kwargs):
        if name == "owned.json":
            raise tx.OwnershipCaptureError("injected same-FD instability")
        return original_capture(dir_fd, name, **kwargs)

    monkeypatch.setattr(tx, "capture_entry", fail_owned)
    snapshot = tx.observe_entry(descriptor, "owned.json")
    assert snapshot.state == "UNSTABLE"
    assert snapshot.device is not None
    assert snapshot.inode is not None
    assert snapshot.mode is not None


def test_owned_exchange_accepts_only_expected_displacement_and_orders_fsync(
    domain,
    directory,
):
    descriptor, path = directory
    old_prepared, expected = _create_public(descriptor, domain, "lifecycle.json", 10, 10)
    new_prepared = _prepare(descriptor, domain, "lifecycle.json", 11, 11)
    report = tx.exchange_owned(descriptor, new_prepared, expected)
    assert report.accepted
    assert report.last_observed == tx.LAST_OBSERVED_YES_AFTER_EXCHANGE
    assert report.public_entry is not None and report.public_entry.ownership is not None
    assert report.public_entry.ownership.same_identity_and_hash(new_prepared.ownership)
    assert report.recovery_entry is not None and report.recovery_entry.ownership is not None
    assert report.recovery_entry.ownership.same_identity_and_hash(expected)
    assert (path / old_prepared.recovery_name).exists()
    assert (path / new_prepared.recovery_name).exists()
    assert report.events.index("after_exchange") < report.events.index("before_directory_fsync")
    assert report.events.index("after_directory_fsync") < report.events.index(
        "before_public_verification"
    )
    assert report.events.index("before_public_verification") < report.events.index(
        "before_displaced_verification"
    )


@pytest.mark.parametrize("collider_serial", [12, 13])
def test_same_and_different_content_inode_colliders_roll_back_unchanged(
    domain,
    directory,
    collider_serial,
):
    descriptor, path = directory
    _, expected = _create_public(descriptor, domain, "lifecycle.json", 12, 12)
    new_prepared = _prepare(descriptor, domain, "lifecycle.json", 14, 14)
    _, collider = _create_public(
        descriptor,
        domain,
        "collider.json",
        collider_serial,
        15 + collider_serial,
    )
    if collider_serial == 12:
        assert collider.sha256 == expected.sha256
        assert collider.inode != expected.inode

    def collide(_context):
        os.replace(
            "collider.json",
            "lifecycle.json",
            src_dir_fd=descriptor,
            dst_dir_fd=descriptor,
        )

    with pytest.raises(tx.OwnedMutationError) as caught:
        tx.exchange_owned(
            descriptor,
            new_prepared,
            expected,
            event_hook=_one_shot("before_exchange", collide),
        )
    report = caught.value.report
    assert report.last_observed == tx.LAST_OBSERVED_NO_AFTER_ROLLBACK
    assert not report.recovery_uncertainty
    current = tx.capture_entry(descriptor, "lifecycle.json")
    assert current.ownership is not None
    assert current.ownership.same_identity_and_hash(collider)
    recovery = tx.capture_entry(descriptor, new_prepared.recovery_name)
    assert recovery.ownership is not None
    assert recovery.ownership.same_identity_and_hash(new_prepared.ownership)
    assert (path / new_prepared.recovery_name).exists()


@pytest.mark.parametrize("collider_kind", ["symlink", "directory", "fifo"])
def test_quiescent_nonregular_colliders_are_rolled_back(
    domain,
    directory,
    collider_kind,
):
    descriptor, _ = directory
    _, expected = _create_public(descriptor, domain, "lifecycle.json", 160, 160)
    new_prepared = _prepare(descriptor, domain, "lifecycle.json", 161, 161)
    if collider_kind == "symlink":
        os.symlink("retained-target", "collider.json", dir_fd=descriptor)
        expected_state = "SYMLINK"
    elif collider_kind == "directory":
        os.mkdir("collider.json", dir_fd=descriptor)
        expected_state = "DIRECTORY"
    else:
        os.mkfifo("collider.json", dir_fd=descriptor)
        expected_state = "NON_REGULAR"

    def collide(_context):
        tx._rename_exchange(descriptor, "collider.json", "lifecycle.json")

    with pytest.raises(tx.OwnedMutationError) as caught:
        tx.exchange_owned(
            descriptor,
            new_prepared,
            expected,
            event_hook=_one_shot("before_exchange", collide),
        )
    report = caught.value.report
    assert report.last_observed == tx.LAST_OBSERVED_NO_AFTER_ROLLBACK
    assert not report.recovery_uncertainty
    public = tx.observe_entry(descriptor, "lifecycle.json")
    assert public.state == expected_state
    recovery = tx.capture_entry(descriptor, new_prepared.recovery_name)
    assert recovery.ownership is not None
    assert recovery.ownership.same_identity_and_hash(new_prepared.ownership)

    if collider_kind == "directory":
        os.rmdir("lifecycle.json", dir_fd=descriptor)
    else:
        os.unlink("lifecycle.json", dir_fd=descriptor)


def test_quiescent_unreadable_regular_collider_is_rolled_back(
    domain,
    directory,
    monkeypatch,
):
    descriptor, _ = directory
    _, expected = _create_public(descriptor, domain, "lifecycle.json", 170, 170)
    new_prepared = _prepare(descriptor, domain, "lifecycle.json", 171, 171)
    _, collider = _create_public(descriptor, domain, "collider.json", 172, 172)
    original_open = tx.os.open

    def deny_collider(name, flags, *args, **kwargs):
        dir_fd = kwargs.get("dir_fd")
        if dir_fd == descriptor and name in {"lifecycle.json", new_prepared.recovery_name}:
            try:
                metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError:
                metadata = None
            if metadata is not None and metadata.st_ino == collider.inode:
                raise PermissionError(errno.EACCES, "injected unreadable collider", name)
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(tx.os, "open", deny_collider)

    def collide(_context):
        tx._rename_exchange(descriptor, "collider.json", "lifecycle.json")

    with pytest.raises(tx.OwnedMutationError) as caught:
        tx.exchange_owned(
            descriptor,
            new_prepared,
            expected,
            event_hook=_one_shot("before_exchange", collide),
        )
    report = caught.value.report
    assert report.last_observed == tx.LAST_OBSERVED_NO_AFTER_ROLLBACK
    assert not report.recovery_uncertainty
    assert report.public_entry is not None
    assert report.public_entry.state == "UNREADABLE"
    assert report.public_entry.inode == collider.inode
    recovery = tx.capture_entry(descriptor, new_prepared.recovery_name)
    assert recovery.ownership is not None
    assert recovery.ownership.same_identity_and_hash(new_prepared.ownership)


def test_mutation_before_displaced_verification_rolls_back_to_observed_collider(
    domain,
    directory,
):
    descriptor, _ = directory
    _, expected = _create_public(descriptor, domain, "lifecycle.json", 20, 20)
    new_prepared = _prepare(descriptor, domain, "lifecycle.json", 21, 21)
    _, collider = _create_public(descriptor, domain, "collider.json", 22, 22)

    def mutate_recovery(context):
        os.replace(
            "collider.json",
            context["name"],
            src_dir_fd=descriptor,
            dst_dir_fd=descriptor,
        )

    with pytest.raises(tx.OwnedMutationError) as caught:
        tx.exchange_owned(
            descriptor,
            new_prepared,
            expected,
            event_hook=_one_shot("before_displaced_verification", mutate_recovery),
        )
    report = caught.value.report
    assert report.last_observed == tx.LAST_OBSERVED_NO_AFTER_ROLLBACK
    assert report.public_entry is not None and report.public_entry.ownership is not None
    assert report.public_entry.ownership.same_identity_and_hash(collider)
    assert report.recovery_entry is not None and report.recovery_entry.ownership is not None
    assert report.recovery_entry.ownership.same_identity_and_hash(new_prepared.ownership)


def test_public_mutation_before_displaced_verification_cannot_be_stale_accepted(
    domain,
    directory,
):
    descriptor, _ = directory
    _, expected = _create_public(descriptor, domain, "lifecycle.json", 23, 23)
    new_prepared = _prepare(descriptor, domain, "lifecycle.json", 24, 24)
    _create_public(descriptor, domain, "collider.json", 25, 25)

    def mutate_public(_context):
        os.replace(
            "collider.json",
            "lifecycle.json",
            src_dir_fd=descriptor,
            dst_dir_fd=descriptor,
        )

    with pytest.raises(tx.OwnedMutationError) as caught:
        tx.exchange_owned(
            descriptor,
            new_prepared,
            expected,
            event_hook=_one_shot("before_displaced_verification", mutate_public),
        )
    report = caught.value.report
    assert not report.accepted
    assert report.last_observed == tx.LAST_OBSERVED_UNKNOWN_AFTER_DISRUPTION
    assert report.recovery_uncertainty


def test_mutation_before_rollback_causes_unknown_and_stops_further_mutation(
    domain,
    directory,
):
    descriptor, _ = directory
    _, actual = _create_public(descriptor, domain, "lifecycle.json", 30, 30)
    expected = replace(actual, inode=actual.inode + 1)
    new_prepared = _prepare(descriptor, domain, "lifecycle.json", 31, 31)
    _create_public(descriptor, domain, "collider.json", 32, 32)

    def mutate_recovery(_context):
        tx._rename_exchange(descriptor, new_prepared.recovery_name, "collider.json")

    with pytest.raises(tx.OwnedMutationError) as caught:
        tx.exchange_owned(
            descriptor,
            new_prepared,
            expected,
            event_hook=_one_shot("before_rollback_exchange", mutate_recovery),
        )
    report = caught.value.report
    assert report.last_observed == tx.LAST_OBSERVED_UNKNOWN_AFTER_DISRUPTION
    assert report.recovery_uncertainty
    assert report.events.count("before_rollback_exchange") == 1
    assert "after_rollback_exchange" in report.events
    assert "before_rollback_recovery_verification" in report.events


def test_mutation_before_rollback_recovery_verification_is_unknown_and_retained(
    domain,
    directory,
):
    descriptor, _ = directory
    _, actual = _create_public(descriptor, domain, "lifecycle.json", 40, 40)
    expected = replace(actual, inode=actual.inode + 1)
    new_prepared = _prepare(descriptor, domain, "lifecycle.json", 41, 41)
    _create_public(descriptor, domain, "collider.json", 42, 42)

    def exchange_recovery(_context):
        tx._rename_exchange(descriptor, new_prepared.recovery_name, "collider.json")

    with pytest.raises(tx.OwnedMutationError) as caught:
        tx.exchange_owned(
            descriptor,
            new_prepared,
            expected,
            event_hook=_one_shot("before_rollback_recovery_verification", exchange_recovery),
        )
    report = caught.value.report
    assert report.last_observed == tx.LAST_OBSERVED_UNKNOWN_AFTER_DISRUPTION
    assert report.recovery_uncertainty
    prepared_elsewhere = tx.capture_entry(descriptor, "collider.json")
    assert prepared_elsewhere.ownership is not None
    assert prepared_elsewhere.ownership.same_identity_and_hash(new_prepared.ownership)


def test_public_mutation_before_rollback_recovery_verification_cannot_be_accepted(
    domain,
    directory,
):
    descriptor, _ = directory
    _, actual = _create_public(descriptor, domain, "lifecycle.json", 43, 43)
    expected = replace(actual, inode=actual.inode + 1)
    new_prepared = _prepare(descriptor, domain, "lifecycle.json", 44, 44)
    _, collider = _create_public(descriptor, domain, "collider.json", 45, 45)

    def exchange_public(_context):
        tx._rename_exchange(descriptor, "lifecycle.json", "collider.json")

    with pytest.raises(tx.OwnedMutationError) as caught:
        tx.exchange_owned(
            descriptor,
            new_prepared,
            expected,
            event_hook=_one_shot("before_rollback_recovery_verification", exchange_public),
        )
    report = caught.value.report
    assert report.last_observed == tx.LAST_OBSERVED_UNKNOWN_AFTER_DISRUPTION
    assert report.recovery_uncertainty
    public = tx.capture_entry(descriptor, "lifecycle.json")
    assert public.ownership is not None
    assert public.ownership.same_identity_and_hash(collider)


def test_exchange_failure_preserves_target_and_prepared_recovery(
    domain,
    directory,
    monkeypatch,
):
    descriptor, _ = directory
    _, expected = _create_public(descriptor, domain, "lifecycle.json", 50, 50)
    new_prepared = _prepare(descriptor, domain, "lifecycle.json", 51, 51)

    def fail_exchange(*_args, **_kwargs):
        raise OSError(errno.EIO, "injected exchange failure")

    monkeypatch.setattr(tx, "_rename_exchange", fail_exchange)
    with pytest.raises(tx.OwnedMutationError) as caught:
        tx.exchange_owned(descriptor, new_prepared, expected)
    report = caught.value.report
    assert report.last_observed == tx.LAST_OBSERVED_NO_BEFORE_MUTATION
    assert not report.recovery_uncertainty
    public = tx.capture_entry(descriptor, "lifecycle.json")
    recovery = tx.capture_entry(descriptor, new_prepared.recovery_name)
    assert public.ownership is not None and public.ownership.same_identity_and_hash(expected)
    assert recovery.ownership is not None
    assert recovery.ownership.same_identity_and_hash(new_prepared.ownership)


@pytest.mark.parametrize("fsync_event", ["before_directory_fsync", "after_directory_fsync"])
def test_fsync_disruption_after_exchange_is_unknown_without_rollback(
    domain,
    directory,
    fsync_event,
):
    descriptor, _ = directory
    _, expected = _create_public(descriptor, domain, "lifecycle.json", 60, 60)
    new_prepared = _prepare(descriptor, domain, "lifecycle.json", 61, 61)

    def fail_owned_fsync(context):
        if context["reason"] == "owned_exchange":
            raise OSError(errno.EIO, "injected directory fsync disruption")

    with pytest.raises(tx.OwnedMutationError) as caught:
        tx.exchange_owned(
            descriptor,
            new_prepared,
            expected,
            event_hook=_one_shot(fsync_event, fail_owned_fsync),
        )
    report = caught.value.report
    assert report.last_observed == tx.LAST_OBSERVED_UNKNOWN_AFTER_DISRUPTION
    assert report.recovery_uncertainty
    assert "before_rollback_exchange" not in report.events
    public = tx.capture_entry(descriptor, "lifecycle.json")
    recovery = tx.capture_entry(descriptor, new_prepared.recovery_name)
    assert public.ownership is not None
    assert public.ownership.same_identity_and_hash(new_prepared.ownership)
    assert recovery.ownership is not None and recovery.ownership.same_identity_and_hash(expected)


def test_capture_disruption_after_exchange_is_unknown_without_rollback(domain, directory):
    descriptor, _ = directory
    _, expected = _create_public(descriptor, domain, "lifecycle.json", 70, 70)
    new_prepared = _prepare(descriptor, domain, "lifecycle.json", 71, 71)

    def fail_capture(_context):
        raise OSError(errno.EIO, "injected public capture disruption")

    with pytest.raises(tx.OwnedMutationError) as caught:
        tx.exchange_owned(
            descriptor,
            new_prepared,
            expected,
            event_hook=_one_shot("before_public_verification", fail_capture),
        )
    report = caught.value.report
    assert report.last_observed == tx.LAST_OBSERVED_UNKNOWN_AFTER_DISRUPTION
    assert report.recovery_uncertainty
    assert "before_rollback_exchange" not in report.events
    assert tx.observe_entry(descriptor, "lifecycle.json").state == "REGULAR"
    assert tx.observe_entry(descriptor, new_prepared.recovery_name).state == "REGULAR"


def test_linux_exchange_capability_and_name_validation(domain, directory):
    descriptor, _ = directory
    tx.require_rename_exchange()
    with pytest.raises(ValueError, match="single"):
        _prepare(descriptor, domain, "nested/result.json", 80, 80)
    with pytest.raises(ValueError, match="nonce"):
        tx.prepare_json(
            descriptor,
            "result.json",
            domain,
            "fault_artifact",
            _receipt(domain, 81),
            nonce="too-short",
        )


@pytest.mark.parametrize("operation", ["exclusive_link", "owned_exchange"])
def test_preexisting_recovery_collider_is_retained_before_public_mutation(
    domain,
    directory,
    operation,
):
    descriptor, _ = directory
    expected = None
    if operation == "owned_exchange":
        _, expected = _create_public(descriptor, domain, "result.json", 90, 90)
    prepared = _prepare(descriptor, domain, "result.json", 91, 91)
    _, collider = _create_public(descriptor, domain, "collider.json", 92, 92)
    tx._rename_exchange(descriptor, prepared.recovery_name, "collider.json")

    with pytest.raises(tx.OwnedMutationError) as caught:
        if operation == "exclusive_link":
            tx.publish_exclusive(descriptor, prepared)
        else:
            assert expected is not None
            tx.exchange_owned(descriptor, prepared, expected)
    report = caught.value.report
    assert report.last_observed == tx.LAST_OBSERVED_NO_BEFORE_MUTATION
    assert not report.recovery_uncertainty
    assert "before_exchange" not in report.events
    assert "before_exclusive_link" not in report.events
    observed_recovery = tx.capture_entry(descriptor, prepared.recovery_name)
    assert observed_recovery.ownership is not None
    assert observed_recovery.ownership.same_identity_and_hash(collider)
    retained_prepared = tx.capture_entry(descriptor, "collider.json")
    assert retained_prepared.ownership is not None
    assert retained_prepared.ownership.same_identity_and_hash(prepared.ownership)
    public = tx.observe_entry(descriptor, "result.json")
    if operation == "exclusive_link":
        assert public.state == "ABSENT"
    else:
        assert public.ownership is not None
        assert public.ownership.same_identity_and_hash(expected)
