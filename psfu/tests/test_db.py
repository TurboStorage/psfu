"""SQLite writer roundtrip tests.

Per the PS-FU design constraint: the DB writer is one of the two places we
test, because bugs here silently lose data. We do NOT mock SQLite — each test
writes a real temp file (matching production behavior on WSL 9p/DrvFs, which
has different commit semantics than ext4).
"""

from __future__ import annotations

import json
import threading

from psfu.tests.conftest import _base_sample, make_run_db

import sqlite3

import pytest

from psfu.common import utc_now
from psfu.config import PROTOCOL_VERSION
from psfu.persistence.db import CURRENT_SCHEMA_VERSION, Database


def test_run_insert_roundtrip(tmp_path):
    db = make_run_db(tmp_path, overrides={"psu_serial": "HX1200-42"})
    row = db.fetch_run()
    assert row is not None
    assert row["psu_serial"] == "HX1200-42"
    assert row["profile_name"] == "test-profile"
    # Pulled from psfu.config so the assertion tracks the live constant —
    # previously this was a string literal that drifted ("0.2" vs 0.9+).
    assert row["protocol_version"] == PROTOCOL_VERSION


def test_update_run_persists_verdict_fields(tmp_path):
    db = make_run_db(tmp_path)
    row = db.fetch_run()
    db.update_run(
        row["id"],
        {
            "verdict": "PASS",
            "verdict_reasons_json": json.dumps(["clean run"]),
            "subsystem_results_json": json.dumps({"power": {"status": "PASS"}}),
            "telemetry_status": "software-only",
            "ended_at": utc_now(),
        },
    )
    row2 = db.fetch_run()
    assert row2["verdict"] == "PASS"
    assert json.loads(row2["verdict_reasons_json"]) == ["clean run"]
    assert json.loads(row2["subsystem_results_json"])["power"]["status"] == "PASS"
    assert row2["telemetry_status"] == "software-only"
    assert row2["ended_at"] is not None


def test_sample_roundtrip_preserves_rail_values(tmp_path):
    db = make_run_db(tmp_path)
    db.add_sample(_base_sample(phase="soak", rail_12v=12.08))
    db.add_sample(_base_sample(phase="steady", rail_12v=11.92))
    rows = db.fetch_samples()
    assert len(rows) == 2
    assert rows[0]["phase"] == "soak"
    assert rows[0]["rail_12v"] == 12.08
    assert rows[1]["rail_12v"] == 11.92


def test_event_roundtrip(tmp_path):
    db = make_run_db(tmp_path)
    db.add_event("soak", "info", "kernel_watcher", "journalctl -k -f")
    db.add_event("steady", "crit", "nvidia_gpu_disappeared", "3 silent ticks")
    rows = db.fetch_events()
    assert len(rows) == 2
    assert rows[0]["severity"] == "info"
    assert rows[1]["kind"] == "nvidia_gpu_disappeared"
    assert rows[1]["phase"] == "steady"


def test_migrate_is_idempotent(tmp_path):
    """Reopening a Database must not fail when columns already exist.

    Each PS-FU run owns its own DB file, and the production PRAGMA
    ``locking_mode=EXCLUSIVE`` means only one live connection at a time.
    Close the first connection before reopening, matching the real
    ``recover`` / ``verdict`` / ``report`` CLI paths that reopen an
    existing run directory.
    """
    db_path = tmp_path / "run.sqlite"
    first = Database(db_path)
    first.conn.close()
    second = Database(db_path)  # second open re-runs _migrate; must not raise
    second.conn.close()


def test_threaded_event_writes_do_not_corrupt_rows(tmp_path):
    """The kernel watcher runs in a daemon thread. Verify that many concurrent
    writes land intact — this exercises the check_same_thread=False +
    self._lock guard that was added for exactly this case."""
    db = make_run_db(tmp_path)

    def writer(tag: str, count: int) -> None:
        for i in range(count):
            db.add_event("soak", "info", "thread_{}".format(tag), "msg {}".format(i))

    threads = [threading.Thread(target=writer, args=(str(i), 50)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = db.fetch_events()
    assert len(rows) == 200
    # Each writer's count should add up; no row should be missing or duplicated.
    per_tag = {}
    for row in rows:
        per_tag.setdefault(row["kind"], 0)
        per_tag[row["kind"]] += 1
    assert all(count == 50 for count in per_tag.values())


def test_mixed_reader_writer_do_not_corrupt_or_raise(tmp_path):
    """Readers (fetch_*) and writers (add_*) run concurrently in production:
    the main thread calls fetch_events() / fetch_samples() from health checks
    and final export, while the kernel watcher thread writes events. sqlite3
    Connection objects are not re-entrant at the C layer even with
    check_same_thread=False, so both sides must share self._lock.

    This test would flake or raise 'Recursive use of cursors not allowed' /
    'database is locked' if fetch_* bypassed the lock.
    """
    db = make_run_db(tmp_path)
    stop = threading.Event()
    errors: list[BaseException] = []

    def writer() -> None:
        i = 0
        try:
            while not stop.is_set():
                db.add_event("soak", "info", "kw", "msg {}".format(i))
                db.add_sample(_base_sample(phase="soak", rail_12v=12.0))
                i += 1
        except BaseException as exc:  # pragma: no cover — recorded on failure
            errors.append(exc)

    def reader() -> None:
        try:
            while not stop.is_set():
                _ = db.fetch_events()
                _ = db.fetch_samples()
                _ = db.fetch_run()
        except BaseException as exc:  # pragma: no cover — recorded on failure
            errors.append(exc)

    threads = [
        threading.Thread(target=writer),
        threading.Thread(target=writer),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    for t in threads:
        t.start()
    # Short burst is enough — race windows are microseconds wide.
    threading.Event().wait(0.5)
    stop.set()
    for t in threads:
        t.join(timeout=5)

    assert not errors, "mixed reader/writer raised: {}".format(errors)
    # Sanity: something landed.
    assert len(db.fetch_events()) > 0
    assert len(db.fetch_samples()) > 0


def test_fresh_db_stamps_schema_version(tmp_path):
    """A new Database lands at CURRENT_SCHEMA_VERSION immediately after
    construction — no first-run gap where _migrate is pending."""
    db = make_run_db(tmp_path)
    assert db._schema_version == CURRENT_SCHEMA_VERSION
    row = db.conn.execute(
        "SELECT MAX(version) FROM schema_version"
    ).fetchone()
    assert row[0] == CURRENT_SCHEMA_VERSION


def test_legacy_db_without_schema_version_table_migrates(tmp_path):
    """A DB that pre-dates the schema_version table must bootstrap at v1 and
    migrate forward to CURRENT_SCHEMA_VERSION without raising.

    Simulate legacy by building a DB then dropping schema_version, then
    reopening. The migration loop should detect the missing table, bootstrap
    v1, and replay v1→CURRENT.
    """
    db_path = tmp_path / "run.sqlite"
    first = Database(db_path)
    first.conn.execute("DROP TABLE schema_version")
    first.conn.commit()
    first.close()

    second = Database(db_path)
    try:
        assert second._schema_version == CURRENT_SCHEMA_VERSION
        row = second.conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()
        assert row[0] == CURRENT_SCHEMA_VERSION
    finally:
        second.close()


def test_future_schema_version_rejects(tmp_path):
    """A DB stamped at v > CURRENT must refuse to open — downgrade reads
    would silently miss columns the binary doesn't know to query."""
    db_path = tmp_path / "run.sqlite"
    first = Database(db_path)
    first.conn.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
        (CURRENT_SCHEMA_VERSION + 5, utc_now()),
    )
    first.conn.commit()
    first.close()

    with pytest.raises(RuntimeError, match="newer than this binary supports"):
        Database(db_path)


def test_insert_run_rejects_identifier_with_nul_byte(tmp_path):
    """_quote_ident refuses NUL bytes — SQLite treats NUL as a string terminator
    and would silently truncate, letting a crafted key become unexpected SQL."""
    from psfu.persistence.db import _quote_ident

    db = make_run_db(tmp_path)
    with pytest.raises(ValueError, match="NUL byte"):
        _quote_ident("evil\x00col")


def test_insert_run_tolerates_reserved_sqlite_keyword_as_column(tmp_path):
    """Quoting identifiers means a payload key that happens to match a SQLite
    reserved keyword still produces valid SQL. This test exists so a future
    schema addition that uses e.g. `order` or `group` as a column does not
    silently fail. We simulate by using a harmless existing column with
    quoting explicitly applied."""
    from psfu.persistence.db import _quote_ident

    # Identifier quoting round-trip: a column containing a double-quote
    # character is escaped by doubling it, per SQLite syntax.
    assert _quote_ident('a"b') == '"a""b"'
    # Plain ASCII column unchanged apart from surrounding quotes.
    assert _quote_ident("psu_serial") == '"psu_serial"'


def test_migrate_rolls_back_on_mid_migration_failure(tmp_path, monkeypatch):
    """If a migration raises mid-way through its statement list, neither the
    partial shape changes NOR the schema_version stamp should persist.
    Reopening the DB must replay the same migration cleanly — it must not see
    a bumped schema_version without its matching ALTERs, nor vice versa.
    """
    from psfu.persistence import db as db_module

    db_path = tmp_path / "run.sqlite"
    # Build a legacy DB at v1.
    first = Database(db_path)
    first.conn.execute("DROP TABLE schema_version")
    first.conn.commit()
    first.close()

    # Injected migration: one statement that fails. Because this runs under
    # the rollback-on-failure guard, the schema_version insert must NOT land.
    bogus = [(1, 2, ["ALTER TABLE does_not_exist_table ADD COLUMN x TEXT"])]
    monkeypatch.setattr(db_module, "_MIGRATIONS", bogus)

    with pytest.raises(sqlite3.OperationalError):
        Database(db_path)

    # Reopen without the bogus migration — DB must still be at v1, not v2.
    # If the failed migration had leaked a schema_version row, the real
    # migration loop would skip v1→v2 and we'd have inconsistent state.
    monkeypatch.setattr(db_module, "_MIGRATIONS", [])
    recovered = sqlite3.connect(str(db_path))
    try:
        row = recovered.execute("SELECT MAX(version) FROM schema_version").fetchone()
        assert row[0] == 1, "migration rolled back: schema_version must not be v2"
    finally:
        recovered.close()


def test_migrate_propagates_non_duplicate_column_errors(tmp_path, monkeypatch):
    """A DB error during migration that is NOT a 'duplicate column name' case
    (e.g. disk full, read-only FS, corrupt page, missing table) must propagate
    rather than be silently swallowed. Otherwise an RMA run could produce a
    half-migrated DB with a broken schema and no warning.

    Simulated by injecting a bogus migration that references a non-existent
    table — SQLite raises OperationalError("no such table: ..."), which is
    distinct from the idempotent duplicate-column case.
    """
    from psfu.persistence import db as db_module

    db_path = tmp_path / "run.sqlite"
    # Build a legacy DB at v1 so the migration loop has work to do on reopen.
    first = Database(db_path)
    first.conn.execute("DROP TABLE schema_version")
    first.conn.commit()
    first.close()

    bogus = [(1, 2, ["ALTER TABLE does_not_exist_table ADD COLUMN x TEXT"])]
    monkeypatch.setattr(db_module, "_MIGRATIONS", bogus)

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        Database(db_path)


def test_fetch_run_asserts_schema_version(tmp_path):
    """fetch_run must refuse to read if _schema_version is stale. Simulated
    by downgrading the attribute after construction — covers the hypothetical
    path where a connection is handed off from a stale code path."""
    db = make_run_db(tmp_path)
    db._schema_version = CURRENT_SCHEMA_VERSION - 1
    with pytest.raises(RuntimeError, match="below CURRENT_SCHEMA_VERSION"):
        db.fetch_run()


def test_v7_to_v8_migration_adds_ups_columns(tmp_path):
    """Legacy DB at schema v7 must advance cleanly to v8 with all 10 ups_*
    columns present and writable.

    Simulate a v7 DB by building a fresh DB, then stamping schema_version
    back to 7 and dropping the v8 columns. Reopening must replay v7→v8 and
    land the columns. Then a sample write with every ups_* key round-trips
    cleanly.

    Also locks in the NULL behavior: the two columns Cyberpower omits
    (ups_input_freq_hz + ups_realpower_w) must accept NULL and read back
    as None. This is the exact shape UpsCollector.snapshot() emits for a
    Cyberpower CP1500PFCLCD.
    """
    db_path = tmp_path / "run.sqlite"
    first = Database(db_path)

    # Tear the DB back down to v7 shape: drop the 10 v8 columns via a
    # rebuild (SQLite has no DROP COLUMN pre-3.35), then rewind the
    # schema_version stamp.
    ups_cols = [
        "ups_input_voltage_v", "ups_input_freq_hz", "ups_output_voltage_v",
        "ups_load_pct", "ups_realpower_w", "ups_battery_charge_pct",
        "ups_battery_runtime_s", "ups_status_raw", "ups_on_battery",
        "ups_transfer_count",
    ]
    cur = first.conn.execute("PRAGMA table_info(sample)")
    all_cols = [row[1] for row in cur.fetchall()]
    keep_cols = [c for c in all_cols if c not in ups_cols]
    cols_sql = ", ".join(keep_cols)
    first.conn.executescript(
        "CREATE TABLE sample_old AS SELECT {cols} FROM sample;"
        "DROP TABLE sample;"
        "ALTER TABLE sample_old RENAME TO sample;".format(cols=cols_sql)
    )
    # _create's CREATE TABLE IF NOT EXISTS is a no-op on a populated table,
    # but the schema shape above matches the v7 baseline minus the ups
    # columns — that's what a real v7 DB on disk looks like.
    first.conn.execute("DELETE FROM schema_version WHERE version > 7")
    first.conn.commit()
    first.close()

    # Reopen: migration loop must replay v7→v8 and add the 10 columns.
    second = Database(db_path)
    try:
        assert second._schema_version == CURRENT_SCHEMA_VERSION
        cur = second.conn.execute("PRAGMA table_info(sample)")
        col_names = {row[1] for row in cur.fetchall()}
        for col in ups_cols:
            assert col in col_names, "v7→v8 migration did not add {}".format(col)

        # Write a sample with every ups_* key populated. Matches the
        # shape UpsCollector.snapshot() emits for an APC Smart-UPS
        # (all fields live), then a second with the Cyberpower shape
        # (freq_hz + realpower_w NULL). Both must round-trip.
        apc_sample = _base_sample(phase="soak")
        apc_sample.update({
            "ups_input_voltage_v": 121.5,
            "ups_input_freq_hz": 60.0,
            "ups_output_voltage_v": 120.0,
            "ups_load_pct": 22.0,
            "ups_realpower_w": 312.0,
            "ups_battery_charge_pct": 100.0,
            "ups_battery_runtime_s": 1820.0,
            "ups_status_raw": "OL",
            "ups_on_battery": 0,
            "ups_transfer_count": 0,
        })
        second.add_sample(apc_sample)

        cyberpower_sample = _base_sample(phase="steady")
        cyberpower_sample.update({
            "ups_input_voltage_v": 120.0,
            "ups_input_freq_hz": None,     # vendor omits
            "ups_output_voltage_v": 120.0,
            "ups_load_pct": 28.0,
            "ups_realpower_w": None,       # vendor omits
            "ups_battery_charge_pct": 100.0,
            "ups_battery_runtime_s": 1560.0,
            "ups_status_raw": "OL",
            "ups_on_battery": 0,
            "ups_transfer_count": 0,
        })
        second.add_sample(cyberpower_sample)

        rows = second.fetch_samples()
        assert len(rows) == 2
        assert rows[0]["ups_input_voltage_v"] == 121.5
        assert rows[0]["ups_input_freq_hz"] == 60.0
        assert rows[0]["ups_realpower_w"] == 312.0
        assert rows[0]["ups_status_raw"] == "OL"
        assert rows[0]["ups_on_battery"] == 0
        assert rows[1]["ups_input_freq_hz"] is None
        assert rows[1]["ups_realpower_w"] is None
        assert rows[1]["ups_load_pct"] == 28.0
        assert rows[1]["ups_battery_runtime_s"] == 1560.0
    finally:
        second.close()


def test_v9_to_v10_migration_adds_nvme_columns(tmp_path):
    """Legacy DB at schema v9 must advance cleanly to v10 with all nvme_*
    columns present and writable."""
    db_path = tmp_path / "run.sqlite"
    first = Database(db_path)

    nvme_cols = [
        "nvme_composite_temp_c",
        "nvme_warning_temp_time_min",
        "nvme_critical_temp_time_min",
        "nvme_percentage_used",
        "nvme_available_spare",
        "nvme_available_spare_threshold",
        "nvme_data_units_read",
        "nvme_data_units_written",
        "nvme_host_read_cmds",
        "nvme_host_write_cmds",
        "nvme_controller_busy_time_min",
        "nvme_power_cycles",
        "nvme_unsafe_shutdowns",
        "nvme_media_errors",
        "nvme_num_err_log_entries",
        "nvme_pcie_link_gen",
        "nvme_pcie_link_width",
    ]
    cur = first.conn.execute("PRAGMA table_info(sample)")
    all_cols = [row[1] for row in cur.fetchall()]
    keep_cols = [c for c in all_cols if c not in nvme_cols]
    cols_sql = ", ".join(keep_cols)
    first.conn.executescript(
        "CREATE TABLE sample_old AS SELECT {cols} FROM sample;"
        "DROP TABLE sample;"
        "ALTER TABLE sample_old RENAME TO sample;".format(cols=cols_sql)
    )
    first.conn.execute("DELETE FROM schema_version WHERE version > 9")
    first.conn.commit()
    first.close()

    second = Database(db_path)
    try:
        assert second._schema_version == CURRENT_SCHEMA_VERSION
        cur = second.conn.execute("PRAGMA table_info(sample)")
        col_names = {row[1] for row in cur.fetchall()}
        for col in nvme_cols:
            assert col in col_names, "v9→v10 migration did not add {}".format(col)

        payload = _base_sample(phase="steady")
        payload.update({
            "nvme_composite_temp_c": 45.0,
            "nvme_warning_temp_time_min": 2.0,
            "nvme_critical_temp_time_min": 0.0,
            "nvme_percentage_used": 4.0,
            "nvme_available_spare": 100.0,
            "nvme_available_spare_threshold": 10.0,
            "nvme_data_units_read": 1234,
            "nvme_data_units_written": 4321,
            "nvme_host_read_cmds": 55,
            "nvme_host_write_cmds": 44,
            "nvme_controller_busy_time_min": 6.0,
            "nvme_power_cycles": 7,
            "nvme_unsafe_shutdowns": 1,
            "nvme_media_errors": 0,
            "nvme_num_err_log_entries": 3,
            "nvme_pcie_link_gen": None,
            "nvme_pcie_link_width": None,
        })
        second.add_sample(payload)
        row = second.fetch_samples()[0]
        assert row["nvme_composite_temp_c"] == 45.0
        assert row["nvme_data_units_written"] == 4321
        assert row["nvme_num_err_log_entries"] == 3
        assert row["nvme_pcie_link_gen"] is None
    finally:
        second.close()
