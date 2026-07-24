r"""Tests for the pluggable checkpoint abstraction."""

import json

from kloosterman_twisted import (
    Checkpoint,
    JSONFileCheckpoint,
    as_checkpoint,
    zeta_kloosterman_run,
)


class InMemoryCheckpoint(Checkpoint):
    """Concrete example: a checkpoint that lives in RAM (useful for tests)."""

    def __init__(self, save_every=1):
        self.save_every = save_every
        self.state = None
        self.saves = 0
        self._i = 0

    def load(self):
        return self.state

    def tick(self):
        self._i += 1
        return self._i % self.save_every == 0

    def save(self, state, *, force=False):
        self.state = json.loads(json.dumps(state))  # deep copy through JSON
        self.saves += 1


def test_null_checkpoint_no_op():
    ckpt = Checkpoint()
    assert ckpt.load() is None
    assert ckpt.tick() is False
    ckpt.save({"a": 1})  # should not raise


def test_as_checkpoint_coercion(tmp_path):
    assert isinstance(as_checkpoint(None), Checkpoint)
    assert isinstance(as_checkpoint(str(tmp_path / "x.json")), JSONFileCheckpoint)
    ck = InMemoryCheckpoint()
    assert as_checkpoint(ck) is ck


def test_json_file_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "ckpt.json"
    ck = JSONFileCheckpoint(str(path), save_every=3)
    assert ck.load() is None
    ck.save({"data": [1, 2, 3]})
    assert path.exists()
    assert ck.load() == {"data": [1, 2, 3]}
    # cadence
    assert (ck.tick(), ck.tick(), ck.tick()) == (False, False, True)


def test_run_with_in_memory_checkpoint_resumes():
    ck = InMemoryCheckpoint(save_every=1)
    # First run: interrupt via time budget after one modulus by using cmax=11
    zeta_kloosterman_run(N=11, mn_pairs=[(1, 5)], cmax=22, checkpoint=ck)
    assert ck.saves > 0
    partial_rows = json.loads(json.dumps(ck.state["data"]["1,5"]))
    n_before = len(partial_rows)
    assert n_before > 0

    # Second run at higher cmax: must resume, not restart
    state = zeta_kloosterman_run(N=11, mn_pairs=[(1, 5)], cmax=66, checkpoint=ck)
    rows = state["data"]["1,5"]
    assert len(rows) > n_before
    cs = [r[0] for r in rows]
    assert cs == sorted(set(cs))            # no duplicate moduli
    assert rows[:n_before] == partial_rows


def test_run_with_path_checkpoint(tmp_path):
    path = tmp_path / "run.json"
    state1 = zeta_kloosterman_run(
        N=11, mn_pairs=[(1, 5)], cmax=44,
        checkpoint=str(path), checkpoint_every=1,
    )
    assert path.exists()
    # Resume: same target -> no extra work, identical rows
    state2 = zeta_kloosterman_run(
        N=11, mn_pairs=[(1, 5)], cmax=44,
        checkpoint=str(path), checkpoint_every=1,
    )
    assert state1["data"]["1,5"] == state2["data"]["1,5"]
