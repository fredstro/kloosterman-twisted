r"""
Portable checkpointing for long-running batch computations.

A checkpoint is a small three-method object -- ``load``, ``tick``, ``save`` --
that hides all storage and cadence concerns from the caller.  A batch loop
then looks like::

    ckpt = JSONFileCheckpoint("run.json", save_every=200)
    state = ckpt.load() or fresh_state()
    for i, item in enumerate(work):
        ... update state ...
        if ckpt.tick():
            ckpt.save(state)
    ckpt.save(state, force=True)

Swap :class:`JSONFileCheckpoint` for :class:`Checkpoint` (the no-op default),
an in-memory stub, or any other backend implementing the same three methods
without touching the loop.
"""

from __future__ import annotations

import json
import os
from typing import Optional


class Checkpoint:
    r"""
    No-op checkpoint.

    Serves both as the default (when checkpointing is off) and as the base
    class documenting the small interface every checkpoint backend must
    implement.

    EXAMPLES::

        sage: from kloosterman_twisted import Checkpoint
        sage: ck = Checkpoint()
        sage: ck.load() is None
        True
        sage: ck.tick()
        False
        sage: ck.save({"a": 1})   # no-op, returns None
    """

    def load(self) -> Optional[dict]:
        r"""
        Return the previously saved state, or ``None`` if none is available.

        OUTPUT: dict or ``None``.
        """
        return None

    def tick(self) -> bool:
        r"""
        Advance the internal counter and signal whether a save is due.

        OUTPUT: ``True`` when the caller should invoke :meth:`save`,
        ``False`` otherwise.
        """
        return False

    def save(self, state: dict, *, force: bool = False) -> None:
        r"""
        Persist ``state``.

        INPUT:

        - ``state`` -- dict to be persisted
        - ``force`` -- if ``True``, bypass any cadence logic
        """
        pass


class JSONFileCheckpoint(Checkpoint):
    r"""
    JSON file checkpoint with atomic writes and a fixed save cadence.

    INPUT:

    - ``path`` -- destination file
    - ``save_every`` -- number of :meth:`tick` calls between automatic saves
      (default ``200``); ``force=True`` bypasses this cadence

    EXAMPLES::

        sage: import os, tempfile
        sage: from kloosterman_twisted import JSONFileCheckpoint
        sage: with tempfile.TemporaryDirectory() as tmp:
        ....:     path = os.path.join(tmp, "ck.json")
        ....:     ck = JSONFileCheckpoint(path, save_every=3)
        ....:     _ = ck.load()   # None
        ....:     ck.save({"data": [1, 2, 3]})
        ....:     loaded = ck.load()
        sage: loaded
        {'data': [1, 2, 3]}
        sage: ck2 = JSONFileCheckpoint("/tmp/ignored", save_every=3)
        sage: (ck2.tick(), ck2.tick(), ck2.tick())
        (False, False, True)
    """

    def __init__(self, path: str, save_every: int = 200):
        self.path = path
        self.save_every = max(1, int(save_every))
        self._i = 0

    def load(self) -> Optional[dict]:
        if not os.path.exists(self.path):
            return None
        with open(self.path) as fp:
            return json.load(fp)

    def tick(self) -> bool:
        self._i += 1
        return self._i % self.save_every == 0

    def save(self, state: dict, *, force: bool = False) -> None:
        # ``force`` is accepted for interface parity; JSON writes are always
        # unconditional here (the caller decides via tick() when to invoke us).
        del force
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fp:
            json.dump(state, fp)
        os.replace(tmp, self.path)


def as_checkpoint(obj, *, save_every: int = 200) -> Checkpoint:
    r"""
    Coerce a caller-supplied ``checkpoint`` argument into a :class:`Checkpoint`.

    INPUT:

    - ``obj`` -- one of

      * ``None``                      -> :class:`Checkpoint` (no-op)
      * ``str`` / ``os.PathLike``     -> :class:`JSONFileCheckpoint`
      * any object with the three-method interface -> returned unchanged

    - ``save_every`` -- cadence forwarded when constructing a
      :class:`JSONFileCheckpoint` from a path

    OUTPUT: a :class:`Checkpoint`-compatible object.

    EXAMPLES::

        sage: from kloosterman_twisted import (
        ....:     as_checkpoint, Checkpoint, JSONFileCheckpoint)
        sage: isinstance(as_checkpoint(None), Checkpoint)
        True
        sage: isinstance(as_checkpoint("/tmp/run.json"), JSONFileCheckpoint)
        True
        sage: existing = JSONFileCheckpoint("/tmp/run.json")
        sage: as_checkpoint(existing) is existing
        True
    """
    if obj is None:
        return Checkpoint()
    if isinstance(obj, (str, os.PathLike)):
        return JSONFileCheckpoint(os.fspath(obj), save_every=save_every)
    return obj
