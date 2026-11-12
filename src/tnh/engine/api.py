"""The Python API under the UCI layer: run the engine's search in a background thread,
read the latest result while it runs, stop it.

    eng = Engine(ChessEvaluator(...), Settings(...))
    h = Handle(eng)
    h.start(seconds=30, on_info=print)      # returns at once
    h.snapshot()                            # the latest Result (lines, values, masses, history)
    h.stop()                                # the final Result

Settings and positions are changed only while idle (`h.busy` is False).
"""

from __future__ import annotations

import threading

from tnh.engine.search import Engine, Result


class Handle:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Result | None = None
        self.error: BaseException | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, *, nodes: int | None = None, seconds: float | None = None, on_info=None,
              on_done=None, info_every: float = 1.0) -> None:
        if self.busy:
            raise RuntimeError("a search is running")
        self._stop.clear()
        self.error = None

        def info(r: Result) -> None:
            with self._lock:
                self._latest = r
            if on_info is not None:
                on_info(r)

        def run() -> None:
            try:
                r = self.engine.search(nodes=nodes, seconds=seconds, stop=self._stop, on_info=info,
                                       info_every=info_every)
                with self._lock:
                    self._latest = r
                if on_done is not None:
                    on_done(r)
            except BaseException as e:          # shown by the caller, not swallowed
                self.error = e
                if on_done is not None:
                    on_done(None)
                raise

        self._thread = threading.Thread(target=run, name="tnh-engine-search", daemon=True)
        self._thread.start()

    def snapshot(self) -> Result | None:
        with self._lock:
            return self._latest

    def wait(self) -> Result | None:
        if self._thread is not None:
            self._thread.join()
        if self.error is not None:
            raise self.error
        return self.snapshot()

    def stop(self) -> Result | None:
        self._stop.set()
        return self.wait()

    def require_idle(self) -> None:
        if self.busy:
            raise RuntimeError("change settings and positions only while no search runs")
