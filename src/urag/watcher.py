"""Watch daemon: debounced incremental re-indexing on file changes."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .config import Config
from .indexer import Indexer

DEBOUNCE_SECONDS = 1.5
UURAG_DIR_NAME = ".urag"
GIT_DIR_NAME = ".git"


class _Handler(FileSystemEventHandler):
    def __init__(self, indexer: Indexer, debounce: float = DEBOUNCE_SECONDS, log=None):
        self.indexer = indexer
        self.debounce = debounce
        self.log = log or print
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def _ignore(self, path: str) -> bool:
        parts = Path(path).parts
        return UURAG_DIR_NAME in parts or GIT_DIR_NAME in parts

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory and event.event_type in ("modified", "closed"):
            return
        if self._ignore(event.src_path):
            return
        with self._lock:
            self._pending.add(event.src_path)
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            paths = list(self._pending)
            self._pending.clear()
            self._timer = None
        if not paths:
            return
        try:
            stats = self.indexer.index_paths(Path(p) for p in paths)
            if stats["changed"] or stats["deleted"]:
                self.log(f"re-indexed {stats['changed']} changed, {stats['deleted']} deleted")
        except Exception as exc:  # keep the daemon alive
            self.log(f"index error: {exc}")


def run_watch(cfg: Config, indexer: Indexer, rescan_minutes: float = 0, log=print) -> None:
    handler = _Handler(indexer, log=log)
    observer = Observer()
    observer.schedule(handler, str(cfg.project_root), recursive=True)
    observer.start()
    log(f"watching {cfg.project_root} (debounce {DEBOUNCE_SECONDS}s)")
    last_rescan = time.monotonic()
    try:
        while True:
            if rescan_minutes > 0 and time.monotonic() - last_rescan > rescan_minutes * 60:
                indexer.index_all()
                last_rescan = time.monotonic()
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
