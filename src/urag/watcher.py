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
    def __init__(
        self,
        indexer: Indexer,
        debounce: float = DEBOUNCE_SECONDS,
        log=None,
        project_root: Path | None = None,
    ):
        self.indexer = indexer
        self.debounce = debounce
        self.log = log or print
        self.project_root = project_root.resolve() if project_root else None
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def _ignore(self, path: str) -> bool:
        parts = Path(path).parts
        if UURAG_DIR_NAME in parts or GIT_DIR_NAME in parts:
            return True
        if self.project_root:
            try:
                Path(path).resolve().relative_to(self.project_root)
            except ValueError:
                return True
        return False

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory and event.event_type in ("modified", "closed"):
            return
        paths = [event.src_path]
        dest_path = getattr(event, "dest_path", None)
        if dest_path:
            paths.append(dest_path)
        paths = [path for path in paths if not self._ignore(path)]
        if not paths:
            return
        with self._lock:
            self._pending.update(paths)
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(
                self.debounce, lambda: self._flush(self._timer)
            )
            self._timer.daemon = True
            self._timer.start()

    def _flush(self, timer: threading.Timer | None = None) -> None:
        with self._flush_lock:
            with self._lock:
                if timer is not None and self._timer is not timer:
                    return
                self._timer = None
                paths = list(self._pending)
                self._pending.clear()
            if not paths:
                return
            try:
                stats = self.indexer.index_paths(Path(p) for p in paths)
                if stats["changed"] or stats["deleted"]:
                    self.log(
                        f"re-indexed {stats['changed']} changed, {stats['deleted']} deleted"
                    )
            except Exception as exc:  # keep the daemon alive
                self.log(f"index error: {exc}")

    def index_all(self):
        with self._flush_lock:
            return self.indexer.index_all()


def run_watch(
    cfg: Config, indexer: Indexer, rescan_minutes: float = 0, log=print
) -> None:
    handler = _Handler(indexer, log=log, project_root=cfg.project_root)
    observer = Observer()
    observer.schedule(handler, str(cfg.project_root), recursive=True)
    observer.start()
    log(f"watching {cfg.project_root} (debounce {DEBOUNCE_SECONDS}s)")
    last_rescan = time.monotonic()
    try:
        while True:
            if (
                rescan_minutes > 0
                and time.monotonic() - last_rescan > rescan_minutes * 60
            ):
                try:
                    handler.index_all()
                except Exception as exc:
                    log(f"rescan error: {exc}")
                finally:
                    last_rescan = time.monotonic()
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
