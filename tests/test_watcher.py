from types import SimpleNamespace

from urag import watcher


class FakeIndexer:
    def __init__(self, fail_rescan=False):
        self.paths = []
        self.fail_rescan = fail_rescan

    def index_paths(self, paths):
        self.paths.append(set(paths))
        return {"changed": 0, "deleted": 0}

    def index_all(self):
        if self.fail_rescan:
            raise RuntimeError("rescan failed")
        return {}


def test_move_events_enqueue_source_and_destination(tmp_path):
    indexer = FakeIndexer()
    handler = watcher._Handler(indexer, debounce=60)
    event = SimpleNamespace(
        is_directory=False,
        event_type="moved",
        src_path=str(tmp_path / "old.py"),
        dest_path=str(tmp_path / "new.py"),
    )

    handler.on_any_event(event)
    with handler._lock:
        pending = set(handler._pending)
        handler._timer.cancel()

    assert pending == {str(tmp_path / "old.py"), str(tmp_path / "new.py")}


def test_move_destination_outside_project_is_ignored(tmp_path):
    indexer = FakeIndexer()
    handler = watcher._Handler(indexer, debounce=60, project_root=tmp_path)
    event = SimpleNamespace(
        is_directory=False,
        event_type="moved",
        src_path=str(tmp_path / "old.py"),
        dest_path=str(tmp_path.parent / "outside.py"),
    )

    handler.on_any_event(event)
    with handler._lock:
        pending = set(handler._pending)
        handler._timer.cancel()

    assert pending == {str(tmp_path / "old.py")}


def test_rescan_failure_is_logged_and_watch_stops_cleanly(tmp_path, monkeypatch):
    indexer = FakeIndexer(fail_rescan=True)
    logs = []

    class FakeObserver:
        def schedule(self, *_args, **_kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def join(self):
            pass

    times = iter((0.0, 61.0, 61.0))
    monkeypatch.setattr(watcher, "Observer", FakeObserver)
    monkeypatch.setattr(watcher.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        watcher.time, "sleep", lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt)
    )

    watcher.run_watch(
        SimpleNamespace(project_root=tmp_path),
        indexer,
        rescan_minutes=1,
        log=logs.append,
    )

    assert "rescan error: rescan failed" in logs
