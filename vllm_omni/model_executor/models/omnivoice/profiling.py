
from __future__ import annotations

import os
import threading
from collections import deque

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def profile_enabled() -> bool:
    return os.environ.get("VLLM_OMNI_OMNIVOICE_PROFILE") == "1"


class _Section:
    __slots__ = ("name", "start_ev", "end_ev")

    def __init__(self, name: str, start_ev: torch.cuda.Event):
        self.name = name
        self.start_ev = start_ev
        self.end_ev: torch.cuda.Event | None = None


class InferenceProfiler:

    _aggregate_lock = threading.Lock()
    _aggregate: dict[int, dict[str, deque]] = {}
    _agg_window = 20
    _tls = threading.local()

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self._sections: list[_Section] = []
        self._stack: list[_Section] = []

    @classmethod
    def maybe_start(cls, batch_size: int) -> "InferenceProfiler | _Noop":
        if not profile_enabled():
            return _NOOP
        return cls(batch_size=batch_size)

    @classmethod
    def scope(cls, batch_size: int) -> "_ScopeCtx":
        return _ScopeCtx(batch_size)

    @classmethod
    def current(cls) -> "InferenceProfiler | _Noop":
        return getattr(cls._tls, "active", _NOOP)

    def section(self, name: str) -> "_SectionCtx":
        return _SectionCtx(self, name)

    def _begin(self, name: str) -> None:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        sec = _Section(name, ev)
        self._sections.append(sec)
        self._stack.append(sec)

    def _end(self) -> None:
        sec = self._stack.pop()
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        sec.end_ev = ev

    def report(self, request_label: str | None = None) -> None:
        if not self._sections:
            return
        torch.cuda.synchronize()
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for sec in self._sections:
            if sec.end_ev is None:
                continue
            ms = sec.start_ev.elapsed_time(sec.end_ev)
            totals[sec.name] = totals.get(sec.name, 0.0) + ms
            counts[sec.name] = counts.get(sec.name, 0) + 1

        forward_ms = totals.get("forward", sum(totals.values()))
        ordered = sorted(totals.items(), key=lambda kv: -kv[1])
        parts = []
        for name, ms in ordered:
            n = counts[name]
            mean = ms / n if n else 0.0
            parts.append(f"{name}={ms:.2f}ms(n={n},mean={mean:.3f})")
        suffix = f" req={request_label}" if request_label else ""
        logger.info(
            "OmniVoice profile B=%d total=%.2fms | %s%s",
            self.batch_size, forward_ms, " | ".join(parts), suffix,
        )

        with InferenceProfiler._aggregate_lock:
            bucket = InferenceProfiler._aggregate.setdefault(
                self.batch_size, {},
            )
            for name, ms in totals.items():
                dq = bucket.setdefault(name, deque(maxlen=self._agg_window))
                dq.append(ms)
            non_empty = [
                (name, dq)
                for name, dq in bucket.items()
                if len(dq) > 0
            ]
            if non_empty and max(len(dq) for _, dq in non_empty) >= self._agg_window:
                summary = []
                for name, dq in non_empty:
                    avg = sum(dq) / len(dq)
                    summary.append(f"{name}_avg={avg:.2f}ms")
                logger.info(
                    "OmniVoice profile rolling-%d B=%d | %s",
                    self._agg_window, self.batch_size, " | ".join(summary),
                )
                for _, dq in non_empty:
                    dq.clear()


class _SectionCtx:
    __slots__ = ("_prof", "_name")

    def __init__(self, prof: InferenceProfiler, name: str):
        self._prof = prof
        self._name = name

    def __enter__(self):
        self._prof._begin(self._name)
        return self._prof

    def __exit__(self, *exc):
        self._prof._end()
        return False


class _ScopeCtx:
    __slots__ = ("_batch_size", "_prof", "_prev")

    def __init__(self, batch_size: int):
        self._batch_size = batch_size
        self._prof: InferenceProfiler | _Noop = _NOOP
        self._prev = None

    def __enter__(self):
        if not profile_enabled():
            self._prof = _NOOP
            return self._prof
        self._prof = InferenceProfiler(batch_size=self._batch_size)
        self._prev = getattr(InferenceProfiler._tls, "active", None)
        InferenceProfiler._tls.active = self._prof
        self._prof._begin("forward")
        return self._prof

    def __exit__(self, *exc):
        if isinstance(self._prof, InferenceProfiler):
            self._prof._end()
            self._prof.report()
            InferenceProfiler._tls.active = self._prev
        return False


class _NoopSection:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Noop:

    def section(self, name: str) -> _NoopSection:
        return _NoopSection()

    def report(self, request_label: str | None = None) -> None:
        pass


_NOOP = _Noop()
