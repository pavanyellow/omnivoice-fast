# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from collections import deque
from collections.abc import Iterable

from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionRequestState,
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    NewRequestData,
    SchedulerInterface,
)

logger = init_logger(__name__)


class _BaseScheduler(SchedulerInterface):
    """Shared queue/state bookkeeping for diffusion schedulers."""

    def __init__(self) -> None:
        self.od_config: OmniDiffusionConfig | None = None
        self._request_states: dict[str, DiffusionRequestState] = {}
        self._request_id_to_sched_req_id: dict[str, str] = {}
        self._step_id: int = 0
        self._waiting: deque[str] = deque()
        self._running: list[str] = []
        self._finished_req_ids: set[str] = set()
        self._max_batch_size: int = 1
        self._batch_strategy: str = "fifo"
        self._duration_bucket_tokens: int = 128
        self._duration_estimator = None
        self._pad_tolerance: float = 1.0

    def initialize(self, od_config: OmniDiffusionConfig) -> None:
        self.od_config = od_config
        self._request_states.clear()
        self._request_id_to_sched_req_id.clear()
        self._step_id = 0
        self._waiting.clear()
        self._running.clear()
        self._finished_req_ids.clear()
        # max_num_running_reqs caps how many requests the scheduler will pull
        # out of WAITING into RUNNING in a single schedule() call. Set via env
        # var VLLM_OMNI_DIFFUSION_BATCH_SIZE; default 1 preserves prior
        # single-request behavior so callers that rely on it stay unaffected.
        env_bs = os.environ.get("VLLM_OMNI_DIFFUSION_BATCH_SIZE")
        if env_bs is not None:
            try:
                self.max_num_running_reqs = max(1, int(env_bs))
            except ValueError:
                logger.warning(
                    "Invalid VLLM_OMNI_DIFFUSION_BATCH_SIZE=%r, "
                    "falling back to 1", env_bs,
                )
                self.max_num_running_reqs = 1
        else:
            self.max_num_running_reqs = 1
        self._batch_strategy = os.environ.get(
            "VLLM_OMNI_DIFFUSION_BATCH_STRATEGY", "fifo",
        ).strip().lower()
        if self._batch_strategy not in ("fifo", "duration_bucket"):
            logger.warning(
                "Invalid VLLM_OMNI_DIFFUSION_BATCH_STRATEGY=%r, falling back to fifo.",
                self._batch_strategy,
            )
            self._batch_strategy = "fifo"

        env_bucket = os.environ.get("VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS")
        if env_bucket is not None:
            try:
                self._duration_bucket_tokens = max(1, int(env_bucket))
            except ValueError:
                logger.warning(
                    "Invalid VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS=%r, "
                    "falling back to 128", env_bucket,
                )
                self._duration_bucket_tokens = 128
        else:
            self._duration_bucket_tokens = 128

        env_pad_tol = os.environ.get("VLLM_OMNI_DIFFUSION_PAD_TOLERANCE")
        if env_pad_tol is not None:
            try:
                self._pad_tolerance = max(1.0, float(env_pad_tol))
            except ValueError:
                logger.warning(
                    "Invalid VLLM_OMNI_DIFFUSION_PAD_TOLERANCE=%r, falling back to 1.0",
                    env_pad_tol,
                )
                self._pad_tolerance = 1.0
        else:
            self._pad_tolerance = 1.0

        self._duration_estimator = None
        if self._batch_strategy == "duration_bucket":
            try:
                from vllm_omni.model_executor.models.omnivoice.duration import (
                    RuleDurationEstimator,
                )

                self._duration_estimator = RuleDurationEstimator()
            except Exception:
                logger.warning(
                    "Duration-bucket batching could not load OmniVoice duration "
                    "estimator; falling back to character-length buckets.",
                    exc_info=True,
                )
            logger.info(
                "Diffusion scheduler using duration_bucket strategy "
                "(bucket_tokens=%d, max_batch=%d, pad_tolerance=%.2f).",
                self._duration_bucket_tokens,
                self.max_num_running_reqs,
                self._pad_tolerance,
            )
        self._reset_scheduler_state()

    def add_request(self, request: OmniDiffusionRequest) -> str:
        sched_req_id = self._make_sched_req_id(request)
        return self._add_request_with_sched_req_id(sched_req_id, request)

    def _add_request_with_sched_req_id(self, sched_req_id: str, request: OmniDiffusionRequest) -> str:
        state = DiffusionRequestState(sched_req_id=sched_req_id, req=request)
        self._request_states[sched_req_id] = state
        self._register_request_ids(request.request_ids, sched_req_id)
        self._waiting.append(sched_req_id)
        logger.debug("%s add_request: %s (waiting=%d)", self.__class__.__name__, sched_req_id, len(self._waiting))
        return sched_req_id

    def schedule(self) -> DiffusionSchedulerOutput:
        scheduled_new_reqs: list[NewRequestData] = []
        scheduled_cached_req_ids: list[str] = []

        # First, schedule the RUNNING request(s)
        for sched_req_id in self._running:
            state = self._request_states.get(sched_req_id)
            if state is not None:
                scheduled_cached_req_ids.append(sched_req_id)

        # Second, schedule WAITING requests while capacity remains.
        capacity = self.max_num_running_reqs - len(self._running)
        if self._batch_strategy == "duration_bucket":
            waiting_to_schedule = self._select_duration_bucket_waiting(
                capacity,
                anchor_req_ids=self._running if self._running else None,
            )
        else:
            waiting_to_schedule = self._select_fifo_waiting(
                capacity,
            )

        for sched_req_id in waiting_to_schedule:
            state = self._request_states.get(sched_req_id)
            if state is None:
                continue
            was_new_request = state.status == DiffusionRequestStatus.WAITING
            state.status = DiffusionRequestStatus.RUNNING
            self._running.append(sched_req_id)
            if was_new_request:
                scheduled_new_reqs.append(NewRequestData.from_state(state))
            else:
                scheduled_cached_req_ids.append(sched_req_id)

        scheduler_output = DiffusionSchedulerOutput(
            step_id=self._step_id,
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=CachedRequestData(sched_req_ids=scheduled_cached_req_ids),
            finished_req_ids=set(self._finished_req_ids),
            num_running_reqs=len(self._running),
            num_waiting_reqs=len(self._waiting),
        )

        # update after schedule
        self._step_id += 1
        self._finished_req_ids.clear()
        return scheduler_output

    def has_requests(self) -> bool:
        return bool(self._waiting or self._running)

    def get_request_state(self, sched_req_id: str) -> DiffusionRequestState | None:
        return self._request_states.get(sched_req_id)

    def get_sched_req_id(self, request_id: str) -> str | None:
        return self._request_id_to_sched_req_id.get(request_id)

    def pop_request_state(self, sched_req_id: str) -> DiffusionRequestState | None:
        self._pop_extra_request_state(sched_req_id)
        state = self._request_states.pop(sched_req_id, None)
        if state is not None:
            self._unregister_request_ids(state.req.request_ids, sched_req_id)
        return state

    def preempt_request(self, sched_req_id: str) -> bool:
        if sched_req_id not in self._request_states:
            return False
        if sched_req_id in self._running:
            self._running.remove(sched_req_id)
            self._waiting.appendleft(sched_req_id)
            self._request_states[sched_req_id].status = DiffusionRequestStatus.PREEMPTED
            return True
        return False

    def finish_requests(self, sched_req_ids: str | list[str], status: DiffusionRequestStatus) -> None:
        assert DiffusionRequestStatus.is_finished(status)
        if isinstance(sched_req_ids, str):
            sched_req_ids = [sched_req_ids]
        self._finish_requests({sched_req_id: status for sched_req_id in sched_req_ids})

    def close(self) -> None:
        self._request_states.clear()
        self._request_id_to_sched_req_id.clear()
        self._waiting.clear()
        self._running.clear()
        self._finished_req_ids.clear()
        self._reset_scheduler_state()

    def _finish_requests(
        self,
        statuses: dict[str, DiffusionRequestStatus],
        errors: dict[str, str | None] | None = None,
    ) -> set[str]:
        if not statuses:
            return set()

        finished_req_ids: set[str] = set()
        running_to_remove: set[str] = set()
        waiting_to_remove: set[str] = set()

        for sched_req_id, status in statuses.items():
            assert DiffusionRequestStatus.is_finished(status)
            state = self._request_states.get(sched_req_id)
            if state is None or state.is_finished():
                continue

            finished_req_ids.add(sched_req_id)
            if sched_req_id in self._running:
                running_to_remove.add(sched_req_id)
            if sched_req_id in self._waiting:
                waiting_to_remove.add(sched_req_id)

        if running_to_remove:
            self._running = [sched_req_id for sched_req_id in self._running if sched_req_id not in running_to_remove]
        if waiting_to_remove:
            self._waiting = deque(
                sched_req_id for sched_req_id in self._waiting if sched_req_id not in waiting_to_remove
            )

        for sched_req_id in finished_req_ids:
            state = self._request_states[sched_req_id]
            status = statuses[sched_req_id]
            state.status = status
            if status == DiffusionRequestStatus.FINISHED_ERROR:
                state.error = None if errors is None else errors.get(sched_req_id)
            else:
                state.error = None

        self._finished_req_ids |= finished_req_ids
        return finished_req_ids

    def _finalize_update_from_output(
        self,
        sched_output: DiffusionSchedulerOutput,
        statuses: dict[str, DiffusionRequestStatus],
        errors: dict[str, str | None] | None = None,
    ) -> set[str]:
        # A scheduled request may be aborted after schedule() but before
        # update_from_output() processes the runner output. It is already
        # marked finished at that point, but we still need to surface its id
        # in this update so the engine can observe the terminal state.
        finished_req_ids = {
            sched_req_id for sched_req_id in sched_output.scheduled_req_ids if sched_req_id in self._finished_req_ids
        }
        finished_req_ids |= self._finish_requests(statuses, errors)
        return finished_req_ids

    def _reset_scheduler_state(self) -> None:
        """Reset subclass-owned state during initialize()/close()."""

    def _pop_extra_request_state(self, sched_req_id: str) -> None:
        """Remove subclass-owned per-request state before popping request state."""

    def _can_schedule_waiting(self, state: DiffusionRequestState) -> bool:
        del state
        return True

    def _prefer_waiting_requests(self) -> bool:
        """Prefer never-run WAITING requests over PREEMPTED continuations."""
        return False

    def _has_preferred_waiting_request(self) -> bool:
        if not self._prefer_waiting_requests():
            return False
        for sched_req_id in self._waiting:
            state = self._request_states.get(sched_req_id)
            if state is not None and state.status == DiffusionRequestStatus.WAITING:
                return True
        return False

    def _select_fifo_waiting(self, capacity: int) -> list[str]:
        selected: list[str] = []
        deferred: list[str] = []
        prefer_waiting_present = self._has_preferred_waiting_request()
        while self._waiting and len(selected) < capacity:
            sched_req_id = self._waiting[0]
            state = self._request_states.get(sched_req_id)
            if state is None:
                self._waiting.popleft()
                continue
            if not self._can_schedule_waiting(state):
                break
            if prefer_waiting_present and state.status != DiffusionRequestStatus.WAITING:
                deferred.append(self._waiting.popleft())
                continue

            self._waiting.popleft()
            selected.append(sched_req_id)
        if deferred:
            self._waiting.extendleft(reversed(deferred))
        return selected

    def _select_duration_bucket_waiting(
        self,
        capacity: int,
        anchor_req_ids: Iterable[str] | None = None,
    ) -> list[str]:
        if capacity <= 0:
            return []

        use_tolerance = self._pad_tolerance > 1.0
        prefer_waiting_present = self._has_preferred_waiting_request()
        anchor_states = [
            state
            for req_id in (anchor_req_ids or [])
            if (state := self._request_states.get(req_id)) is not None
        ]

        if anchor_states:
            if use_tolerance:
                anchor_tokens = [
                    max(1, self._estimate_target_tokens(state))
                    for state in anchor_states
                ]
                batch_max = max(anchor_tokens)
                batch_min = min(anchor_tokens)
                anchor_bucket = -1  # unused
                if (batch_max / batch_min) > self._pad_tolerance:
                    return []
            else:
                running_buckets = {
                    self._duration_bucket(state)
                    for state in anchor_states
                }
                if len(running_buckets) != 1:
                    return []
                anchor_bucket = next(iter(running_buckets))
        else:
            while self._waiting and self._request_states.get(self._waiting[0]) is None:
                self._waiting.popleft()
            if not self._waiting:
                return []

            anchor_id = self._waiting[0]
            if self._prefer_waiting_requests():
                for candidate_id in self._waiting:
                    candidate_state = self._request_states.get(candidate_id)
                    if candidate_state is None:
                        continue
                    if candidate_state.status == DiffusionRequestStatus.WAITING:
                        anchor_id = candidate_id
                        break
            anchor_state = self._request_states.get(anchor_id)
            if anchor_state is None or not self._can_schedule_waiting(anchor_state):
                return []

            if use_tolerance:
                anchor_tokens = max(1, self._estimate_target_tokens(anchor_state))
                batch_max = anchor_tokens
                batch_min = anchor_tokens
                anchor_bucket = -1  # unused
            else:
                anchor_bucket = self._duration_bucket(anchor_state)

        selected: list[str] = []
        deferred: list[str] = []

        while self._waiting and len(selected) < capacity:
            sched_req_id = self._waiting.popleft()
            state = self._request_states.get(sched_req_id)
            if state is None:
                continue

            if prefer_waiting_present and state.status != DiffusionRequestStatus.WAITING:
                deferred.append(sched_req_id)
                continue

            if use_tolerance:
                t = max(1, self._estimate_target_tokens(state))
                new_max = batch_max if t <= batch_max else t
                new_min = batch_min if t >= batch_min else t
                if (new_max / new_min) <= self._pad_tolerance:
                    selected.append(sched_req_id)
                    batch_max = new_max
                    batch_min = new_min
                else:
                    deferred.append(sched_req_id)
            else:
                if self._duration_bucket(state) == anchor_bucket:
                    selected.append(sched_req_id)
                else:
                    deferred.append(sched_req_id)

        if deferred:
            self._waiting.extendleft(reversed(deferred))
        return selected

    def _duration_bucket(self, state: DiffusionRequestState) -> int:
        target_tokens = self._estimate_target_tokens(state)
        return max(0, target_tokens // self._duration_bucket_tokens)

    def _estimate_target_tokens(self, state: DiffusionRequestState) -> int:
        texts = list(self._request_texts(state.req.prompts))
        if not texts:
            return 0
        text = " ".join(texts)
        if self._duration_estimator is not None:
            try:
                return max(1, int(self._duration_estimator.estimate_duration(
                    text,
                    "Nice to meet you.",
                    25,
                )))
            except Exception:
                logger.debug("Duration estimator failed; using character length.", exc_info=True)
        return max(1, len(text))

    def _request_texts(self, prompts: Iterable) -> Iterable[str]:
        for prompt in prompts:
            if isinstance(prompt, str):
                yield prompt
            elif isinstance(prompt, dict):
                value = prompt.get("input", prompt.get("text", ""))
                if value:
                    yield str(value)
            elif prompt is not None:
                yield str(prompt)

    def _register_request_ids(self, request_ids: list[str], sched_req_id: str) -> None:
        for request_id in request_ids:
            existing = self._request_id_to_sched_req_id.get(request_id)
            if existing is not None and existing != sched_req_id:
                raise ValueError(f"request_id {request_id!r} is already mapped to active sched_req_id {existing!r}.")
            self._request_id_to_sched_req_id[request_id] = sched_req_id

    def _unregister_request_ids(self, request_ids: list[str], sched_req_id: str) -> None:
        for request_id in request_ids:
            if self._request_id_to_sched_req_id.get(request_id) == sched_req_id:
                self._request_id_to_sched_req_id.pop(request_id, None)
