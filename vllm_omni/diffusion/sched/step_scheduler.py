# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import _BaseScheduler
from vllm_omni.diffusion.sched.interface import (
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.utils import RunnerOutput

logger = init_logger(__name__)


@dataclass
class _StepProgress:
    current_step: int
    total_steps: int


class StepScheduler(_BaseScheduler):
    """Placeholder scheduler that advances a request one denoise step per update."""

    def __init__(self) -> None:
        super().__init__()
        self._request_progress: dict[str, _StepProgress] = {}

    def _reset_scheduler_state(self) -> None:
        self._request_progress.clear()

    def add_request(self, request: OmniDiffusionRequest) -> str:
        sched_req_id = self._make_sched_req_id(request)
        total_steps = self._get_total_steps(request)
        if total_steps <= 0:
            raise ValueError(f"Diffusion request {sched_req_id} must have positive total_steps, got {total_steps}")

        current_step = request.sampling_params.step_index or 0
        if current_step < 0 or current_step >= total_steps:
            raise ValueError(
                f"Diffusion request {sched_req_id} has invalid initial step_index {current_step} "
                f"for total_steps={total_steps}"
            )

        request.sampling_params.step_index = current_step
        sched_req_id = self._add_request_with_sched_req_id(sched_req_id, request)
        self._request_progress[sched_req_id] = _StepProgress(current_step=current_step, total_steps=total_steps)
        logger.debug(
            "StepScheduler add_request: %s (step=%d/%d, waiting=%d)",
            sched_req_id,
            current_step,
            total_steps,
            len(self._waiting),
        )
        return sched_req_id

    def schedule(self) -> DiffusionSchedulerOutput:
        return super().schedule()

    def _prefer_waiting_requests(self) -> bool:
        # In streaming step mode, a PREEMPTED request usually means a
        # continuation that has already emitted audio. Prefer never-run
        # requests so TTFA does not tail behind continuation blocks.
        return True

    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: RunnerOutput) -> set[str]:
        scheduled_req_ids = sched_output.scheduled_req_ids
        if not scheduled_req_ids:
            return set()

        terminal_statuses: dict[str, DiffusionRequestStatus] = {}
        terminal_errors: dict[str, str | None] = {}
        per_request_results = getattr(output, "per_request_results", None) or {}
        per_request_finished = getattr(output, "per_request_finished", None) or {}
        per_request_step_indices = getattr(output, "per_request_step_indices", None) or {}
        output_error = output.result.error if output.result is not None else None
        yielded_req_ids: list[str] = []
        for sched_req_id in scheduled_req_ids:
            state = self._request_states.get(sched_req_id)
            progress = self._request_progress.get(sched_req_id)
            if state is None or progress is None or state.is_finished():
                continue

            result = per_request_results.get(sched_req_id, output.result)
            result_error = result.error if result is not None else output_error
            if result_error is not None:
                terminal_statuses[sched_req_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[sched_req_id] = result_error
                continue

            step_index = per_request_step_indices.get(sched_req_id, output.step_index)
            if step_index is None:
                logger.warning(
                    "Received RunnerOutput with no step_index for request %s, treating as error",
                    sched_req_id,
                )
                terminal_statuses[sched_req_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[sched_req_id] = "Missing step_index in RunnerOutput"
                continue

            # We assume that the decoding stage is executed immediately after the denoising stage completes.
            progress.current_step = step_index
            state.req.sampling_params.step_index = step_index
            if per_request_finished.get(sched_req_id, output.finished):
                terminal_statuses[sched_req_id] = DiffusionRequestStatus.FINISHED_COMPLETED
                terminal_errors[sched_req_id] = None
            else:
                state.error = None
                if sched_req_id in per_request_results:
                    yielded_req_ids.append(sched_req_id)

        finished_req_ids = self._finalize_update_from_output(sched_output, terminal_statuses, terminal_errors)
        self._yield_intermediate_results(yielded_req_ids)
        return finished_req_ids

    def _yield_intermediate_results(self, sched_req_ids: list[str]) -> None:
        """Yield chunk emitters that would otherwise block first audio or shorter continuations."""
        prefer_first_audio = self._has_preferred_waiting_request()
        for sched_req_id in sched_req_ids:
            state = self._request_states.get(sched_req_id)
            if state is None or state.is_finished():
                continue
            if sched_req_id not in self._running:
                continue
            if not prefer_first_audio and not self._should_yield_to_shorter_waiting_continuation(state):
                continue
            self._running.remove(sched_req_id)
            state.status = DiffusionRequestStatus.PREEMPTED
            self._waiting.append(sched_req_id)

    def _should_yield_to_shorter_waiting_continuation(self, state: Any) -> bool:
        if self._batch_strategy != "duration_bucket":
            return False
        if not self._waiting:
            return False

        running_bucket = self._duration_bucket(state)
        for waiting_req_id in self._waiting:
            waiting_state = self._request_states.get(waiting_req_id)
            if waiting_state is None or not self._can_schedule_waiting(waiting_state):
                continue
            if self._duration_bucket(waiting_state) < running_bucket:
                return True
        return False

    def _pop_extra_request_state(self, sched_req_id: str) -> None:
        self._request_progress.pop(sched_req_id, None)

    def _get_total_steps(self, request: OmniDiffusionRequest) -> int:
        sampling = request.sampling_params

        if sampling.timesteps is not None:
            return self._sequence_length(sampling.timesteps)
        if sampling.sigmas is not None:
            return len(sampling.sigmas)
        return int(sampling.num_inference_steps)

    @staticmethod
    def _sequence_length(values: Any) -> int:
        ndim = getattr(values, "ndim", None)
        if ndim == 0:
            return 1

        shape = getattr(values, "shape", None)
        if shape is not None:
            return int(shape[0])

        return len(values)
