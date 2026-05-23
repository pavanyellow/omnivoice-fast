# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import inspect
import os
import queue
import threading
import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import PIL.Image
import torch
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.data import (
    DiffusionOutput,
    DiffusionRequestAbortedError,
    OmniDiffusionConfig,
)
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.registry import (
    DiffusionModelRegistry,
    get_diffusion_post_process_func,
    get_diffusion_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import RequestScheduler, SchedulerInterface, StepScheduler
from vllm_omni.diffusion.sched.interface import DiffusionRequestStatus
from vllm_omni.diffusion.worker.utils import RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


def supports_image_input(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_image_input", False))


def supports_audio_input(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_audio_input", False))


def image_color_format(model_class_name: str) -> str:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    return getattr(model_cls, "color_format", "RGB")


def supports_audio_output(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_audio_output", False))


class DiffusionEngine:
    """The diffusion engine for vLLM-Omni diffusion models."""

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        scheduler: SchedulerInterface | None = None,
    ):
        """Initialize the diffusion engine.

        Args:
            config: The configuration for the diffusion engine.
        """
        self.od_config = od_config

        self.post_process_func = get_diffusion_post_process_func(od_config)
        self.pre_process_func = get_diffusion_pre_process_func(od_config)
        # Cache whether the model-specific postprocess accepts request-level
        # sampling params so step() can support both legacy and extended hooks.
        self._post_process_accepts_sampling_params = bool(
            self.post_process_func is not None
            and "sampling_params" in inspect.signature(self.post_process_func).parameters
        )

        executor_class = DiffusionExecutor.get_class(od_config)
        self.executor = executor_class(od_config)
        self.step_execution = bool(getattr(od_config, "step_execution", False))
        self.scheduler: SchedulerInterface = scheduler or (
            StepScheduler() if self.step_execution else RequestScheduler()
        )
        self.scheduler.initialize(od_config)
        self._rpc_lock = threading.RLock()
        self.abort_queue: queue.Queue[str] = queue.Queue()
        self.execute_fn = self.executor.execute_step if self.step_execution else self.executor.execute_request

        # Concurrent submission path: HTTP threads enqueue + wait on per-
        # request events while a single driver thread runs the
        # schedule/execute/distribute loop. Off by default so callers that
        # depend on the historical synchronous semantics stay unaffected.
        self._concurrent_mode = (
            os.environ.get("VLLM_OMNI_DIFFUSION_CONCURRENT") == "1"
        )
        self._engine_lock = threading.Lock()
        self._driver_thread: threading.Thread | None = None
        self._driver_running = False
        self._driver_started = False
        self._pending_results: dict[str, DiffusionOutput] = {}
        self._pending_runner_outputs: dict[str, RunnerOutput] = {}
        self._req_events: dict[str, threading.Event] = {}
        self._stream_queues: dict[str, queue.Queue[tuple[DiffusionOutput, bool]]] = {}
        self._driver_idle_sleep = float(
            os.environ.get("VLLM_OMNI_DIFFUSION_DRIVER_IDLE_SLEEP", "0.0005")
        )
        # Coalesce wait: when there is a request ready to schedule but the
        # current batch is below the configured maximum, sleep briefly to
        # let more arrivals join the same batch instead of letting the
        # first request run solo. Adds at most this many ms of latency at
        # c=1. Default 0 (disabled).
        self._driver_batch_wait_ms = float(
            os.environ.get("VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS", "0")
        )

        try:
            self._dummy_run()
        except Exception as e:
            logger.error(f"Dummy run failed: {e}")
            self.close()
            raise e

    def step(self, request: OmniDiffusionRequest) -> list[OmniRequestOutput]:
        diffusion_engine_start_time = time.perf_counter()

        # Apply pre-processing if available
        preprocess_time = 0.0
        if self.pre_process_func is not None:
            preprocess_start_time = time.perf_counter()
            request = self.pre_process_func(request)
            preprocess_time = time.perf_counter() - preprocess_start_time
            logger.info(f"Pre-processing completed in {preprocess_time:.4f} seconds")

        exec_start_time = time.perf_counter()
        output = self.add_req_and_wait_for_response(request)
        exec_total_time = time.perf_counter() - exec_start_time

        if output.aborted:
            raise DiffusionRequestAbortedError(output.abort_message or "Diffusion request aborted.")
        if output.error:
            raise RuntimeError(output.error)
        logger.info("Generation completed successfully.")

        if output.output is None:
            logger.warning("Output is None, returning empty OmniRequestOutput")
            return [
                OmniRequestOutput.from_diffusion(
                    request_id=request.request_ids[i] if i < len(request.request_ids) else "",
                    images=[],
                    prompt=prompt,
                    metrics={},
                    latents=None,
                )
                for i, prompt in enumerate(request.prompts)
            ]

        # When CPU offload is enabled, move output to CPU before
        # post-processing to avoid device OOM — model weights may still
        # reside on the device and leave no headroom for intermediates.
        output_data = output.output
        if (
            self.od_config.enable_cpu_offload
            and isinstance(output_data, torch.Tensor)
            and output_data.device.type != "cpu"
        ):
            output_data = output_data.cpu()

        postprocess_start_time = time.perf_counter()
        if self.post_process_func is not None:
            # Some video pipelines need request-level controls during
            # postprocess (for example worker-side frame interpolation).
            if self._post_process_accepts_sampling_params:
                outputs = self.post_process_func(output_data, sampling_params=request.sampling_params)
            else:
                outputs = self.post_process_func(output_data)
        else:
            outputs = output_data
        audio_payload = None
        custom_output = output.custom_output or {}
        model_audio_sample_rate = None
        model_fps = None
        if isinstance(outputs, dict):
            audio_payload = outputs.get("audio")
            custom_output.update(outputs.get("custom_output") or {})
            model_audio_sample_rate = outputs.get("audio_sample_rate", outputs.get("sr"))
            model_fps = outputs.get("fps")
            outputs = outputs.get("video", outputs)
        postprocess_time = time.perf_counter() - postprocess_start_time
        logger.info(f"Post-processing completed in {postprocess_time:.4f} seconds")

        step_total_ms = (time.perf_counter() - diffusion_engine_start_time) * 1000
        logger.info(
            "DiffusionEngine.step breakdown: preprocess=%.2f ms, "
            "add_req_and_wait=%.2f ms, postprocess=%.2f ms, total=%.2f ms",
            preprocess_time * 1000,
            exec_total_time * 1000,
            postprocess_time * 1000,
            step_total_ms,
        )

        # Convert to OmniRequestOutput format
        # Ensure outputs is a list
        if not isinstance(outputs, list):
            outputs = [outputs] if outputs is not None else []

        metrics = {
            "preprocess_time_ms": preprocess_time * 1000,
            "diffusion_engine_exec_time_ms": exec_total_time * 1000,
            "diffusion_engine_total_time_ms": step_total_ms,
            "image_num": int(request.sampling_params.num_outputs_per_prompt),
            "resolution": int(request.sampling_params.resolution),
            "postprocess_time_ms": postprocess_time * 1000,
        }

        # Handle single request or multiple requests
        is_audio_output = supports_audio_output(self.od_config.model_class_name)
        if len(request.prompts) == 1:
            # Single request: return single OmniRequestOutput
            prompt = request.prompts[0]
            request_id = request.request_ids[0] if request.request_ids else ""

            if is_audio_output:
                request_audio_payload = (
                    audio_payload
                    if audio_payload is not None
                    else (outputs[0] if len(outputs) == 1 else outputs)
                )
                multimodal_output = {"audio": request_audio_payload}
                if model_audio_sample_rate is not None:
                    multimodal_output["sr"] = model_audio_sample_rate
                    multimodal_output["audio_sample_rate"] = model_audio_sample_rate
                return [
                    OmniRequestOutput.from_diffusion(
                        request_id=request_id,
                        images=[],
                        prompt=prompt,
                        metrics=metrics,
                        latents=output.trajectory_latents,
                        trajectory_latents=output.trajectory_latents,
                        trajectory_timesteps=output.trajectory_timesteps,
                        trajectory_log_probs=output.trajectory_log_probs,
                        trajectory_decoded=output.trajectory_decoded,
                        multimodal_output=multimodal_output,
                        final_output_type="audio",
                        stage_durations=output.stage_durations,
                        peak_memory_mb=output.peak_memory_mb,
                    ),
                ]
            else:
                mm_output = {}
                if audio_payload is not None:
                    mm_output["audio"] = audio_payload
                if model_audio_sample_rate is not None:
                    mm_output["audio_sample_rate"] = model_audio_sample_rate
                if model_fps is not None:
                    mm_output["fps"] = model_fps
                return [
                    OmniRequestOutput.from_diffusion(
                        request_id=request_id,
                        images=outputs,
                        prompt=prompt,
                        metrics=metrics,
                        latents=output.trajectory_latents,
                        trajectory_latents=output.trajectory_latents,
                        trajectory_timesteps=output.trajectory_timesteps,
                        trajectory_log_probs=output.trajectory_log_probs,
                        trajectory_decoded=output.trajectory_decoded,
                        custom_output=custom_output,
                        multimodal_output=mm_output,
                        stage_durations=output.stage_durations,
                        peak_memory_mb=output.peak_memory_mb,
                    ),
                ]
        else:
            # Multiple requests: return list of OmniRequestOutput
            # Split images based on num_outputs_per_prompt for each request
            results = []
            output_idx = 0

            for i, prompt in enumerate(request.prompts):
                request_id = request.request_ids[i] if i < len(request.request_ids) else ""

                # Get images for this request
                num_outputs = request.sampling_params.num_outputs_per_prompt
                start_idx = output_idx
                end_idx = start_idx + num_outputs
                request_outputs = outputs[start_idx:end_idx] if output_idx < len(outputs) else []
                output_idx = end_idx

                if is_audio_output:
                    request_audio_payload = request_outputs[0] if len(request_outputs) == 1 else request_outputs
                    results.append(
                        OmniRequestOutput.from_diffusion(
                            request_id=request_id,
                            images=[],
                            prompt=prompt,
                            metrics=metrics,
                            latents=output.trajectory_latents,
                            trajectory_latents=output.trajectory_latents,
                            trajectory_timesteps=output.trajectory_timesteps,
                            trajectory_log_probs=output.trajectory_log_probs,
                            trajectory_decoded=output.trajectory_decoded,
                            multimodal_output={"audio": request_audio_payload},
                            final_output_type="audio",
                            stage_durations=output.stage_durations,
                            peak_memory_mb=output.peak_memory_mb,
                        ),
                    )
                else:
                    mm_output = {}
                    if audio_payload is not None:
                        sliced_audio = audio_payload
                        if isinstance(audio_payload, (list, tuple)):
                            sliced_audio = audio_payload[start_idx:end_idx]
                            if len(sliced_audio) == 1:
                                sliced_audio = sliced_audio[0]
                        elif hasattr(audio_payload, "shape") and getattr(audio_payload, "shape", None) is not None:
                            if len(audio_payload.shape) > 0 and audio_payload.shape[0] >= end_idx:
                                sliced_audio = audio_payload[start_idx:end_idx]
                                if num_outputs == 1:
                                    sliced_audio = sliced_audio[0]
                        mm_output["audio"] = sliced_audio
                    if model_audio_sample_rate is not None:
                        mm_output["audio_sample_rate"] = model_audio_sample_rate
                    if model_fps is not None:
                        mm_output["fps"] = model_fps
                    results.append(
                        OmniRequestOutput.from_diffusion(
                            request_id=request_id,
                            images=request_outputs,
                            prompt=prompt,
                            metrics=metrics,
                            latents=output.trajectory_latents,
                            trajectory_latents=output.trajectory_latents,
                            trajectory_timesteps=output.trajectory_timesteps,
                            trajectory_log_probs=output.trajectory_log_probs,
                            trajectory_decoded=output.trajectory_decoded,
                            custom_output=custom_output,
                            multimodal_output=mm_output,
                            stage_durations=output.stage_durations,
                            peak_memory_mb=output.peak_memory_mb,
                        ),
                    )

            return results

    @staticmethod
    def make_engine(
        config: OmniDiffusionConfig,
        scheduler: SchedulerInterface | None = None,
    ) -> DiffusionEngine:
        """Factory method to create a DiffusionEngine instance.

        Args:
            config: The configuration for the diffusion engine.

        Returns:
            An instance of DiffusionEngine.
        """
        return DiffusionEngine(config, scheduler=scheduler)

    def add_req_and_wait_for_response(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        if getattr(self, "_concurrent_mode", False):
            return self._add_req_and_wait_concurrent(request)
        return self._add_req_and_wait_legacy(request)

    def stream(self, request: OmniDiffusionRequest) -> Iterable[OmniRequestOutput]:
        """Stream request outputs from step-wise diffusion execution.

        This path is used by block-wise OmniVoice serving. It keeps requests in
        the same scheduler/driver loop as non-streaming diffusion so concurrent
        streams can still share batched model forwards.
        """
        if not (self.step_execution and self._concurrent_mode):
            for output in self.step(request):
                yield output
            return

        preprocess_time = 0.0
        if self.pre_process_func is not None:
            preprocess_start_time = time.perf_counter()
            request = self.pre_process_func(request)
            preprocess_time = time.perf_counter() - preprocess_start_time
        for prompt in request.prompts:
            if isinstance(prompt, dict) and prompt.get("_stream_audio_blocks"):
                prompt["_stream_output_enabled"] = True

        exec_start_time = time.perf_counter()
        for output, finished in self.add_req_and_stream_response(request):
            if output.aborted:
                raise DiffusionRequestAbortedError(
                    output.abort_message or "Diffusion request aborted."
                )
            if output.error:
                raise RuntimeError(output.error)
            exec_total_time = time.perf_counter() - exec_start_time
            yield self._make_stream_omni_output(
                request=request,
                output=output,
                finished=finished,
                preprocess_time=preprocess_time,
                exec_total_time=exec_total_time,
            )

    def _make_stream_omni_output(
        self,
        *,
        request: OmniDiffusionRequest,
        output: DiffusionOutput,
        finished: bool,
        preprocess_time: float,
        exec_total_time: float,
    ) -> OmniRequestOutput:
        request_id = request.request_ids[0] if request.request_ids else ""
        prompt = request.prompts[0] if request.prompts else None
        payload = output.output
        if isinstance(payload, dict):
            multimodal_output = dict(payload)
        else:
            multimodal_output = {"audio": payload}

        metrics = {
            "preprocess_time_ms": preprocess_time * 1000,
            "diffusion_engine_exec_time_ms": exec_total_time * 1000,
            "diffusion_engine_total_time_ms": exec_total_time * 1000,
            "postprocess_time_ms": 0.0,
        }
        omni_output = OmniRequestOutput.from_diffusion(
            request_id=request_id,
            images=[],
            prompt=prompt,
            metrics=metrics,
            multimodal_output=multimodal_output,
            final_output_type="audio",
            stage_durations=output.stage_durations,
            peak_memory_mb=output.peak_memory_mb,
        )
        omni_output.finished = finished
        return omni_output

    def add_req_and_stream_response(
        self, request: OmniDiffusionRequest,
    ) -> Iterable[tuple[DiffusionOutput, bool]]:
        if not self._concurrent_mode:
            final_output = self._add_req_and_wait_legacy(request)
            yield final_output, True
            return

        stream_queue: queue.Queue[tuple[DiffusionOutput, bool]] = queue.Queue()
        with self._engine_lock:
            self._ensure_driver_started()
            sched_req_id = self.scheduler.add_request(request)
            self._stream_queues[sched_req_id] = stream_queue

        try:
            while True:
                output, finished = stream_queue.get()
                yield output, finished
                if finished:
                    break
        finally:
            with self._engine_lock:
                self._stream_queues.pop(sched_req_id, None)

    def _add_req_and_wait_legacy(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        with self._rpc_lock:
            target_sched_req_id = self.scheduler.add_request(request)

            # keep scheduling and executing until the target request is finished
            while True:
                self._process_aborts_queue()
                sched_output = self.scheduler.schedule()
                if sched_output.is_empty:
                    if target_sched_req_id in sched_output.finished_req_ids:
                        return self._finalize_finished_request(target_sched_req_id)
                    if not self.scheduler.has_requests():
                        raise RuntimeError("Diffusion scheduler has no runnable requests.")
                    continue

                # NOTE: legacy single-flight path; scheduler returns one
                # scheduled request because max_num_running_reqs defaults
                # to 1. Concurrent mode handles multi-request schedules.
                sched_req_id = sched_output.scheduled_req_ids[0]
                try:
                    runner_output = self.execute_fn(sched_output)
                except EngineDeadError:
                    raise
                except Exception as exc:
                    logger.error("Execution failed for diffusion request %s", sched_req_id, exc_info=True)
                    runner_output = RunnerOutput(
                        req_id=sched_req_id,
                        step_index=None,
                        finished=True,
                        result=DiffusionOutput(error=str(exc)),
                    )

                self._process_aborts_queue()

                finished_req_ids = self.scheduler.update_from_output(sched_output, runner_output)
                if target_sched_req_id in finished_req_ids:
                    return self._finalize_finished_request(
                        target_sched_req_id,
                        runner_output=runner_output,
                        missing_result_error="Diffusion execution finished without a final output.",
                    )

    def _ensure_driver_started(self) -> None:
        if self._driver_started:
            return
        self._driver_running = True
        self._driver_thread = threading.Thread(
            target=self._driver_loop,
            daemon=True,
            name="DiffusionEngineDriver",
        )
        self._driver_thread.start()
        self._driver_started = True

    def _add_req_and_wait_concurrent(
        self, request: OmniDiffusionRequest,
    ) -> DiffusionOutput:
        event = threading.Event()
        with self._engine_lock:
            self._ensure_driver_started()
            sched_req_id = self.scheduler.add_request(request)
            self._req_events[sched_req_id] = event

        event.wait()

        with self._engine_lock:
            result = self._pending_results.pop(sched_req_id, None)
            self._pending_runner_outputs.pop(sched_req_id, None)
            self._req_events.pop(sched_req_id, None)

        if result is None:
            return DiffusionOutput(error="Driver returned no result")
        return result

    def _driver_loop(self) -> None:
        idle_sleep = self._driver_idle_sleep
        batch_wait_ms = self._driver_batch_wait_ms
        while self._driver_running:
            with self._engine_lock:
                self._process_aborts_queue()
                has_reqs = self.scheduler.has_requests()
                waiting_count = len(getattr(self.scheduler, "_waiting", []))
                running_count = len(getattr(self.scheduler, "_running", []))
                max_batch = self.scheduler.max_num_running_reqs

            # Coalesce wait: if at least one request is ready and the
            # current scheduled batch would be smaller than the configured
            # maximum, sleep briefly to let more in-flight HTTP coroutines
            # hit add_request before we commit. This is the difference
            # between c=4 batching all 4 from the start vs running req0
            # alone first and then batching the remaining 3.
            if (
                batch_wait_ms > 0
                and waiting_count > 0
                and running_count == 0
                and waiting_count < max_batch
            ):
                deadline = time.monotonic() + batch_wait_ms / 1000.0
                step = max(0.0005, batch_wait_ms / 4000.0)
                while time.monotonic() < deadline:
                    time.sleep(step)
                    with self._engine_lock:
                        new_waiting = len(getattr(self.scheduler, "_waiting", []))
                    if new_waiting >= max_batch or new_waiting == waiting_count:
                        # Already full or no new arrivals in the latest tick.
                        if new_waiting >= max_batch:
                            break
                    waiting_count = new_waiting

            with self._engine_lock:
                if not has_reqs and not self.scheduler.has_requests():
                    sched_output = None
                else:
                    sched_output = self.scheduler.schedule()
                    if logger.isEnabledFor(10):  # DEBUG
                        new_n = len(sched_output.scheduled_new_reqs)
                        logger.debug(
                            "driver schedule(): batch_size=%d max=%d",
                            new_n, self.scheduler.max_num_running_reqs,
                        )

            if sched_output is None or sched_output.is_empty:
                # Surface aborted-only finishes that may have arrived without
                # any scheduled work (the legacy path returns from inside
                # the synchronous loop; the driver path needs to do it here).
                if sched_output is not None:
                    self._driver_dispatch_aborts(sched_output.finished_req_ids)
                if idle_sleep > 0:
                    time.sleep(idle_sleep)
                continue

            try:
                runner_output = self.execute_fn(sched_output)
            except EngineDeadError:
                self._driver_running = False
                # Wake any waiters with an error so HTTP threads return.
                self._driver_signal_engine_dead()
                raise
            except Exception as exc:
                logger.error(
                    "Execution failed for diffusion batch %s",
                    sched_output.scheduled_req_ids,
                    exc_info=True,
                )
                runner_output = RunnerOutput(
                    req_id=(sched_output.scheduled_req_ids[0]
                            if sched_output.scheduled_req_ids else ""),
                    step_index=None,
                    finished=True,
                    result=DiffusionOutput(error=str(exc)),
                )

            with self._engine_lock:
                self._process_aborts_queue()
                finished_req_ids = self.scheduler.update_from_output(
                    sched_output, runner_output,
                )
                skip_stream_ids = {
                    sched_req_id
                    for sched_req_id in finished_req_ids
                    if (
                        (state := self.scheduler.get_request_state(sched_req_id)) is not None
                        and state.status != DiffusionRequestStatus.FINISHED_COMPLETED
                    )
                }
                stream_finals = self._driver_dispatch_step_outputs(
                    runner_output,
                    skip_stream_ids,
                )
                self._driver_dispatch_finished(finished_req_ids, runner_output, stream_finals)

    def _driver_dispatch_step_outputs(
        self,
        runner_output: RunnerOutput,
        skip_req_ids: set[str] | None = None,
    ) -> set[str]:
        """Publish non-final and final step outputs to streaming waiters."""
        skip_req_ids = skip_req_ids or set()
        stream_finals: set[str] = set()
        per_request = runner_output.per_request_results
        per_request_finished = runner_output.per_request_finished or {}

        if per_request is None:
            if runner_output.result is None:
                return stream_finals
            if runner_output.req_id in skip_req_ids:
                return stream_finals
            stream_queue = self._stream_queues.get(runner_output.req_id)
            if stream_queue is None:
                return stream_finals
            finished = per_request_finished.get(runner_output.req_id, runner_output.finished)
            stream_queue.put((runner_output.result, finished))
            if finished:
                stream_finals.add(runner_output.req_id)
            return stream_finals

        for sched_req_id, result in per_request.items():
            if sched_req_id in skip_req_ids:
                continue
            stream_queue = self._stream_queues.get(sched_req_id)
            if stream_queue is None:
                continue
            finished = per_request_finished.get(sched_req_id, runner_output.finished)
            stream_queue.put((result, finished))
            if finished:
                stream_finals.add(sched_req_id)
        return stream_finals

    def _driver_dispatch_finished(
        self,
        finished_req_ids: set[str],
        runner_output: RunnerOutput,
        stream_finals: set[str] | None = None,
    ) -> None:
        stream_finals = stream_finals or set()
        per_request = runner_output.per_request_results
        for sched_req_id in finished_req_ids:
            state = self.scheduler.pop_request_state(sched_req_id)

            if state is not None and state.status == DiffusionRequestStatus.FINISHED_ABORTED:
                request_id = state.req.request_ids[0] if state.req.request_ids else sched_req_id
                result = DiffusionOutput(
                    aborted=True,
                    abort_message=f"Request {request_id} aborted.",
                )
            elif per_request is not None and sched_req_id in per_request:
                result = per_request[sched_req_id]
            elif runner_output.result is not None:
                result = runner_output.result
            else:
                result = DiffusionOutput(error="Diffusion execution finished without a final output.")

            stream_queue = self._stream_queues.pop(sched_req_id, None)
            if stream_queue is not None:
                if sched_req_id not in stream_finals:
                    stream_queue.put((result, True))
                continue

            self._pending_results[sched_req_id] = result
            self._pending_runner_outputs[sched_req_id] = runner_output
            event = self._req_events.get(sched_req_id)
            if event is not None:
                event.set()

    def _driver_dispatch_aborts(self, finished_req_ids: set[str]) -> None:
        if not finished_req_ids:
            return
        with self._engine_lock:
            for sched_req_id in finished_req_ids:
                if sched_req_id in self._pending_results:
                    continue
                state = self.scheduler.pop_request_state(sched_req_id)
                if state is None:
                    continue
                request_id = state.req.request_ids[0] if state.req.request_ids else sched_req_id
                result = DiffusionOutput(
                    aborted=True, abort_message=f"Request {request_id} aborted.",
                )
                stream_queue = self._stream_queues.pop(sched_req_id, None)
                if stream_queue is not None:
                    stream_queue.put((result, True))
                    continue
                self._pending_results[sched_req_id] = result
                event = self._req_events.get(sched_req_id)
                if event is not None:
                    event.set()

    def _driver_signal_engine_dead(self) -> None:
        with self._engine_lock:
            for sched_req_id, event in list(self._req_events.items()):
                if sched_req_id not in self._pending_results:
                    self._pending_results[sched_req_id] = DiffusionOutput(
                        error="Diffusion engine died.",
                    )
                event.set()
            for sched_req_id, stream_queue in list(self._stream_queues.items()):
                stream_queue.put((DiffusionOutput(error="Diffusion engine died."), True))
                self._stream_queues.pop(sched_req_id, None)

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        """Start or stop profiling on all diffusion workers.

        Args:
            is_start: True to start profiling, False to stop.
            profile_prefix: Optional prefix for trace filename.
        """
        if is_start:
            if profile_prefix is None:
                profile_prefix = f"diffusion_{int(time.time())}"
            logger.info(f"Starting diffusion profiling with prefix: {profile_prefix}")
        else:
            logger.info("Stopping diffusion profiling...")

        try:
            self.collective_rpc(method="profile", args=(is_start, profile_prefix))
        except Exception as e:
            action = "start" if is_start else "stop"
            logger.error(f"Failed to {action} profiling on workers", exc_info=True)
            if is_start:
                raise RuntimeError(f"Could not {action} profiler: {e}") from e

    def _dummy_run(self):
        """A dummy run to warm up the model."""
        num_inference_steps = 1
        height = 512
        width = 512
        if supports_image_input(self.od_config.model_class_name):
            # Provide a dummy image input if the model supports it
            color_format = image_color_format(self.od_config.model_class_name)
            dummy_image = PIL.Image.new(color_format, (width, height))
        else:
            dummy_image = None

        if supports_audio_input(self.od_config.model_class_name):
            audio_sr = 16000
            dummy_audio = np.random.randn(audio_sr * 2).astype(np.float32)
        else:
            dummy_audio = None

        prompt: OmniTextPrompt = {
            "prompt": "dummy run",
            "multi_modal_data": {"image": dummy_image, "audio": dummy_audio},
        }
        req = OmniDiffusionRequest(
            prompts=[prompt],
            request_ids=["dummy_req_id"],
            sampling_params=OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                # Keep warmup path minimal and robust across text encoders.
                # Some models may fail when warmup implicitly triggers
                # classifier-free guidance with an empty negative prompt.
                guidance_scale=0.0,
                num_outputs_per_prompt=1,
                # Disable CFG for warmup to avoid triggering CFG parallel
                # validation when cfg_parallel_size > 1.
                extra_args={"cfg_text_scale": 1.0, "cfg_img_scale": 1.0},
            ),
        )
        logger.info("dummy run to warm up the model")
        request = self.pre_process_func(req) if self.pre_process_func is not None else req
        output = self.add_req_and_wait_for_response(request)
        if output.error:
            raise RuntimeError(f"Dummy run failed: {output.error}")

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Call a method on worker processes and get results immediately.

        Args:
            method: The method name (str) to execute on workers
            timeout: Optional timeout in seconds
            args: Positional arguments for the method
            kwargs: Keyword arguments for the method
            unique_reply_rank: If set, only get reply from this rank

        Returns:
            Single result if unique_reply_rank is provided, otherwise list of results
        """
        assert isinstance(method, str), "Only string method names are supported for now"

        deadline = None if timeout is None else time.monotonic() + timeout
        acquired = False
        try:
            if deadline is None:
                self._rpc_lock.acquire()
                acquired = True
            else:
                lock_timeout = max(0, deadline - time.monotonic())
                acquired = self._rpc_lock.acquire(timeout=lock_timeout)
            if not acquired:
                raise TimeoutError(f"RPC call to {method} timed out waiting for engine lock.")

            rpc_timeout = None if deadline is None else max(0, deadline - time.monotonic())
            if deadline is not None and rpc_timeout <= 0:
                raise TimeoutError(f"RPC call to {method} timed out.")

            return self.executor.collective_rpc(
                method=method,
                timeout=rpc_timeout,
                args=args,
                kwargs=kwargs,
                unique_reply_rank=unique_reply_rank,
            )
        finally:
            if acquired:
                self._rpc_lock.release()

    def close(self) -> None:
        if getattr(self, "_driver_running", False):
            self._driver_running = False
            if self._driver_thread is not None:
                # Wake any stragglers waiting on per-request events.
                self._driver_signal_engine_dead()
                self._driver_thread.join(timeout=2.0)
        if hasattr(self, "scheduler"):
            self.scheduler.close()
        if hasattr(self, "executor"):
            self.executor.shutdown()

    def abort(self, request_id: str | Iterable[str]) -> None:
        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)
        for req_id in request_ids:
            self.abort_queue.put(req_id)

    def _process_aborts_queue(self) -> None:
        if self.abort_queue.empty():
            return

        request_ids: list[str] = []
        while not self.abort_queue.empty():
            ids = self.abort_queue.get_nowait()
            request_ids.extend((ids,) if isinstance(ids, str) else ids)

        self._abort_requests(request_ids)

    def _abort_requests(self, request_ids: str | Iterable[str]) -> None:
        request_ids = [request_ids] if isinstance(request_ids, str) else list(request_ids)

        sched_req_ids: list[str] = []
        for request_id in dict.fromkeys(request_ids):
            sched_req_id = self.scheduler.get_sched_req_id(request_id)
            if sched_req_id is not None:
                sched_req_ids.append(sched_req_id)

        for sched_req_id in dict.fromkeys(sched_req_ids):
            if self.scheduler.get_request_state(sched_req_id) is not None:
                self.scheduler.finish_requests(sched_req_id, DiffusionRequestStatus.FINISHED_ABORTED)

    def _finalize_finished_request(
        self,
        sched_req_id: str,
        runner_output: RunnerOutput | None = None,
        missing_result_error: str = "Diffusion scheduler finished target request without execution output.",
    ) -> DiffusionOutput:
        state = self.scheduler.get_request_state(sched_req_id)
        popped_state = self.scheduler.pop_request_state(sched_req_id)
        state = state or popped_state

        if state is None:
            raise RuntimeError(f"Diffusion scheduler lost state for request {sched_req_id}.")

        if state.status == DiffusionRequestStatus.FINISHED_ABORTED:
            request_id = state.req.request_ids[0] if state.req.request_ids else sched_req_id
            return DiffusionOutput(
                aborted=True,
                abort_message=f"Request {request_id} aborted.",
            )

        if runner_output is not None:
            per_request = runner_output.per_request_results or {}
            if sched_req_id in per_request:
                return per_request[sched_req_id]
            if runner_output.result is not None:
                return runner_output.result

        return DiffusionOutput(error=missing_result_error)
