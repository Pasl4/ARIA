"""ARIA on top of AutoGen's MagenticOne multi-agent team.

`AriaOrchestrator` calls the ARIA controller before each ledger step, before
the final answer and before a stall-triggered replan; with `controller=None`
no contact point fires. `_orchestrate_step` and `_aria_prepare_final_answer`
mirror autogen-agentchat 0.7.5 (pinned in pyproject.toml) with ARIA additions
in marked blocks; re-check both methods when upgrading autogen-agentchat.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from autogen_agentchat.base import Response, TerminationCondition
from autogen_agentchat.messages import (
    BaseChatMessage,
    MessageFactory,
    StopMessage,
    TextMessage,
)
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.teams._group_chat._events import (
    GroupChatAgentResponse,
    GroupChatMessage,
    GroupChatRequestPublish,
    GroupChatStart,
)
from autogen_agentchat.teams._group_chat._magentic_one._magentic_one_orchestrator import (
    MagenticOneOrchestrator,
)
from autogen_agentchat.teams._group_chat._magentic_one._prompts import LedgerEntry
from autogen_core import CancellationToken, DefaultTopicId, MessageContext, rpc
from autogen_core.models import ChatCompletionClient, UserMessage
from autogen_core.utils import extract_json_from_str

from ...core.types import ContactPointKind
from ...telemetry.token_meter import call_purpose
from .teams import LIAISON_NAME
from .translate import thread_to_records

logger = logging.getLogger(__name__)

# Bounded retries for a filtered/empty final answer (see _aria_prepare_final_answer).
_FINAL_ANSWER_RETRIES = 3

if TYPE_CHECKING:
    from ...detectors.base import StepRecord
    from ...orchestration.controller import InterventionController

# The pseudo-instruction classified at the pre-delivery contact point.
FINAL_ANSWER_INSTRUCTION = "Synthesize and state the final answer to the task."

# The pseudo-instruction classified at the stall/replan contact point;
# "plan" is a FRAME marker, so the decision gates the replan as a FRAME step.
STALL_REPLAN_INSTRUCTION = "Revise the plan: progress has stalled."

STOP_COMPLETED = "completed"
STOP_MAX_ROUNDS = "max_rounds"


class AriaOrchestrator(MagenticOneOrchestrator):
    """MagenticOne orchestrator with the ARIA contact-point hook."""

    def __init__(
        self,
        *args: Any,
        controller: InterventionController | None = None,
        gate_revision_rounds: int = 0,
        organic_liaison: bool = False,
    ) -> None:
        # Gate revision: an off_track verdict at the pre-delivery gate
        # re-opens the chat for bounded revision instead of finalizing (once
        # per episode).
        self._gate_revision_rounds = gate_revision_rounds
        self._gate_revisions_used = 0
        # Organic only: set from config.organic_consult, the same flag that
        # attaches the liaison. At most one liaison turn is granted at finalization.
        self._organic_liaison = organic_liaison
        self._organic_final_turn_used = False
        super().__init__(*args)
        self._aria_controller = controller
        # Records archived across stall-triggered thread resets, so error
        # streaks and step indices survive replans.
        self._aria_archive: list[StepRecord] = []
        self._aria_prev_cur: list[StepRecord] = []
        # Guidance from a STALL-consult, stashed for _update_task_ledger.
        self._aria_replan_guidance: str | None = None

    # ------------------------------------------------------------------
    # Planning calls (task-ledger facts/plan), tagged for the token meter
    # ------------------------------------------------------------------

    @rpc
    async def handle_start(self, message: GroupChatStart, ctx: MessageContext) -> None:  # type: ignore[override]
        # Same @rpc message type as upstream, so routing is unchanged. Context
        # variables do not cross runtime task boundaries, so participant calls
        # keep their own tag; nested tags override this one.
        with call_purpose("planning"):
            # @rpc erases the (message, ctx) signature, so the super() call is
            # untypable; upstream ignores it on its own override as well.
            await super().handle_start(message, ctx)  # pyright: ignore[reportCallIssue]

    async def _update_task_ledger(self, cancellation_token: CancellationToken) -> None:
        # Inject STALL-consult guidance as a thread message, since the upstream
        # body builds the replan context from the thread. _reenter_outer_loop
        # clears the thread right after and _aria_prev_cur is snapshotted
        # earlier, so the guidance never reaches the StepRecords or the state.
        if self._aria_replan_guidance:
            await self.update_message_thread(
                [
                    TextMessage(
                        content=self._aria_replan_guidance,
                        source=f"{self._name}-advisor",
                    )
                ]
            )
            self._aria_replan_guidance = None
        with call_purpose("planning"):
            await super()._update_task_ledger(cancellation_token)

    # ------------------------------------------------------------------
    # ARIA record log (persistent across replans)
    # ------------------------------------------------------------------

    def _aria_records(self) -> list[StepRecord]:
        markers = self._aria_controller.markers if self._aria_controller else None
        cur = thread_to_records(list(self._message_thread), self._name, markers=markers)
        if len(cur) < len(self._aria_prev_cur):
            # A replan cleared the thread; archive the previous records.
            self._aria_archive.extend(self._aria_prev_cur)
        self._aria_prev_cur = cur
        merged = self._aria_archive + cur
        return [replace(r, index=i) for i, r in enumerate(merged)]

    # ------------------------------------------------------------------
    # Inner loop (mirrors autogen-agentchat 0.7.5 + marked ARIA blocks)
    # ------------------------------------------------------------------

    async def _orchestrate_step(self, cancellation_token: CancellationToken) -> None:
        # Check if we reached the maximum number of rounds
        if self._max_turns is not None and self._n_rounds >= self._max_turns:
            await self._aria_prepare_final_answer(
                "Max rounds reached.", cancellation_token, stop_reason=STOP_MAX_ROUNDS
            )
            return
        self._n_rounds += 1

        # Update the progress ledger
        context = self._thread_to_context()
        progress_ledger_prompt = self._get_progress_ledger_prompt(
            self._task, self._team_description, self._participant_names
        )
        context.append(UserMessage(content=progress_ledger_prompt, source=self._name))
        progress_ledger: dict[str, Any] = {}
        assert self._max_json_retries > 0
        key_error: bool = False
        for _ in range(self._max_json_retries):
            # Token-meter purpose tag: these are orchestrator ledger calls.
            with call_purpose("ledger"):
                if self._model_client.model_info.get("structured_output", False):
                    response = await self._model_client.create(
                        self._get_compatible_context(context), json_output=LedgerEntry
                    )
                elif self._model_client.model_info.get("json_output", False):
                    response = await self._model_client.create(
                        self._get_compatible_context(context),
                        cancellation_token=cancellation_token,
                        json_output=True,
                    )
                else:
                    response = await self._model_client.create(
                        self._get_compatible_context(context),
                        cancellation_token=cancellation_token,
                    )
            ledger_str = response.content
            try:
                assert isinstance(ledger_str, str)
                output_json = extract_json_from_str(ledger_str)
                if len(output_json) != 1:
                    raise ValueError(
                        f"Progress ledger should contain a single JSON object, "
                        f"but found: {len(output_json)}"
                    )
                progress_ledger = output_json[0]
                if len(self._participant_names) == 1:
                    progress_ledger["next_speaker"] = {
                        "reason": "The team consists of only one agent.",
                        "answer": self._participant_names[0],
                    }
                required_keys = [
                    "is_request_satisfied",
                    "is_progress_being_made",
                    "is_in_loop",
                    "instruction_or_question",
                    "next_speaker",
                ]
                key_error = False
                for key in required_keys:
                    if (
                        key not in progress_ledger
                        or not isinstance(progress_ledger[key], dict)
                        or "answer" not in progress_ledger[key]
                        or "reason" not in progress_ledger[key]
                    ):
                        key_error = True
                        break
                # Guarded on key_error: upstream indexes these keys even when the
                # check above found one missing, which raises instead of retrying.
                if not key_error and (
                    not progress_ledger["is_request_satisfied"]["answer"]
                    and progress_ledger["next_speaker"]["answer"] not in self._participant_names
                ):
                    key_error = True
                    break
                if not key_error:
                    break
                await self._log_message(
                    f"Failed to parse ledger information, retrying: {ledger_str}"
                )
            # AssertionError is caught too: a filtered response (for example
            # MALFORMED_FUNCTION_CALL) is not a str and should use a retry.
            except (json.JSONDecodeError, TypeError, AssertionError, KeyError):
                key_error = True
                await self._log_message("Invalid ledger format encountered, retrying...")
                continue
        if key_error:
            raise ValueError("Failed to parse ledger information after multiple retries.")
        await self._log_message(f"Progress Ledger: {progress_ledger}")
        # ARIA trace telemetry: one snapshot per parsed ledger, with the
        # stall count as it stood before this round's stall accounting.
        if self._aria_controller is not None:
            self._aria_controller.note_ledger(
                {"round": self._n_rounds, "n_stalls": self._n_stalls, "ledger": progress_ledger}
            )

        # Check for task completion
        if progress_ledger["is_request_satisfied"]["answer"]:
            await self._log_message("Task completed, preparing final answer...")
            # -------------------- ARIA hook: pre-delivery -------------
            # Consult before declaring done: the decision gates the
            # final-answer synthesis itself.
            guidance: str | None = None
            if self._aria_controller is not None:
                guidance = await self._aria_controller.on_contact_point(
                    plan=str(self._plan),
                    records=self._aria_records(),
                    upcoming_instruction=FINAL_ANSWER_INSTRUCTION,
                    kind=ContactPointKind.PRE_DELIVERY,
                )
            # Gate revision: an off_track verdict turns "finalize now" into
            # continued work (once per episode). The guidance is broadcast as an
            # orchestrator message, the next ledger round drives the revision,
            # and max_turns still bounds the episode.
            if (
                guidance
                and self._gate_revision_rounds > 0
                and self._gate_revisions_used == 0
                and self._aria_controller is not None
                and self._aria_controller.consults
                and self._aria_controller.consults[-1].get("assessment") == "off_track"
            ):
                self._gate_revisions_used = 1
                await self._log_message("Gate revision triggered by off_track verdict")
                revision = TextMessage(
                    content=(
                        "Hold the final answer — an expert review found the "
                        f"work incomplete:\n\n{guidance}\n\nAddress this "
                        "before finishing."
                    ),
                    source=self._name,
                )
                await self.update_message_thread([revision])
                await self.publish_message(
                    GroupChatMessage(message=revision),
                    topic_id=DefaultTopicId(type=self._output_topic_type),
                )
                await self._output_message_queue.put(revision)
                await self.publish_message(
                    GroupChatAgentResponse(
                        response=Response(chat_message=revision), name=self._name
                    ),
                    topic_id=DefaultTopicId(type=self._group_topic_type),
                    cancellation_token=cancellation_token,
                )
                await self._orchestrate_step(cancellation_token)
                return
            # ------------------------ end ARIA hook --------------------

            # ---------- ORGANIC-ONLY: final turn for the liaison ----------
            # Upstream goes from is_request_satisfied straight to synthesis, so a requested
            # liaison turn would never run; this block is organic's counterpart to PRE_DELIVERY.
            # Needs organic_liaison, a liaison participant and the ledger naming it; at most once.
            if (
                self._organic_liaison
                and LIAISON_NAME in self._participant_name_to_topic_type
                and progress_ledger["next_speaker"]["answer"] == LIAISON_NAME
                and not self._organic_final_turn_used
            ):
                self._organic_final_turn_used = True
                await self._log_message(
                    "Organic: granting the requested advisor_liaison turn before finalizing"
                )
                ask = TextMessage(
                    content=progress_ledger["instruction_or_question"]["answer"],
                    source=self._name,
                )
                await self.update_message_thread([ask])
                await self.publish_message(
                    GroupChatMessage(message=ask),
                    topic_id=DefaultTopicId(type=self._output_topic_type),
                )
                await self._output_message_queue.put(ask)
                await self.publish_message(
                    GroupChatAgentResponse(response=Response(chat_message=ask), name=self._name),
                    topic_id=DefaultTopicId(type=self._group_topic_type),
                    cancellation_token=cancellation_token,
                )
                await self.publish_message(
                    GroupChatRequestPublish(),
                    topic_id=DefaultTopicId(
                        type=self._participant_name_to_topic_type[LIAISON_NAME]
                    ),
                    cancellation_token=cancellation_token,
                )
                return  # the liaison replies; the next ledger round finalizes
            # ------------------- end ORGANIC-ONLY block -----------------

            await self._aria_prepare_final_answer(
                progress_ledger["is_request_satisfied"]["reason"],
                cancellation_token,
                guidance=guidance,
                stop_reason=STOP_COMPLETED,
            )
            return

        # Check for stalling
        if not progress_ledger["is_progress_being_made"]["answer"]:
            self._n_stalls += 1
        elif progress_ledger["is_in_loop"]["answer"]:
            self._n_stalls += 1
        else:
            self._n_stalls = max(0, self._n_stalls - 1)

        # Too much stalling
        if self._n_stalls >= self._max_stalls:
            await self._log_message("Stall count exceeded, re-planning with the outer loop...")
            # -------------------- ARIA hook: stall/replan --------------
            # Resolves the pending decision against the failing records before the
            # replan clears them, with STALL as successor so the transition chain stays
            # intact. On CONSULT the guidance is stashed for _update_task_ledger.
            if self._aria_controller is not None:
                self._aria_replan_guidance = await self._aria_controller.on_contact_point(
                    plan=str(self._plan),
                    records=self._aria_records(),
                    upcoming_instruction=STALL_REPLAN_INSTRUCTION,
                    kind=ContactPointKind.STALL,
                )
                self._aria_controller.note_ledger(
                    {"event": "replan", "round": self._n_rounds, "n_stalls": self._n_stalls}
                )
            # ------------------------ end ARIA hook --------------------
            await self._update_task_ledger(cancellation_token)
            await self._reenter_outer_loop(cancellation_token)
            return

        instruction_content: str = progress_ledger["instruction_or_question"]["answer"]

        # ---------------------- ARIA hook: ledger round ----------------
        # The next instruction is known; the decision gates exactly that
        # step, so the contact point classifies the upcoming instruction.
        if self._aria_controller is not None:
            guidance = await self._aria_controller.on_contact_point(
                plan=str(self._plan),
                records=self._aria_records(),
                upcoming_instruction=instruction_content,
            )
            if guidance:
                instruction_content = f"{instruction_content}\n\n{guidance}"
        # -------------------------- end ARIA hook ----------------------

        # Broadcast the next step
        message = TextMessage(content=instruction_content, source=self._name)
        await self.update_message_thread([message])  # My copy

        await self._log_message(f"Next Speaker: {progress_ledger['next_speaker']['answer']}")
        await self.publish_message(
            GroupChatMessage(message=message),
            topic_id=DefaultTopicId(type=self._output_topic_type),
        )
        await self._output_message_queue.put(message)
        await self.publish_message(
            GroupChatAgentResponse(response=Response(chat_message=message), name=self._name),
            topic_id=DefaultTopicId(type=self._group_topic_type),
            cancellation_token=cancellation_token,
        )

        # Request that the step be completed
        next_speaker = progress_ledger["next_speaker"]["answer"]
        if next_speaker not in self._participant_name_to_topic_type:
            raise ValueError(
                f"Invalid next speaker: {next_speaker} from the ledger, "
                f"participants are: {self._participant_names}"
            )
        participant_topic_type = self._participant_name_to_topic_type[next_speaker]
        await self.publish_message(
            GroupChatRequestPublish(),
            topic_id=DefaultTopicId(type=participant_topic_type),
            cancellation_token=cancellation_token,
        )

        if self._emit_team_events:
            from autogen_agentchat.messages import SelectSpeakerEvent

            select_msg = SelectSpeakerEvent(content=[next_speaker], source=self._name)
            await self.publish_message(
                GroupChatMessage(message=select_msg),
                topic_id=DefaultTopicId(type=self._output_topic_type),
            )
            await self._output_message_queue.put(select_msg)

    # ------------------------------------------------------------------
    # Final answer (mirrors _prepare_final_answer + guidance + capture)
    # ------------------------------------------------------------------

    async def _aria_prepare_final_answer(
        self,
        reason: str,
        cancellation_token: CancellationToken,
        guidance: str | None = None,
        stop_reason: str = STOP_COMPLETED,
    ) -> None:
        context = self._thread_to_context()
        final_answer_prompt = self._get_final_answer_prompt(self._task)
        if guidance:
            final_answer_prompt = f"{final_answer_prompt}\n\n{guidance}"
        context.append(UserMessage(content=final_answer_prompt, source=self._name))

        # With typed delivery the score is computed from this string. A filtered
        # response (for example MALFORMED_FUNCTION_CALL) is not a str, so fall
        # back to `.thought` instead of upstream's str assertion, and retry an
        # empty answer up to _FINAL_ANSWER_RETRIES times before delivering it.
        final_text = ""
        for attempt in range(_FINAL_ANSWER_RETRIES + 1):
            with call_purpose("final_answer"):
                response = await self._model_client.create(
                    self._get_compatible_context(context),
                    cancellation_token=cancellation_token,
                )
            content = response.content
            final_text = content if isinstance(content, str) else (response.thought or "")
            if final_text.strip():
                break
            logger.warning(
                "final answer empty (finish_reason=%s, content=%s), retry %d/%d",
                response.finish_reason,
                type(content).__name__,
                attempt + 1,
                _FINAL_ANSWER_RETRIES,
            )
        else:
            logger.error("final answer still empty after %d attempts", _FINAL_ANSWER_RETRIES + 1)
        message = TextMessage(content=final_text, source=self._name)

        # -------------------- ARIA: explicit final-answer capture ------
        if self._aria_controller is not None:
            self._aria_controller.note_final(final_text, stop_reason)
        # ----------------------------------------------------------------

        await self.update_message_thread([message])  # My copy
        await self.publish_message(
            GroupChatMessage(message=message),
            topic_id=DefaultTopicId(type=self._output_topic_type),
        )
        await self._output_message_queue.put(message)
        await self.publish_message(
            GroupChatAgentResponse(response=Response(chat_message=message), name=self._name),
            topic_id=DefaultTopicId(type=self._group_topic_type),
            cancellation_token=cancellation_token,
        )

        if self._termination_condition is not None:
            await self._termination_condition.reset()
        await self._signal_termination(StopMessage(content=reason, source=self._name))


class AriaGroupChat(MagenticOneGroupChat):
    """MagenticOneGroupChat wired to build an AriaOrchestrator."""

    def __init__(
        self,
        *args: Any,
        controller: InterventionController | None = None,
        gate_revision_rounds: int = 0,
        organic_liaison: bool = False,
        **kwargs: Any,
    ):
        self._aria_controller = controller
        self._gate_revision_rounds_cfg = gate_revision_rounds
        self._organic_liaison_cfg = organic_liaison
        super().__init__(*args, **kwargs)

    def _create_group_chat_manager_factory(
        self,
        name: str,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: list[str],
        participant_names: list[str],
        participant_descriptions: list[str],
        output_message_queue: Any,
        termination_condition: TerminationCondition | None,
        max_turns: int | None,
        message_factory: MessageFactory,
    ) -> Callable[[], MagenticOneOrchestrator]:
        return lambda: AriaOrchestrator(
            name,
            group_topic_type,
            output_topic_type,
            participant_topic_types,
            participant_names,
            participant_descriptions,
            max_turns,
            message_factory,
            self._model_client,
            self._max_stalls,
            self._final_answer_prompt,
            output_message_queue,
            termination_condition,
            self._emit_team_events,
            controller=self._aria_controller,
            gate_revision_rounds=self._gate_revision_rounds_cfg,
            organic_liaison=self._organic_liaison_cfg,
        )


def extract_final_answer(
    messages: Sequence[Any], orchestrator_name: str = "MagenticOneOrchestrator"
) -> str | None:
    """Return the orchestrator's last text message, or None.

    Positional fallback; prefer `controller.final_answer`, which is captured
    explicitly in `_aria_prepare_final_answer`.
    """
    for msg in reversed(messages):
        if isinstance(msg, BaseChatMessage) and msg.source == orchestrator_name:
            if isinstance(msg, TextMessage):
                return msg.content
    return None


__all__ = [
    "AriaGroupChat",
    "AriaOrchestrator",
    "ChatCompletionClient",
    "FINAL_ANSWER_INSTRUCTION",
    "STALL_REPLAN_INSTRUCTION",
    "STOP_COMPLETED",
    "STOP_MAX_ROUNDS",
    "extract_final_answer",
]
