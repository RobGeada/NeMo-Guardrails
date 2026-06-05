# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Dict, List, Optional

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import ActionResult, action
from nemoguardrails.actions.llm.utils import llm_call, warn_if_truncated
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.library.self_check.utils import (
    SELF_CHECK_INPUT_DEFAULT_TASK,
    SELF_CHECK_INPUT_FLOW,
    SELF_CHECK_INPUT_TASK_PARAM,
    get_self_check_llm,
    parse_self_check_output,
    resolve_self_check_task,
)
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.types import LLMModel
from nemoguardrails.utils import new_event_dict

log = logging.getLogger(__name__)

DEFAULT_TASK = SELF_CHECK_INPUT_DEFAULT_TASK


@action(is_system_action=True)
async def self_check_input(
    llms: Dict[str, LLMModel],
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    events: Optional[List[dict]] = None,
    llm: Optional[LLMModel] = None,
    config: Optional[RailsConfig] = None,
    task: Optional[str] = None,
    **kwargs,
):
    """Checks the input from the user.

    Prompt the LLM, using the `check_input` task prompt, to determine if the input
    from the user should be allowed or not.

    Returns:
        True if the input should be allowed, False otherwise.
    """

    _MAX_TOKENS = 1024
    context = context or {}
    user_input = context.get("user_message")

    task = resolve_self_check_task(
        task,
        context,
        events,
        triggered_rail_key="triggered_input_rail",
        start_rail_event_type="StartInputRail",
        flow_id=SELF_CHECK_INPUT_FLOW,
        task_param=SELF_CHECK_INPUT_TASK_PARAM,
        default_task=DEFAULT_TASK,
    )

    llm = get_self_check_llm(llms, task, default_task=DEFAULT_TASK, main_llm=llm)

    if user_input:
        prompt = llm_task_manager.render_task_prompt(
            task=task,
            context={
                "user_input": user_input,
            },
        )
        stop = llm_task_manager.get_stop_tokens(task=task)
        max_tokens = llm_task_manager.get_max_tokens(task=task)
        max_tokens = max_tokens or _MAX_TOKENS

        # Initialize the LLMCallInfo object
        llm_call_info_var.set(LLMCallInfo(task=task))

        llm_response = await llm_call(
            llm,
            prompt,
            stop=stop,
            llm_params={
                "temperature": config.lowest_temperature,
                "max_tokens": max_tokens,
            },
        )
        warn_if_truncated(llm_response, task)
        response = llm_response.content

        if task == DEFAULT_TASK:
            log.info(f"Input self-checking result is: `{response}`.")
        else:
            log.info(f"Input self-checking result for task={task} is: `{response}`.")

        result = parse_self_check_output(llm_task_manager, task, response)

        is_safe = result[0]

        if not is_safe:
            return ActionResult(
                return_value=False,
                events=[new_event_dict("mask_prev_user_message", intent="unanswerable message")],
            )

        return is_safe
