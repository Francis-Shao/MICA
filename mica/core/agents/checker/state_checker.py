import json
from typing import Any, Dict, List

from mica.core.agents.checker.base_checker import BaseChecker
from mica.core.agents.analyzer.behavior_extractor import BehaviorExtractor
from mica.core.agents.analyzer.state_extractor import StateExtractor
from mica.core.models.behavior import BehaviorNode
from mica.core.models.state import StateNode, BehaviorStateNode
from mica.core.models.trace_context import TraceContext


class StateChecker(BaseChecker):
    VALID_POSITIONS = StateExtractor.VALID_POSITIONS
    VALID_SPEEDS = StateExtractor.VALID_SPEEDS

    def __init__(self, context_pool=None):
        super().__init__(
            name="State Checker",
            context_pool=context_pool,
        )

    def run(
        self,
        write_back: bool = True,
    ) -> Dict[str, List[BehaviorStateNode]]:

        self._require_context_pool()

        report = self._get_report()
        behavior_timelines = self.context_pool.get_behavior_timelines()
        state_timelines = self.context_pool.get_state_timelines()

        if not behavior_timelines:
            raise ValueError("No behavior timelines found in ContextPool.")

        if not state_timelines:
            raise ValueError("No state timelines found in ContextPool.")

        behavior_information = BehaviorExtractor.to_prompt_format(
            behavior_timelines
        )

        state_information = StateExtractor.to_prompt_format(
            state_timelines
        )

        state_hints = self._build_state_hints(
            behavior_timelines=behavior_timelines,
            state_timelines=state_timelines,
        )

        prompt = self._build_prompt(
            accident_report=report,
            behavior_information=behavior_information,
            state_information=state_information,
            state_hints=state_hints,
        )

        raw_check_result = self._query_json(prompt)

        self._validate_check_result(raw_check_result)

        corrected_raw = raw_check_result["corrected_state_timelines"]

        corrected_state_timelines = self._parse_state_timelines(
            raw_result=corrected_raw,
            behavior_timelines=behavior_timelines,
        )

        self._store_check_result(
            "raw_state_check_result",
            raw_check_result,
        )

        self._store_check_result(
            "state_check_hints",
            state_hints,
        )

        if write_back:
            self.context_pool.set_state_timelines(corrected_state_timelines)
            self.context_pool.set(
                "checked_state_result",
                corrected_raw,
            )

        return corrected_state_timelines

    def _build_state_hints(
        self,
        behavior_timelines: Dict[str, List[BehaviorNode]],
        state_timelines: Dict[str, List[BehaviorStateNode]],
    ) -> List[Dict[str, Any]]:

        hints: List[Dict[str, Any]] = []

        for vehicle_name, behavior_state_nodes in state_timelines.items():
            previous_position = None
            previous_speed = None

            for behavior_state_node in behavior_state_nodes:
                behavior = behavior_state_node.behavior

                for state_index, state in enumerate(behavior_state_node.states):
                    position = state.position
                    speed = state.speed

                    expected_speed = self._expected_speed_for_behavior(
                        behavior
                    )

                    if expected_speed is not None and speed != expected_speed:
                        hints.append(
                            {
                                "vehicle_name": vehicle_name,
                                "behavior": behavior,
                                "state_index": state_index,
                                "hint_type": "behavior_speed_inconsistency",
                                "current_speed": speed,
                                "suggested_speed": expected_speed,
                                "description": (
                                    f"Behavior '{behavior}' usually requires "
                                    f"Speed '{expected_speed}', but the current "
                                    f"state uses Speed '{speed}'."
                                ),
                            }
                        )

                    if (
                        previous_position is not None
                        and previous_position != position
                    ):
                        hints.append(
                            {
                                "vehicle_name": vehicle_name,
                                "behavior": behavior,
                                "state_index": state_index,
                                "hint_type": "position_transition_check",
                                "previous_position": previous_position,
                                "current_position": position,
                                "description": (
                                    "The vehicle position changes from "
                                    f"'{previous_position}' to '{position}'. "
                                    "Check whether the report evidence supports "
                                    "this position transition."
                                ),
                            }
                        )

                    if (
                        previous_speed is not None
                        and previous_speed != speed
                    ):
                        hints.append(
                            {
                                "vehicle_name": vehicle_name,
                                "behavior": behavior,
                                "state_index": state_index,
                                "hint_type": "speed_transition_check",
                                "previous_speed": previous_speed,
                                "current_speed": speed,
                                "description": (
                                    "The vehicle speed state changes from "
                                    f"'{previous_speed}' to '{speed}'. "
                                    "Check whether the behavior and report evidence "
                                    "support this speed transition."
                                ),
                            }
                        )

                    previous_position = position
                    previous_speed = speed

        return hints

    def _expected_speed_for_behavior(
        self,
        behavior: str,
    ) -> str | None:
        if behavior == "Stationary":
            return "Stopped"

        if behavior == "Brake":
            return "Decelerate"

        if behavior in {
            "Proceed Straight",
            "Turn Left",
            "Turn Right",
            "Make U-Turn",
            "Back",
            "Change Lane",
            "Enter Opposite Lane",
            "Park",
            "Unpark",
        }:
            return "Maintain"

        return None

    def _build_prompt(
        self,
        accident_report: str,
        behavior_information: Dict[str, List[Dict[str, Any]]],
        state_information: Dict[str, List[Dict[str, Any]]],
        state_hints: List[Dict[str, Any]] | None = None,
    ) -> str:
        behavior_json = json.dumps(
            behavior_information,
            ensure_ascii=False,
            indent=2,
        )

        state_json = json.dumps(
            state_information,
            ensure_ascii=False,
            indent=2,
        )

        state_hints_json = json.dumps(
            state_hints or [],
            ensure_ascii=False,
            indent=2,
        )

        valid_positions = "\n".join(
            f"- {position}"
            for position in sorted(self.VALID_POSITIONS)
        )

        valid_speeds = "\n".join(
            f"- {speed}"
            for speed in sorted(self.VALID_SPEEDS)
        )

        return f"""
You are an expert in accident report analysis and extraction result validation.

Your task is to check whether the extracted state timelines are correct according to the accident report and the given behavior timelines, and revise them when necessary.

Accident Report Description:
{accident_report}

Behavior Timelines:
{behavior_json}

Extracted State Timelines:
{state_json}

Potential State Inconsistency Hints:
{state_hints_json}

Valid Position Values:
{valid_positions}

Valid Speed Values:
{valid_speeds}

State Definition:
A state is defined by Position and Speed.

Position:
Outside Intersection: The vehicle is located outside an intersection.
Inside Intersection: The vehicle is located inside an intersection.
Parking Lot: The vehicle is located inside a parking area or parking lot.

Speed:
Stopped: The vehicle is not moving.
Maintain: The vehicle is moving without explicit acceleration or deceleration.
Accelerate: The vehicle is increasing its speed.
Decelerate: The vehicle is decreasing its speed.

Mandatory Checking Procedure:
Step 1: Check whether each state belongs to the corresponding behavior in Behavior Timelines.

Step 2: Check whether the behavior order and behavior names in Extracted State Timelines exactly match the given Behavior Timelines.

Step 3: Check whether Position and Speed values are selected only from the valid value lists.

Step 4: Check whether Speed is consistent with the behavior:
- Stationary should have Speed "Stopped".
- Brake should have Speed "Decelerate".
- Proceed Straight, Turn Left, Turn Right, Make U-Turn, Back, Change Lane, Enter Opposite Lane, Park, and Unpark should usually have Speed "Maintain" unless acceleration or deceleration is explicitly stated.

Step 5: Check whether Position is supported by the report:
- If the vehicle is waiting at a red light, stopped before entering an intersection, approaching an intersection, or traveling outside an intersection, use "Outside Intersection".
- If the vehicle enters or moves within an intersection, use "Inside Intersection".
- If the vehicle is in a parking lot or parking area, use "Parking Lot".

Step 6: Check whether any Position transition is supported by evidence.
For example, "entered the intersection" supports a transition from Outside Intersection to Inside Intersection.

Step 7: Check whether each source span actually supports the corresponding state of the same vehicle and behavior.

Step 8: Use Potential State Inconsistency Hints as hints, but verify them against the accident report before revising.

Step 9: Return the complete corrected state timelines, not only the changed parts.

Important Rules:
(1) Do not create, remove, rename, or reorder behaviors.
(2) The corrected state timelines must preserve exactly the same vehicles, behaviors, and behavior order as Behavior Timelines.
(3) You may revise states under each behavior.
(4) Each behavior must contain at least one state.
(5) A behavior may contain multiple states only if Position or Speed changes during that behavior.
(6) Do not infer lane numbers, coordinates, distances, exact speed values, or vehicle directions.
(7) If Position cannot be determined for a behavior, inherit the most recent Position of the same vehicle.
(8) If Speed cannot be determined for a behavior, infer Speed from the behavior type.
(9) Evidence must be copied verbatim from the report and must not be paraphrased.
(10) Source sentences must be complete sentences copied verbatim from the report.
(11) If a source sentence contains actions of multiple vehicles, assign each source span only to the vehicle and behavior it describes.
(12) Do not use a source span describing one vehicle as evidence for another vehicle.

Output Constraints:
(1) The output must be valid JSON.
(2) Do not include explanations outside JSON.
(3) Use the following format:

{{
  "is_valid": true or false,
  "errors": [
    {{
      "vehicle_name": "Vehicle Name",
      "behavior": "Behavior Name",
      "error_type": "wrong_position | wrong_speed | unsupported_state | wrong_evidence | behavior_mismatch | unsupported_transition",
      "description": "Brief description of the error.",
      "evidence": [
        "Relevant report span"
      ],
      "suggested_fix": "Brief description of the fix."
    }}
  ],
  "corrected_state_timelines": {{
    "Vehicle Name": [
      {{
        "behavior": "Behavior Name",
        "states": [
          {{
            "position": "Position Value",
            "speed": "Speed Value",
            "source_spans": [
              "supporting span"
            ],
            "source_sentences": [
              "complete supporting sentence"
            ]
          }}
        ]
      }}
    ]
  }}
}}

Answer:
""".strip()

    def _validate_check_result(
        self,
        raw_check_result: Dict[str, Any],
    ):
        required_keys = {
            "is_valid",
            "errors",
            "corrected_state_timelines",
        }

        missing_keys = required_keys - set(raw_check_result.keys())

        if missing_keys:
            raise ValueError(
                f"StateChecker output missing keys: {missing_keys}"
            )

        if not isinstance(raw_check_result["is_valid"], bool):
            raise ValueError(
                "StateChecker field 'is_valid' must be boolean."
            )

        if not isinstance(raw_check_result["errors"], list):
            raise ValueError(
                "StateChecker field 'errors' must be a list."
            )

        if not isinstance(
            raw_check_result["corrected_state_timelines"],
            dict,
        ):
            raise ValueError(
                "StateChecker field 'corrected_state_timelines' "
                "must be a dict."
            )

    def _parse_state_timelines(
        self,
        raw_result: Dict[str, List[Dict[str, Any]]],
        behavior_timelines: Dict[str, List[BehaviorNode]],
    ) -> Dict[str, List[BehaviorStateNode]]:
        state_timelines: Dict[str, List[BehaviorStateNode]] = {}

        for vehicle_name, behavior_nodes in behavior_timelines.items():
            if vehicle_name not in raw_result:
                raise ValueError(
                    f"Vehicle '{vehicle_name}' is missing in StateChecker output."
                )

            raw_behavior_items = raw_result[vehicle_name]

            if len(raw_behavior_items) != len(behavior_nodes):
                raise ValueError(
                    f"Behavior count mismatch for vehicle '{vehicle_name}'. "
                    f"Expected {len(behavior_nodes)}, got {len(raw_behavior_items)}."
                )

            state_timelines[vehicle_name] = []

            for idx, behavior_node in enumerate(behavior_nodes):
                raw_behavior_item = raw_behavior_items[idx]

                raw_behavior_name = raw_behavior_item.get("behavior")
                expected_behavior_name = behavior_node.behavior

                if raw_behavior_name != expected_behavior_name:
                    raise ValueError(
                        f"Behavior mismatch for vehicle '{vehicle_name}' "
                        f"at index {idx}. Expected '{expected_behavior_name}', "
                        f"got '{raw_behavior_name}'."
                    )

                raw_states = raw_behavior_item.get("states", [])

                if not raw_states:
                    raise ValueError(
                        f"No states found for vehicle '{vehicle_name}', "
                        f"behavior '{expected_behavior_name}'."
                    )

                states: List[StateNode] = []

                for state_item in raw_states:
                    position = state_item.get("position")
                    speed = state_item.get("speed")

                    if position not in self.VALID_POSITIONS:
                        raise ValueError(
                            f"Invalid position '{position}' for vehicle "
                            f"'{vehicle_name}', behavior "
                            f"'{expected_behavior_name}'."
                        )

                    if speed not in self.VALID_SPEEDS:
                        raise ValueError(
                            f"Invalid speed '{speed}' for vehicle "
                            f"'{vehicle_name}', behavior "
                            f"'{expected_behavior_name}'."
                        )

                    source_spans = state_item.get("source_spans", [])
                    source_sentences = state_item.get(
                        "source_sentences",
                        [],
                    )

                    if not isinstance(source_spans, list):
                        raise ValueError(
                            f"source_spans must be a list for vehicle "
                            f"'{vehicle_name}', behavior "
                            f"'{expected_behavior_name}'."
                        )

                    if not isinstance(source_sentences, list):
                        raise ValueError(
                            f"source_sentences must be a list for vehicle "
                            f"'{vehicle_name}', behavior "
                            f"'{expected_behavior_name}'."
                        )

                    context = TraceContext(
                        source_spans=source_spans,
                        source_sentences=source_sentences,
                    )

                    state_node = StateNode(
                        position=position,
                        speed=speed,
                        context=context,
                    )

                    states.append(state_node)

                behavior_state_node = BehaviorStateNode(
                    vehicle_name=vehicle_name,
                    behavior=behavior_node.behavior,
                    behavior_order=behavior_node.order,
                    states=states,
                )

                state_timelines[vehicle_name].append(
                    behavior_state_node
                )

        return state_timelines