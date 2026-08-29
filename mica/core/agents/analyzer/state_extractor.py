import json
from typing import Dict, List, Any

from mica.core.agents.base_agent import BaseAgent
from mica.core.agents.analyzer.behavior_extractor import BehaviorExtractor
from mica.core.models.trace_context import TraceContext
from mica.core.models.behavior import BehaviorNode
from mica.core.models.state import StateNode, BehaviorStateNode


class StateExtractor(BaseAgent):
    VALID_POSITIONS = {
        "Outside Intersection",
        "Inside Intersection",
        "Parking Lot",
    }

    VALID_SPEEDS = {
        "Stopped",
        "Maintain",
        "Accelerate",
        "Decelerate",
    }

    def __init__(self, context_pool=None):
        super().__init__(name="State Extractor", context_pool=context_pool)

    def run(self) -> Dict[str, List[BehaviorStateNode]]:
        if self.context_pool is None:
            raise ValueError("StateExtractor requires a ContextPool.")

        report = self.context_pool.get_report()
        behavior_timelines = self.context_pool.get_behavior_timelines()

        if report is None:
            raise ValueError("No accident report found in ContextPool.")

        if not behavior_timelines:
            raise ValueError("No behavior timelines found in ContextPool.")

        behavior_information = BehaviorExtractor.to_prompt_format(
            behavior_timelines
        )

        prompt = self._build_prompt(
            accident_report=report,
            behavior_information=behavior_information,
        )

        messages = self.build_messages(prompt)
        response_text = self.query_llm(messages)

        raw_result = self.parse_json_response(response_text)

        state_timelines = self._parse_state_timelines(
            raw_result=raw_result,
            behavior_timelines=behavior_timelines,
        )

        self.context_pool.set_state_timelines(state_timelines)
        self.context_pool.set("raw_state_result", raw_result)

        return state_timelines

    def _build_prompt(
        self,
        accident_report: str,
        behavior_information: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        behavior_json = json.dumps(
            behavior_information,
            ensure_ascii=False,
            indent=2,
        )

        return f"""
You are an expert in accident report analysis and accident information extraction. Your task is analysing the following accident report and behavior information, and answer the question below in the given format.

Accident Report Description:
{accident_report}

Behavior Information:
{behavior_json}

Question:
What state timeline can be derived for each existing behavior of each vehicle?

State Definition:
A state is defined by the following two attributes.

Position:
Outside Intersection: The vehicle is located outside an intersection.
Inside Intersection: The vehicle is located inside an intersection.
Parking Lot: The vehicle is located inside a parking area or parking lot.

Speed:
Stopped: The vehicle is not moving.
Maintain: The vehicle is moving without explicit acceleration or deceleration.
Accelerate: The vehicle is increasing its speed.
Decelerate: The vehicle is decreasing its speed.

Extraction Rules:
(1) Extract states in chronological order.
(2) Only extract states under the behaviors provided in Behavior Information.
(3) Do not create, remove, rename, or reorder behaviors.
(4) Each behavior must contain at least one state.
(5) A behavior may contain multiple states if Position or Speed changes during the execution of that behavior.
(6) Create a new state only when Position or Speed changes.
(7) States must remain associated with the behavior from which they are derived.
(8) Use the accident report as contextual information for identifying state changes within each given behavior.
(9) Use the provided source_spans and source_sentences as anchors for the corresponding behavior.
(10) If the behavior is "Stationary", the Speed should be "Stopped".
(11) If the behavior is "Brake", the Speed should be "Decelerate".
(12) If the behavior is "Proceed Straight", "Turn Left", "Turn Right", "Change Lane", "Enter Opposite Lane", "Back", "Park", or "Unpark", and no acceleration or deceleration is stated, the Speed should be "Maintain".
(13) If the vehicle enters or moves within an intersection, the Position should be "Inside Intersection".
(14) If the vehicle is waiting at a traffic signal, stopped before entering an intersection, approaching an intersection, or traveling outside an intersection, the Position should be "Outside Intersection".
(15) If the vehicle is located in a parking area or parking lot, the Position should be "Parking Lot".
(16) If Position cannot be determined for a state, inherit the most recent Position of the same vehicle.
(17) If Speed cannot be determined for a state, inherit the most recent Speed of the same vehicle.
(18) Do not infer lane numbers, coordinates, distances, exact speed values, or vehicle directions.
(19) For each state, provide the report text span(s) and sentence(s) used to derive the state.

Output Constraints:
(1) The output must be valid JSON format.
(2) Do not include explanations or additional text.
(3) Use the following format:
{{
    "Vehicle Name": [
        {{
            "behavior": "Behavior Name",
            "states": [
                {{
                    "position": "Position Value",
                    "speed": "Speed Value",
                    "source_spans": [
                        "Supporting Span"
                    ],
                    "source_sentences": [
                        "Supporting Sentence"
                    ]
                }}
            ]
        }}
    ]
}}

Answer:
""".strip()

    def _parse_state_timelines(
        self,
        raw_result: Dict[str, List[Dict[str, Any]]],
        behavior_timelines: Dict[str, List[BehaviorNode]],
    ) -> Dict[str, List[BehaviorStateNode]]:
        state_timelines: Dict[str, List[BehaviorStateNode]] = {}

        for vehicle_name, behavior_nodes in behavior_timelines.items():
            if vehicle_name not in raw_result:
                raise ValueError(
                    f"Vehicle '{vehicle_name}' is missing in StateExtractor output."
                )

            raw_behavior_items = raw_result[vehicle_name]

            raw_behavior_map = {
                item["behavior"]: item
                for item in raw_behavior_items
            }

            ordered_behavior_items = []

            for behavior_node in behavior_nodes:

                if behavior_node.behavior in raw_behavior_map:
                    ordered_behavior_items.append(
                        raw_behavior_map[behavior_node.behavior]
                    )

                else:
                    ordered_behavior_items.append(
                        {
                            "behavior": behavior_node.behavior,
                            "states": [
                                {
                                    "position": "Outside Intersection",
                                    "speed": "Maintain",
                                    "source_spans": [],
                                    "source_sentences": []
                                }
                            ]
                        }
                    )

            raw_behavior_items = ordered_behavior_items

            state_timelines[vehicle_name] = []

            for idx, behavior_node in enumerate(behavior_nodes):
                raw_behavior_item = raw_behavior_items[idx]

                raw_behavior_name = raw_behavior_item.get("behavior")
                expected_behavior_name = behavior_node.behavior

                if raw_behavior_name != expected_behavior_name:
                    raise ValueError(
                        f"Behavior mismatch for vehicle '{vehicle_name}' at index {idx}. "
                        f"Expected '{expected_behavior_name}', got '{raw_behavior_name}'."
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
                            f"Invalid position '{position}' for vehicle '{vehicle_name}', "
                            f"behavior '{expected_behavior_name}'."
                        )

                    if speed not in self.VALID_SPEEDS:
                        raise ValueError(
                            f"Invalid speed '{speed}' for vehicle '{vehicle_name}', "
                            f"behavior '{expected_behavior_name}'."
                        )

                    context = TraceContext(
                        source_spans=state_item.get("source_spans", []),
                        source_sentences=state_item.get("source_sentences", []),
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

                state_timelines[vehicle_name].append(behavior_state_node)

        return state_timelines

    @staticmethod
    def to_prompt_format(
        state_timelines: Dict[str, List[BehaviorStateNode]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        result = {}

        for vehicle_name, behavior_state_nodes in state_timelines.items():
            result[vehicle_name] = []

            for behavior_state_node in behavior_state_nodes:
                result[vehicle_name].append({
                    "behavior": behavior_state_node.behavior,
                    "states": [
                        {
                            "position": state.position,
                            "speed": state.speed,
                            "source_spans": state.context.source_spans,
                            "source_sentences": state.context.source_sentences,
                        }
                        for state in behavior_state_node.states
                    ],
                })

        return result

    @staticmethod
    def to_flat_prompt_format(
        state_timelines: Dict[str, List[BehaviorStateNode]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        result = {
            "states": []
        }

        for vehicle_name, behavior_state_nodes in state_timelines.items():
            safe_vehicle_name = (
                vehicle_name
                .replace(" ", "")
                .replace("-", "")
                .replace("/", "")
                .replace("\"", "")
                .replace("'", "")
            )

            for behavior_state_node in behavior_state_nodes:
                behavior_order = behavior_state_node.behavior_order
                behavior = behavior_state_node.behavior

                for state_index, state in enumerate(behavior_state_node.states):
                    state_id = (
                        f"{safe_vehicle_name}"
                        f"_B{behavior_order}"
                        f"_S{state_index}"
                    )

                    result["states"].append(
                        {
                            "state_id": state_id,
                            "vehicle_name": vehicle_name,
                            "behavior": behavior,
                            "behavior_order": behavior_order,
                            "state_index": state_index,
                            "position": state.position,
                            "speed": state.speed,
                            "source_spans": state.context.source_spans,
                            "source_sentences": state.context.source_sentences,
                        }
                    )

        return result