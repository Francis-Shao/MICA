import json
from dataclasses import asdict
from typing import Dict, List, Any

from mica.core.agents.base_agent import BaseAgent

from mica.core.models.behavior import BehaviorNode
from mica.core.models.trace_context import TraceContext


class BehaviorExtractor(BaseAgent):

    VALID_BEHAVIORS = {
        "Stationary",
        "Proceed Straight",
        "Turn Left",
        "Turn Right",
        "Make U-Turn",
        "Back",
        "Brake",
        "Change Lane",
        "Enter Opposite Lane",
        "Park",
        "Unpark",
    }

    VALID_DIRECTIONS = {
        "Eastbound",
        "Westbound",
        "Southbound",
        "Northbound",
        None,
    }

    def __init__(self, context_pool=None):
        super().__init__(
            name="Behavior Extractor",
            context_pool=context_pool
        )

    def run(self, accident_report: str) -> Dict[str, Any]:

        prompt = self._build_prompt(accident_report)
        messages = self.build_messages(prompt)

        response_text = self.query_llm(messages)
        raw_result = self.parse_json_response(response_text)
        print(json.dumps(raw_result, indent=4))

        vehicle_directions = self._parse_vehicle_directions(
            raw_result.get("vehicles", {})
        )

        behavior_timelines = self._parse_behavior_timelines(
            raw_result.get("vehicles", {})
        )

        if self.context_pool is not None:
            self.context_pool.set_report(accident_report)

            self.context_pool.set_vehicle_directions(
                vehicle_directions
            )

            self.context_pool.set_behavior_timelines(
                behavior_timelines
            )

            self.context_pool.set(
                "raw_behavior_result",
                raw_result
            )

        return behavior_timelines


    def _build_prompt(self, accident_report: str) -> str:

        return f"""
You are an expert in accident report analysis and accident information extraction.

Your task is to identify the collision-involved vehicles in the accident report,
extract their travel directions, and extract their behavior timelines.

Accident Report Description:
{accident_report}


Question:
Which vehicles physically collided with each other in the final collision?
Only extract these vehicles.
Ignore vehicles involved only in earlier interactions but not in the collision.
For each collision vehicle, extract its travel direction and behavior timeline.


Behavior List and Explanation:

Stationary:
The vehicle is stopped and not moving.

Proceed Straight:
The vehicle moves forward along its current road or lane without turning.

Turn Left:
The vehicle performs a left turn at an intersection, junction, or roadway.

Turn Right:
The vehicle performs a right turn at an intersection, junction, or roadway.

Make U-Turn:
The vehicle reverses its driving direction by making a U-turn.

Back:
The vehicle moves backward or reverses.

Brake:
The vehicle slows down or attempts to stop.

Change Lane:
The vehicle moves from one lane to another lane traveling in the same direction.

Enter Opposite Lane:
The vehicle crosses into a lane intended for vehicles traveling in the opposite direction.

Park:
The vehicle moves into a parked condition.

Unpark:
The vehicle leaves a parked position and starts moving.


Travel Direction:

Eastbound:
The vehicle is traveling toward the east.

Westbound:
The vehicle is traveling toward the west.

Southbound:
The vehicle is traveling toward the south.

Northbound:
The vehicle is traveling toward the north.


Extraction Rules:

(1) Identify collision participants based on the final contact event.
Only vehicles that physically made contact during the collision should be extracted.
(2) Ignore vehicles that only affected the situation before the collision, such as vehicles changing lanes, yielding, or passing nearby.
(3) Extract behaviors in chronological order.
(4) Only extract behaviors explicitly stated or strongly implied in the report.
(5) Do not infer unsupported behaviors.
(6) If a vehicle is described as traveling along a road without turning,
classify it as "Proceed Straight".
(7) If multiple consecutive descriptions correspond to the same behavior,
do not repeat the behavior.
(8) Each vehicle must have an independent behavior timeline.
(9) Use only behaviors from the predefined behavior list.
(10) For each behavior, provide the shortest text span supporting the behavior.
(11) Evidence must be copied verbatim from the report.
(12) Temporal clauses such as "as", "when", "after", and "before" must be used to determine behavior order.
(13) Extract the travel direction of each collision-involved vehicle.
(14) Travel direction must always be one of: Eastbound, Westbound, Southbound, Northbound.
(15) Use explicitly stated directions whenever available. If the direction is not explicitly stated, infer the most likely direction based on vehicle movement, lane relationship, road geometry, and collision context.
(16) Do not output "Unknown", "Unclear", "N/A", or any other value outside the four allowed directions.
(17) Do not infer direction only from turning behavior. A turn describes a maneuver, not necessarily the original travel direction.
Output Constraints:
(1) Output must be valid JSON.
(2) Do not include explanations.

Use the following format:

{{
    "vehicles": {{
        "Vehicle Name": {{
            "travel_direction": "Direction",
            "behaviors": [
                {{
                    "behavior": "Behavior Name",
                    "source_spans": [
                        "supporting span"
                    ],
                    "source_sentences": [
                        "complete supporting sentence"
                    ]
                }}
            ]
        }}
    }}
}}


Answer:
""".strip()

    def _parse_vehicle_directions(
            self,
            raw_vehicles
    ):

        vehicle_directions = {}

        for vehicle_name, vehicle_info in raw_vehicles.items():

            direction = vehicle_info.get(
                "travel_direction"
            )

            if direction == "":
                direction = None

            if direction not in self.VALID_DIRECTIONS:
                raise ValueError(
                    f"Invalid direction '{direction}' "
                    f"for vehicle '{vehicle_name}'."
                )

            vehicle_directions[vehicle_name] = direction

        return vehicle_directions

    def _parse_behavior_timelines(
        self,
        raw_vehicles: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[BehaviorNode]]:

        behavior_timelines = {}

        for vehicle_name, vehicle_info in raw_vehicles.items():

            behaviors = vehicle_info.get(
                "behaviors",
                []
            )

            behavior_timelines[vehicle_name] = []

            for order, item in enumerate(behaviors):

                behavior = item.get(
                    "behavior"
                )

                if behavior not in self.VALID_BEHAVIORS:
                    raise ValueError(
                        f"Invalid behavior '{behavior}' "
                        f"for vehicle '{vehicle_name}'."
                    )

                context = TraceContext(
                    source_spans=item.get(
                        "source_spans",
                        []
                    ),
                    source_sentences=item.get(
                        "source_sentences",
                        []
                    ),
                )

                node = BehaviorNode(
                    vehicle_name=vehicle_name,
                    behavior=behavior,
                    order=order,
                    context=context,
                )

                behavior_timelines[vehicle_name].append(
                    node
                )

        return behavior_timelines



    @staticmethod
    def to_dict(
            behavior_timelines: Dict[str, List[BehaviorNode]]
    ) -> Dict[str, List[Dict]]:

        return {
            vehicle_name: [
                asdict(node)
                for node in nodes
            ]
            for vehicle_name, nodes in behavior_timelines.items()
        }



    @staticmethod
    def to_prompt_format(
            behavior_timelines: Dict[str, List[BehaviorNode]]
    ) -> Dict:

        result = {}

        for vehicle_name, nodes in behavior_timelines.items():

            result[vehicle_name] = []

            for node in nodes:

                result[vehicle_name].append(
                    {
                        "behavior": node.behavior,
                        "source_spans": node.context.source_spans,
                        "source_sentences": node.context.source_sentences,
                    }
                )

        return result