import json
from typing import Any, Dict, List

from mica.core.agents.checker.base_checker import BaseChecker
from mica.core.agents.analyzer.behavior_extractor import BehaviorExtractor
from mica.core.models.behavior import BehaviorNode
from mica.core.models.trace_context import TraceContext


class BehaviorChecker(BaseChecker):
    VALID_BEHAVIORS = BehaviorExtractor.VALID_BEHAVIORS

    def __init__(self, context_pool=None):
        super().__init__(
            name="Behavior Checker",
            context_pool=context_pool,
        )

    def run(
            self,
            write_back: bool = True,
    ) -> Dict[str, List[BehaviorNode]]:

        self._require_context_pool()

        report = self._get_report()

        behavior_timelines = (
            self.context_pool.get_behavior_timelines()
        )

        if not behavior_timelines:
            raise ValueError(
                "No behavior timelines found in ContextPool."
            )

        vehicle_directions = (
            self.context_pool.get_vehicle_directions()
        )

        behavior_information = BehaviorExtractor.to_prompt_format(
            behavior_timelines
        )

        prompt = self._build_prompt(
            accident_report=report,
            behavior_information=behavior_information,
            vehicle_directions=vehicle_directions,
        )

        raw_check_result = self._query_json(prompt)

        self._validate_check_result(
            raw_check_result
        )

        corrected_raw = (
            raw_check_result[
                "corrected_behavior_timelines"
            ]
        )

        corrected_timelines = (
            self._parse_behavior_timelines(
                corrected_raw
            )
        )

        self._store_check_result(
            "raw_behavior_check_result",
            raw_check_result,
        )

        if write_back:
            self.context_pool.set_behavior_timelines(
                corrected_timelines
            )

            self.context_pool.set(
                "checked_behavior_result",
                corrected_raw,
            )

        return corrected_timelines


    def _build_prompt(
            self,
            accident_report: str,
            behavior_information,
            vehicle_directions,
    ) -> str:

        behavior_json = json.dumps(
            behavior_information,
            ensure_ascii=False,
            indent=2,
        )

        direction_json = json.dumps(
            vehicle_directions or {},
            ensure_ascii=False,
            indent=2,
        )

        return f"""
    You are an expert in accident report behavior validation.

    Your task is to validate the extracted vehicle behavior timelines.
    Do not perform new behavior extraction.
    Only revise the extracted results when clear errors exist.

    Accident Report:
    {accident_report}


    Collision-Involved Vehicle Directions:
    {direction_json}


    Extracted Behavior Timelines:
    {behavior_json}


    Validation Rules:
    (1) Check whether each behavior is supported by the report.
    (2) Consider normal traffic semantics.
    For example:
    - "yielded to another vehicle" may indicate slowing down or stopping.
    - "was traveling on a road without turning" may indicate Proceed Straight.
    (3) Do not remove a behavior only because the exact action word is absent,
    if the behavior is strongly implied by traffic context.
    (4) Do not infer a specific maneuver only from collision words.
        Use the overall vehicle movement context when validating behaviors.
    (5) Check whether behavior order follows the temporal order in the report.
    (6) Check whether source spans reasonably support the behavior.
    (7) Do not introduce new vehicles.
    (8) Do not add new behaviors unless the omission is obvious.
    (9) A vehicle described as traveling on a road or involved in a collision without any turning, lane-changing, or reversing action may be considered Proceed Straight.


    Revision Rules:

    - Preserve correct behaviors.
    - Remove only clearly unsupported behaviors.
    - Correct only obvious evidence or ordering errors.
    - Keep the original behavior timeline whenever possible.


    Output Format:

    {{
      "is_valid": true or false,
      "errors": [
        {{
          "vehicle_name": "Vehicle Name",
          "error_type": "wrong_order | wrong_behavior | wrong_evidence",
          "description": "Brief description.",
          "evidence": [
            "Relevant report span"
          ],
          "suggested_fix": "Brief fix."
        }}
      ],
      "corrected_behavior_timelines": {{
        "Vehicle Name": [
          {{
            "behavior": "Behavior Name",
            "source_spans": [
              "supporting span"
            ],
            "source_sentences": [
              "complete sentence"
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
            "corrected_behavior_timelines",
        }

        missing_keys = required_keys - set(raw_check_result.keys())

        if missing_keys:
            raise ValueError(
                f"BehaviorChecker output missing keys: {missing_keys}"
            )

        if not isinstance(raw_check_result["is_valid"], bool):
            raise ValueError(
                "BehaviorChecker field 'is_valid' must be boolean."
            )

        if not isinstance(raw_check_result["errors"], list):
            raise ValueError(
                "BehaviorChecker field 'errors' must be a list."
            )

        if not isinstance(
            raw_check_result["corrected_behavior_timelines"],
            dict,
        ):
            raise ValueError(
                "BehaviorChecker field 'corrected_behavior_timelines' "
                "must be a dict."
            )

    def _parse_behavior_timelines(
        self,
        raw_result: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[BehaviorNode]]:
        behavior_timelines: Dict[str, List[BehaviorNode]] = {}

        for vehicle_name, behavior_items in raw_result.items():
            if not isinstance(behavior_items, list):
                raise ValueError(
                    f"Behavior timeline for vehicle '{vehicle_name}' "
                    f"must be a list."
                )

            behavior_timelines[vehicle_name] = []

            for order, item in enumerate(behavior_items):
                behavior = item.get("behavior")

                if behavior not in self.VALID_BEHAVIORS:
                    raise ValueError(
                        f"Invalid behavior '{behavior}' "
                        f"for vehicle '{vehicle_name}'."
                    )

                source_spans = item.get("source_spans", [])
                source_sentences = item.get("source_sentences", [])

                if not isinstance(source_spans, list):
                    raise ValueError(
                        f"source_spans must be a list for vehicle "
                        f"'{vehicle_name}', behavior '{behavior}'."
                    )

                if not isinstance(source_sentences, list):
                    raise ValueError(
                        f"source_sentences must be a list for vehicle "
                        f"'{vehicle_name}', behavior '{behavior}'."
                    )

                context = TraceContext(
                    source_spans=source_spans,
                    source_sentences=source_sentences,
                )

                node = BehaviorNode(
                    vehicle_name=vehicle_name,
                    behavior=behavior,
                    order=order,
                    context=context,
                )

                behavior_timelines[vehicle_name].append(node)

        return behavior_timelines