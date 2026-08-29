import json
from typing import Any, Dict, List, Set

from mica.core.agents.base_agent import BaseAgent


class SpatialAnalyzer(BaseAgent):
    VALID_RELATIONS = {
        "Same-Lane-Front",
        "Same-Lane-Rear",

        "Adjacent-Lane-Front-Left",
        "Adjacent-Lane-Left",
        "Adjacent-Lane-Rear-Left",

        "Adjacent-Lane-Front-Right",
        "Adjacent-Lane-Right",
        "Adjacent-Lane-Rear-Right",

        "Lateral-Lane-Left",
        "Lateral-Lane-Right",

        "Opposite-Lane",
    }

    PERPENDICULAR_DIRECTION_PAIRS = {
        ("Northbound", "Eastbound"),
        ("Northbound", "Westbound"),

        ("Southbound", "Eastbound"),
        ("Southbound", "Westbound"),

        ("Eastbound", "Northbound"),
        ("Eastbound", "Southbound"),

        ("Westbound", "Northbound"),
        ("Westbound", "Southbound"),
    }

    OPPOSITE_DIRECTION_PAIRS = {
        ("Northbound", "Southbound"),
        ("Southbound", "Northbound"),

        ("Eastbound", "Westbound"),
        ("Westbound", "Eastbound"),
    }

    INVERSE_RELATIONS = {
        "Same-Lane-Front": "Same-Lane-Rear",
        "Same-Lane-Rear": "Same-Lane-Front",

        "Adjacent-Lane-Front-Left": "Adjacent-Lane-Rear-Right",
        "Adjacent-Lane-Left": "Adjacent-Lane-Right",
        "Adjacent-Lane-Rear-Left": "Adjacent-Lane-Front-Right",

        "Adjacent-Lane-Front-Right": "Adjacent-Lane-Rear-Left",
        "Adjacent-Lane-Right": "Adjacent-Lane-Left",
        "Adjacent-Lane-Rear-Right": "Adjacent-Lane-Front-Left",

        "Lateral-Lane-Left": "Lateral-Lane-Right",
        "Lateral-Lane-Right": "Lateral-Lane-Left",

        "Opposite-Lane": "Opposite-Lane",
    }

    def __init__(self, context_pool=None):
        super().__init__(
            name="Spatial Analyzer",
            context_pool=context_pool,
        )

    def run(self) -> Dict[str, List[Dict[str, Any]]]:

        if self.context_pool is None:
            raise ValueError(
                "SpatialAnalyzer requires a ContextPool."
            )

        report = self.context_pool.get_report()

        time_slices = (
            self.context_pool.get_time_slices()
        )

        vehicle_directions = (
            self.context_pool.get_vehicle_directions()
        )

        print("RAW VEHICLE DIRECTIONS")
        print(vehicle_directions)

        vehicle_directions = (
            self._normalize_vehicle_directions(
                vehicle_directions,
                time_slices,
            )
        )

        print("NORMALIZED VEHICLE DIRECTIONS")
        print(vehicle_directions)

        if report is None:
            raise ValueError(
                "No accident report found in ContextPool."
            )

        if not time_slices:
            raise ValueError(
                "No time slices found in ContextPool."
            )

        slice_information = (
            self._build_slice_information(
                time_slices
            )
        )

        (
            rule_based_relations,
            llm_time_slices,
        ) = self._generate_direction_based_relations(
            time_slices,
            vehicle_directions,
        )

        llm_relations = []

        if llm_time_slices:
            llm_slice_information = (
                self._build_slice_information(
                    llm_time_slices
                )
            )

            prompt = self._build_prompt(
                accident_report=report,
                slice_information=
                llm_slice_information,
            )

            messages = self.build_messages(
                prompt
            )

            response_text = (
                self.query_llm(messages)
            )

            raw_result = (
                self.parse_json_response(
                    response_text
                )
            )

            llm_relations = (
                raw_result.get(
                    "spatial_relations",
                    []
                )
            )

        spatial_relations = (
            self._merge_spatial_relations(
                rule_based_relations,
                llm_relations
            )
        )

        final_result = {
            "spatial_relations":
                spatial_relations
        }

        self._validate_spatial_result(
            raw_result=final_result,
            slice_information=slice_information,
        )

        self.context_pool.set_spatial_relations(
            spatial_relations
        )

        self.context_pool.set(
            "raw_spatial_result",
            final_result,
        )

        return final_result

    def _classify_direction_relation(
            self,
            reference_direction,
            target_direction,
    ):

        pair = (
            reference_direction,
            target_direction,
        )

        if pair in self.OPPOSITE_DIRECTION_PAIRS:
            return "Opposite"

        if pair in self.PERPENDICULAR_DIRECTION_PAIRS:
            return "Lateral"

        if reference_direction == target_direction:
            return "Same"

        return None


    def _create_spatial_relation(
            self,
            slice_item,
            reference_state,
            target_state,
            relation_type,
    ):
        return {
            "slice_id": slice_item["slice_id"],

            "reference_state_id":
                reference_state["state_id"],

            "reference_vehicle":
                reference_state["vehicle_name"],

            "reference_behavior":
                reference_state["behavior"],

            "reference_behavior_order":
                reference_state["behavior_order"],

            "reference_state_index":
                reference_state["state_index"],

            "target_state_id":
                target_state["state_id"],

            "target_vehicle":
                target_state["vehicle_name"],

            "target_behavior":
                target_state["behavior"],

            "target_behavior_order":
                target_state["behavior_order"],

            "target_state_index":
                target_state["state_index"],

            "relation":
                relation_type,

            "source_spans": [],
            "source_sentences": [],
        }

    def _generate_direction_based_relations(
            self,
            time_slices,
            vehicle_directions,
    ):

        for slice_item in time_slices:
            print("SLICE", slice_item["slice_id"])

            for p in slice_item["participants"]:
                print(
                    p["vehicle_name"],
                    p["behavior"]
                )

        rule_relations = []
        llm_slices = []

        for slice_item in time_slices:
            participants = slice_item.get(
                "participants",
                []
            )
            slice_has_llm_case = False
            for i in range(len(participants)):
                for j in range(i + 1, len(participants)):
                    ref = participants[i]
                    target = participants[j]
                    ref_dir = vehicle_directions.get(
                        ref["vehicle_name"]
                    )
                    target_dir = vehicle_directions.get(
                        target["vehicle_name"]
                    )

                    print(ref_dir, target_dir)

                    # direction missing
                    if (
                            ref_dir is None
                            or target_dir is None
                    ):
                        slice_has_llm_case = True
                        continue
                    relation_type = (
                        self._classify_direction_relation(
                            ref_dir,
                            target_dir,
                        )
                    )
                    # opposite
                    if relation_type == "Opposite":
                        rule_relations.append(
                            self._create_spatial_relation(
                                slice_item,
                                ref,
                                target,
                                "Opposite-Lane",
                            )
                        )
                        continue
                    # lateral
                    if relation_type == "Lateral":
                        lateral_relation = (
                            self._infer_lateral_relation(
                                ref_dir,
                                target_dir,
                            )
                        )
                        if lateral_relation:
                            rule_relations.append(
                                self._create_spatial_relation(
                                    slice_item,
                                    ref,
                                    target,
                                    lateral_relation,
                                )
                            )
                        continue
                    # same direction
                    if relation_type == "Same":
                        slice_has_llm_case = True
            if slice_has_llm_case:
                llm_slices.append(
                    slice_item
                )
        return (
            rule_relations,
            llm_slices,
        )


    def _infer_lateral_relation(
            self,
            reference_direction: str,
            target_direction: str,
    ):
        if reference_direction == "Northbound":
            if target_direction == "Westbound":
                return "Lateral-Lane-Right"
            if target_direction == "Eastbound":
                return "Lateral-Lane-Left"
        if reference_direction == "Southbound":
            if target_direction == "Eastbound":
                return "Lateral-Lane-Right"
            if target_direction == "Westbound":
                return "Lateral-Lane-Left"
        if reference_direction == "Eastbound":
            if target_direction == "Northbound":
                return "Lateral-Lane-Right"
            if target_direction == "Southbound":
                return "Lateral-Lane-Left"
        if reference_direction == "Westbound":
            if target_direction == "Southbound":
                return "Lateral-Lane-Right"
            if target_direction == "Northbound":
                return "Lateral-Lane-Left"
        return None


    def _build_slice_information(
        self,
        time_slices: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        result = {
            "time_slices": []
        }

        for slice_item in time_slices:
            result["time_slices"].append(
                {
                    "slice_id": slice_item["slice_id"],
                    "slice_type": slice_item.get("slice_type"),
                    "participants": slice_item.get("participants", []),
                    "source_relations": slice_item.get("source_relations", []),
                    "notes": slice_item.get("notes", []),
                }
            )

        return result

    def _build_prompt(
            self,
            accident_report: str,
            slice_information: Dict[str, List[Dict[str, Any]]],
    ) -> str:

        slice_json = json.dumps(
            slice_information,
            ensure_ascii=False,
            indent=2,
        )

        valid_relations_json = json.dumps(
            [
                "Same-Lane-Front",
                "Same-Lane-Rear",

                "Adjacent-Lane-Front-Left",
                "Adjacent-Lane-Left",
                "Adjacent-Lane-Rear-Left",

                "Adjacent-Lane-Front-Right",
                "Adjacent-Lane-Right",
                "Adjacent-Lane-Rear-Right",
            ],
            ensure_ascii=False,
            indent=2,
        )

        return f"""
    You are an expert in accident report analysis and spatial relation extraction.

    Your task is to identify lane-level and collision-oriented spatial relations between vehicles within the provided time slices.

    Only extract:
    {valid_relations_json}

    Accident Report:
    {accident_report}

    Time Slices:
    {slice_json}

    Spatial Relation Definition:

    Same-Lane-Front:
    The target vehicle is in the same lane and ahead of the reference vehicle.

    Same-Lane-Rear:
    The target vehicle is in the same lane and behind the reference vehicle.

    Adjacent-Lane-Left:
    The target vehicle is in the adjacent left lane of the reference vehicle.

    Adjacent-Lane-Right:
    The target vehicle is in the adjacent right lane of the reference vehicle.

    Adjacent-Lane-Front-Left:
    The target vehicle is in the adjacent left lane and ahead of the reference vehicle.

    Adjacent-Lane-Rear-Left:
    The target vehicle is in the adjacent left lane and behind the reference vehicle.

    Adjacent-Lane-Front-Right:
    The target vehicle is in the adjacent right lane and ahead of the reference vehicle.

    Adjacent-Lane-Rear-Right:
    The target vehicle is in the adjacent right lane and behind the reference vehicle.


    Extraction Rules:
    (1) Only extract relations between vehicles appearing in the same time slice.
    (2) A relation describes the target vehicle relative to the reference vehicle.
    (3) Extract a relation only when it is explicitly described or can be reliably inferred from:
    - lane information,
    - relative position,
    - vehicle behavior,
    - road configuration.
    (4) Do not infer relations between two vehicles through a third vehicle.
    Example:
    If:
    - BMW is right of Cruise AV.
    - Acura is in front of Cruise AV.
    Do NOT infer:
    Acura -> BMW = Adjacent-Lane-Right.
    (5) Do not infer front/rear from left/right information.
    (6) Do not infer left/right from front/rear information.
    (7) Prefer Adjacent-Lane relations when vehicles are explicitly described as being in neighboring lanes.
    (8) Do not output inverse duplicate relations.
    Example:
    If:
    Cruise AV -> BMW = Adjacent-Lane-Right
    Do not output:
    BMW -> Cruise AV = Adjacent-Lane-Left.
    (9) Collision-oriented spatial cues:
    - "rear-ended"
    - "struck the rear"
    - "hit the rear"
    - "contact with the rear"
    indicate that the reference vehicle is behind the target vehicle.
    Example:
    "A rear-ended B"
    Output:
    A -> B = Same-Lane-Front
    - "rear corner", "rear fascia" indicate that the contacting vehicle is behind the target vehicle.
    Do not infer adjacent lane only from corner contact.
    (10) If no reliable relation exists, output nothing.
    (11) Copy all IDs exactly from the given time slices.
    (12) source_spans must be copied from the accident report.
    (13) source_sentences must be complete sentences copied from the accident report.


    Output Format:
    {{
      "spatial_relations": [
        {{
          "slice_id": "S0",

          "reference_state_id": "StateID",
          "reference_vehicle": "Vehicle Name",
          "reference_behavior": "Behavior",
          "reference_behavior_order": 0,
          "reference_state_index": 0,

          "target_state_id": "StateID",
          "target_vehicle": "Vehicle Name",
          "target_behavior": "Behavior",
          "target_behavior_order": 0,
          "target_state_index": 0,

          "relation": "Same-Lane-Front",

          "source_spans": [
            "Supporting span"
          ],

          "source_sentences": [
            "Supporting sentence"
          ]
        }}
      ]
    }}

    Answer:
    """.strip()

    def _merge_spatial_relations(
            self,
            rule_relations,
            llm_relations,
    ):

        result = []
        seen = set()

        for relation in (
                rule_relations + llm_relations
        ):

            key = (
                relation["slice_id"],
                relation["reference_state_id"],
                relation["target_state_id"],
                relation["relation"],
            )

            if key not in seen:
                result.append(relation)
                seen.add(key)

        return result


    def _validate_spatial_result(
        self,
        raw_result: Dict[str, Any],
        slice_information: Dict[str, List[Dict[str, Any]]],
    ):
        if "spatial_relations" not in raw_result:
            raise ValueError(
                "SpatialAnalyzer output missing key: spatial_relations"
            )

        spatial_relations = raw_result["spatial_relations"]

        if not isinstance(spatial_relations, list):
            raise ValueError(
                "SpatialAnalyzer field 'spatial_relations' must be a list."
            )

        slice_by_id = {
            slice_item["slice_id"]: slice_item
            for slice_item in slice_information["time_slices"]
        }

        state_by_slice = self._build_state_by_slice(slice_information)

        seen_relations = set()

        for idx, relation_item in enumerate(spatial_relations):
            self._validate_single_relation(
                relation_item=relation_item,
                relation_index=idx,
                slice_by_id=slice_by_id,
                state_by_slice=state_by_slice,
                seen_relations=seen_relations,
            )

    def _build_state_by_slice(
        self,
        slice_information: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        state_by_slice = {}

        for slice_item in slice_information["time_slices"]:
            slice_id = slice_item["slice_id"]

            state_by_slice[slice_id] = {
                participant["state_id"]: participant
                for participant in slice_item.get("participants", [])
            }

        return state_by_slice

    def _validate_single_relation(
        self,
        relation_item: Dict[str, Any],
        relation_index: int,
        slice_by_id: Dict[str, Dict[str, Any]],
        state_by_slice: Dict[str, Dict[str, Dict[str, Any]]],
        seen_relations: Set,
    ):
        required_keys = {
            "slice_id",

            "reference_state_id",
            "reference_vehicle",
            "reference_behavior",
            "reference_behavior_order",
            "reference_state_index",

            "target_state_id",
            "target_vehicle",
            "target_behavior",
            "target_behavior_order",
            "target_state_index",

            "relation",
            "source_spans",
            "source_sentences",
        }

        missing_keys = required_keys - set(relation_item.keys())
        if missing_keys:
            raise ValueError(
                f"Spatial relation at index {relation_index} "
                f"missing keys: {missing_keys}"
            )

        slice_id = relation_item["slice_id"]

        if slice_id not in slice_by_id:
            raise ValueError(
                f"Invalid slice_id '{slice_id}' at spatial relation "
                f"index {relation_index}."
            )

        relation = relation_item["relation"]

        if relation not in self.VALID_RELATIONS:
            raise ValueError(
                f"Invalid spatial relation '{relation}' at index "
                f"{relation_index}."
            )

        reference_state_id = relation_item["reference_state_id"]
        target_state_id = relation_item["target_state_id"]

        if reference_state_id == target_state_id:
            raise ValueError(
                f"Spatial relation at index {relation_index} has identical "
                f"reference and target state."
            )

        slice_states = state_by_slice[slice_id]

        if reference_state_id not in slice_states:
            raise ValueError(
                f"reference_state_id '{reference_state_id}' does not belong "
                f"to slice '{slice_id}' at spatial relation index "
                f"{relation_index}."
            )

        if target_state_id not in slice_states:
            raise ValueError(
                f"target_state_id '{target_state_id}' does not belong "
                f"to slice '{slice_id}' at spatial relation index "
                f"{relation_index}."
            )

        reference_state = slice_states[reference_state_id]
        target_state = slice_states[target_state_id]

        if reference_state["vehicle_name"] == target_state["vehicle_name"]:
            raise ValueError(
                f"Spatial relation at index {relation_index} connects states "
                f"of the same vehicle."
            )

        self._check_endpoint_consistency(
            relation_item=relation_item,
            relation_index=relation_index,
            prefix="reference",
            state=reference_state,
        )

        self._check_endpoint_consistency(
            relation_item=relation_item,
            relation_index=relation_index,
            prefix="target",
            state=target_state,
        )

        source_spans = relation_item["source_spans"]
        source_sentences = relation_item["source_sentences"]

        if not isinstance(source_spans, list):
            raise ValueError(
                f"source_spans must be a list at spatial relation "
                f"index {relation_index}."
            )

        if not isinstance(source_sentences, list):
            raise ValueError(
                f"source_sentences must be a list at spatial relation "
                f"index {relation_index}."
            )

        self._check_duplicate_or_inverse_relation(
            relation_item=relation_item,
            relation_index=relation_index,
            seen_relations=seen_relations,
        )

    def _normalize_vehicle_directions(
            self,
            vehicle_directions,
            time_slices,
    ):
        normalized = {}

        for slice_item in time_slices:
            for participant in slice_item.get(
                    "participants",
                    []
            ):
                vehicle_name = participant["vehicle_name"]

                if vehicle_name in vehicle_directions:
                    normalized[vehicle_name] = (
                        vehicle_directions[vehicle_name]
                    )
                    continue

                for raw_name, direction in vehicle_directions.items():

                    if (
                            vehicle_name.lower()
                            in raw_name.lower()
                            or
                            raw_name.lower()
                            in vehicle_name.lower()
                    ):
                        normalized[vehicle_name] = direction
                        break

        return normalized


    def _check_duplicate_or_inverse_relation(
        self,
        relation_item: Dict[str, Any],
        relation_index: int,
        seen_relations: Set,
    ):
        slice_id = relation_item["slice_id"]
        reference_state_id = relation_item["reference_state_id"]
        target_state_id = relation_item["target_state_id"]
        relation = relation_item["relation"]

        relation_key = (
            slice_id,
            reference_state_id,
            target_state_id,
            relation,
        )

        if relation_key in seen_relations:
            raise ValueError(
                f"Duplicate spatial relation found at index {relation_index}."
            )

        inverse_relation = self.INVERSE_RELATIONS.get(relation)

        inverse_relation_key = (
            slice_id,
            target_state_id,
            reference_state_id,
            inverse_relation,
        )

        if inverse_relation_key in seen_relations:
            raise ValueError(
                f"Redundant inverse spatial relation found at index "
                f"{relation_index}."
            )

        seen_relations.add(relation_key)

    def _check_endpoint_consistency(
        self,
        relation_item: Dict[str, Any],
        relation_index: int,
        prefix: str,
        state: Dict[str, Any],
    ):
        vehicle_key = f"{prefix}_vehicle"
        behavior_key = f"{prefix}_behavior"
        behavior_order_key = f"{prefix}_behavior_order"
        state_index_key = f"{prefix}_state_index"

        if relation_item[vehicle_key] != state["vehicle_name"]:
            raise ValueError(
                f"{vehicle_key} mismatch at spatial relation index "
                f"{relation_index}. Expected '{state['vehicle_name']}', "
                f"got '{relation_item[vehicle_key]}'."
            )

        if relation_item[behavior_key] != state["behavior"]:
            raise ValueError(
                f"{behavior_key} mismatch at spatial relation index "
                f"{relation_index}. Expected '{state['behavior']}', "
                f"got '{relation_item[behavior_key]}'."
            )

        if relation_item[behavior_order_key] != state["behavior_order"]:
            raise ValueError(
                f"{behavior_order_key} mismatch at spatial relation index "
                f"{relation_index}. Expected '{state['behavior_order']}', "
                f"got '{relation_item[behavior_order_key]}'."
            )

        if relation_item[state_index_key] != state["state_index"]:
            raise ValueError(
                f"{state_index_key} mismatch at spatial relation index "
                f"{relation_index}. Expected '{state['state_index']}', "
                f"got '{relation_item[state_index_key]}'."
            )