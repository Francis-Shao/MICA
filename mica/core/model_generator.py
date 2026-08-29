# trace/core/model_generator.py

from typing import Any, Dict, List, Set
from collections import defaultdict

from mica.core.agents.analyzer.state_extractor import StateExtractor


class ModelGenerator:
    def __init__(self, context_pool=None):
        self.context_pool = context_pool

    def run(
        self,
        scenario_id: str = "scenario_0",
    ) -> Dict[str, Any]:
        if self.context_pool is None:
            raise ValueError("ModelGenerator requires a ContextPool.")

        state_timelines = self.context_pool.get_state_timelines()
        temporal_relations = self.context_pool.get_temporal_relations()
        time_slices = self.context_pool.get_time_slices()
        spatial_relations = self.context_pool.get_spatial_relations()

        if not state_timelines:
            raise ValueError("No state timelines found in ContextPool.")

        if time_slices is None:
            time_slices = []

        if temporal_relations is None:
            temporal_relations = []

        if spatial_relations is None:
            spatial_relations = []

        state_information = StateExtractor.to_flat_prompt_format(
            state_timelines
        )

        states = state_information["states"]

        state_by_id = {
            state["state_id"]: state
            for state in states
        }

        self._validate_references(
            state_by_id=state_by_id,
            time_slices=time_slices,
            temporal_relations=temporal_relations,
            spatial_relations=spatial_relations,
        )

        enriched_time_slices = self._build_enriched_time_slices(
            time_slices=time_slices,
            spatial_relations=spatial_relations,
        )

        scenario_model = {
            "scenario_id": scenario_id,
            "states": states,
            "time_slices": enriched_time_slices,
            "temporal_relations": temporal_relations,
            "spatial_relations": spatial_relations,
            "metadata": {
                "num_states": len(states),
                "num_time_slices": len(enriched_time_slices),
                "num_temporal_relations": len(temporal_relations),
                "num_spatial_relations": len(spatial_relations),
            },
        }

        self.context_pool.set_scenario_model(scenario_model)
        self.context_pool.set("raw_scenario_model", scenario_model)

        return scenario_model

    def _build_enriched_time_slices(
        self,
        time_slices: List[Dict[str, Any]],
        spatial_relations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        spatial_by_slice = defaultdict(list)

        for relation in spatial_relations:
            slice_id = relation.get("slice_id")
            spatial_by_slice[slice_id].append(relation)

        enriched_slices = []

        for slice_item in time_slices:
            slice_id = slice_item["slice_id"]

            enriched_slice = {
                "slice_id": slice_id,
                "slice_type": slice_item.get("slice_type"),
                "participants": slice_item.get("participants", []),
                "spatial_relations": spatial_by_slice.get(slice_id, []),
                "temporal_source_relations": slice_item.get(
                    "source_relations",
                    [],
                ),
                "notes": slice_item.get("notes", []),
            }

            enriched_slices.append(enriched_slice)

        return enriched_slices

    def _validate_references(
        self,
        state_by_id: Dict[str, Dict[str, Any]],
        time_slices: List[Dict[str, Any]],
        temporal_relations: List[Dict[str, Any]],
        spatial_relations: List[Dict[str, Any]],
    ):
        valid_state_ids = set(state_by_id.keys())

        slice_by_id = {
            slice_item["slice_id"]: slice_item
            for slice_item in time_slices
        }

        valid_slice_ids = set(slice_by_id.keys())

        self._validate_time_slice_references(
            time_slices=time_slices,
            valid_state_ids=valid_state_ids,
        )

        self._validate_temporal_references(
            temporal_relations=temporal_relations,
            valid_state_ids=valid_state_ids,
        )

        self._validate_spatial_references(
            spatial_relations=spatial_relations,
            valid_state_ids=valid_state_ids,
            valid_slice_ids=valid_slice_ids,
            slice_by_id=slice_by_id,
        )


    def _validate_time_slice_references(
        self,
        time_slices: List[Dict[str, Any]],
        valid_state_ids: Set[str],
    ):
        for slice_item in time_slices:
            slice_id = slice_item.get("slice_id")

            if slice_id is None:
                raise ValueError("A time slice is missing slice_id.")

            for participant in slice_item.get("participants", []):
                state_id = participant.get("state_id")

                if state_id not in valid_state_ids:
                    raise ValueError(
                        f"Time slice '{slice_id}' references invalid "
                        f"state_id '{state_id}'."
                    )

    def _validate_temporal_references(
        self,
        temporal_relations: List[Dict[str, Any]],
        valid_state_ids: Set[str],
    ):
        for idx, relation in enumerate(temporal_relations):
            source_state_id = relation.get("source_state_id")
            target_state_id = relation.get("target_state_id")

            if source_state_id not in valid_state_ids:
                raise ValueError(
                    f"Temporal relation at index {idx} references invalid "
                    f"source_state_id '{source_state_id}'."
                )

            if target_state_id not in valid_state_ids:
                raise ValueError(
                    f"Temporal relation at index {idx} references invalid "
                    f"target_state_id '{target_state_id}'."
                )

    def _validate_spatial_references(
        self,
        spatial_relations: List[Dict[str, Any]],
        valid_state_ids: Set[str],
        valid_slice_ids: Set[str],
        slice_by_id: Dict[str, Dict[str, Any]],
    ):
        for idx, relation in enumerate(spatial_relations):
            slice_id = relation.get("slice_id")
            reference_state_id = relation.get("reference_state_id")
            target_state_id = relation.get("target_state_id")

            if slice_id not in valid_slice_ids:
                raise ValueError(
                    f"Spatial relation at index {idx} references invalid "
                    f"slice_id '{slice_id}'."
                )

            if reference_state_id not in valid_state_ids:
                raise ValueError(
                    f"Spatial relation at index {idx} references invalid "
                    f"reference_state_id '{reference_state_id}'."
                )

            if target_state_id not in valid_state_ids:
                raise ValueError(
                    f"Spatial relation at index {idx} references invalid "
                    f"target_state_id '{target_state_id}'."
                )

            slice_state_ids = {
                participant["state_id"]
                for participant in slice_by_id[slice_id].get(
                    "participants",
                    [],
                )
            }

            if reference_state_id not in slice_state_ids:
                raise ValueError(
                    f"Spatial relation at index {idx} references "
                    f"reference_state_id '{reference_state_id}', which does "
                    f"not belong to slice '{slice_id}'."
                )

            if target_state_id not in slice_state_ids:
                raise ValueError(
                    f"Spatial relation at index {idx} references "
                    f"target_state_id '{target_state_id}', which does not "
                    f"belong to slice '{slice_id}'."
                )