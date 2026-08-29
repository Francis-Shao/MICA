from typing import Any, Dict, List, Set, Tuple
from collections import defaultdict, deque

from mica.core.agents.analyzer.state_extractor import StateExtractor


class TimeSlicer:
    def __init__(self, context_pool=None):
        self.context_pool = context_pool


    def run(self) -> Dict[str, List[Dict[str, Any]]]:

        if self.context_pool is None:
            raise ValueError(
                "TimeSlicer requires a ContextPool."
            )

        state_timelines = (
            self.context_pool.get_state_timelines()
        )

        temporal_relations = (
            self.context_pool.get_temporal_relations()
        )

        if temporal_relations is None:
            temporal_relations = []

        if not state_timelines:
            raise ValueError(
                "No state timelines found."
            )


        state_information = (
            StateExtractor.to_flat_prompt_format(
                state_timelines
            )
        )

        states = state_information["states"]


        state_by_id = {
            state["state_id"]: state
            for state in states
        }


        # 1. Construct slices from When relations
        slices = (
            self._build_initial_slices_from_when_relations(
                temporal_relations,
                state_by_id,
            )
        )


        # 2. Add uncovered states
        slices = self._add_uncovered_states(
            slices,
            states,
        )


        # 3. Complete every slice with latest context
        slices = self._complete_slice_context(
            slices,
            states,
        )


        slices = self._deduplicate_slices(
            slices
        )


        slices = self._assign_slice_ids(
            slices
        )


        slices = self._sort_slices(
            slices,
            states,
            temporal_relations,
            state_by_id,
        )


        slices = self._assign_slice_ids(
            slices
        )


        result = {
            "time_slices": slices
        }


        self.context_pool.set_time_slices(
            slices
        )

        self.context_pool.set(
            "raw_time_slicing_result",
            result
        )

        return result



    def _build_initial_slices_from_when_relations(
        self,
        temporal_relations,
        state_by_id,
    ):

        slices = []


        for relation_index, relation in enumerate(
            temporal_relations
        ):

            if relation.get("relation") != "When":
                continue


            source_id = relation.get(
                "source_state_id"
            )

            target_id = relation.get(
                "target_state_id"
            )


            if (
                source_id not in state_by_id
                or target_id not in state_by_id
            ):
                continue


            state_ids = {
                source_id,
                target_id
            }


            source_relation = (
                self._brief_relation(
                    relation,
                    relation_index
                )
            )


            merged = False


            for existing in slices:

                if self._can_add_states_to_slice(
                    existing,
                    state_ids,
                    state_by_id,
                ):

                    self._add_states_to_slice(
                        existing,
                        state_ids,
                        state_by_id,
                    )

                    existing["source_relations"].append(
                        source_relation
                    )

                    merged = True
                    break


            if not merged:

                slices.append(
                    self._create_slice(
                        state_ids,
                        state_by_id,
                        "explicit_when",
                        [source_relation],
                    )
                )


        return slices



    def _add_uncovered_states(
        self,
        slices,
        states,
    ):

        covered = set()


        for slice_item in slices:
            for participant in slice_item["participants"]:
                covered.add(
                    participant["state_id"]
                )


        for state in states:

            if state["state_id"] in covered:
                continue


            slices.append(
                {
                    "slice_id": None,
                    "slice_type":
                        "state_transition",
                    "participants": [
                        self._brief_state(state)
                    ],
                    "source_relations": [],
                    "notes": [
                        "Context will be completed "
                        "using previous vehicle states."
                    ],
                }
            )


        return slices



    def _complete_slice_context(
        self,
        slices,
        states,
    ):

        states_by_vehicle = defaultdict(list)


        for state in states:
            states_by_vehicle[
                state["vehicle_name"]
            ].append(state)


        for vehicle_states in states_by_vehicle.values():

            vehicle_states.sort(
                key=lambda x:
                (
                    x["behavior_order"],
                    x["state_index"]
                )
            )


        vehicles = list(
            states_by_vehicle.keys()
        )


        for slice_item in slices:


            existing = {
                p["vehicle_name"]
                : p
                for p in slice_item["participants"]
            }


            for vehicle in vehicles:


                if vehicle in existing:
                    continue


                previous_state = (
                    self._find_latest_state(
                        vehicle,
                        slice_item,
                        states_by_vehicle,
                    )
                )


                if previous_state is None:
                    continue


                slice_item["participants"].append(
                    self._brief_state(
                        previous_state
                    )
                )


            slice_item["participants"] = (
                self._sort_participants(
                    slice_item["participants"]
                )
            )


        return slices



    def _find_latest_state(
        self,
        vehicle,
        slice_item,
        states_by_vehicle,
    ):

        if vehicle not in states_by_vehicle:
            return None


        reference_orders = []


        for participant in slice_item["participants"]:

            reference_orders.append(
                (
                    participant["behavior_order"],
                    participant["state_index"]
                )
            )


        if not reference_orders:
            return None


        reference = max(
            reference_orders
        )


        candidates = []


        for state in states_by_vehicle[vehicle]:

            order = (
                state["behavior_order"],
                state["state_index"]
            )


            if order <= reference:
                candidates.append(
                    state
                )


        if not candidates:
            return None


        return max(
            candidates,
            key=lambda x:
            (
                x["behavior_order"],
                x["state_index"]
            )
        )



    def _can_add_states_to_slice(
        self,
        slice_item,
        new_state_ids,
        state_by_id,
    ):

        vehicle_states = {}

        for participant in slice_item["participants"]:

            vehicle_states[
                participant["vehicle_name"]
            ] = participant["state_id"]


        for state_id in new_state_ids:

            state = state_by_id[state_id]

            vehicle = state["vehicle_name"]


            if (
                vehicle in vehicle_states
                and vehicle_states[vehicle]
                != state_id
            ):
                return False


        return True



    def _add_states_to_slice(
        self,
        slice_item,
        new_state_ids,
        state_by_id,
    ):

        existing = {
            p["state_id"]
            for p in slice_item["participants"]
        }


        for state_id in new_state_ids:

            if state_id in existing:
                continue


            slice_item["participants"].append(
                self._brief_state(
                    state_by_id[state_id]
                )
            )


        slice_item["participants"] = (
            self._sort_participants(
                slice_item["participants"]
            )
        )



    def _create_slice(
        self,
        state_ids,
        state_by_id,
        slice_type,
        source_relations,
    ):

        return {
            "slice_id": None,
            "slice_type": slice_type,
            "participants":
                self._sort_participants(
                    [
                        self._brief_state(
                            state_by_id[s]
                        )
                        for s in state_ids
                    ]
                ),
            "source_relations":
                source_relations,
            "notes": [],
        }



    def _deduplicate_slices(
        self,
        slices,
    ):

        result = []
        seen = set()


        for slice_item in slices:

            key = tuple(
                sorted(
                    p["state_id"]
                    for p in slice_item["participants"]
                )
            )


            if key in seen:
                continue


            seen.add(key)
            result.append(slice_item)


        return result



    def _sort_slices(
        self,
        slices,
        states,
        temporal_relations,
        state_by_id,
    ):

        return sorted(
            slices,
            key=self._slice_sort_key
        )



    def _slice_sort_key(
        self,
        slice_item,
    ):

        return (
            min(
                p["behavior_order"]
                for p in slice_item["participants"]
            ),
            min(
                p["state_index"]
                for p in slice_item["participants"]
            ),
        )



    def _assign_slice_ids(
        self,
        slices,
    ):

        for i, slice_item in enumerate(slices):
            slice_item["slice_id"] = f"S{i}"

        return slices



    def _brief_state(
        self,
        state,
    ):

        return {
            "state_id":
                state["state_id"],
            "vehicle_name":
                state["vehicle_name"],
            "behavior":
                state["behavior"],
            "behavior_order":
                state["behavior_order"],
            "state_index":
                state["state_index"],
            "position":
                state["position"],
            "speed":
                state["speed"],
            "source_spans":
                state.get("source_spans", []),
            "source_sentences":
                state.get("source_sentences", []),
        }



    def _sort_participants(
        self,
        participants,
    ):

        return sorted(
            participants,
            key=lambda p:
            (
                p["vehicle_name"],
                p["behavior_order"],
                p["state_index"],
            )
        )



    def _brief_relation(
        self,
        relation,
        relation_index,
    ):

        return {
            "relation_id":
                f"T{relation_index}",
            "relation":
                relation.get("relation"),
            "source_state_id":
                relation.get("source_state_id"),
            "target_state_id":
                relation.get("target_state_id"),
            "source_spans":
                relation.get("source_spans", []),
            "source_sentences":
                relation.get("source_sentences", []),
        }