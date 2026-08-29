from __future__ import annotations

from itertools import permutations
from typing import Any, Dict, List, Optional, Tuple

from mica.core.models.pattern import PatternMatch


class PatternIdentifier:

    def identify_from_model(self, scenario_model: Dict[str, Any]) -> Dict[str, Any]:
        slices = (
            scenario_model.get("time_slices")
            or scenario_model.get("slices")
            or scenario_model.get("temporal_slices")
            or []
        )

        collision_events = (
            scenario_model.get("collision_events")
            or scenario_model.get("collisions")
            or []
        )

        global_spatial_relations = scenario_model.get("spatial_relations") or []

        return self.identify(
            slices=slices,
            collision_events=collision_events,
            global_spatial_relations=global_spatial_relations,
        )

    def identify(
        self,
        slices: List[Dict[str, Any]],
        collision_events: List[Dict[str, Any]],
        global_spatial_relations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:

        global_spatial_relations = global_spatial_relations if False else (global_spatial_relations or [])
        collision_pairs = self._get_collision_pairs(collision_events)

        active_relations: Dict[Tuple[str, str], Dict[str, Any]] = {}
        all_matches: List[PatternMatch] = []
        pattern_sequence: List[Dict[str, Any]] = []

        for slice_order, slice_data in enumerate(slices):
            slice_id = self._get_slice_id(slice_data, slice_order)

            current_relations = self._collect_relations_for_slice(
                slice_data=slice_data,
                slice_id=slice_id,
                global_spatial_relations=global_spatial_relations,
            )
            active_relations.update(current_relations)

            slice_matches = self._match_slice(
                slice_data=slice_data,
                slice_order=slice_order,
                active_relations=active_relations,
            )

            all_matches.extend(slice_matches)

            pattern_sequence.append(
                {
                    "slice_id": slice_id,
                    "slice_order": slice_order,
                    "patterns": [self._to_dict(m) for m in slice_matches],
                }
            )

        key_event = self._select_key_event(
            matches=all_matches,
            collision_pairs=collision_pairs,
        )

        return {
            "pattern_sequence": pattern_sequence,
            "pattern_matches": [self._to_dict(m) for m in all_matches],
            "key_event": self._to_dict(key_event) if key_event else None,
        }

    def _match_slice(
        self,
        slice_data: Dict[str, Any],
        slice_order: int,
        active_relations: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> List[PatternMatch]:

        slice_id = self._get_slice_id(slice_data, slice_order)
        vehicle_states = self._index_vehicle_states(slice_data)

        matches: List[PatternMatch] = []
        vehicles = list(vehicle_states.keys())

        for subject, target in permutations(vehicles, 2):
            subject_state = vehicle_states[subject]
            target_state = vehicle_states[target]

            relation = self._get_pair_relation(
                active_relations=active_relations,
                subject=subject,
                target=target,
            )

            pair_matches = self._match_pair(
                slice_id=slice_id,
                subject=subject,
                target=target,
                subject_state=subject_state,
                target_state=target_state,
                relation=relation,
            )

            matches.extend(pair_matches)

        return matches

    def _match_pair(
        self,
        slice_id: str,
        subject: str,
        target: str,
        subject_state: Dict[str, Any],
        target_state: Dict[str, Any],
        relation: Dict[str, Any],
    ) -> List[PatternMatch]:

        sb = self._behavior(subject_state)
        tb = self._behavior(target_state)

        ss = self._speed(subject_state)
        ts = self._speed(target_state)

        sp = self._position(subject_state)
        tp = self._position(target_state)

        rel = self._relation_type(relation)

        matched_pattern_ids: List[str] = []

        # --------------------------------------------------------------
        # 1. Backing into Vehicle
        # Behavior: Back
        # Spatial: target behind subject
        # --------------------------------------------------------------
        if sb == "back" and rel in {
            "same-lane-rear",
            "adjacent-lane-rear-left",
            "adjacent-lane-rear-right",
        }:
            matched_pattern_ids.append("BACKING_INTO_VEHICLE")

        # --------------------------------------------------------------
        # 2. Parking/Same Direction
        # Behavior: Park / Unpark
        # --------------------------------------------------------------
        if sb in {"park", "unpark"}:
            matched_pattern_ids.append("PARKING_SAME_DIRECTION")

        # --------------------------------------------------------------
        # 3. Turning/Same Direction
        # Left turn: target is on subject's left adjacent lane.
        # Right turn: target is on subject's right adjacent lane.
        # U-turn: target is same lane or adjacent lane.
        # --------------------------------------------------------------
        if sb == "turn left" and rel in {
            "adjacent-lane-front-left",
            "adjacent-lane-left",
            "adjacent-lane-rear-left",
        }:
            matched_pattern_ids.append("TURNING_SAME_DIRECTION")

        if sb == "turn right" and rel in {
            "adjacent-lane-front-right",
            "adjacent-lane-right",
            "adjacent-lane-rear-right",
        }:
            matched_pattern_ids.append("TURNING_SAME_DIRECTION")

        if sb == "make u-turn" and rel in {
            "adjacent-lane-front-left",
            "adjacent-lane-left",
            "adjacent-lane-rear-left"
        }:
            matched_pattern_ids.append("TURNING_SAME_DIRECTION")

        # --------------------------------------------------------------
        # 4. Changing Lanes/Same Direction
        # Behavior: Change Lane
        # Spatial: adjacent lane
        # --------------------------------------------------------------
        if sb == "change lane" and rel in {
            "same-lane-rear",
            "adjacent-lane-front-left",
            "adjacent-lane-left",
            "adjacent-lane-rear-left",
            "adjacent-lane-front-right",
            "adjacent-lane-right",
            "adjacent-lane-rear-right",
        }:
            matched_pattern_ids.append("CHANGING_LANES_SAME_DIRECTION")

        # --------------------------------------------------------------
        # 5. Opposite Direction/Maneuver
        # Behavior: maneuver
        # Spatial: opposite lane
        # --------------------------------------------------------------
        if sb in {
            "enter opposite lane"
        } and rel == "opposite-lane":
            matched_pattern_ids.append("OPPOSITE_DIRECTION_MANEUVER")

        # --------------------------------------------------------------
        # 6. Rear-End family
        # subject is following vehicle.
        # target is lead vehicle.
        # Spatial: target is ahead in same lane.
        # --------------------------------------------------------------

        if rel == "same-lane-front" and ss != "stopped":
            # Lead vehicle stopped / braking
            if ts == "stopped" or tb in {"brake", "stationary"}:
                matched_pattern_ids.append("LVS")
            elif sb == "change lane":
                matched_pattern_ids.append("FVM")
            elif ts == "accelerate":
                matched_pattern_ids.append("LVA")
            elif ts == "maintain":
                matched_pattern_ids.append("LVM")
            elif ts == "decelerate":
                matched_pattern_ids.append("LVD")
            elif ts == "unmentioned":
                matched_pattern_ids.append("LVM")
        elif rel is None:
            if (
                    sb == "proceed straight"
                    and (ts == "stopped" or tb in {"brake", "stationary"})
            ):
                matched_pattern_ids.append("LVS")

        # --------------------------------------------------------------
        # 7. Straight Crossing Paths
        # Both vehicles proceed straight.
        # Position: at least one inside intersection.
        # Spatial: lateral relation.
        # --------------------------------------------------------------
        if (
            sb == "proceed straight"
            and tb == "proceed straight"
            and rel in {"lateral-lane-left", "lateral-lane-right"}
        ):
            matched_pattern_ids.append("SCP")

        # --------------------------------------------------------------
        # 8. Right-turn intersection patterns
        # You can adjust lateral-side mapping here.
        # --------------------------------------------------------------
        if (
            sb == "turn right"
            and rel == "lateral-lane-left"
        ):
            matched_pattern_ids.append("RTIP")

        if (
            sb == "turn right"
            and rel == "lateral-lane-right"
        ):
            matched_pattern_ids.append("RTAP")

        # --------------------------------------------------------------
        # 9. Left-turn intersection patterns
        # --------------------------------------------------------------
        if (
            sb == "turn left"
            and rel == "opposite-lane"
        ):
            matched_pattern_ids.append("LTAP_OD")

        if (
            sb == "turn left"
            and rel == "lateral-lane-left"
        ):
            matched_pattern_ids.append("LTAP_LD")

        if (
            sb == "turn left"
            and rel == "lateral-lane-right"
        ):
            matched_pattern_ids.append("LTIP")

        return [
            self._build_match(
                pattern_id=pattern_id,
                slice_id=slice_id,
                subject=subject,
                target=target,
                subject_state=subject_state,
                target_state=target_state,
                relation=relation,
            )
            for pattern_id in matched_pattern_ids
        ]


    def _collect_relations_for_slice(
        self,
        slice_data: Dict[str, Any],
        slice_id: str,
        global_spatial_relations: List[Dict[str, Any]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:

        relation_items: List[Dict[str, Any]] = []

        relation_items.extend(
            slice_data.get("spatial_relations")
            or slice_data.get("relations")
            or slice_data.get("pair_relations")
            or []
        )

        for relation in global_spatial_relations:
            if str(relation.get("slice_id", "")) == str(slice_id):
                relation_items.append(relation)

        return self._index_relations(relation_items)

    def _index_relations(
        self,
        relations: List[Dict[str, Any]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:

        indexed: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for relation in relations:
            source = self._relation_source(relation)
            target = self._relation_target(relation)

            if not source or not target:
                continue

            normalized = dict(relation)
            self._set_relation_type(normalized, self._relation_type(normalized))

            indexed[(source, target)] = normalized

            reversed_relation = self._reverse_relation(normalized)
            if reversed_relation:
                reversed_source = self._relation_source(reversed_relation)
                reversed_target = self._relation_target(reversed_relation)

                if reversed_source and reversed_target:
                    indexed[(reversed_source, reversed_target)] = reversed_relation

        return indexed

    def _get_pair_relation(
        self,
        active_relations: Dict[Tuple[str, str], Dict[str, Any]],
        subject: str,
        target: str,
    ) -> Dict[str, Any]:

        direct = active_relations.get((subject, target))
        if direct:
            return direct

        reverse = active_relations.get((target, subject))
        if reverse:
            reversed_relation = self._reverse_relation(reverse)
            if reversed_relation:
                return reversed_relation

        return {}

    def _reverse_relation(self, relation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rel = self._relation_type(relation)

        reverse_map = {
            "same-lane-front": "same-lane-rear",
            "same-lane-rear": "same-lane-front",

            "adjacent-lane-front-left": "adjacent-lane-rear-right",
            "adjacent-lane-left": "adjacent-lane-right",
            "adjacent-lane-rear-left": "adjacent-lane-front-right",

            "adjacent-lane-front-right": "adjacent-lane-rear-left",
            "adjacent-lane-right": "adjacent-lane-left",
            "adjacent-lane-rear-right": "adjacent-lane-front-left",

            "lateral-lane-left": "lateral-lane-right",
            "lateral-lane-right": "lateral-lane-left",

            "opposite-lane": "opposite-lane",
        }

        if rel not in reverse_map:
            return None

        source = self._relation_source(relation)
        target = self._relation_target(relation)

        if not source or not target:
            return None

        reversed_relation = dict(relation)

        self._set_relation_source(reversed_relation, target)
        self._set_relation_target(reversed_relation, source)
        self._set_relation_type(reversed_relation, reverse_map[rel])

        if "reference_state_id" in reversed_relation and "target_state_id" in reversed_relation:
            reversed_relation["reference_state_id"] = relation.get("target_state_id")
            reversed_relation["target_state_id"] = relation.get("reference_state_id")

        if "reference_behavior" in reversed_relation and "target_behavior" in reversed_relation:
            reversed_relation["reference_behavior"] = relation.get("target_behavior")
            reversed_relation["target_behavior"] = relation.get("reference_behavior")

        reversed_relation["is_reversed"] = True

        return reversed_relation


    def _select_key_event(
        self,
        matches: List[PatternMatch],
        collision_pairs: List[Tuple[str, str]],
    ) -> Optional[PatternMatch]:

        if not matches:
            return None

        candidates = [
            m for m in matches
            if (m.subject_vehicle, m.target_vehicle) in collision_pairs
        ]

        if not candidates:
            candidates = matches

        latest_order = max(self._slice_order(m.slice_id) for m in candidates)

        latest_matches = [
            m for m in candidates
            if self._slice_order(m.slice_id) == latest_order
        ]

        return latest_matches[-1] if latest_matches else None


    def _index_vehicle_states(self, slice_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        participants = (
            slice_data.get("participants")
            or slice_data.get("states")
            or slice_data.get("vehicles")
            or []
        )

        result = {}

        for item in participants:
            vehicle = (
                item.get("vehicle_name")
                or item.get("vehicle")
                or item.get("name")
            )
            if vehicle:
                result[vehicle] = item

        return result

    def _get_collision_pairs(self, collision_events: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        pairs = []

        for event in collision_events:
            v1 = event.get("vehicle_1")
            v2 = event.get("vehicle_2")

            if v1 and v2:
                pairs.append((v1, v2))
                pairs.append((v2, v1))

        return pairs

    def _get_slice_id(self, slice_data: Dict[str, Any], slice_order: int) -> str:
        return str(slice_data.get("slice_id") or slice_data.get("id") or f"S{slice_order}")

    def _slice_order(self, slice_id: str) -> int:
        digits = "".join(ch for ch in str(slice_id) if ch.isdigit())
        return int(digits) if digits else 0

    def _behavior(self, state: Dict[str, Any]) -> str:
        return self._norm_behavior(
            state.get("behavior")
            or state.get("behavior_type")
            or state.get("action")
        )

    def _speed(self, state: Dict[str, Any]) -> str:
        return self._norm_speed(
            state.get("speed")
            or state.get("speed_state")
            or state.get("motion")
        )

    def _position(self, state: Dict[str, Any]) -> str:
        return self._norm_position(
            state.get("position")
            or state.get("location")
        )

    def _relation_source(self, relation: Dict[str, Any]) -> Optional[str]:
        return (
            relation.get("reference_vehicle")
            or relation.get("vehicle_1")
            or relation.get("source")
            or relation.get("subject")
        )

    def _relation_target(self, relation: Dict[str, Any]) -> Optional[str]:
        return (
            relation.get("target_vehicle")
            or relation.get("vehicle_2")
            or relation.get("target")
            or relation.get("object")
        )

    def _relation_type(self, relation: Dict[str, Any]) -> str:
        return self._norm_relation(
            relation.get("relation")
            or relation.get("relation_type")
            or relation.get("spatial_relation")
            or relation.get("type")
        )

    def _set_relation_source(self, relation: Dict[str, Any], vehicle: str) -> None:
        if "reference_vehicle" in relation:
            relation["reference_vehicle"] = vehicle
        elif "vehicle_1" in relation:
            relation["vehicle_1"] = vehicle
        elif "source" in relation:
            relation["source"] = vehicle
        elif "subject" in relation:
            relation["subject"] = vehicle
        else:
            relation["reference_vehicle"] = vehicle

    def _set_relation_target(self, relation: Dict[str, Any], vehicle: str) -> None:
        if "target_vehicle" in relation:
            relation["target_vehicle"] = vehicle
        elif "vehicle_2" in relation:
            relation["vehicle_2"] = vehicle
        elif "target" in relation:
            relation["target"] = vehicle
        elif "object" in relation:
            relation["object"] = vehicle
        else:
            relation["target_vehicle"] = vehicle

    def _set_relation_type(self, relation: Dict[str, Any], relation_type: str) -> None:
        if "relation" in relation:
            relation["relation"] = relation_type
        elif "relation_type" in relation:
            relation["relation_type"] = relation_type
        elif "spatial_relation" in relation:
            relation["spatial_relation"] = relation_type
        elif "type" in relation:
            relation["type"] = relation_type
        else:
            relation["relation"] = relation_type


    def _norm_behavior(self, value: Any) -> str:
        if value is None:
            return ""

        value = str(value).strip().lower().replace("_", "-")

        mapping = {
            "stationary": "stationary",
            "proceed-straight": "proceed straight",
            "turn-left": "turn left",
            "turn-right": "turn right",
            "make-u-turn": "make u-turn",
            "u-turn": "make u-turn",
            "back": "back",
            "backing": "back",
            "brake": "brake",
            "braking": "brake",
            "change-lane": "change lane",
            "changing-lane": "change lane",
            "enter-opposite-lane": "enter opposite lane",
            "park": "park",
            "unpark": "unpark",
        }

        return mapping.get(value, value.replace("-", " "))

    def _norm_speed(self, value: Any) -> str:
        if value is None:
            return "unmentioned"

        value = str(value).strip().lower().replace("_", "-")

        mapping = {
            "maintain": "maintain",
            "moving": "maintain",
            "accelerate": "accelerate",
            "accelerating": "accelerate",
            "decelerate": "decelerate",
            "decelerating": "decelerate",
            "brake": "decelerate",
            "braking": "decelerate",
            "stopped": "stopped",
            "stationary": "stopped",
            "unmentioned": "unmentioned",
        }

        return mapping.get(value, value)

    def _norm_position(self, value: Any) -> str:
        if value is None:
            return ""

        value = str(value).strip().lower().replace("_", "-")

        mapping = {
            "inside-intersection": "inside intersection",
            "outside-intersection": "outside intersection",
            "parking-lot": "parking lot",
        }

        return mapping.get(value, value.replace("-", " "))

    def _norm_relation(self, value: Any) -> str:
        if value is None:
            return ""

        value = str(value).strip().lower().replace("_", "-")

        mapping = {
            "same-lane-front": "same-lane-front",
            "same-lane-rear": "same-lane-rear",

            "adjacent-lane-front-left": "adjacent-lane-front-left",
            "adjacent-lane-fl": "adjacent-lane-front-left",

            "adjacent-lane-left": "adjacent-lane-left",

            "adjacent-lane-rear-left": "adjacent-lane-rear-left",
            "adjacent-lane-rl": "adjacent-lane-rear-left",

            "adjacent-lane-front-right": "adjacent-lane-front-right",
            "adjacent-lane-fr": "adjacent-lane-front-right",

            "adjacent-lane-right": "adjacent-lane-right",

            "adjacent-lane-rear-right": "adjacent-lane-rear-right",
            "adjacent-lane-rr": "adjacent-lane-rear-right",

            "lateral-lane-left": "lateral-lane-left",
            "lateral-lane-right": "lateral-lane-right",

            "opposite-lane": "opposite-lane",
        }

        return mapping.get(value, value)


    def _build_match(
        self,
        pattern_id: str,
        slice_id: str,
        subject: str,
        target: str,
        subject_state: Dict[str, Any],
        target_state: Dict[str, Any],
        relation: Dict[str, Any],
    ) -> PatternMatch:

        return PatternMatch(
            pattern_id=pattern_id,
            pattern_name=self._pattern_name(pattern_id),
            slice_id=slice_id,
            subject_vehicle=subject,
            target_vehicle=target,
            matched_conditions=[
                f"subject_behavior={self._behavior(subject_state)}",
                f"target_behavior={self._behavior(target_state)}",
                f"relation={self._relation_type(relation)}",
                f"subject_speed={self._speed(subject_state)}",
                f"target_speed={self._speed(target_state)}",
                f"subject_position={self._position(subject_state)}",
                f"target_position={self._position(target_state)}",
            ],
            evidence={
                "subject_state_id": subject_state.get("state_id") or subject_state.get("id"),
                "target_state_id": target_state.get("state_id") or target_state.get("id"),
                "relation": relation,
            },
        )

    def _pattern_name(self, pattern_id: str) -> str:
        names = {
            "BACKING_INTO_VEHICLE": "Backing into Vehicle",
            "TURNING_SAME_DIRECTION": "Turning/Same Direction",
            "PARKING_SAME_DIRECTION": "Parking/Same Direction",
            "CHANGING_LANES_SAME_DIRECTION": "Changing Lanes/Same Direction",
            "OPPOSITE_DIRECTION_MANEUVER": "Opposite Direction/Maneuver",
            "FVM": "Rear-End/Striking Maneuver (FVM)",
            "LVA": "Rear-End/Lead Vehicle Accelerating (LVA)",
            "LVM": "Rear-End/Lead Vehicle Moving (LVM)",
            "LVD": "Rear-End/Lead Vehicle Decelerating (LVD)",
            "LVS": "Rear-End/Lead Vehicle Stopped (LVS)",
            "RTIP": "Right Turn Into Path (RTIP)",
            "RTAP": "Right Turn Across Path (RTAP)",
            "SCP": "Straight Crossing Paths (SCP)",
            "LTAP_LD": "Left Turn Across Path, Lateral Direction (LTAP/LD)",
            "LTIP": "Left Turn Into Path (LTIP)",
            "LTAP_OD": "Left Turn Across Path/Opposite Direction (LTAP/OD)",
        }

        return names.get(pattern_id, pattern_id)

    def _inside_intersection(self, pos1: str, pos2: str) -> bool:
        return pos1 == "inside intersection" or pos2 == "inside intersection"

    def _to_dict(self, match: Optional[PatternMatch]) -> Optional[Dict[str, Any]]:
        if match is None:
            return None

        if hasattr(match, "to_dict"):
            return match.to_dict()

        return dict(match.__dict__)