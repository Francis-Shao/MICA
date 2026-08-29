import json
from typing import Any, Dict, List, Set

from mica.core.agents.base_agent import BaseAgent
from mica.core.agents.analyzer.state_extractor import StateExtractor


class TemporalAnalyzer(BaseAgent):
    VALID_RELATIONS = {
        "When"
    }

    TEMPORAL_CUES = [
        "when",
        "while",
        "as",
        "during",
        "meanwhile",
        "simultaneously",
        "at the same time"
    ]

    PARALLEL_CUES = [
        "parallel",
        "stayed parallel",
        "alongside",
        "next to",
        "beside",
    ]

    def __init__(self, context_pool=None):
        super().__init__(
            name="Temporal Analyzer",
            context_pool=context_pool,
        )

    def run(self) -> Dict[str, List[Dict[str, Any]]]:
        if self.context_pool is None:
            raise ValueError("TemporalAnalyzer requires a ContextPool.")

        report = self.context_pool.get_report()
        state_timelines = self.context_pool.get_state_timelines()

        if report is None:
            raise ValueError("No accident report found in ContextPool.")

        if not state_timelines:
            raise ValueError("No state timelines found in ContextPool.")

        state_information = StateExtractor.to_flat_prompt_format(
            state_timelines
        )

        temporal_candidates = self._build_temporal_candidates(
            accident_report=report,
            state_information=state_information,
        )

        prompt = self._build_prompt(
            accident_report=report,
            state_information=state_information,
            temporal_candidates=temporal_candidates,
        )

        messages = self.build_messages(prompt)
        response_text = self.query_llm(messages)
        raw_result = self.parse_json_response(response_text)

        self._validate_temporal_result(
            raw_result=raw_result,
            state_information=state_information,
        )

        temporal_relations = raw_result["temporal_relations"]

        self.context_pool.set_temporal_relations(temporal_relations)
        self.context_pool.set("raw_temporal_result", raw_result)
        self.context_pool.set("temporal_state_information", state_information)
        self.context_pool.set("temporal_candidates", temporal_candidates)

        return raw_result

    def _build_temporal_candidates(
        self,
        accident_report: str,
        state_information: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Build lightweight candidate groups for possible temporal relations.

        These candidates are not final relations. They are used to help the LLM
        notice potentially missing cross-vehicle temporal dependencies.
        """
        candidates: List[Dict[str, Any]] = []

        states = state_information.get("states", [])
        sentences = self._split_sentences(accident_report)

        candidates.extend(
            self._build_initial_scene_candidates(states)
        )

        candidates.extend(
            self._build_explicit_temporal_candidates(
                sentences=sentences,
                states=states,
            )
        )

        candidates.extend(
            self._build_parallel_movement_candidates(
                sentences=sentences,
                states=states,
            )
        )

        return self._deduplicate_candidates(candidates)

    def _build_initial_scene_candidates(
        self,
        states: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        initial_states = []

        for state in states:
            if (
                state.get("behavior_order") == 0
                and state.get("state_index") == 0
                and (
                    state.get("behavior") == "Stationary"
                    or state.get("speed") == "Stopped"
                )
            ):
                initial_states.append(state)

        vehicle_names = {
            state["vehicle_name"]
            for state in initial_states
        }

        if len(vehicle_names) < 2:
            return []

        return [
            {
                "candidate_type": "initial_scene_synchronization",
                "suggested_relation": "When",
                "candidate_state_ids": [
                    state["state_id"]
                    for state in initial_states
                ],
                "candidate_states": [
                    self._brief_state(state)
                    for state in initial_states
                ],
                "description": (
                    "These states appear to describe the initial traffic scene "
                    "before vehicles start moving. If they are located in the "
                    "same initial traffic context, they may occur When each other."
                ),
            }
        ]

    def _build_explicit_temporal_candidates(
        self,
        sentences: List[str],
        states: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Candidate groups triggered by explicit temporal expressions such as
        'as', 'when', 'while', or 'in response to'.
        """
        candidates = []

        for sentence in sentences:
            sentence_lower = sentence.lower()

            matched_cues = [
                cue
                for cue in self.TEMPORAL_CUES
                if cue in sentence_lower
            ]

            if not matched_cues:
                continue

            matched_states = self._find_states_supported_by_sentence(
                sentence=sentence,
                states=states,
            )

            if not self._has_multiple_vehicles(matched_states):
                continue

            candidates.append(
                {
                    "candidate_type": "explicit_temporal_expression",
                    "suggested_relation": "When",
                    "temporal_expressions": matched_cues,
                    "source_sentence": sentence,
                    "candidate_state_ids": [
                        state["state_id"]
                        for state in matched_states
                    ],
                    "candidate_states": [
                        self._brief_state(state)
                        for state in matched_states
                    ],
                    "description": (
                        "This sentence contains explicit temporal expression(s) "
                        "and references states of multiple vehicles."
                    ),
                }
            )

        return candidates

    def _build_parallel_movement_candidates(
        self,
        sentences: List[str],
        states: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        candidates = []

        for sentence in sentences:
            sentence_lower = sentence.lower()

            matched_cues = [
                cue
                for cue in self.PARALLEL_CUES
                if cue in sentence_lower
            ]

            if not matched_cues:
                continue

            sentence_states = self._find_states_supported_by_sentence(
                sentence=sentence,
                states=states,
            )

            mentioned_vehicle_states = self._find_states_of_mentioned_vehicles(
                sentence=sentence,
                states=states,
            )

            candidate_states = self._merge_states(
                sentence_states,
                mentioned_vehicle_states,
            )

            candidate_states = [
                state
                for state in candidate_states
                if state.get("behavior") != "Stationary"
            ]

            if not self._has_multiple_vehicles(candidate_states):
                continue

            candidates.append(
                {
                    "candidate_type": "parallel_movement_synchronization",
                    "suggested_relation": "When",
                    "temporal_expressions": matched_cues,
                    "source_sentence": sentence,
                    "candidate_state_ids": [
                        state["state_id"]
                        for state in candidate_states
                    ],
                    "candidate_states": [
                        self._brief_state(state)
                        for state in candidate_states
                    ],
                    "description": (
                        "The sentence describes vehicles moving or being positioned "
                        "parallel/alongside each other, which may indicate concurrent "
                        "movement states."
                    ),
                }
            )

        return candidates

    def _find_states_supported_by_sentence(
        self,
        sentence: str,
        states: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        matched_states = []
        sentence_lower = sentence.lower()

        for state in states:
            source_sentences = state.get("source_sentences", [])
            source_spans = state.get("source_spans", [])

            sentence_matched = any(
                self._normalize_text(src_sentence)
                == self._normalize_text(sentence)
                for src_sentence in source_sentences
            )

            span_matched = any(
                span.lower() in sentence_lower
                for span in source_spans
            )

            if sentence_matched or span_matched:
                matched_states.append(state)

        return matched_states

    def _find_states_of_mentioned_vehicles(
        self,
        sentence: str,
        states: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        sentence_lower = sentence.lower()
        matched_states = []

        vehicles_in_sentence = {
            state["vehicle_name"]
            for state in states
            if state["vehicle_name"].lower() in sentence_lower
        }

        for vehicle_name in vehicles_in_sentence:
            vehicle_states = [
                state
                for state in states
                if state["vehicle_name"] == vehicle_name
            ]

            movement_states = [
                state
                for state in vehicle_states
                if state.get("behavior") != "Stationary"
            ]

            if movement_states:
                matched_states.append(
                    sorted(
                        movement_states,
                        key=lambda item: (
                            item.get("behavior_order", 0),
                            item.get("state_index", 0),
                        ),
                    )[0]
                )

        return matched_states

    def _has_multiple_vehicles(
        self,
        states: List[Dict[str, Any]],
    ) -> bool:
        vehicle_names = {
            state.get("vehicle_name")
            for state in states
        }
        return len(vehicle_names) >= 2

    def _merge_states(
        self,
        states_a: List[Dict[str, Any]],
        states_b: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        state_by_id = {}

        for state in states_a + states_b:
            state_by_id[state["state_id"]] = state

        return list(state_by_id.values())

    def _deduplicate_candidates(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        unique_candidates = []
        seen = set()

        for candidate in candidates:
            candidate_key = (
                candidate.get("candidate_type"),
                tuple(sorted(candidate.get("candidate_state_ids", []))),
                candidate.get("source_sentence", ""),
            )

            if candidate_key in seen:
                continue

            seen.add(candidate_key)
            unique_candidates.append(candidate)

        return unique_candidates

    def _brief_state(
        self,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "state_id": state["state_id"],
            "vehicle_name": state["vehicle_name"],
            "behavior": state["behavior"],
            "behavior_order": state["behavior_order"],
            "state_index": state["state_index"],
            "position": state["position"],
            "speed": state["speed"],
        }

    def _split_sentences(
        self,
        text: str,
    ) -> List[str]:
        sentences = []

        current = []
        for char in text:
            current.append(char)

            if char in {".", "!", "?"}:
                sentence = "".join(current).strip()
                if sentence:
                    sentences.append(sentence)
                current = []

        rest = "".join(current).strip()
        if rest:
            sentences.append(rest)

        return sentences

    def _normalize_text(
        self,
        text: str,
    ) -> str:
        return " ".join(text.lower().split())

    def _build_prompt(
        self,
        accident_report: str,
        state_information: Dict[str, List[Dict[str, Any]]],
        temporal_candidates: List[Dict[str, Any]],
    ) -> str:
        state_json = json.dumps(
            state_information,
            ensure_ascii=False,
            indent=2,
        )

        candidate_json = json.dumps(
            temporal_candidates,
            ensure_ascii=False,
            indent=2,
        )

        return f"""
You are an expert in accident report analysis and accident information extraction. Your task is to analyze the following accident report and checked state information, and extract temporal relations between states belonging to different vehicles.

Accident Report Description:
{accident_report}

Checked State Information:
{state_json}

Potential Temporal Relation Candidates:
{candidate_json}

Question:
What temporal relations can be derived between states belonging to different vehicles?

Temporal Relation Definition:

When:
Two states occur simultaneously or their active periods overlap.

Typical indicators include:
- when
- while
- as
- during
- meanwhile
- simultaneously
- at the same time
- in response to the same event
- shared initial traffic scene
- parallel or alongside movement

Initial Scene Synchronization:
If multiple vehicles are described as being located in the same initial traffic scene, such as waiting at the same red light, stopped at the same intersection, or positioned relative to each other before movement begins, their initial states may be treated as occurring "When" each other, even if no explicit word such as "when" or "while" appears.

State Reference Rules:
(1) Each temporal relation must reference existing states from Checked State Information.
(2) Use source_state_id and target_state_id to identify relation endpoints.
(3) Do not create new states, new vehicles, or new behaviors.
(4) Do not use a behavior or state that is not listed in Checked State Information.
(5) The source_state_id and target_state_id must be copied exactly from Checked State Information.
(6) The source and target states must belong to different vehicles.
(7) The source_vehicle, source_behavior, source_behavior_order, and source_state_index must match the source_state_id.
(8) The target_vehicle, target_behavior, target_behavior_order, and target_state_index must match the target_state_id.

Candidate Usage Rules:
(1) Potential Temporal Relation Candidates are not final answers.
(2) Use the candidates to check potentially missing temporal relations.
(3) Output a candidate relation only if it is supported by the accident report and both endpoints are existing states.
(4) For an initial_scene_synchronization candidate, output a compact set of When relations using one anchor state rather than all possible pairwise combinations.
(5) For example, if three initial states A, B, and C occur in the same initial scene, output A When B and A When C, rather than A When B, A When C, and B When C.
(6) For a parallel_movement_synchronization candidate, output When only if the states represent movement or continuing movement of different vehicles.

Extraction Rules:
(1) Only extract temporal relations between states belonging to different vehicles.
(2) Do not extract temporal relations between states belonging to the same vehicle.
(3) Only extract temporal relations that are explicitly stated or clearly implied by the accident report.
(4) A temporal relation must be supported by a temporal expression, temporal phrase, shared initial traffic scene, or clearly described temporal dependency.
(5) Do not infer temporal relations from common traffic knowledge or assumptions.
(6) Do not create temporal relations solely based on sentence order.
(7) If no explicit or clearly implied temporal relation exists between two states, do not create a relation.
(8) Match the temporal relation to the most specific states supported by the report.
(9) Do not create duplicate temporal relations.
(10) For each temporal relation, provide the report text span(s) and sentence(s) used to derive the relation.
(11) Use "When" when the report indicates simultaneous, overlapping, concurrent, initially synchronized, or parallel state occurrences.
(12) If multiple states could match the same evidence, select the state that is most directly referenced by the temporal expression or candidate.
(13) Returning fewer relations is preferred over introducing unsupported relations.
(14) Do not create Collision as a behavior, state, or temporal relation endpoint.
(15)  If a sentence describes a collision using phrases such as "made contact", "hit", "struck", or "collided", do not extract the collision event in this task.
(16) However, temporal relations between non-collision states in the same sentence may still be extracted if explicitly supported.
(17) Do not extract same-vehicle temporal order, because same-vehicle order is already represented by behavior_order and state_index.
(18) The output should focus on cross-vehicle temporal dependencies needed for scenario synchronization.
(19) Initial states of different vehicles may be linked by "When" if they describe the same initial traffic scene.
(20) A shared initial scene may be indicated by phrases such as "at a red light", "at the intersection", "in front of", "to the right of", or other relative-position descriptions before vehicles start moving.
(21) Do not create temporal relations involving states that only describe collision occurrence, contact, impact, or damage.
(22) A temporal relation should represent synchronization or ordering between vehicle motion states, not the collision event itself.

Output Constraints:
(1) The output must be valid JSON format.
(2) Do not include explanations or additional text.
(3) Use only relation values from: "When".
(4) Use the following format:

{{
  "temporal_relations": [
    {{
      "source_state_id": "StateID",
      "target_state_id": "StateID",

      "source_vehicle": "Vehicle Name",
      "source_behavior": "Behavior Name",
      "source_behavior_order": 0,
      "source_state_index": 0,

      "target_vehicle": "Vehicle Name",
      "target_behavior": "Behavior Name",
      "target_behavior_order": 0,
      "target_state_index": 0,

      "relation": "When",

      "source_spans": [
        "Supporting Span"
      ],

      "source_sentences": [
        "Supporting Sentence"
      ]
    }}
  ]
}}

Example Output:
{{
  "temporal_relations": [
    {{
      "source_state_id": "CruiseAV_B0_S0",
      "target_state_id": "AcuraSedan_B0_S0",

      "source_vehicle": "Cruise AV",
      "source_behavior": "Stationary",
      "source_behavior_order": 0,
      "source_state_index": 0,

      "target_vehicle": "Acura Sedan",
      "target_behavior": "Stationary",
      "target_behavior_order": 0,
      "target_state_index": 0,

      "relation": "When",

      "source_spans": [
        "at a complete stop",
        "was stopped in front of the Cruise AV"
      ],

      "source_sentences": [
        "A Cruise autonomous vehicle (\\\"Cruise AV\\\"), operating in driverless autonomous mode, was at a complete stop in response to a red light on southbound Masonic Avenue at the intersection with Oak Street.",
        "At the intersection, an Acura Sedan was stopped in front of the Cruise AV in the left southbound lane."
      ]
    }},
    {{
      "source_state_id": "CruiseAV_B1_S0",
      "target_state_id": "BMWSedan_B2_S0",

      "source_vehicle": "Cruise AV",
      "source_behavior": "Proceed Straight",
      "source_behavior_order": 1,
      "source_state_index": 0,

      "target_vehicle": "BMW Sedan",
      "target_behavior": "Turn Left",
      "target_behavior_order": 2,
      "target_state_index": 0,

      "relation": "When",

      "source_spans": [
        "As the Cruise AV entered the intersection",
        "the BMW Sedan made a prohibited left turn"
      ],

      "source_sentences": [
        "As the Cruise AV entered the intersection, the BMW Sedan made a prohibited left turn from the right southbound lane on Masonic Avenue with no turn signal."
      ]
    }}
  ]
}}

Answer:
""".strip()

    def _validate_temporal_result(
        self,
        raw_result: Dict[str, Any],
        state_information: Dict[str, List[Dict[str, Any]]],
    ):
        if "temporal_relations" not in raw_result:
            raise ValueError(
                "TemporalAnalyzer output missing key: temporal_relations"
            )

        temporal_relations = raw_result["temporal_relations"]

        if not isinstance(temporal_relations, list):
            raise ValueError(
                "TemporalAnalyzer field 'temporal_relations' must be a list."
            )

        state_by_id = {
            state["state_id"]: state
            for state in state_information["states"]
        }

        valid_state_ids: Set[str] = set(state_by_id.keys())
        seen_relations = set()

        valid_temporal_relations = []

        for idx, relation_item in enumerate(temporal_relations):

            is_valid = self._validate_single_relation(
                relation_item=relation_item,
                relation_index=idx,
                valid_state_ids=valid_state_ids,
                state_by_id=state_by_id,
                seen_relations=seen_relations,
            )

            if is_valid:
                valid_temporal_relations.append(relation_item)

        raw_result["temporal_relations"] = valid_temporal_relations

    def _validate_single_relation(
        self,
        relation_item: Dict[str, Any],
        relation_index: int,
        valid_state_ids: Set[str],
        state_by_id: Dict[str, Dict[str, Any]],
        seen_relations: Set,
    ):
        required_keys = {
            "source_state_id",
            "target_state_id",
            "source_vehicle",
            "source_behavior",
            "source_behavior_order",
            "source_state_index",
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
            print(
                f"Temporal relation at index {relation_index} "
                f"missing keys: {missing_keys}"
            )
            return False

        source_state_id = relation_item["source_state_id"]
        target_state_id = relation_item["target_state_id"]
        relation = relation_item["relation"]

        if source_state_id not in valid_state_ids:
            print(
                f"Invalid source_state_id '{source_state_id}' "
                f"at temporal relation index {relation_index}."
            )
            return False

        if target_state_id not in valid_state_ids:
            print(
                f"Invalid target_state_id '{target_state_id}' "
                f"at temporal relation index {relation_index}."
            )
            return False

        if source_state_id == target_state_id:
            print(
                f"Temporal relation at index {relation_index} "
                f"has identical source and target state."
            )
            return False

        if relation not in self.VALID_RELATIONS:
            print(
                f"Invalid temporal relation '{relation}' "
                f"at index {relation_index}."
            )
            return False

        source_state = state_by_id[source_state_id]
        target_state = state_by_id[target_state_id]

        if source_state["vehicle_name"] == target_state["vehicle_name"]:
            print(
                f"Temporal relation at index {relation_index} "
                f"connects states of the same vehicle."
            )
            return False

        self._check_endpoint_consistency(
            relation_item=relation_item,
            relation_index=relation_index,
            prefix="source",
            state=source_state,
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
            print(
                f"source_spans must be a list at temporal relation "
                f"index {relation_index}."
            )
            return False

        if not isinstance(source_sentences, list):
            print(
                f"source_sentences must be a list at temporal relation "
                f"index {relation_index}."
            )
            return False

        relation_key = (
            source_state_id,
            target_state_id,
            relation,
        )

        reverse_relation_key = (
            target_state_id,
            source_state_id,
            relation,
        )

        if relation == "When":
            if (
                relation_key in seen_relations
                or reverse_relation_key in seen_relations
            ):
                raise ValueError(
                    f"Duplicate temporal relation found at index "
                    f"{relation_index}."
                )
        else:
            if relation_key in seen_relations:
                raise ValueError(
                    f"Duplicate temporal relation found at index "
                    f"{relation_index}."
                )

        seen_relations.add(relation_key)
        return True

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
                f"{vehicle_key} mismatch at temporal relation "
                f"index {relation_index}. Expected '{state['vehicle_name']}', "
                f"got '{relation_item[vehicle_key]}'."
            )

        if relation_item[behavior_key] != state["behavior"]:
            raise ValueError(
                f"{behavior_key} mismatch at temporal relation "
                f"index {relation_index}. Expected '{state['behavior']}', "
                f"got '{relation_item[behavior_key]}'."
            )

        if relation_item[behavior_order_key] != state["behavior_order"]:
            raise ValueError(
                f"{behavior_order_key} mismatch at temporal relation "
                f"index {relation_index}. Expected "
                f"'{state['behavior_order']}', "
                f"got '{relation_item[behavior_order_key]}'."
            )

        if relation_item[state_index_key] != state["state_index"]:
            raise ValueError(
                f"{state_index_key} mismatch at temporal relation "
                f"index {relation_index}. Expected "
                f"'{state['state_index']}', "
                f"got '{relation_item[state_index_key]}'."
            )