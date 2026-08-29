from typing import Any, Dict, List
from dataclasses import asdict, is_dataclass

from mica.core.models.behavior import BehaviorNode
from mica.core.models.state import BehaviorStateNode


class ContextPool:

    def __init__(self):

        self.report_text = None
        self.vehicle_directions: Dict[str, str] = {}

        # Behavior-level information
        self.behavior_timelines: Dict[
            str,
            List[BehaviorNode]
        ] = {}

        # State-level information
        self.state_timelines: Dict[
            str,
            List[BehaviorStateNode]
        ] = {}

        self.temporal_relations = []
        self.time_slices = []
        self.spatial_relations = []
        self.collision_events = []

        self.scenario_model = None
        self.identified_events = []

        # Temporary or intermediate data
        self.extra = {}

    def set_report(self, report_text: str):
        self.report_text = report_text


    def get_report(self):
        return self.report_text

    def set_vehicle_directions(
            self,
            directions: Dict[str, str]
    ):
        self.vehicle_directions = directions


    def get_vehicle_directions(self):
        return self.vehicle_directions


    def set_behavior_timelines(
            self,
            timelines: Dict[str, Any]
    ):
        self.behavior_timelines = timelines


    def get_behavior_timelines(self):
        return self.behavior_timelines


    def set_state_timelines(
            self,
            timelines: Dict[str, Any]
    ):
        self.state_timelines = timelines


    def get_state_timelines(self):
        return self.state_timelines


    def set_temporal_relations(
            self,
            relations
    ):
        self.temporal_relations = relations


    def get_temporal_relations(self):
        return self.temporal_relations


    def set_time_slices(
            self,
            time_slices
    ):
        self.time_slices = time_slices


    def get_time_slices(self):
        return self.time_slices

    def set_spatial_relations(
            self,
            relations
    ):
        self.spatial_relations = relations


    def get_spatial_relations(self):
        return self.spatial_relations


    def set_scenario_model(
            self,
            model
    ):
        self.scenario_model = model


    def get_scenario_model(self):
        return self.scenario_model


    def set_identified_events(
            self,
            events
    ):
        self.identified_events = events


    def get_identified_events(self):
        return self.identified_events

    def set(
            self,
            key: str,
            value: Any
    ):
        self.extra[key] = value


    def get(
            self,
            key: str,
            default=None
    ):
        return self.extra.get(key, default)


    def to_dict(self):

        return {
            "report_text": self.report_text,

            "vehicle_directions": self._serialize(
                self.vehicle_directions
            ),

            "behavior_timelines": self._serialize(
                self.behavior_timelines
            ),

            "state_timelines": self._serialize(
                self.state_timelines
            ),

            "temporal_relations": self._serialize(
                self.temporal_relations
            ),

            "time_slices": self._serialize(
                self.time_slices
            ),

            "spatial_relations": self._serialize(
                self.spatial_relations
            ),

            "collision_events": self._serialize(
                self.collision_events
            ),

            "scenario_model": self._serialize(
                self.scenario_model
            ),

            "identified_events": self._serialize(
                self.identified_events
            ),

            "extra": self._serialize(
                self.extra
            ),
        }


    def _serialize(
            self,
            obj: Any
    ):

        if is_dataclass(obj):
            return asdict(obj)

        if isinstance(obj, dict):
            return {
                key: self._serialize(value)
                for key, value in obj.items()
            }

        if isinstance(obj, list):
            return [
                self._serialize(item)
                for item in obj
            ]

        return obj