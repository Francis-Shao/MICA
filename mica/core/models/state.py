from dataclasses import dataclass
from typing import List

from mica.core.models.trace_context import TraceContext


@dataclass
class StateNode:
    position: str
    speed: str
    context: TraceContext


@dataclass
class BehaviorStateNode:
    vehicle_name: str
    behavior: str
    behavior_order: int
    states: List[StateNode]