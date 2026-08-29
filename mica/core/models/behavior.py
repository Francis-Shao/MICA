from dataclasses import dataclass

from mica.core.models.trace_context import TraceContext


@dataclass
class BehaviorNode:
    vehicle_name: str
    behavior: str
    order: int
    context: TraceContext