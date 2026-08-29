from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PatternMatch:
    pattern_id: str
    pattern_name: str
    slice_id: str
    subject_vehicle: str
    target_vehicle: str
    confidence: float = 1.0
    matched_conditions: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_name": self.pattern_name,
            "slice_id": self.slice_id,
            "subject_vehicle": self.subject_vehicle,
            "target_vehicle": self.target_vehicle,
            "confidence": self.confidence,
            "matched_conditions": self.matched_conditions,
            "evidence": self.evidence,
        }