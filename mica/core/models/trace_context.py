from dataclasses import dataclass
from typing import List


@dataclass
class TraceContext:
    source_spans: List[str]
    source_sentences: List[str]