from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pathlib import Path
from dataclasses import asdict, is_dataclass
import json

from mica.utils.llm_util import request_response


class BaseAgent(ABC):
    def __init__(self, name: str, context_pool=None):
        self.name = name
        self.context_pool = context_pool

    def query_llm(self, messages: List[Dict[str, str]]) -> str:
        response = request_response(messages)
        return response.choices[0].message.content

    def build_messages(
        self,
        user_prompt: str,
        system_prompt: str | None = None
    ) -> List[Dict[str, str]]:
        if system_prompt is None:
            system_prompt = (
                f"You are {self.name}, an expert agent for accident report analysis."
            )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def parse_json_response(self, response_text: str) -> Any:
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"[{self.name}] Invalid JSON response:\n{response_text}"
            ) from e

    def export_json(self, data: Any, filename: str):
        output_dir = self._get_output_dir()
        output_path = output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                self._serialize(data),
                f,
                ensure_ascii=False,
                indent=2,
            )

        return output_path

    def export_agent_result(self, data: Any, filename: str | None = None):
        if filename is None:
            filename = f"{self._safe_name()}_result.json"

        return self.export_json(data, filename)

    def export_context_pool(self, filename: str | None = None):
        if self.context_pool is None:
            return None

        if filename is None:
            filename = f"{self._safe_name()}_context_pool.json"

        return self.export_json(self.context_pool.to_dict(), filename)

    def _get_output_dir(self) -> Path:
        current_file = Path(__file__).resolve()

        # trace/core/agents/base_agent.py
        # parents[0] = agents
        # parents[1] = core
        # parents[2] = trace
        # parents[3] = project root
        project_root = current_file.parents[3]

        output_dir = project_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir

    def _safe_name(self) -> str:
        return self.name.lower().replace(" ", "_")

    def _serialize(self, obj: Any) -> Any:
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

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        pass