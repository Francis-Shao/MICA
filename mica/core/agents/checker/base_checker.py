from abc import abstractmethod
from typing import Any

from mica.core.agents.base_agent import BaseAgent


class BaseChecker(BaseAgent):

    def __init__(self, name: str, context_pool=None):
        super().__init__(name=name, context_pool=context_pool)

    def _require_context_pool(self):
        if self.context_pool is None:
            raise ValueError(f"{self.name} requires a ContextPool.")

    def _get_report(self) -> str:
        self._require_context_pool()

        report = self.context_pool.get_report()
        if not report:
            raise ValueError(
                f"No accident report found in ContextPool for {self.name}."
            )

        return report

    def _query_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> Any:
        messages = self.build_messages(
            user_prompt=prompt,
            system_prompt=system_prompt,
        )
        response_text = self.query_llm(messages)
        return self.parse_json_response(response_text)

    def _store_check_result(self, key: str, value: Any):
        self._require_context_pool()
        self.context_pool.set(key, value)

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        pass