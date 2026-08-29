import json
from pathlib import Path
from typing import Any, Union
import pandas as pd


def read_json_file(file_path: Union[str, Path]) -> Any:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Format error in JSON file: {e}", e.doc, e.pos)


def write_json_file(file_path: Union[str, Path], data: Any,
                    ensure_ascii: bool = False, indent: int = 4) -> None:
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=ensure_ascii, indent=indent)
    except IOError as e:
        raise IOError(f"Error writing to file: {e}")


def load_excel(file_path):
    return pd.read_excel(file_path, sheet_name=None)


