# MICA-demo

This repository contains the implementation of our paper *"Model-Based Critical Pre-crash Interaction Identification via Multi-Agent Collaboration"*. It provides a demo of the proposed approach for identifying critical pre-crash interactions from textual accident reports.

### Requirements

- Python version: 3.11 or higher.
- Operation system: Windows 11
- Basic dependencies: see environment.yaml

### Usage

To run the demo, follow the steps below:

- Clone this project and create a virtual environment:

  ```bash
  conda env create -f environment.yaml
  conda activate mica
  ```

- Configure the LLM settings in `configs/configs.yaml` by providing your own credentials (`base_url`, `api_key`, `llm_model`).

- Run the following script:

  `mica/run_mica.py`

  The generated intermediate outputs will be saved in the `outputs/` directory, while the final results will be saved in the `results/` directory.

### Project Structure

```
MICA/
├── configs
├── data
├── mica
│   ├── core
│   │   ├── agents
│   │   │   ├── analyzer
│   │   │   │   ├── __init__.py
│   │   │   │   ├── behavior_extractor.py
│   │   │   │   ├── spatial_analyzer.py
│   │   │   │   ├── state_extractor.py
│   │   │   │   └── temporal_analyzer.py
│   │   │   ├── checker
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base_checker.py
│   │   │   │   ├── behavior_checker.py
│   │   │   │   └── state_checker.py
│   │   │   ├── __init__.py
│   │   │   └── base_agent.py
│   │   ├── context
│   │   │   ├── __init__.py
│   │   │   └── context_pool.py
│   │   ├── models
│   │   │   ├── __init__.py
│   │   │   ├── behavior.py
│   │   │   ├── pattern.py
│   │   │   ├── state.py
│   │   │   └── trace_context.py
│   │   ├── __init__.py
│   │   ├── event_identifier.py
│   │   ├── model_generator.py
│   │   └── time_slicer.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── file_util.py
│   │   └── llm_util.py
│   ├── __init__.py
│   ├── mica_runner.py
│   └── run_mica.py
├── output
│   └── dmv_6
│       ├── behavior
│       ├── final
│       ├── model
│       ├── pattern
│       ├── slicing
│       ├── spatial
│       ├── state
│       └── temporal
├── prompts
│   ├── baseline_encoding_mapping
│   ├── baseline_single_llm
│   └── mica
├── results
```