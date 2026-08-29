from pathlib import Path

from mica.core.context.context_pool import ContextPool

from mica.core.agents.analyzer.behavior_extractor import BehaviorExtractor
from mica.core.agents.checker.behavior_checker import BehaviorChecker

from mica.core.agents.analyzer.state_extractor import StateExtractor
from mica.core.agents.checker.state_checker import StateChecker

from mica.core.agents.analyzer.temporal_analyzer import TemporalAnalyzer
from mica.core.time_slicer import TimeSlicer

from mica.core.agents.analyzer.spatial_analyzer import SpatialAnalyzer

from mica.core.model_generator import ModelGenerator

from mica.core.event_identifier import PatternIdentifier

from mica.utils.file_util import write_json_file



class MICA:
    def __init__(self, output_dir):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # =====================================================
    # Save
    # =====================================================
    def save(self, *parts, data):

        file_path = self.output_dir.joinpath(
            *parts
        )

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        write_json_file(
            file_path,
            data
        )

    # =====================================================
    # Initialize
    # =====================================================
    def initialize(self):

        context_pool = ContextPool()

        behavior_agent = BehaviorExtractor(
            context_pool=context_pool
        )

        behavior_checker = BehaviorChecker(
            context_pool=context_pool
        )

        state_agent = StateExtractor(
            context_pool=context_pool
        )

        state_checker = StateChecker(
            context_pool=context_pool
        )

        temporal_agent = TemporalAnalyzer(
            context_pool=context_pool
        )

        time_slicer = TimeSlicer(
            context_pool=context_pool
        )

        spatial_agent = SpatialAnalyzer(
            context_pool=context_pool
        )


        model_generator = ModelGenerator(
            context_pool=context_pool
        )

        return {
            "context_pool": context_pool,

            "behavior_agent": behavior_agent,
            "behavior_checker": behavior_checker,

            "state_agent": state_agent,
            "state_checker": state_checker,

            "temporal_agent": temporal_agent,

            "time_slicer": time_slicer,

            "spatial_agent": spatial_agent,

            "model_generator": model_generator
        }

    # =====================================================
    # Pattern Identification
    # =====================================================
    def analyze_pattern(
            self,
            scenario_model
    ):
        identifier = PatternIdentifier()

        pattern_result = identifier.identify_from_model(
            scenario_model
        )

        self.save(
            "pattern",
            "pattern_result.json",
            data=pattern_result
        )

        return pattern_result

    # =====================================================
    # Full MICA
    # =====================================================

    def run(
            self,
            report,
            scenario_id
    ):
        agents = self.initialize()
        context_pool = agents["context_pool"]

        # =================================================
        # Step 1 Behavior Extraction
        # =================================================

        behavior_timelines = agents[
            "behavior_agent"
        ].run(
            report
        )

        self.save(
            "behavior",
            "raw",
            "result.json",
            data=BehaviorExtractor.to_dict(
                behavior_timelines
            )
        )

        self.save(
            "behavior",
            "raw",
            "context.json",
            data=context_pool.to_dict()
        )

        # =================================================
        # Step 2 Behavior Checking
        # =================================================
        checked_behavior_timelines = agents[
            "behavior_checker"
        ].run(
            write_back=True
        )

        self.save(
            "behavior",
            "check",
            "hints.json",
            data=context_pool.get(
                "behavior_coverage_hints"
            )
        )

        self.save(
            "behavior",
            "check",
            "check_result.json",
            data=context_pool.get(
                "raw_behavior_check_result"
            )
        )

        self.save(
            "behavior",
            "checked",
            "result.json",
            data=BehaviorExtractor.to_dict(
                checked_behavior_timelines
            )
        )

        self.save(
            "behavior",
            "checked",
            "context.json",
            data=context_pool.to_dict()
        )

        # =================================================
        # Step 3 State Extraction
        # =================================================
        state_timelines = agents[
            "state_agent"
        ].run()

        self.save(
            "state",
            "raw",
            "result.json",
            data=StateExtractor.to_prompt_format(
                state_timelines
            )
        )

        self.save(
            "state",
            "raw",
            "flat_result.json",
            data=StateExtractor.to_flat_prompt_format(
                state_timelines
            )
        )

        # =================================================
        # Step 4 State Checking
        # =================================================
        checked_state_timelines = agents[
            "state_checker"
        ].run(
            write_back=True
        )

        self.save(
            "state",
            "check",
            "hints.json",
            data=context_pool.get(
                "state_check_hints"
            )
        )

        self.save(
            "state",
            "check",
            "check_result.json",
            data=context_pool.get(
                "raw_state_check_result"
            )
        )

        self.save(
            "state",
            "checked",
            "result.json",
            data=StateExtractor.to_prompt_format(
                checked_state_timelines
            )
        )

        # =================================================
        # Step 5 Temporal
        # =================================================
        temporal_result = agents[
            "temporal_agent"
        ].run()

        self.save(
            "temporal",
            "raw",
            "result.json",
            data=temporal_result
        )

        self.save(
            "temporal",
            "raw",
            "state_information.json",
            data=context_pool.get(
                "temporal_state_information"
            )
        )

        self.save(
            "temporal",
            "raw",
            "candidates.json",
            data=context_pool.get(
                "temporal_candidates"
            )
        )

        # =================================================
        # Step 6 Time Slicing
        # =================================================
        slicing_result = agents[
            "time_slicer"
        ].run()


        self.save(
            "slicing",
            "result.json",
            data=slicing_result
        )

        self.save(
            "slicing",
            "context.json",
            data=context_pool.to_dict()
        )

        # =================================================
        # Step 7 Spatial Analysis
        # =================================================
        spatial_result = agents[
            "spatial_agent"
        ].run()

        self.save(
            "spatial",
            "raw",
            "slice_information.json",
            data=context_pool.get(
                "spatial_slice_information"
            )
        )

        self.save(
            "spatial",
            "raw",
            "result.json",
            data=spatial_result
        )

        self.save(
            "spatial",
            "raw",
            "context.json",
            data=context_pool.to_dict()
        )

        # =================================================
        # Step 9 Model Generation
        # =================================================
        scenario_model = agents[
            "model_generator"
        ].run(scenario_id)

        self.save(
            "model",
            "scenario_model.json",
            data=scenario_model
        )

        self.save(
            "model",
            "context.json",
            data=context_pool.to_dict()
        )

        # =================================================
        # Step 10 Pattern Identification
        # =================================================
        pattern_result = self.analyze_pattern(
            scenario_model
        )

        # =================================================
        # Final Context
        # =================================================
        self.save(
            "final",
            "context.json",
            data=context_pool.to_dict()
        )

        return pattern_result