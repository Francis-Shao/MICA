from pathlib import Path

from mica.mica_runner import MICA
from mica.utils.file_util import write_json_file, load_excel

mica_root = Path(__file__).parent
project_root = mica_root.parent
input_data_folder = project_root / "data"

mica_output_data_folder = project_root / "output"
mica_results_folder = project_root / "results"

input_data_file_name = {
    "nhtsa": "mica_nhtsa.xlsx",
    "dmv": "mica_dmv.xlsx"
}


def run_mica(report_id, report_description, annotated_pattern):
    print(f"Running: MICA, Report ID: {report_id}")
    mica = MICA(
        output_dir=mica_output_data_folder / f"{report_id}"
    )
    mica_result = mica.run(
        report_description,
        report_id
    )
    output = {}
    if mica_result.get("key_event") is None:
        output = {
            "report_id": report_id,
            "report_description": report_description,
            "subject_vehicle": None,
            "target_vehicle": None,
            "pattern": "Unmatched"
        }
    elif mica_result.get("key_event") is not None:
        output = {
            "report_id": report_id,
            "report_description": report_description,
            "subject_vehicle": mica_result['key_event']['subject_vehicle'],
            "target_vehicle": mica_result['key_event']['target_vehicle'],
            "pattern": mica_result['key_event']['pattern_name'],
            "annotated_pattern": annotated_pattern
        }

    write_json_file(
        mica_results_folder / f"{report_id}.json",
        output
    )


def run_trace_on_selected_case(dataset, case_no):
    input_file_path = input_data_folder / input_data_file_name[dataset]
    input_file = load_excel(input_file_path)
    case_df = input_file["Sheet1"]
    for index, row in case_df.iterrows():
        if index != case_no:
            continue
        report_description = row["Case Summary"]
        annotated_pattern = row["Pre-crash Pattern"]
        report_id = f"{dataset}_{str(index)}"

        run_mica(report_id, report_description, annotated_pattern)


if __name__ == "__main__":
    # target_dataset = "nhtsa"
    target_dataset = "dmv"

    case_no = 6

    run_trace_on_selected_case(target_dataset, case_no)

