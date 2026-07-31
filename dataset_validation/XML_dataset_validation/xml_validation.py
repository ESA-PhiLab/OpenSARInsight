import os
import xml.etree.ElementTree as ET
import importlib.util
import re
import csv
from datetime import datetime

# Helper function for natural sorting (e.g., "file2.xml" comes before "file10.xml")
def natural_key(string_):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', string_)]

# Configuration dictionary for each dataset type (RFI, Vessel, Flood)
# Includes paths to test cases, XML directories, and output log directories
script_dir = os.path.abspath(os.path.dirname(__file__))

CONFIG = {
    "RFI": {
        "test_cases_dir": os.path.join(script_dir, "test_cases", "RFI"),
        "log_dir": os.path.join(script_dir, "validation_restults", "RFI"),
        "xml_base": "/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/rfi/rfi_aresys_dataset_FR_v1",
        "xml_paths": {
            "test": "test/labels/xml_labelling",
            "train": "train/labels/xml_labelling",
            "val": "val/labels/xml_labelling",
        }
    },
    "Vessel": {
        "test_cases_dir": os.path.join(script_dir, "test_cases", "Vessel"),
        "log_dir": os.path.join(script_dir, "validation_restults", "Vessel"),
        "xml_base": "/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/vessels/vessel_viewsar3_dataset_FR_v1",
        "xml_paths": {
            "test": "test/labels",
            "train": "train/labels",
            "val": "val/labels",
        }
    },
    "Flood": {
        "test_cases_dir": os.path.join(script_dir, "test_cases", "Flood"),
        "log_dir": os.path.join(script_dir, "validation_restults", "Flood"),
        "xml_base": "/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/flood/flood_kurosiwo_dataset_FR_v1",
        "xml_paths": {
            "test": "test/labels/xml_labelling",
            "train": "train/labels/xml_labelling",
            "val": "val/labels/xml_labelling",
        }
    }
}

# Dynamically load Python test case modules from the test_cases directory
def load_test_cases(test_cases_dir):
    test_cases = []
    for filename in os.listdir(test_cases_dir):
        if filename.endswith(".py") and filename.startswith("test_case_"):
            module_name = filename[:-3]
            file_path = os.path.join(test_cases_dir, filename)
            print(file_path)
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            test_cases.append(module)
    return test_cases

# Core validation function for a dataset split (test/train/val)
def validate(xml_dir, dataset_label, test_cases_dir, log_dir, dataset_name):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{dataset_name}_validation_results_{dataset_label}.log")
    csv_file = os.path.join(log_dir, f"{dataset_name}_validation_results_{dataset_label}.csv")

    # Get and sort XML files naturally
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
    xml_files.sort(key=natural_key)
    total_files = len(xml_files)

    test_cases = load_test_cases(test_cases_dir)

    final_status = 'PASS'
    passed_count = 0
    failed_count = 0
    warnings_count = 0

    # Open log and CSV file for writing results
    with open(log_file, 'w', encoding='utf-8') as log, open(csv_file, 'w', newline='', encoding='utf-8') as csv_out:
        log.write(f"🕒 Validation started at: {datetime.now()}\n")
        log.write(f"📁 Dataset: {dataset_label}\n\n")

        csv_writer = csv.writer(csv_out)
        csv_writer.writerow(["Test Case", "XML File", "Status", "Missing Elements", "Warning Details"])

        # Loop through each test case
        for tc_index, test_case in enumerate(test_cases, start=1):
            tc_name = f"TC{tc_index}"

            # Print description or fallback info
            if hasattr(test_case, 'DESCRIPTION'):
                log.write(test_case.DESCRIPTION + "\n")
            else:
                log.write(f" - Required: {test_case.REQUIRED}\n")
                if hasattr(test_case, 'RANGES'):
                    log.write(f" - Ranges:\n")
                    for k, v in test_case.RANGES.items():
                        log.write(f"    {k}: {v}\n")

            ok = 0
            ko = 0

            # Run validation for each XML file
            for xml_file in xml_files:
                xml_path = os.path.join(xml_dir, xml_file)
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                except Exception as e:
                    # Parsing error
                    ko += 1
                    msg = f"[{tc_name}] ❌ {xml_file} - Parsing error: {e}"
                    log.write(msg + "\n")
                    csv_writer.writerow([tc_name, xml_file, "FAIL", f"Parsing error: {e}", ""])
                    continue

                # Run test logic from test case
                result = test_case.run_test(root)
                result.setdefault('details', {})['file'] = xml_file

                status = result.get('status', 'FAIL')
                missing = result['details'].get('missing', [])
                warnings_list = result['details'].get('warnings', [])

                if isinstance(missing, str):
                    missing = [missing]
                if isinstance(warnings_list, str):
                    warnings_list = [warnings_list]

                # Log results based on status
                if status == 'PASS':
                    ok += 1
                    csv_writer.writerow([tc_name, xml_file, "PASS", "", ""])
                elif status == 'WARNING':
                    warnings_count += 1
                    ok += 1
                    log.write(f"[{tc_name}] ⚠️ {xml_file} warnings: {warnings_list}\n")
                    csv_writer.writerow([tc_name, xml_file, "WARNING", "", ", ".join(warnings_list)])
                else:
                    ko += 1
                    log.write(f"[{tc_name}] ❌ {xml_file} failed: {missing}\n")
                    csv_writer.writerow([tc_name, xml_file, "FAIL", ", ".join(missing), ""])

            # Log summary per test case
            log.write(f"[{tc_name} Results] ✅ OK: {ok} | ❌ KO: {ko} | ⚠️ Warnings: {warnings_count} | Total: {total_files}\n")
            if ko > 0:
                log.write(f" ❌[{tc_name}] failed.\n\n")
                final_status = 'FAIL'
                failed_count += 1
            else:
                log.write(f" ✅[{tc_name}] passed successfully.\n\n")
                passed_count += 1

        # Final summary of all test cases
        log.write("🔎 Final Summary:\n")
        log.write(f" ✅ Passed Test Cases: {passed_count}\n")
        log.write(f" ❌ Failed Test Cases: {failed_count}\n")
        log.write(f" ⚠️ Warnings: {warnings_count}\n")
        log.write(f" 📊 Total Test Cases: {len(test_cases)}\n\n")

        # Final status print
        if final_status == 'FAIL':
            print(f"❌ Validation FAILED for {dataset_label} in {dataset_name}. Check log and CSV: {log_file}, {csv_file}")
            log.write(f"❌ {dataset_name} Validation FAILED.\n")
        else:
            print(f"✅ Validation SUCCESSFUL for {dataset_label} in {dataset_name}.")
            log.write(f"✅ {dataset_name} Validation SUCCESSFUL.\n")

# Submenu runner for one dataset with interactive options
def run_submenu(dataset_key):
    cfg = CONFIG[dataset_key]
    base_path = cfg["xml_base"]
    paths = {
        "1": ("test", os.path.join(base_path, cfg["xml_paths"]["test"])),
        "2": ("train", os.path.join(base_path, cfg["xml_paths"]["train"])),
        "3": ("val", os.path.join(base_path, cfg["xml_paths"]["val"])),
    }

    while True:
        print(f"\n{dataset_key} XML dataset validation.\nSelect an option from the list:")
        print(f"1) {dataset_key} XML validation on \"{paths['1'][1]}\" directory")
        print(f"2) {dataset_key} XML validation on \"{paths['2'][1]}\" directory")
        print(f"3) {dataset_key} XML validation on \"{paths['3'][1]}\" directory")
        print(f"4) {dataset_key} XML validation on all directories")
        print(f"5) Exit")

        choice = input("Enter your choice (1-5): ").strip()

        if choice in ["1", "2", "3"]:
            label, path = paths[choice]
            print(f"\n▶️ Run {dataset_key} XML validation on \"{path}\" directory")
            validate(path, label, cfg["test_cases_dir"], cfg["log_dir"], dataset_key)
            break
        elif choice == "4":
            for label, path in paths.values():
                print(f"\n▶️ Run {dataset_key} XML validation on \"{path}\" directory")
                validate(path, label, cfg["test_cases_dir"], cfg["log_dir"], dataset_key)
            break
        elif choice == "5":
            print(f"👋 Exiting {dataset_key} validation submenu.")
            return
        else:
            print("⚠️  Wrong input. Select again.")

# Run validation for all splits without user prompt (used by "Run All")
def run_submenu_all(dataset_key):
    cfg = CONFIG[dataset_key]
    base_path = cfg["xml_base"]
    paths = {
        "test": os.path.join(base_path, cfg["xml_paths"]["test"]),
        "train": os.path.join(base_path, cfg["xml_paths"]["train"]),
        "val": os.path.join(base_path, cfg["xml_paths"]["val"]),
    }
    for label, path in paths.items():
        print(f"\n▶️ Run {dataset_key} XML validation on \"{path}\" directory")
        validate(path, label, cfg["test_cases_dir"], cfg["log_dir"], dataset_key)

# Main menu interface for user to select dataset or run all
def main_menu():
    while True:
        print("\nXML Validation.\nSelect an option from the list:")
        print("1) RFI XML Validation")
        print("2) Vessel XML Validation")
        print("3) Flood XML Validation")
        print("4) Run All")
        print("5) Exit")

        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            run_submenu("RFI")
            break
        elif choice == "2":
            run_submenu("Vessel")
            break
        elif choice == "3":
            run_submenu("Flood")
            break
        elif choice == "4":
            for dataset_key in ["RFI", "Vessel", "Flood"]:
                print(f"\n▶️ Running all validations for {dataset_key}")
                run_submenu_all(dataset_key)
            break
        elif choice == "5":
            print("👋 Exiting program.")
            break
        else:
            print("⚠️  Wrong input. Select again.")

# Entry point of the script
if __name__ == "__main__":
    main_menu()
