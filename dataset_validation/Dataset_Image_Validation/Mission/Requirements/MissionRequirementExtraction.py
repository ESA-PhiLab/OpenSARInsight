import pandas as pd
import os

def fcn_ExtractDatasetRequirements(excel_file_path):
    try:
        # Check if the file exists
        if not os.path.isfile(excel_file_path):
            print("Error: Excel file is not found.")
            return pd.DataFrame()  # Return an empty DataFrame
        # Load the Excel file
        xls = pd.ExcelFile(excel_file_path)
        # Check if "Requirements" sheet exists
        if "Requirements" not in xls.sheet_names:
            print("No 'Requirements' sheet found in the Excel file.")
            return pd.DataFrame()  # Return an empty DataFrame
        # Read only necessary columns from the "Requirements" sheet
        usecols = ["ID", "RequirementDetails", "Aplicability"]
        df = xls.parse("Requirements", usecols=usecols)
        # Ensure required columns exist
        required_columns = set(usecols)
        if not required_columns.issubset(df.columns):
            print("Missing required columns in the 'Requirements' sheet.")
            return pd.DataFrame()  # Return an empty DataFrame
        # Filter rows where "Aplicability" is "DS"
        ds_requirements = df[df["Aplicability"].astype(str).str.strip() == "DS"]
        # Extract both "ID" and "Requirement Details"
        requirement_pairs = ds_requirements[["ID", "RequirementDetails"]].dropna()
        return requirement_pairs
    except Exception as e:
        print(f"Error processing the Excel file: {e}")
        return pd.DataFrame()  # Return an empty DataFrame
