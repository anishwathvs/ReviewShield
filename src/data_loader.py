"""Dataset loading and verification module.

Provides flexible loading and validation for Kaggle / custom review datasets.
Supports both CSV (.csv, .tsv) and Excel (.xlsx, .xls) files.
Does not assume hardcoded column names; includes auto-detection,
validation, and clear configuration options.
"""

import os
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd

# Default expected column candidate names (case-insensitive matching)
CANDIDATE_TEXT_COLUMNS = [
    "review_text", "review", "text", "text_", "review_body", "content",
    "statement", "comment", "verified_reviews", "review_content",
    "review_description", "body"
]

CANDIDATE_LABEL_COLUMNS = [
    "label", "target", "category", "class", "rating", "is_fake",
    "deceptive", "fake", "status", "label_num", "ground_truth"
]


def _read_file(filepath: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Reads CSV or Excel file safely based on file extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(filepath, nrows=nrows)
    elif ext == ".tsv":
        return pd.read_csv(filepath, sep="\t", nrows=nrows)
    else:
        # Default to standard CSV reader
        return pd.read_csv(filepath, nrows=nrows)


def find_column_match(df_columns: List[str], candidates: List[str]) -> Optional[str]:
    """Finds matching column name from a list of candidates."""
    cols_lower = {col.lower().strip(): col for col in df_columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    return None


def get_available_datasets(dataset_dir: str = "dataset") -> List[str]:
    """Lists all recognized dataset files (.csv, .tsv, .xlsx, .xls) inside the directory."""
    if not os.path.exists(dataset_dir):
        return []
    valid_exts = {".csv", ".tsv", ".xlsx", ".xls"}
    return [
        f for f in os.listdir(dataset_dir)
        if os.path.splitext(f)[1].lower() in valid_exts and not f.startswith(".")
    ]


def inspect_dataset_file(filepath: str) -> dict:
    """Inspects a dataset file and reports its schema and suitability.

    Args:
        filepath: Path to the dataset file (CSV or Excel)

    Returns:
        Dictionary with status, columns, sample shape, detected text and label cols.
    """
    if not os.path.exists(filepath):
        return {
            "exists": False,
            "error": f"Dataset file not found at: {filepath}"
        }

    try:
        df = _read_file(filepath, nrows=5)
        text_col = find_column_match(list(df.columns), CANDIDATE_TEXT_COLUMNS)
        label_col = find_column_match(list(df.columns), CANDIDATE_LABEL_COLUMNS)

        return {
            "exists": True,
            "filepath": filepath,
            "columns": list(df.columns),
            "sample_rows": len(df),
            "detected_text_column": text_col,
            "detected_label_column": label_col
        }
    except Exception as e:
        return {
            "exists": True,
            "error": f"Failed to read file: {str(e)}"
        }


def validate_dataset(
    filepath: str,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None
) -> Dict[str, Any]:
    """Validates the dataset without modifying the file or training any model.

    Checks:
    - File existence & dimensions
    - Column identification
    - Missing values breakdown
    - Duplicate rows & duplicate review texts
    - Unique label categories and distribution
    - Non-text values in review column
    - Irrelevant extra columns

    Returns:
        Dictionary containing comprehensive validation report.
    """
    if not os.path.exists(filepath):
        return {
            "is_valid": False,
            "error": f"File does not exist: {filepath}"
        }

    ext = os.path.splitext(filepath)[1].lower()
    df = _read_file(filepath)

    total_rows, total_cols = df.shape
    columns = list(df.columns)

    # Resolve text & label columns
    resolved_text_col = text_column if text_column in df.columns else find_column_match(columns, CANDIDATE_TEXT_COLUMNS)
    resolved_label_col = label_column if label_column in df.columns else find_column_match(columns, CANDIDATE_LABEL_COLUMNS)

    # Identify potential extra/irrelevant columns
    extra_cols = [c for c in columns if c not in (resolved_text_col, resolved_label_col)]

    # Missing values
    missing_by_col = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())

    # Duplicates
    full_duplicates = int(df.duplicated().sum())

    report: Dict[str, Any] = {
        "is_valid": True,
        "filename": os.path.basename(filepath),
        "filepath": filepath,
        "file_type": ext,
        "total_rows": total_rows,
        "total_columns": total_cols,
        "columns": columns,
        "review_text_column": resolved_text_col,
        "label_column": resolved_label_col,
        "extra_columns": extra_cols,
        "total_missing_values": total_missing,
        "missing_per_column": missing_by_col,
        "duplicate_rows": full_duplicates,
        "text_duplicates": 0,
        "missing_text_count": 0,
        "missing_label_count": 0,
        "unique_labels": [],
        "label_distribution": {},
        "first_records": df.head(3).to_dict(orient="records"),
        "validation_warnings": []
    }

    if not resolved_text_col:
        report["is_valid"] = False
        report["validation_warnings"].append(
            f"Could not identify a review-text column. Candidates checked: {CANDIDATE_TEXT_COLUMNS}. Available columns: {columns}."
        )

    if not resolved_label_col:
        report["is_valid"] = False
        report["validation_warnings"].append(
            f"Could not identify a label column. Candidates checked: {CANDIDATE_LABEL_COLUMNS}. Available columns: {columns}."
        )

    if resolved_text_col and resolved_label_col:
        missing_text = int(df[resolved_text_col].isnull().sum())
        missing_labels = int(df[resolved_label_col].isnull().sum())
        report["missing_text_count"] = missing_text
        report["missing_label_count"] = missing_labels

        # Text duplicates
        text_dups = int(df[resolved_text_col].dropna().duplicated().sum())
        report["text_duplicates"] = text_dups

        # Label distribution
        label_counts = df[resolved_label_col].value_counts(dropna=False).to_dict()
        report["label_distribution"] = {str(k): int(v) for k, v in label_counts.items()}
        report["unique_labels"] = [str(k) for k in label_counts.keys() if pd.notnull(k)]

        # Check for non-string / empty reviews
        non_str = int((~df[resolved_text_col].apply(lambda x: isinstance(x, str))).sum())
        if non_str > 0:
            report["validation_warnings"].append(
                f"Found {non_str} rows with non-string values in review column '{resolved_text_col}'."
            )

        # Check class balance
        if len(report["unique_labels"]) > 1:
            counts = list(report["label_distribution"].values())
            ratio = max(counts) / (min(counts) if min(counts) > 0 else 1)
            if ratio > 4.0:
                report["validation_warnings"].append(
                    f"Noticeable class imbalance detected (ratio: {ratio:.1f}:1)."
                )

        if missing_text > 0:
            report["validation_warnings"].append(
                f"{missing_text} records have missing review text."
            )
        if missing_labels > 0:
            report["validation_warnings"].append(
                f"{missing_labels} records have missing labels."
            )

    return report


def load_dataset(
    filepath: str,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    drop_duplicates: bool = True
) -> Tuple[pd.DataFrame, str, str]:
    """Loads and cleans dataset for training/evaluation without mutating original file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. Please place your dataset file inside the dataset/ directory."
        )

    df = _read_file(filepath)

    # Resolve text column
    if text_column and text_column in df.columns:
        chosen_text_col = text_column
    else:
        chosen_text_col = find_column_match(list(df.columns), CANDIDATE_TEXT_COLUMNS)

    # Resolve label column
    if label_column and label_column in df.columns:
        chosen_label_col = label_column
    else:
        chosen_label_col = find_column_match(list(df.columns), CANDIDATE_LABEL_COLUMNS)

    if not chosen_text_col:
        raise ValueError(
            f"Could not identify the review text column among {list(df.columns)}. "
            f"Please specify 'text_column' explicitly."
        )

    if not chosen_label_col:
        raise ValueError(
            f"Could not identify the label/target column among {list(df.columns)}. "
            f"Please specify 'label_column' explicitly."
        )

    # Handle missing values: keep rows with valid text and label
    df_clean = df.dropna(subset=[chosen_text_col, chosen_label_col]).copy()
    df_clean = df_clean[df_clean[chosen_text_col].astype(str).str.strip() != ""]

    # Deduplicate review texts to prevent data leakage between train and test splits
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates(subset=[chosen_text_col]).reset_index(drop=True)

    if len(df_clean) == 0:
        raise ValueError(f"Dataset in '{filepath}' has 0 valid non-null rows.")

    return df_clean, chosen_text_col, chosen_label_col



if __name__ == "__main__":
    import sys
    available = get_available_datasets()
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    elif available:
        target_path = os.path.join("dataset", available[0])
    else:
        target_path = None

    print("=" * 65)
    print("AI-Powered Fake Product Review Detection - Dataset Validator")
    print("=" * 65)

    if not target_path or not os.path.exists(target_path):
        print("\n[!] No dataset file found in the 'dataset/' folder.")
        print("    Available files in dataset/:", os.listdir("dataset") if os.path.exists("dataset") else "Directory missing")
        print("\nACTION REQUIRED:")
        print("Please copy your CSV or XLSX dataset file into 'dataset/'")
        print("Example: dataset/reviews.csv")
    else:
        print(f"\nRunning validation on: {target_path}")
        report = validate_dataset(target_path)
        print(f"File Name:             {report.get('filename')}")
        print(f"File Format:           {report.get('file_type')}")
        print(f"Total Rows:            {report.get('total_rows')}")
        print(f"Total Columns:         {report.get('total_columns')}")
        print(f"Columns:               {report.get('columns')}")
        print(f"Review Text Column:    {report.get('review_text_column')}")
        print(f"Label Column:          {report.get('label_column')}")
        print(f"Extra Columns:         {report.get('extra_columns')}")
        print(f"Missing Values:        {report.get('total_missing_values')}")
        print(f"Duplicate Rows:        {report.get('duplicate_rows')}")
        print(f"Duplicate Texts:       {report.get('text_duplicates')}")
        print(f"Unique Labels:         {report.get('unique_labels')}")
        print(f"Label Distribution:    {report.get('label_distribution')}")
        if report.get("validation_warnings"):
            print("\nWarnings:")
            for w in report["validation_warnings"]:
                print(f" - {w}")
        print(f"\nReady for Training:    {report.get('is_valid')}")
    print("=" * 65)
