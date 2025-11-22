import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


def clean_text(df):
    """Lowercase + strip all string columns for consistency."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.lower().str.strip()
    return df


def build_preprocessing_pipeline(categorical_cols, numerical_cols):
    """Create ColumnTransformer for categorical + numerical preprocessing."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                ),
                categorical_cols,
            ),
            ("numerical", StandardScaler(), numerical_cols),
        ]
    )


def preprocess_dataset(raw_csv_path, output_csv_path, preprocess_path):
    """Load dataset, clean, preprocess, and save transformed CSV + pipeline."""
    
    print("Loading dataset...")
    df = pd.read_csv(raw_csv_path)

    # Clean object columns
    df = clean_text(df)

    # Define target
    target_col = "Accident Severity"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Feature groups
    categorical_cols = [
        "Country", "Month", "Day of Week", "Time of Day", "Urban/Rural",
        "Road Type", "Weather Conditions", "Driver Age Group", "Driver Gender",
        "Driver Alcohol Level", "Driver Fatigue", "Vehicle Condition",
        "Road Condition", "Accident Cause", "Region"
    ]
    numerical_cols = [
        "Year", "Visibility Level", "Number of Vehicles Involved", "Speed Limit",
        "Pedestrians Involved", "Cyclists Involved", "Number of Injuries",
        "Number of Fatalities", "Emergency Response Time", "Traffic Volume",
        "Insurance Claims", "Medical Cost", "Economic Loss",
        "Population Density"
    ]

    # Separate X and y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Build preprocessing transformer
    preprocessor = build_preprocessing_pipeline(categorical_cols, numerical_cols)

    preprocessing_pipeline = Pipeline([
        ("preprocessor", preprocessor)
    ])

    print("Fitting preprocessing pipeline...")
    X_processed = preprocessing_pipeline.fit_transform(X)

    print("Saving preprocessing pipeline...")
    joblib.dump(preprocessing_pipeline, preprocess_path)

    print("Saving processed dataset...")
    processed_df = pd.DataFrame(X_processed)
    processed_df[target_col] = y.values
    processed_df.to_csv(output_csv_path, index=False)

    print("Preprocessing completed successfully.")
    return processed_df


if __name__ == "__main__":
    preprocess_dataset(
        raw_csv_path="GlobalRoadAccidents_raw/road_accident_dataset.csv",
        output_csv_path="preprocessing/processed_dataset.csv",
        preprocess_path="preprocessing/preprocessing_pipeline.pkl"
    )
