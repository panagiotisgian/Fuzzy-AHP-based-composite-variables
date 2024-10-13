import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.stats import pearsonr

# Load dataset
def load_dataset(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    elif file_path.endswith('.data'):
        delimiter = input("Please specify the delimiter used in the .data file (e.g., ',' for comma, ' ' for space, '\\t' for tab): ")
        return pd.read_csv(file_path, delimiter=delimiter)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv, .json, or .data file.")

# Convert categorical columns to numeric using Label Encoding
def encode_categorical_columns(df):
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < len(df) * 0.1:  # Αν η στήλη έχει λιγότερες μοναδικές τιμές από το 10% των συνολικών τιμών, θεωρείται κατηγορική
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            print(f"Column '{col}' was encoded with classes: {list(le.classes_)}")
    return df, label_encoders

# Remove non-numeric columns
def filter_numeric_columns(df):
    return df.select_dtypes(include=[np.number])

# Normalize data
def normalize_data(data):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Get user input for target variable
def get_target_variable(df):
    print("Available columns: ", df.columns)
    target_var = input("Please specify the target variable: ")
    if target_var not in df.columns:
        raise ValueError("Invalid target variable")
    return target_var

# Calculate correlation-based weights for features using AHP method
def calculate_weights(df, target):
    correlations = {}
    for col in df.columns:
        if col != target:
            # Υπολογισμός συσχετισμού κάθε χαρακτηριστικού με την target μεταβλητή
            corr, _ = pearsonr(df[col], df[target])
            correlations[col] = abs(corr)

    # Ταξινόμηση των συσχετισμών
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    # Υπολογισμός συνολικού βάρους και επιλογή χαρακτηριστικών που συνεισφέρουν το 90%
    total_weight = sum(corr for _, corr in sorted_corr)
    cumulative_weight = 0
    selected_features = []
    for col, weight in sorted_corr:
        cumulative_weight += weight
        selected_features.append((col, weight))
        if cumulative_weight / total_weight >= 0.9:
            break
    
    weights = {col: weight for col, weight in selected_features}
    return weights

# Create composite dataset with top 90% contributing features
def create_composite_dataset(df, weights):
    composite_df = pd.DataFrame()
    for col, weight in weights.items():
        composite_df[col] = df[col] * weight
    return composite_df

def main():
    # Load dataset
    file_path = input("Enter the file path of the dataset (CSV, JSON, or DATA): ")
    df = load_dataset(file_path)
    
    # Encode categorical columns
    df_encoded, encoders = encode_categorical_columns(df)
    
    # Filter only numeric columns
    df_numeric = filter_numeric_columns(df_encoded)
    print("Filtered dataset with numeric columns: ", df_numeric.columns)
    
    # Normalize data
    df_normalized = normalize_data(df_numeric)
    
    # Get target variable
    target_var = get_target_variable(df_normalized)
    
    # Calculate weights using AHP
    weights = calculate_weights(df_normalized, target_var)
    print(f"Weights based on AHP for top 90% features: {weights}")
    
    # Create composite dataset
    composite_df = create_composite_dataset(df_normalized, weights)
    print("Composite dataset created successfully.")
    
    # Save the composite dataset
    output_file = input("Enter the full file path to save the composite dataset (CSV format): ")
    composite_df.to_csv(output_file, index=False)
    print(f"Composite dataset saved to {output_file}.")

if __name__ == "__main__":
    main()
