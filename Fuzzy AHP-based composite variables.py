import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.stats import pearsonr

# Load and preprocess the dataset
def load_and_preprocess_dataset(file_path):
    # Load dataset
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    elif file_path.endswith('.data'):
        delimiter = input("Please specify the delimiter used in the .data file (e.g., ',' for comma, ' ' for space, '\\t' for tab): ")
        df = pd.read_csv(file_path, delimiter=delimiter)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv, .json, or .data file.")
    
    # Exclude ID-like columns
    id_like_columns = [col for col in df.columns if df[col].nunique() == len(df)]
    if id_like_columns:
        print(f"ID-like columns excluded: {id_like_columns}")
    df = df.drop(columns=id_like_columns)
    
    # Identify target variable
    print("Available columns: ", df.columns.tolist())
    target_var = input("Please specify the target variable: ")
    if target_var not in df.columns:
        raise ValueError("Invalid target variable")
    
    # Encode categorical variables (including target variable if necessary)
    df, encoders, encoded_columns, metadata_columns = encode_categorical_columns(df.copy(), target_var)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    return df, target_var, encoders, metadata_columns

# Encode categorical columns
def encode_categorical_columns(df, target_var):
    label_encoders = {}
    encoded_columns = []
    metadata_columns = []
    for col in df.select_dtypes(include=['object']).columns:
        unique_values = df[col].nunique()
        if unique_values <= 10 or col == target_var:
            # If the column has 10 or fewer unique values or it's the target variable, encode it
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            encoded_columns.append(col)
            print(f"Column '{col}' was encoded with classes: {list(le.classes_)}")
        else:
            # Otherwise, treat it as metadata
            metadata_columns.append(col)
            print(f"Column '{col}' was treated as metadata and will be included as is.")
    return df, label_encoders, encoded_columns, metadata_columns

# Handle missing values
def handle_missing_values(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                mode = df[col].mode()[0]
                df[col].fillna(mode, inplace=True)
                print(f"Missing values in column '{col}' filled with mode: {mode}")
            else:
                mean = df[col].mean()
                df[col].fillna(mean, inplace=True)
                print(f"Missing values in column '{col}' filled with mean: {mean}")
    return df

# Calculate Pearson correlations
def calculate_correlations(X, y):
    correlations = {}
    for col in X.columns:
        try:
            corr, _ = pearsonr(X[col], y)
            if np.isnan(corr):
                corr = 0
                print(f"Correlation between '{col}' and target variable resulted in NaN and was set to zero.")
            correlations[col] = abs(corr)  # Use absolute value
        except Exception as e:
            correlations[col] = 0
            print(f"Correlation between '{col}' and target variable could not be computed. Error: {e}")
    return correlations

# Calculate Importance based on correlations
def calculate_importance(correlations):
    max_corr = max(correlations.values())
    min_corr = min(correlations.values())
    s = (max_corr - min_corr) / 9  # Step size

    if s == 0:
        print("All correlations are equal. Assigning equal importance.")
        Importance = {k: 5 for k in correlations.keys()}  # Assign middle importance
        return Importance

    Importance = {}
    for var, corr in correlations.items():
        r = int((corr - min_corr) / s) + 1  # Assign importance between 1 and 10
        if r > 9:
            r = 9
        Importance[var] = r
    return Importance

# Construct pairwise comparison matrix A
def construct_pairwise_matrix(Importance):
    features = list(Importance.keys())
    n = len(features)
    A = np.ones((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = abs(Importance[features[i]] - Importance[features[j]])
            if Importance[features[i]] > Importance[features[j]]:
                A[i][j] = d + 1
                A[j][i] = 1 / (d + 1)
            else:
                A[i][j] = 1 / (d + 1)
                A[j][i] = d + 1
    return A, features

# Calculate weights from pairwise comparison matrix A using fuzzy AHP
def calculate_weights_from_A(A, features):
    n = A.shape[0]
    # Initialize fuzzy comparison matrix with triangular fuzzy numbers
    fuzzy_A = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            a_ij = A[i, j]
            if a_ij == 1:
                fuzzy_A[i, j] = (1, 1, 1)
            elif a_ij > 1:
                l = a_ij - 0.5
                m = a_ij
                u = a_ij + 0.5
                fuzzy_A[i, j] = (l, m, u)
            else:
                inv = 1 / a_ij
                l = 1 / (inv + 0.5)
                m = a_ij
                u = 1 / (inv - 0.5)
                fuzzy_A[i, j] = (l, m, u)
    
    # Compute fuzzy synthetic extent values
    fuzzy_sums = []
    for i in range(n):
        sum_l = sum([fuzzy_A[i, j][0] for j in range(n)])
        sum_m = sum([fuzzy_A[i, j][1] for j in range(n)])
        sum_u = sum([fuzzy_A[i, j][2] for j in range(n)])
        fuzzy_sums.append((sum_l, sum_m, sum_u))
    
    total_l = sum([fs[0] for fs in fuzzy_sums])
    total_m = sum([fs[1] for fs in fuzzy_sums])
    total_u = sum([fs[2] for fs in fuzzy_sums])

    fuzzy_weights = []
    for fs in fuzzy_sums:
        l = fs[0] / total_u
        m = fs[1] / total_m
        u = fs[2] / total_l
        fuzzy_weights.append((l, m, u))
    
    # Defuzzify using Center of Area (CoA)
    defuzzified_weights = []
    for fw in fuzzy_weights:
        defuzzified_weight = (fw[0] + fw[1] + fw[2]) / 3
        defuzzified_weights.append(defuzzified_weight)
    
    # Normalize the weights
    sum_weights = sum(defuzzified_weights)
    normalized_weights = [w / sum_weights for w in defuzzified_weights]
    
    # Create weights dictionary
    weights_dict = dict(zip(features, normalized_weights))
    return weights_dict

# Select top variables contributing to 90% cumulative weight
def select_top_variables(weights_dict):
    sorted_weights = sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)
    total_weight = sum([w for f, w in sorted_weights])
    cumulative_weight = 0
    selected_features = []
    for f, w in sorted_weights:
        cumulative_weight += w
        selected_features.append(f)
        if cumulative_weight / total_weight >= 0.9:
            break
    return selected_features

# Create composite dataset
def create_composite_dataset(X_original, weights_dict, selected_features, metadata_df, target_var, target_series):
    composite_df = pd.DataFrame()
    # Include target variable as the first column
    composite_df[f"{target_var} (target)"] = target_series
    for feature in selected_features:
        # Include weight in column name
        composite_df[f"{feature} ({weights_dict[feature]:.4f})"] = X_original[feature] * weights_dict[feature]
    # Include metadata columns as is
    for col in metadata_df.columns:
        composite_df[col] = metadata_df[col]
    return composite_df

def main():
    # Load and preprocess dataset
    file_path = input("Enter the file path of the dataset (CSV, JSON, or DATA): ")
    df, target_var, encoders, metadata_columns = load_and_preprocess_dataset(file_path)
    
    # Separate features and target
    X = df.drop(columns=[target_var])
    y = df[target_var]
    
    # Handle metadata columns
    metadata_df = df[metadata_columns] if metadata_columns else pd.DataFrame()
    
    # Filter numeric columns for X
    X_numeric = X.select_dtypes(include=[np.number])
    
    # Store original data before normalization
    X_original = X_numeric.copy()
    
    # Normalize data for weight calculation
    scaler = MinMaxScaler()
    X_normalized = pd.DataFrame(scaler.fit_transform(X_numeric), columns=X_numeric.columns)
    
    # Calculate Pearson correlations
    correlations = calculate_correlations(X_normalized, y)
    
    # Exclude features with zero correlation
    correlations = {k: v for k, v in correlations.items() if v != 0}
    if not correlations:
        print("No valid correlations could be computed. Please check your data.")
        return
    
    # Calculate Importance
    Importance = calculate_importance(correlations)
    
    # Construct pairwise comparison matrix A
    A, features = construct_pairwise_matrix(Importance)
    
    # Calculate weights from A using fuzzy AHP
    weights_dict = calculate_weights_from_A(A, features)
    
    # Select top variables
    selected_features = select_top_variables(weights_dict)
    print(f"Selected features: {selected_features}")
    
    # Create composite dataset
    composite_df = create_composite_dataset(X_original, weights_dict, selected_features, metadata_df, target_var, y)
    print("Composite dataset created successfully.")
    
    # Save the composite dataset
    output_file = input("Enter the full file path to save the composite dataset (CSV format): ")
    composite_df.to_csv(output_file, index=False)
    print(f"Composite dataset saved to {output_file}.")

if __name__ == "__main__":
    main()
