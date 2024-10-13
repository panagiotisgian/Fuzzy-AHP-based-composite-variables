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

# Identify and exclude ID-like variables
def exclude_id_like_columns(df):
    id_like_columns = []
    for col in df.columns:
        if df[col].nunique() == len(df):
            id_like_columns.append(col)
            print(f"Column '{col}' is identified as an ID-like variable and will be excluded.")
    df = df.drop(columns=id_like_columns)
    return df, id_like_columns

# Encode categorical columns
def encode_categorical_columns(df, target_var=None):
    label_encoders = {}
    encoded_columns = []
    metadata_columns = []
    for col in df.select_dtypes(include=['object']).columns:
        unique_values = df[col].nunique()
        if unique_values <= 10:
            # If the column has 10 or fewer unique values, encode it
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))  # Convert to string in case of mixed types
            label_encoders[col] = le
            encoded_columns.append(col)
            print(f"Column '{col}' was encoded with classes: {list(le.classes_)}")
        else:
            # Otherwise, treat it as metadata
            metadata_columns.append(col)
            print(f"Column '{col}' was treated as metadata and will be included as is.")
    return df, label_encoders, encoded_columns, metadata_columns

# Remove non-numeric columns except metadata
def filter_numeric_columns(df, metadata_columns):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_columns], df[metadata_columns]

# Normalize data
def normalize_data(data):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Get user input for target variable
def get_target_variable(df):
    print("Available columns: ", df.columns.tolist())
    target_var = input("Please specify the target variable: ")
    if target_var not in df.columns:
        raise ValueError("Invalid target variable")
    return target_var

# Calculate weights using fuzzy AHP
def calculate_weights(df, target):
    import numpy as np
    from scipy.stats import pearsonr

    # Step 1: Calculate correlations between features and target variable
    correlations = {}
    for col in df.columns:
        if col != target:
            try:
                corr, _ = pearsonr(df[col], df[target])
                if np.isnan(corr):
                    correlations[col] = 0  # Assign zero if correlation is NaN
                    print(f"Correlation between '{col}' and '{target}' resulted in NaN and was set to zero.")
                else:
                    correlations[col] = abs(corr)  # Use absolute value of Pearson correlation
            except Exception as e:
                correlations[col] = 0  # If correlation cannot be computed
                print(f"Correlation between '{col}' and '{target}' could not be computed. Error: {e}")

    # Exclude features with zero correlation
    correlations = {k: v for k, v in correlations.items() if v != 0}

    if not correlations:
        print("No valid correlations could be computed. Please check your data.")
        return {}

    features = list(correlations.keys())

    # Step 2: Build fuzzy pairwise comparison matrix
    def ratio_to_fuzzy_number(ratio):
        if ratio == 1:
            return (1, 1, 1)
        elif 1 < ratio <= 2:
            return (1, 3, 5)
        elif 2 < ratio <= 3:
            return (3, 5, 7)
        elif 3 < ratio <= 4:
            return (5, 7, 9)
        elif ratio > 4:
            return (7, 9, 9)
        elif 0 < ratio < 1:
            reciprocal_ratio = 1 / ratio
            l, m, u = ratio_to_fuzzy_number(reciprocal_ratio)
            return (1/u, 1/m, 1/l)
        elif ratio == 0:
            # Handle zero ratio, assign lowest possible value
            return (1/9, 1/9, 1/9)
        elif ratio == float('inf'):
            # Handle infinite ratio, assign highest possible value
            return (9, 9, 9)
        else:
            raise ValueError(f"Invalid ratio value: {ratio}")

    fuzzy_matrix = {}
    for i in features:
        fuzzy_matrix[i] = {}
        for j in features:
            if i == j:
                fuzzy_matrix[i][j] = (1, 1, 1)
            else:
                # Calculate ratio carefully to handle zero correlations
                if correlations[j] == 0:
                    if correlations[i] == 0:
                        ratio = 1  # Both correlations are zero
                    else:
                        ratio = float('inf')  # Division by zero, infinite ratio
                else:
                    ratio = correlations[i] / correlations[j]

                # Handle ratio
                try:
                    fuzzy_number = ratio_to_fuzzy_number(ratio)
                    fuzzy_matrix[i][j] = fuzzy_number
                except ValueError as e:
                    print(f"Invalid ratio between '{i}' and '{j}': {e}")
                    return {}

    # Step 3: Sum the fuzzy numbers in each row
    row_sums = {}
    for i in features:
        l_sum = 0
        m_sum = 0
        u_sum = 0
        for j in features:
            l, m, u = fuzzy_matrix[i][j]
            l_sum += l
            m_sum += m
            u_sum += u
        row_sums[i] = (l_sum, m_sum, u_sum)

    # Step 4: Compute total sum
    total_l_sum = sum([l for l, m, u in row_sums.values()])
    total_m_sum = sum([m for l, m, u in row_sums.values()])
    total_u_sum = sum([u for l, m, u in row_sums.values()])
    total_sum = (total_l_sum, total_m_sum, total_u_sum)

    # Step 5: Compute synthetic extent values
    synthetic_extent = {}
    for i in features:
        l1, m1, u1 = row_sums[i]
        l2, m2, u2 = total_sum
        s_l = l1 / u2
        s_m = m1 / m2
        s_u = u1 / l2
        synthetic_extent[i] = (s_l, s_m, s_u)

    # Step 6: Defuzzify synthetic extent values using CoA
    defuzzified_extents = {}
    for i in features:
        l, m, u = synthetic_extent[i]
        defuzzified_extents[i] = (l + m + u) / 3

    # Step 7: Normalize the defuzzified extents to get weights
    total_defuzzified_extent = sum(defuzzified_extents.values())
    normalized_weights = {i: de / total_defuzzified_extent for i, de in defuzzified_extents.items()}

    # Step 8: Sort the weights
    sorted_weights = sorted(normalized_weights.items(), key=lambda x: x[1], reverse=True)

    # Step 9: Select features contributing to 90% cumulative weight
    cumulative_weight = 0
    selected_features = []
    for col, weight in sorted_weights:
        cumulative_weight += weight
        selected_features.append((col, weight))
        if cumulative_weight >= 0.9:
            break

    weights = {col: weight for col, weight in selected_features}
    return weights

# Create composite dataset with top 90% contributing features
def create_composite_dataset(df_original, weights, metadata_df, target_var, df_encoded):
    composite_df = pd.DataFrame()
    # Include target variable as the first column
    composite_df[f"{target_var} (target)"] = df_encoded[target_var]
    for col, weight in weights.items():
        # Include weight in column name
        composite_df[f"{col} ({weight:.4f})"] = df_original[col] * weight
    # Include metadata columns as is
    for col in metadata_df.columns:
        composite_df[col] = metadata_df[col]
    return composite_df

def main():
    # Load dataset
    file_path = input("Enter the file path of the dataset (CSV, JSON, or DATA): ")
    df = load_dataset(file_path)

    # Exclude ID-like variables
    df, id_like_columns = exclude_id_like_columns(df)

    # Get target variable
    print("Available columns: ", df.columns.tolist())
    target_var = input("Please specify the target variable: ")
    if target_var not in df.columns:
        raise ValueError("Invalid target variable")

    # Encode categorical columns (including target variable if necessary)
    df_encoded, encoders, encoded_columns, metadata_columns = encode_categorical_columns(df.copy(), target_var)

    # Filter only numeric columns and separate metadata columns
    df_numeric, metadata_df = filter_numeric_columns(df_encoded, metadata_columns)
    print("Filtered dataset with numeric columns: ", df_numeric.columns.tolist())

    # Exclude variables with zero variance
    zero_variance_columns = df_numeric.columns[df_numeric.nunique() <= 1].tolist()
    if zero_variance_columns:
        df_numeric = df_numeric.drop(columns=zero_variance_columns)
        print(f"Columns with zero variance and removed: {zero_variance_columns}")

    # **Store original numeric data before normalization**
    df_numeric_original = df_numeric.copy()

    # Normalize data
    df_normalized = normalize_data(df_numeric)

    # Re-check if target variable was encoded; if not, raise an error
    if target_var in metadata_columns:
        raise ValueError(f"Target variable '{target_var}' is non-numeric and was not encoded. Please ensure it has 10 or fewer unique values to be encoded.")

    # Calculate weights using fuzzy AHP
    weights = calculate_weights(df_normalized, target_var)
    if not weights:
        print("Weights could not be calculated due to insufficient data.")
        return
    print(f"Weights based on fuzzy AHP for top 90% features: {weights}")

    # **Create composite dataset using original data and weights, including target variable as first column**
    composite_df = create_composite_dataset(df_numeric_original, weights, metadata_df, target_var, df_encoded)
    print("Composite dataset created successfully.")

    # Save the composite dataset
    output_file = input("Enter the full file path to save the composite dataset (CSV format): ")
    composite_df.to_csv(output_file, index=False)
    print(f"Composite dataset saved to {output_file}.")

if __name__ == "__main__":
    main()
