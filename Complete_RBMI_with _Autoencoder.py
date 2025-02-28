# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 20:06:44 2024

@author: Deep
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from pygrinder import mcar, mar_logistic, mnar_x, mnar_t
from scipy.stats import pearsonr


from pypots.data import load_specific_dataset



plt.rcParams['font.sans-serif'] = ['SimHei']  # Supports Chinese characters
plt.rcParams['axes.unicode_minus'] = False  # Avoid issues with negative numbers

# Load the dataset
#df = pd.read_excel("mis_iris_data.xlsx")
#df = pd.read_excel("test.xlsx")
#df = pd.read_csv("aki_icu_dataset.csv")
df = pd.read_csv("processed_aki_dataset.csv")
#df = pd.read_csv("aki_processed_dataset_unix.csv")
#df = pd.read_csv("collected_processed_dataset_unix.csv")

new_data = df.copy()

#new_data.fillna(new_data.mean(), inplace=True)
#print(new_data)

# X = data['X']
# num_samples = len(X['RecordID'].unique())
# X = X.drop(['RecordID', 'Time'], axis = 1)
X = df # Features
num_samples, n_features = X.shape
#scaler = StandardScaler()
#X = scaler.fit_transform(X.to_numpy())

X = X.to_numpy()
X_ori = X  # keep X_ori for validation

X = X.reshape(num_samples, 1, -1)


# Define missingness probability
p = 0.2  # 20% probability of missing values

#Create completely random missing values (MCAR case).
# Apply MCAR missing value generation with 20% missing data
#X = mcar(X, p)

# Define missingness parameters
obs_rate = 0.3  # 30% of variables are fully observed
missing_rate = 0.2  # 20% missing values in remaining variables

# Create random missing values (MAR case) with a logistic model. 
#X = mar_logistic(X, obs_rate, missing_rate)

X = mnar_t(X, cycle=20, pos=10, scale=1)


#print(X.shape)  # (11988, 48, 37), 11988 samples and each sample has 48 time steps, 37 features

X = X.reshape(X.shape[0], -1)

df = pd.DataFrame(X)


original_df = pd.DataFrame(X_ori)
#original_df.fillna(original_df.mean(), inplace=True)

# # Define conditions for intentional missingness
# intentional_mask = (
#     (df[['uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr', 'aki_stage_uo']].notnull().any(axis=1)) &
#     (df[['creat_low_past_7day', 'creat_low_past_48hr', 'creat', 'aki_stage_creat']].isnull().all(axis=1))
# ) | (
#     (df[['creat_low_past_7day', 'creat_low_past_48hr', 'creat', 'aki_stage_creat']].notnull().any(axis=1)) &
#     (df[['uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr', 'aki_stage_uo']].isnull().all(axis=1))
# ) | (
#     df['dod'].isnull()  # Marking intentional gaps for date of death
# )

# # Mark rows with intentional missingness (you can set these values to NaN or perform any other transformation)
# df.loc[intentional_mask] = np.nan


#new_data = aki_data.copy()
# #df.columns = [col.replace(" (cm)", "").replace(" ", "_") for col in df.columns]  # Simplify column names
# # #Step 1, 
# # # Fill NaN in the first row of each column with the first non-NaN value from below
# # for col in df.columns:
# #     if pd.isna(df.loc[0, col]):
# #         first_non_nan = df[col].loc[1:].first_valid_index()  # Finds the first non-NaN index below the first row
# #         if first_non_nan is not None:
# #             df.loc[0, col] = df.loc[first_non_nan, col]
# # # Fill NaN in the last row of each column with the last non-NaN value from above
# # for col in df.columns:
# #     if pd.isna(df.loc[df.index[-1], col]):
# #         last_non_nan = df[col].iloc[:-1].last_valid_index()  # Finds the last non-NaN index above the last row
# #         if last_non_nan is not None:
# #             df.loc[df.index[-1], col] = df.loc[last_non_nan, col]

# new_data = df.copy()

# def remove_empty_rows_and_columns(df):
#     # Remove completely empty columns
#     df_cleaned = df.dropna(axis=1, how='all')
    
#     # # Remove completely empty rows
#     # df_cleaned = df_cleaned.dropna(axis=0, how='all')
    
#     return df_cleaned

# # Remove completely missing columns and rows
# df = remove_empty_rows_and_columns(df)

# # def count_complete_columns(df):
# #     # Count columns where there are no missing values
# #     complete_columns = [col for col in df.columns if df[col].notna().all()]
    
# #     return len(complete_columns), complete_columns

# # # Count complete columns
# # num_complete_cols, complete_cols = count_complete_columns(df)


# # print(f"Number of complete columns: {num_complete_cols}")
# # print(f"Complete columns: {complete_cols}")


# # # Check the data types of all columns before any conversion
# # print("Initial Data Types:\n", df.dtypes)


# # # Convert valid date-time columns
# # for col in df.columns:
# #     try:
# #         df[col] = pd.to_datetime(df[col])  # Convert if it's a date column
# #     except Exception:
# #         pass  # Ignore non-date columns

# # # Convert datetime columns to Unix timestamp
# # for col in df.select_dtypes(include=['datetime64[ns]']):
# #     df[f'{col}_timestamp'] = df[col].astype('int64') // 10**9  # Convert to seconds

# # print(df)
# # # Drop columns with datetime dtype
# # df = df.drop(columns=df.select_dtypes(include=['datetime']).columns)
# # Convert a specific column to datetime
# #df['timestamp'] = pd.to_datetime(df['上机时间'])
# #df['timestamp_unix'] = df['timestamp'].astype('int64') // 10**9  # Convert nanoseconds to seconds

# # Fill columns that are completely empty (all NaN) with 0
# df_cleaned = df.apply(lambda col: col.fillna(0) if col.isna().all() else col, axis=0)

# print("DataFrame after filling completely empty columns with 0:\n", df_cleaned)

# # Convert non-numeric columns to numeric
# # Step 1: Identify non-numeric columns
# non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
# print("\nNon-Numeric Columns:", non_numeric_cols)

# df.drop(['subject_id', 'hadm_id', 'stay_id','charttime_aki', 'charttime_rrt', 'dod', 'admittime', 'dischtime', 'icu_intime', 'icu_outtime'], axis = 1)

# encoder = LabelEncoder()

# for col in df.select_dtypes(include=['object', 'category']).columns:
#     df[col] = df[col].astype(str)  # Convert all to string
#     #df[col] = encoder.fit_transform(df[col])

# for col in non_numeric_cols:
#     df[col] = encoder.fit_transform(df[col])

# # Step 3: Handle any other non-numeric columns (if any) by coercing errors to NaN
# df = df.apply(pd.to_numeric, errors='coerce')

# # Check the data types again after conversion
# print("\nData Types After Conversion:\n", df.dtypes)


# df = df.drop(df.columns[df.notna().all()], axis=1)


# for column in df.columns:
#     try:
#         # Attempt to use argmin on the column
#         min_index = df[column].argmin()
#         print(f"Column '{column}' is numeric and argmin worked. Min index: {min_index}")
#     except TypeError:
#         print(f"Column '{column}' caused TypeError: argmin not allowed for this dtype.")
#     except Exception as e:
#         print(f"Column '{column}' caused an error: {e}")

# # Check for NaN values
# print(df.isna().sum())

# # # Check for infinite values
# # print((df == float('inf')).sum())
# # print((df == -float('inf')).sum())



# # # Check for object types in the DataFrame
# # print(df.select_dtypes(include=['object']).dtypes)
# # df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

# #df = df.iloc[:, :26]
# # # First 15 columns
# # first_15_columns = df.iloc[:, :15]

# # # Last 15 columns
# # last_15_columns = df.iloc[:, -15:]

# # # Concatenate first 15 and last 15 columns
# # df = pd.concat([first_15_columns, last_15_columns], axis=1)

# # Save the processed dataset
# df.to_csv('processed_dataset_unix.csv', encoding='utf-8-sig', index=False)

# df = data_mar
df.to_csv('processed_dataset_unix.csv', encoding='utf-8-sig', index=False)


# original_df = X_ori


def compute_min_gap(column_data, row_idx):
    """
    Finds the smallest gap (in row indices) from `row_idx` to a non-NaN value in `column_data`.
    Returns the index of the closest non-NaN value relative to `row_idx`.
    """
    # Get non-NaN values
    non_nan_indices = column_data.index[column_data.notna()].tolist()
    
    if not non_nan_indices:
        return np.nan

    # Find the closest index
    min_gap_index = min(non_nan_indices, key=lambda x: abs(x - row_idx))
    
    return min_gap_index - row_idx

# Identify rows that are entirely NaN
empty_row_index = df[df.isna().all(axis=1)].index.tolist()

# Initialize a DataFrame for storing gaps
gap = pd.DataFrame(index=empty_row_index, columns=df.columns)

# Compute the gap for each empty row
for row_idx in empty_row_index:
    for col in df.columns:
        gap.loc[row_idx, col] = compute_min_gap(df[col], row_idx)

# Ensure all columns in the DataFrame are numeric before performing operations like idxmin
abs_gap = gap.abs().apply(pd.to_numeric, errors='coerce')
min_gap_indices = abs_gap.idxmin(axis=1)

# Fill empty rows based on the calculated minimum gap
for row_idx in empty_row_index:
    col_idx = min_gap_indices[row_idx]  # Column with the smallest gap for this empty row
    line_idx = row_idx + gap.loc[row_idx, col_idx]

    # Check if line_index is within bounds before assignment
    if 0 <= line_idx < df.shape[0]:
        df.at[row_idx, col_idx] = df.at[line_idx, col_idx]

# Step 1: Compute pairwise correlations with missing data
correlations = df.corr(method='pearson', min_periods=1)

# Step 2: Sort each feature's correlations individually (highest to lowest)
sorted_feature_correlations = {col: correlations[col].abs().sort_values(ascending=False).index.tolist() 
                               for col in correlations.columns}

# Step 3: Reorder the DataFrame columns for each feature based on sorted correlations
reordered_data = {feature: df[sorted_columns] for feature, sorted_columns in sorted_feature_correlations.items()}   

# Step 4: Compute mean of reordered data (used later in data fusion step)
mean_values = {col: df[col].mean(skipna=True) for col in df.columns}
# mean_values = {feature: reordered_df.mean() for feature, reordered_df in reordered_data.items()}

# Step 4: Impute missing values in the first and last rows within each DataFrame in reordered_data
for feature, reordered_df in reordered_data.items():
    # Use the average for the feature as the target imputation value
    target_mean = mean_values[feature]

    # Iterate over each column in the reordered DataFrame for this feature
    for col in reordered_df.columns:
        # Get the mean for the current reference column
        reference_mean = mean_values[col]

        # Check the first row for NaNs in the current column
        if pd.isna(reordered_df.at[0, col]):
            if pd.notna(reordered_df.at[0, feature]):
                reordered_df.at[0, col] = (reordered_df.at[0, feature] / target_mean) * reference_mean
            else:
                # Fallback to nearest non-NaN value if the direct reference is NaN
                nearest_non_nan = reordered_df[feature].dropna().iloc[0] if not reordered_df[feature].dropna().empty else np.nan
                if nearest_non_nan is not np.nan:
                    reordered_df.at[0, col] = (nearest_non_nan / target_mean) * reference_mean

        # Check the last row for NaNs in the current column
        last_row = reordered_df.index[-1]
        if pd.isna(reordered_df.at[last_row, col]):
            if pd.notna(reordered_df.at[last_row, feature]):
                reordered_df.at[last_row, col] = (reordered_df.at[last_row, feature] / target_mean) * reference_mean
            else:
                # Fallback to nearest non-NaN value if the direct reference is NaN
                nearest_non_nan = reordered_df[feature].dropna().iloc[-1] if not reordered_df[feature].dropna().empty else np.nan
                if nearest_non_nan is not np.nan:
                    reordered_df.at[last_row, col] = (nearest_non_nan / target_mean) * reference_mean

    # Update reordered_data with the fully imputed DataFrame for the feature
    reordered_data[feature] = reordered_df

#Step 5: Data fusion function
# def data_fusion(reordered_df, mean_values):
#     fused_data = reordered_df.copy()
#     for i in range(len(fused_data)):
#         dependent_value = np.nan
#         for col in fused_data.columns:
#             if not np.isnan(fused_data.loc[i, col]):
#                 dependent_value = fused_data.loc[i, col]
#                 break
#         if not np.isnan(dependent_value):
#             for col in fused_data.columns:
#                 if col != fused_data.columns[0] and not np.isnan(fused_data.loc[i, col]):
#                     fused_data.loc[i, col] = (
#                         dependent_value / fused_data.loc[i, col]
#                     ) * (mean_values[col] / mean_values[fused_data.columns[0]])
#     return fused_data

# mean_values = {feature: reordered_df.mean() for feature, reordered_df in reordered_data.items()}


def data_fusion(reordered_df, mean_values):
    fused_data = reordered_df.copy()
    for i in range(len(fused_data)):
        dependent_value = np.nan
        for col in fused_data.columns:
            if not np.isnan(fused_data.loc[i, col]):
                dependent_value = fused_data.loc[i, col]
                break
        if not np.isnan(dependent_value):
            for col in fused_data.columns:
                if col != fused_data.columns[0] and not np.isnan(fused_data.loc[i, col]):
                    # Ensure no division by zero and NaN values
                    if fused_data.loc[i, col] != 0:
                        fused_data.loc[i, col] = (
                            dependent_value / fused_data.loc[i, col]
                        ) * (mean_values[col] / mean_values[fused_data.columns[0]])
                    else:
                        # Handle division by zero by setting to a default value (e.g., 0 or the original value)
                        fused_data.loc[i, col] = np.nan  # or some other logic you prefer
    return fused_data

# Assuming reordered_data is a dictionary where the keys are feature names and values are dataframes
mean_values = {feature: reordered_df.mean() for feature, reordered_df in reordered_data.items()}

# Step 6: Apply data fusion
fused_results = {feature: data_fusion(reordered_df, mean_values[feature])
                 for feature, reordered_df in reordered_data.items()}
# Step 7: Optimized Linear Interpolation
def fast_linear_interpolation(data):
    interpolated_data = data.copy()
    for col in range(interpolated_data.shape[1]):
        column_data = interpolated_data[:, col]
        nans = np.isnan(column_data)
        valid = ~nans
        valid_indices = np.where(valid)[0]
        
        if nans.any() and valid.any():
            interpolated_data[nans, col] = np.interp(
                np.flatnonzero(nans),
                valid_indices,
                column_data[valid]
            )
    
    return interpolated_data

# Step 8: Apply optimized linear interpolation to the fused results
interpolated_fused_results = {
    feature: pd.DataFrame(fast_linear_interpolation(fused_results[feature].values), 
                          columns=fused_results[feature].columns)
    for feature in fused_results
}

# Step 9: Define function to compute adjustment matrix based on correlations
def compute_adjustment_matrix(correlations):
    def get_adjustment_value(p):
        if 0 <= abs(p) <= 0.2:
            return 0.5
        elif 0.2 < abs(p) <= 0.4:
            return 0.6
        elif 0.4 < abs(p) <= 0.6:
            return 0.7
        elif 0.6 < abs(p) <= 0.8:
            return 0.8
        elif 0.8 < abs(p) <= 1.0:
            return 0.9
        return 1.0

    num_features = correlations.shape[0]
    adjustment_matrix = np.zeros((num_features, num_features))

    for i in range(num_features):
        for j in range(num_features):
            if i != j:
                adjustment_matrix[i, j] = get_adjustment_value(correlations.iloc[i, j])
            else:
                adjustment_matrix[i, j] = 1  # Identity for self-correlation

    return adjustment_matrix

# Compute the adjustment matrix
adjustment_matrix = compute_adjustment_matrix(correlations)

# Step 10: Define the data recovery function with value capping
def recover_data(fused_data, reordered_df, mean_values, adjustment_matrix, original_min, original_max):
    row, column = fused_data.shape
    recovered_data = np.full((row, column), np.nan)
    
    for i in range(row):
        for j in range(column):
            if not np.isnan(reordered_df.values[i][j]):
                recovered_data[i][j] = reordered_df.values[i][j]
            else:
                for k in range(column):
                    if k != j and not np.isnan(reordered_df.values[i][k]):
                        recovered_value = (
                            fused_data[i][j] * reordered_df.values[i][k] * 
                            mean_values[j] / mean_values[k] * adjustment_matrix[j][k]
                        )
                        recovered_data[i][j] = np.clip(recovered_value, original_min[j], original_max[j])
                        break
    return recovered_data

# Step 11: Compute original min and max values for each feature
# original_min = df.min().values
# original_max = df.max().values

# Step 11: Ensure that original_min and original_max are aligned with the columns in reordered_df
original_min = df.min(axis=0).values  # Min per feature
original_max = df.max(axis=0).values  # Max per feature

# Step 12: Apply data recovery with value capping
recovered_results = {
    feature: pd.DataFrame(
        recover_data(interpolated_fused_results[feature].values, reordered_df, 
                     mean_values[feature], adjustment_matrix, original_min, original_max), 
        columns=reordered_df.columns
    )
    for feature, reordered_df in reordered_data.items()
}

# Step 12: Scale each feature in recovered_results using MinMaxScaler
scaled_recovered_results = {}
scalers = {}  # Dictionary to store scalers for each feature to use on test data later

for feature, recovered_df in recovered_results.items():
    scaler = MinMaxScaler()
    scaled_recovered_results[feature] = pd.DataFrame(
        scaler.fit_transform(recovered_df),
        columns=recovered_df.columns,
        index=recovered_df.index
    )
    scalers[feature] = scaler  # Store the scaler for this feature


# Step 13: Split each DataFrame in reordered_data into training and testing based on NaNs
train_test_data = {}

for feature, reordered_df in reordered_data.items():
    train_test_data[feature] = {}

    for col in reordered_df.columns:
        # Training Data: Rows with observed (non-NaN) values in the current column
        train_data = reordered_df[reordered_df[col].notna()]
        train_indices = train_data.index
        
        # Testing Data: Rows with NaN values in the current column
        test_data = reordered_df[reordered_df[col].isna()]
        test_indices = test_data.index
        
        train_test_data[feature][col] = {
            'train': train_data.reset_index(drop=True),
            'train_indices': train_indices,
            'test': test_data.reset_index(drop=True),
            'test_indices': test_indices
        }

# Initialize a dictionary to store the final train and test data from recovered_results
final_train_test_data = {}

for feature, feature_data in train_test_data.items():
    final_train_test_data[feature] = {}

    # Retrieve the recovered DataFrame for the current feature
    recovered_df = recovered_results[feature]

    for col, split_indices in feature_data.items():
        # Extract train and test indices for the current column from train_test_data
        train_indices = split_indices['train_indices']
        test_indices = split_indices['test_indices']
        
        # Select the training and testing data from recovered_df using the exact indices from train_test_data
        train_data = recovered_df.loc[train_indices]
        test_data = recovered_df.loc[test_indices]
        
        # Store the train and test data for each column in final_train_test_data without altering indices
        final_train_test_data[feature][col] = {
            'train': train_data,
            'train_indices': train_indices,
            'test': test_data,
            'test_indices': test_indices
        }

# Step 2: Combine all scaled training data from final_train_test_data into a single DataFrame
all_train_data = []

for feature, feature_data in final_train_test_data.items():
    for col, data in feature_data.items():
        # Scale the training data using the corresponding scaler
        train_data = scalers[feature].transform(data['train'])
        all_train_data.append(pd.DataFrame(train_data, columns=data['train'].columns))

# Concatenate all training data for BiLSTM training
combined_train_data = pd.concat(all_train_data, axis=0).reset_index(drop=True)

# Reshape for LSTM input [samples, time steps, features]
X_train = combined_train_data.values.reshape((combined_train_data.shape[0], 1, combined_train_data.shape[1]))

# Step 3: Define the BiLSTM Autoencoder Model
input_dim = X_train.shape[2]  # Number of features

# Define MAE metric
def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# Create the model
model = keras.Sequential([
    layers.Input(shape=(1, input_dim)),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.RepeatVector(1),
    layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.TimeDistributed(layers.Dense(input_dim))
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=[mae])

model.summary()

# Step 4: Train the Model
history = model.fit(X_train, X_train, epochs=10, batch_size=32, validation_split=0.2)

# Plot Loss, MAE, and RMSE
plt.figure(figsize=(12,5))

# Loss Plot
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss (MSE)')

# MAE & RMSE Plot
plt.subplot(1,2,2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.title('MAE')

plt.show()

# Step 5: Prepare the test data for BiLSTM prediction
all_test_data = []

for feature, feature_data in final_train_test_data.items():
    for col, data in feature_data.items():
        # Scale the test data using the corresponding scaler
        test_data = scalers[feature].transform(data['test'])
        all_test_data.append(pd.DataFrame(test_data, columns=data['test'].columns))

# Concatenate all scaled test data for BiLSTM prediction
combined_test_data = pd.concat(all_test_data, axis=0).reset_index(drop=True)

# Reshape for LSTM input [samples, time steps, features]
X_test = combined_test_data.values.reshape((combined_test_data.shape[0], 1, combined_test_data.shape[1]))

# Step 6: Make predictions with the BiLSTM model
predicted_test_data = model.predict(X_test)

# Reshape back to original structure for inverse transformation
predicted_test_data = predicted_test_data.reshape((predicted_test_data.shape[0], predicted_test_data.shape[2]))

# Step 7: Inverse transform predicted data to original scale
# Create a dictionary for imputed results structured like recovered_results
# imputed_results = {feature: pd.DataFrame(np.nan, index=recovered_df.index, columns=recovered_df.columns)
                   # for feature, recovered_df in recovered_results.items()}
imputed_results = {feature: recovered_df.copy() for feature, recovered_df in recovered_results.items()}
# Populate `imputed_results` by inserting imputed values for each feature and column
start_idx = 0

for feature, feature_data in final_train_test_data.items():
    scaler = scalers[feature]
    for col, data in feature_data.items():
        num_rows = data['test'].shape[0]
        
        # Apply inverse transform for the predicted test data segment
        imputed_test_segment = scaler.inverse_transform(predicted_test_data[start_idx:start_idx + num_rows])
        imputed_test_df = pd.DataFrame(imputed_test_segment, columns=data['test'].columns, index=data['test'].index)
        
        # Assign imputed values in the original imputed_results structure
        imputed_results[feature].loc[data['test_indices'], col] = imputed_test_df[col].values
        
        start_idx += num_rows

# Display final imputed results
# for feature, imputed_df in imputed_results.items():
#     print(f"Imputed Results for Feature '{feature}':")
#     print(imputed_df)


##########************###############
# Initialize a dictionary to store each column's imputed results grouped together
combined_imputed_dict = {}

# Track the suffix count for each column name to ensure unique names
column_suffix_count = {}

# Iterate through each feature's DataFrame in imputed_results
for feature, imputed_df in imputed_results.items():
    for col in imputed_df.columns:
        # Initialize the suffix count for each unique column
        if col not in column_suffix_count:
            column_suffix_count[col] = 1
        else:
            column_suffix_count[col] += 1

        # Generate a new column name with a unique suffix
        suffix = column_suffix_count[col]
        new_col_name = f"{col}_{suffix}"
        
        # Initialize the DataFrame in combined_imputed_dict for each column if it doesn't exist
        if col not in combined_imputed_dict:
            combined_imputed_dict[col] = pd.DataFrame()
        
        # Add the imputed column to the respective DataFrame in combined_imputed_dict
        combined_imputed_dict[col][new_col_name] = imputed_df[col]

# Display the dictionary with grouped DataFrames
# for col_name, df in combined_imputed_dict.items():
#     print(f"DataFrame for column '{col_name}':")
#     print(df)
#     print("\n")

# # Initialize a dictionary to store the final imputed results (only the last round for each column)
# final_imputed_dict = {}

# # Iterate through each feature's DataFrame in imputed_results
# for feature, imputed_df in imputed_results.items():
#     for col in imputed_df.columns:
#         # Overwrite with the last imputed result for the column
#         final_imputed_dict[col] = imputed_df[col]

# # Combine the final imputed results for each column into a single DataFrame
# final_imputed_df = pd.DataFrame(final_imputed_dict)

# # Save the final imputed results to a CSV file
# final_imputed_df.to_csv('final_imputed_results.csv', encoding='utf-8-sig', index=False)


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize a dictionary to store the final imputed results (only the last round for each column)
final_imputed_dict = {}
error_metrics = {}

# Iterate through each feature's DataFrame in imputed_results
for feature, imputed_df in imputed_results.items():
    for col in imputed_df.columns:
        # Overwrite with the last imputed result for the column
        final_imputed_dict[col] = imputed_df[col]

        # Compute performance metrics (assuming original_df contains the ground truth)
        if col in original_df.columns:
            true_values = original_df[col].dropna()  # Drop NaNs from original values
            imputed_values = imputed_df[col].loc[true_values.index]  # Align indices
            
            # Compute errors only if there are valid values to compare
            if len(true_values) > 0:
                mae = mean_absolute_error(true_values, imputed_values)
                mse = mean_squared_error(true_values, imputed_values)
                rmse = np.sqrt(mse)
                r2 = r2_score(true_values, imputed_values)

                # Compute Pearson Correlation if possible
                pearson_corr = pearsonr(true_values, imputed_values)[0] if len(true_values) > 1 else np.nan

                error_metrics[col] = {
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "R2": r2,
                    "Pearson Correlation": pearson_corr
                }
# Combine the final imputed results for each column into a single DataFrame
final_imputed_df = pd.DataFrame(final_imputed_dict)

# Identify the column with the best performance based on MAE (or change to RMSE/R²)
best_column = min(error_metrics, key=lambda x: error_metrics[x]["MSE"])

# Display results
print(f"Column with the best imputation performance: {best_column}")
print(f"Metrics: {error_metrics[best_column]}")


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from scipy.stats import pearsonr

def evaluate_imputation_whole_data(original, imputed, df, scaler):
    """
    Compute evaluation metrics between original and imputed datasets on the whole data.
    
    Metrics:
    - Pearson Correlation Coefficient
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    
    Args:
        original (pd.DataFrame): The original dataset with missing values.
        imputed (pd.DataFrame): The dataset after imputation.

    Returns:
        dict: Dictionary of computed metric values for the whole dataset.
    """
    # Inverse transform both datasets to revert scaling
    #original = pd.DataFrame(scaler.inverse_transform(original), columns=original.columns)
    #imputed = pd.DataFrame(scaler.inverse_transform(imputed), columns=imputed.columns)

    # Make a copy of the original dataset to replace missing values
    imputed_df = original.copy()

    # Replace the missing values in the original dataset with imputed values
    for col in df.columns:
        missing_indices = df[col].isna()
        imputed_df.loc[missing_indices, col] = imputed[col][missing_indices]  # Substitute imputed values

    # Flatten the data to evaluate all columns as a whole
    original_values = original.values.flatten()
    imputed_values = imputed_df.values.flatten()

    # Remove NaN values (which are in the original data for the missing entries)
    valid_indices = ~np.isnan(original_values)
    original_values = original_values[valid_indices]
    imputed_values = imputed_values[valid_indices]

    # Pearson Correlation Coefficient
    pearson_corr, _ = pearsonr(original_values, imputed_values)

    # Return the results as a dictionary
    return {
        'Pearson Correlation': pearson_corr,
    }
# Assuming `original_df` is your original DataFrame with missing values and
# `imputed_test_data` is your imputed dataset (as a DataFrame)
evaluation_results = evaluate_imputation_whole_data(original_df, pd.DataFrame(final_imputed_dict, columns=original_df.columns), df, scaler)

# Display the results
print(f"Pearson Correlation: {evaluation_results['Pearson Correlation']:.4f}")



# # Display the results
# for feature, metrics in evaluation_results.items():
#     print(f"Feature: {feature}")
#     for metric_name, value in metrics.items():
#         print(f"  {metric_name}: {value:.4f}")
#     print("-" * 40)



# 5. Visual Inspection (Using Plots)
# Boxplot to compare distributions
plt.figure(figsize=(8, 6))
sns.boxplot(data=[original_df.values.flatten(), final_imputed_df.values.flatten()], 
            orient='h', 
            palette="Set2")
plt.yticks([0, 1], ['Original', 'Imputed'])
plt.title('Boxplot Comparison: Original vs Imputed Data')
plt.show()

# Histogram & KDE (Kernel Density Estimation) plots
plt.figure(figsize=(8, 6))
sns.histplot(original_df.values.flatten(), color='blue', kde=True, label='Original', alpha=0.6)
sns.histplot(final_imputed_df.values.flatten(), color='red', kde=True, label='Imputed', alpha=0.6)
plt.legend()
plt.title('Histogram & KDE Comparison')
plt.show()

# Scatter plot to compare original vs imputed data
plt.figure(figsize=(8, 6))
sns.scatterplot(x=original_df.values.flatten(), y=final_imputed_df.values.flatten())
plt.xlabel('Original Values')
plt.ylabel('Imputed Values')
plt.title('Scatter Plot: Original vs Imputed')
plt.show()

# Missingness Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(original_df.isna(), cbar=False, cmap="Blues", cbar_kws={'label': 'Missing Data'})
plt.title('Missingness Heatmap (Original Data)')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.isna(), cbar=False, cmap="Blues", cbar_kws={'label': 'Missing Data'})
plt.title('Missingness Heatmap (MNAR Data)')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(final_imputed_df.isna(), cbar=False, cmap="Blues", cbar_kws={'label': 'Missing Data'})
plt.title('Missingness Heatmap (Imputed Data)')
plt.show()

# # Clustering Heatmap
# sns.clustermap(original_df.corr(), cmap="coolwarm", annot=True)
# plt.title("Hierarchical Clustering Heatmap")
# plt.show()


feature_importances = np.random.rand(original_df.shape[1])  # Replace with actual feature importance values
plt.figure(figsize=(10, 4))
sns.heatmap([feature_importances], annot=False, cmap="coolwarm", xticklabels=original_df.columns, yticklabels=['Feature Importance'])
plt.title("Feature Importance Heatmap")
plt.show()


plt.figure(figsize=(12, 6))
sns.heatmap(original_df.T, cmap="coolwarm", cbar=True)
plt.title("Time-Series Heatmap")
plt.xlabel("Observations")
plt.ylabel("Features")
plt.show()


# cross_tab = pd.crosstab(original_df['Category1'], original_df['Category2'])
# plt.figure(figsize=(8, 6))
# sns.heatmap(cross_tab, annot=True, cmap="coolwarm", linewidths=0.5)
# plt.title("Category Co-Occurrence Heatmap")
# plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(original_df.cov(), annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Covariance Heatmap")
plt.show()


from scipy.spatial.distance import pdist, squareform

distance_matrix = squareform(pdist(original_df.fillna(0), metric="euclidean"))  # Fill NaNs with 0 to compute distance
plt.figure(figsize=(10, 6))
sns.heatmap(distance_matrix, cmap="viridis")
plt.title("Pairwise Distance Heatmap")
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(original_df.corr(), annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
