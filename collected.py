import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pypots.data import load_specific_dataset
from pygrinder import mcar
from pypots.utils.metrics import calc_mae
from sklearn.metrics import mean_squared_error
import seaborn as sns
from pypots.imputation import SAITS, iTransformer, CSDI, FreTS, DLinear, FiLM, USGAN


plt.rcParams['font.sans-serif'] = ['SimHei']  # Supports Chinese characters
plt.rcParams['axes.unicode_minus'] = False  # Avoid issues with negative numbers


# Step 1: Mount Google Drive
#from google.colab import drive
#drive.mount('/content/drive')

# Step 2: Load the Excel file from Google Drive
#file_path = '/content/drive/MyDrive/mis_iris_data.xlsx'

# Use pandas to read the Excel file
#df = pd.read_excel("test.xlsx")
df = pd.read_csv("collected_processed_dataset_unix.csv")
#df = pd.read_csv("aki_processed_dataset_unix.csv")

# Convert non-numeric columns to numeric
# Step 1: Identify non-numeric columns
non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
print("\nNon-Numeric Columns:", non_numeric_cols)

#df['timestamp'] = pd.to_datetime(df['上机时间'])
#df['timestamp_unix'] = df['timestamp'].astype('int64') // 10**9  # Convert nanoseconds to seconds

encoder = LabelEncoder()

for col in df.select_dtypes(include=['object', 'category']).columns:
    df[col] = df[col].astype(str)  # Convert all to string
    #df[col] = encoder.fit_transform(df[col])

for col in non_numeric_cols:
    df[col] = encoder.fit_transform(df[col])

# Step 3: Handle any other non-numeric columns (if any) by coercing errors to NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Check the data types again after conversion
print("\nData Types After Conversion:\n", df.dtypes)

#df = df.drop(df.columns[df.notna().all()], axis=1)
# Step 3: Preprocess the data
# Extract the feature columns
X_ori = df.iloc[:].values  # Features
#y = df.iloc[:].values   # Target
# Standardize the features
X = StandardScaler().fit_transform(X_ori)


print(X.shape)

# Reshape the data to fit the input format of the SAITS model (num_samples, 1, n_features)
num_samples, n_features = X.shape
X = X.reshape(num_samples, 1, n_features)

# Prepare dataset for model input
dataset = {"X": X}

# Fill NaN values with the mean of each column
df = df.apply(lambda col: col.fillna(col.mean()), axis=0)



# # Step 5: Initialize the SAITS model
# saits = SAITS(
#     n_steps=1,  # One time step per sample
#     n_features=n_features,  # Number of features
#     n_layers=2,  # Number of layers in the model
#     d_model=128,  # Model dimensionality
#     n_heads=4,  # Number of attention heads
#     d_k=64,  # Key dimension
#     d_v=64,  # Value dimension
#     d_ffn=128,  # Feed-forward network dimension
#     dropout=0.1,  # Dropout rate
#     epochs=10  # Number of epochs for training
# )

# # Step 6: Train the model on the dataset
# saits.fit(dataset)

# # Step 7: Impute the missing values
# imputation = saits.impute(dataset)

# # Step 5: Initialize the SAITS model
# itransformer = iTransformer(
#     # n_steps=1,  # One time step per sample
#     # n_features=n_features,  # Number of features
#     # n_layers=2,  # Number of layers in the model
#     # llm_model_type=LLaMA,
#     # d_model=128,  # Model dimensionality
#     # n_heads=4,  # Number of attention heads
#     # d_k=64,  # Key dimension
#     # d_v=64,  # Value dimension
#     # d_ffn=128,  # Feed-forward network dimension
#     # patch_len=8,
#     # stride=4,
#     # d_llm=4096,
#     # dropout=0.1,
#     # domain_prompt_content="An imputation model for robust handling of missing data in multivariate longitudinal medical datasets.",
#     # epochs=100,  # Number of epochs for training
#     # batch_size=32

#     n_steps=1, n_features=n_features, n_layers=2, d_model=128, n_heads=4, d_k=64, d_v=64, d_ffn=128, dropout=0,  epochs=100,  # Number of epochs for training
#     batch_size=32

#     )
# # Step 6: Train the model on the dataset
# itransformer.fit(dataset)

# # Step 7: Impute the missing values
# imputation = itransformer.impute(dataset)


# csdi = CSDI(
#     n_steps=1,  # One time step per sample
#     n_features=n_features,  # Number of features
#     n_layers=2,  # Number of layers in the model
#     n_heads=4,  # Number of attention heads
#     n_channels=64,
#     d_time_embedding=16,
#     d_feature_embedding=32,
#     d_diffusion_embedding=64,
#     epochs=10,  # Number of epochs for training
#     batch_size=32

#     )
# # Step 6: Train the model on the datasets
# csdi.fit(dataset)

# # Step 7: Impute the missing values
# imputation = csdi.impute(dataset)

# Initialize FreTS model
# frets = FreTS(
#     n_steps=1,
#     n_features=n_features,
#     embed_size=128,        # Size of the embedding layer
#     hidden_size=256,       # Hidden layer size
#     channel_independence=False,  # Option to use channel independence
#     ORT_weight=1,          # ORT loss weight
#     MIT_weight=1,          # MIT loss weight
#     batch_size=32,         # Batch size
#     epochs=10,            # Number of epochs for training

# )

# frets.fit(dataset)
# imputation = frets.impute(dataset)

# dlinear = DLinear(
#     n_steps=1,  # Number of time steps in your time-series data, adjust as per your dataset
#     n_features=n_features,  # Number of features (variables) in your time-series data, adjust accordingly
#     moving_avg_window_size=5,  # Size of the moving average window
#     individual=False,  # Whether to make a linear layer for each feature individually, False means shared
#     d_model=64,  # Embedding dimension for non-individual mode, adjust based on your model complexity
#     ORT_weight=1.0,  # Weight for the ORT loss function, adjust if necessary
#     MIT_weight=1.0,  # Weight for the MIT loss function, adjust as needed
#     batch_size=32,  # Batch size for training
#     epochs=100,  # Number of training epochs

# )

# dlinear.fit(dataset)
# imputation = dlinear.impute(dataset)

# film = FiLM(
#     n_steps=1,  # The number of time steps in the time-series data sample (adjust based on your sequence length)
#     n_features=n_features,  # The number of features in the time-series data sample (adjust based on your dataset)
#     window_size = [3, 5, 7],  # Example of window sizes for the HiPPO projection layers
#     multiscale = [1, 2, 4],  # Example of multiscale factors for the HiPPO projection layers
#     dropout = 0.3,  # Dropout ratio for HiPPO projection layers
#     mode_type = 1,  # Mode type of SpectralConv1d layers (0, 1, or 2)
#     d_model = 128,  # Dimension of the model
#     ORT_weight = 1.0,  # Weight for the ORT loss
#     MIT_weight = 1.0,  # Weight for the MIT loss
#     batch_size = 32,  # Batch size for training
#     epochs = 100,  # Number of epochs for training
 
# )

# # Fit the model with training and validati

# film.fit(dataset)
# imputation = film.impute(dataset)

usgan = USGAN(
    n_steps=1,                      # Number of time steps in the time-series data sample. Adjust based on your dataset.
    n_features=n_features,                   # Number of features in the time-series data sample. Adjust as needed.
    rnn_hidden_size=64,              # Hidden size of the RNN cell. A typical value could be 64 or 128 depending on model complexity.
    lambda_mse=1.0,                  # The weight of the reconstruction loss. Typically set to 1, but can be adjusted.
    hint_rate=0.7,                   # The hint rate for the discriminator. Can be adjusted based on experimentation.
    dropout=0.2,                     # Dropout rate for the last layer in the Discriminator. This helps prevent overfitting.
    G_steps=1,                       # Number of steps to train the generator per iteration. Typically 1, but you can experiment.
    D_steps=1,                       # Number of steps to train the discriminator per iteration. Also typically 1.
    batch_size=32,                   # Batch size for training. Typical values are 32 or 64, depending on memory.
    epochs=10,                      # Number of epochs for training. This depends on your dataset size and convergence.
)

usgan.fit(dataset)
imputation = usgan.impute(dataset)


# Step 8: Create a mask indicating which values were originally missing
indicating_mask = np.isnan(X) ^ np.isnan(X_ori)

# Step 9: Calculate the Mean Absolute Error (MAE) for imputation performance
# Reshape the imputed data to match the ground truth shape
imputation_reshaped = imputation.reshape(num_samples, n_features)

# Create a mask indicating which values were originally missing (should match the shape of imputation_reshaped)
indicating_mask = np.isnan(X_ori)

# Now calculate MAE using the reshaped imputed data and original ground truth
mae = calc_mae(imputation_reshaped, np.nan_to_num(X_ori), indicating_mask)

# Calculate MSE and RMSE
mse = mean_squared_error(np.nan_to_num(X_ori)[indicating_mask], imputation_reshaped[indicating_mask])
rmse = np.sqrt(mse)  # RMSE is the square root of MSE

# Print the metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Define the folder in Google Drive where you want to save the model
#save_path = '/content/drive/MyDrive/test/saits_imputed_model.pypots'

# Step 10: Save the trained model for future use
#saits.save(save_path)

# Reload the model if needed
# saits.load(save_path)

# Step 11: Reconstruct the dataset after imputation

# Reconstruct the dataset by replacing missing values with imputed values
X_imputed = X_ori.copy()

# Fill the missing values with the imputed values where the original data was NaN
X_imputed[np.isnan(X_ori)] = imputation_reshaped[np.isnan(X_ori)]

# Convert the imputed dataset back to a DataFrame
df_imputed = pd.DataFrame(X_imputed, columns=df.columns[:])

# Append the target column (y) if needed
#df_imputed['Target'] = y

# Output the imputed dataset
print("Imputed Dataset:")
print(df_imputed.head())

# Save the imputed dataset to a file (e.g., CSV or Excel)
#df_imputed.to_csv('/content/drive/MyDrive/test/imputed_iris_data.csv', index=False)
#df_imputed.to_excel('/content/drive/MyDrive/test/imputed_iris_data.xlsx', index=False)

df_imputed.to_excel('imputed_iris_data.xlsx', index=False)

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# Example: Assuming `original_data` and `imputed_data` are your original and imputed datasets.
# These should be 1D arrays or Series of the same length.
original_data = df.values.flatten()  # Flatten the original dataset
imputed_data = df_imputed.values.flatten()  # Flatten the imputed dataset


# Remove NaN values from both arrays
mask = ~np.isnan(original_data) & ~np.isnan(imputed_data)
valid_original = original_data[mask]
valid_imputed = imputed_data[mask]

# Compute Pearson correlation coefficient
if len(valid_original) > 1 and len(valid_imputed) > 1:  # Ensure enough valid data
    pearson_corr, _ = pearsonr(valid_original, valid_imputed)
    print(f"Pearson Correlation Coefficient: {pearson_corr}")
else:
    print("Not enough valid data to compute Pearson correlation.")



# Boxplot to compare distributions
plt.figure(figsize=(8, 6))
sns.boxplot(data=[df.values.flatten(), df_imputed.values.flatten()], 
            orient='h', 
            palette="Set2")
plt.yticks([0, 1], ['Original', 'Imputed'])
plt.title('Boxplot Comparison: Original vs Imputed Data')
plt.show()

# Histogram & KDE (Kernel Density Estimation) plots
plt.figure(figsize=(8, 6))
sns.histplot(df.values.flatten(), color='blue', kde=True, label='Original', alpha=0.6)
sns.histplot(df_imputed.values.flatten(), color='red', kde=True, label='Imputed', alpha=0.6)
plt.legend()
plt.title('Histogram & KDE Comparison')
plt.show()

# Scatter plot to compare original vs imputed data
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df.values.flatten(), y=df_imputed.values.flatten())
plt.xlabel('Original Values')
plt.ylabel('Imputed Values')
plt.title('Scatter Plot: Original vs Imputed')
plt.show()

# Missingness Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.isna(), cbar=False, cmap="Blues", cbar_kws={'label': 'Missing Data'})
plt.title('Missingness Heatmap (Original Data)')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df_imputed.isna(), cbar=False, cmap="Blues", cbar_kws={'label': 'Missing Data'})
plt.title('Missingness Heatmap (Imputed Data)')
plt.show()

# # Clustering Heatmap
# sns.clustermap(original_df.corr(), cmap="coolwarm", annot=True)
# plt.title("Hierarchical Clustering Heatmap")
# plt.show()