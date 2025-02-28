import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pypots.data import load_specific_dataset
from pygrinder import mcar, mnar_x, mar_logistic, mnar_t
from pypots.imputation import SAITS, iTransformer, CSDI, FreTS, DLinear, FiLM, USGAN
from pypots.utils.metrics import calc_mae
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # Supports Chinese characters
plt.rcParams['axes.unicode_minus'] = False  # Avoid issues with negative numbers

# Step 1: Mount Google Drive
#from google.colab import drive
#drive.mount('/content/drive')

# Step 2: Load the Excel file from Google Drive
#file_path = '/content/drive/MyDrive/mis_iris_data.xlsx'

# Use pandas to read the Excel file
#df = pd.read_excel("test.xlsx")
df = pd.read_csv("processed_aki_dataset.csv")
#df = pd.read_csv("aki_processed_dataset_unix.csv")

# # Step 3: Preprocess the data
# # Extract the feature columns
# X_ori = df.iloc[:].values  # Features
# #y = df.iloc[:].values   # Target
# # Standardize the features
# X = StandardScaler().fit_transform(X_ori)

# # Step 4: randomly hold out 10% observed values as ground truth
# X = mcar(X, 0.2)

# print(X.shape)

# # Reshape the data to fit the input format of the SAITS model (num_samples, 1, n_features)
# num_samples, n_features = X.shape
# X = X.reshape(num_samples, 1, n_features)

# # Prepare dataset for model input
# dataset = {"X": X}


# X = data['X']
# num_samples = len(X['RecordID'].unique())
# X = X.drop(['RecordID', 'Time'], axis = 1)
X = df # Features
num_samples, n_features = X.shape
X = StandardScaler().fit_transform(X.to_numpy())
X = X.reshape(num_samples, 1, -1)
#X_ori = X.reshape(num_samples, 1, -1) # keep X_ori for validation
X_ori = X
#X = mcar(X, 0.2)  # randomly hold out 10% observed values as ground truth
#X = mnar_x(X, offset=0.2)
obs_rate = 0.3  # 30% of variables are fully observed
missing_rate = 0.2  # 20% missing values in remaining variables

#X = mar_logistic(X, obs_rate, missing_rate)
X = mnar_t(X, cycle=20, pos=10, scale=1)
#X = X.reshape(num_samples, 1, -1)



dataset = {"X": X}  # X for model input
print(X.shape)  # (11988, 48, 37), 11988 samples and each sample has 48 time steps, 37 features


# Step 5: Initialize the SAITS model
saits = SAITS(
    n_steps=1,  # One time step per sample
    n_features=n_features,  # Number of features
    n_layers=2,  # Number of layers in the model
    d_model=128,  # Model dimensionality
    n_heads=4,  # Number of attention heads
    d_k=64,  # Key dimension
    d_v=64,  # Value dimension
    d_ffn=128,  # Feed-forward network dimension
    dropout=0.1,  # Dropout rate
    epochs=10  # Number of epochs for training
)

# Step 6: Train the model on the dataset
saits.fit(dataset)

# Step 7: Impute the missing values
imputation = saits.impute(dataset)

# # Step 5: Initialize the SAITS model
# itransformer = iTransformer(
#     n_steps=1, 
#     n_features=n_features,
#     n_layers=2, 
#     d_model=128, 
#     n_heads=4, 
#     d_k=64, 
#     d_v=64, 
#     d_ffn=128, 
#     dropout=0,  
#     epochs=10,  # Number of epochs for training
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

# # Initialize FreTS model
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
#     epochs=10,  # Number of training epochs

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
#     epochs = 10,  # Number of epochs for training
 
# )

# # Fit the model with training and validati

# film.fit(dataset)
# imputation = film.impute(dataset)

# usgan = USGAN(
#     n_steps=1,                      # Number of time steps in the time-series data sample. Adjust based on your dataset.
#     n_features=n_features,                   # Number of features in the time-series data sample. Adjust as needed.
#     rnn_hidden_size=64,              # Hidden size of the RNN cell. A typical value could be 64 or 128 depending on model complexity.
#     lambda_mse=1.0,                  # The weight of the reconstruction loss. Typically set to 1, but can be adjusted.
#     hint_rate=0.7,                   # The hint rate for the discriminator. Can be adjusted based on experimentation.
#     dropout=0.2,                     # Dropout rate for the last layer in the Discriminator. This helps prevent overfitting.
#     G_steps=1,                       # Number of steps to train the generator per iteration. Typically 1, but you can experiment.
#     D_steps=1,                       # Number of steps to train the discriminator per iteration. Also typically 1.
#     batch_size=32,                   # Batch size for training. Typical values are 32 or 64, depending on memory.
#     epochs=10,                      # Number of epochs for training. This depends on your dataset size and convergence.
# )

# usgan.fit(dataset)
# imputation = usgan.impute(dataset)
indicating_mask = np.isnan(X_ori)  # Corrected mask for imputation error calculation

# Create a mask indicating which values were originally missing
indicating_mask = np.isnan(X) ^ np.isnan(X_ori)  # indicating mask for imputation error calculation

#indicating_mask = np.isnan(df.values)  # Ensuring it's based on the original dataframe
print(f"Total Missing Values: {np.sum(indicating_mask)}")  # Should be > 0

# Reshape the imputed data to match the ground truth shape
imputation_reshaped = imputation.reshape(num_samples, n_features)

# Extract the valid indices where the original data was not missing
valid_indices = np.where(~indicating_mask)

# # Calculate MAE using the reshaped imputed data and original ground truth
# mae = calc_mae(imputation, np.nan_to_num(X_ori), indicating_mask)

# Calculate MSE and RMSE only on valid indices
if valid_indices[0].size > 0:  # Check if there are valid indices
    mae = mean_absolute_error(np.nan_to_num(X_ori)[valid_indices], imputation[valid_indices])
    mse = mean_squared_error(np.nan_to_num(X_ori)[valid_indices], imputation[valid_indices])
    rmse = np.sqrt(mse)  # RMSE is the square root of MSE

    # Print the metrics
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
else:
    print("No valid indices to calculate MSE and RMSE.")


# Define the folder in Google Drive where you want to save the model
#save_path = '/content/drive/MyDrive/test/saits_imputed_model.pypots'

# Step 10: Save the trained model for future use
#saits.save(save_path)

# Reload the model if needed
# saits.load(save_path)

# # Step 11: Reconstruct the dataset after imputation

# # Reconstruct the dataset by replacing missing values with imputed values
# X_imputed = X_ori.copy()

# # Fill the missing values with the imputed values where the original data was NaN
# X_imputed[np.isnan(X_ori)] = imputation[np.isnan(X_ori)]

# X_imputed = X_imputed.reshape(num_samples, n_features)
# # Convert the imputed dataset back to a DataFrame
# df_imputed = pd.DataFrame(X_imputed, columns=df.columns[:])


# Step 11: Reconstruct the dataset after imputation

# Reconstruct the dataset by replacing missing values with imputed values
X_imputed = X_ori.copy()

# Fill the missing values with the imputed values where the original data was NaN
X_imputed[np.isnan(X)] = imputation[np.isnan(X)]

X_imputed = X_imputed.reshape(X_imputed.shape[0], -1)


import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# Example: Assuming `original_data` and `imputed_data` are your original and imputed datasets.
# These should be 1D arrays or Series of the same length.
original_data = X_ori.flatten()  # Flatten the original dataset
imputed_data = X_imputed.flatten()  # Flatten the imputed dataset


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



from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Assuming `scaler` is the StandardScaler used for initial transformation
scaler = StandardScaler()

# Convert 3D to 2D
X_ori = X_ori.reshape(X_ori.shape[0], -1)  # Shape: (samples, features)
scaler.fit(X_ori)  # Fit on original data before scaling



# Revert scaling
X_ori_unscaled = scaler.inverse_transform(X_ori)  # Original values in real-world units
X_imputed_unscaled = scaler.inverse_transform(X_imputed)  # Imputed values in real-world units

# Compute errors in the original scale
mse_original = mean_squared_error(X_ori_unscaled[~np.isnan(X_ori_unscaled)], X_imputed_unscaled[~np.isnan(X_ori_unscaled)])
rmse_original = np.sqrt(mse_original)
mae_original = mean_absolute_error(X_ori_unscaled[~np.isnan(X_ori_unscaled)], X_imputed_unscaled[~np.isnan(X_ori_unscaled)])

print(f"Original Scale - MSE: {mse_original}, RMSE: {rmse_original}, MAE: {mae_original}")

from scipy.stats import pearsonr

# Flatten and remove NaN values
mask = ~np.isnan(X_ori_unscaled) & ~np.isnan(X_imputed_unscaled)
valid_ori = X_ori_unscaled[mask]
valid_imputed = X_imputed_unscaled[mask]

# Compute Pearson correlation
pearson_corr, _ = pearsonr(valid_ori, valid_imputed)
print(f"Pearson Correlation (Original Scale): {pearson_corr}")




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

# # Clustering Heatmap
# sns.clustermap(original_df.corr(), cmap="coolwarm", annot=True)
# plt.title("Hierarchical Clustering Heatmap")
# plt.show()