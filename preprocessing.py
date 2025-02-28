import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pygrinder import mcar, mar_logistic, mnar_x, mnar_t


from scipy import stats
#from pandas_profiling import ProfileReport

plt.rcParams['font.sans-serif'] = ['SimHei']  # Supports Chinese characters
plt.rcParams['axes.unicode_minus'] = False  # Avoid issues with negative numbers

# Load the dataset
#df = pd.read_excel("mis_iris_data.xlsx")
#df = pd.read_excel("test.xlsx")
aki_data = pd.read_csv("aki_icu_dataset.csv")
#df = pd.read_csv("collected_processed_dataset_unix.csv")
#df = pd.read_csv("aki_processed_dataset_unix.csv")

aki_data.info()

# rename columns, if necessary
# changing all column names to lower case, and remove special characters and spacing.
aki_data.columns = aki_data.columns.str.lower()
aki_data.columns = aki_data.columns.str.replace(' |/','_')
aki_data.columns

#missing data
pd.set_option('display.max_columns', 100)
aki_data.head()


aki_data.drop(columns=[ 'subject_id',
                        'hadm_id',
                       'stay_id',
                       'gender','dod',
                       'admittime','dischtime','ethnicity','first_hosp_stay',
                       'icu_intime','icu_outtime','first_icu_stay',
                       'charttime_aki', 'charttime_rrt'
                       ],inplace=True)

# Print the filtered DataFrame or use it for further processing
print(aki_data.head())


columns_to_fill = [
    'creat_low_past_7day',	
    'creat_low_past_48hr',
    'creat',
    'aki_stage_creat',	
    'uo_rt_6hr',	
    'uo_rt_12hr',
    'uo_rt_24hr',
    'aki_stage_uo'
]

aki_data[columns_to_fill] = aki_data[columns_to_fill].fillna(0)

# clinician's input

# List of columns to filter
desired_columns = [
    'lactate_max', 
    'ph_min', 
    'platelets_min', 
    'inr_max', 
    'calcium_min', 
    'ptt_max',    
    'creat_low_past_7day',	
    'creat_low_past_48hr',
    'creat',
    'aki_stage_creat',	
    'uo_rt_6hr',	
    'uo_rt_12hr',
    'uo_rt_24hr',
    'aki_stage_uo'
]

# Filter the DataFrame to include only the desired columns
aki_data = aki_data[desired_columns]

# Print the filtered DataFrame or use it for further processing
print(aki_data.head())

#missing data
pd.isna(aki_data).sum()


aki_data.shape

aki_data = aki_data.dropna(axis=0)

# Count columns with complete (non-missing) data
complete_columns = (aki_data.isna().sum() == 0).sum()

print(f"Number of columns with complete data: {complete_columns}")

# Drop columns with any missing values
aki_data = aki_data.dropna(axis=1)


# Save the processed dataset
aki_data.to_csv('processed_aki_dataset.csv', encoding='utf-8-sig', index=False)


# Load the processed dataset
aki_data = pd.read_csv('processed_aki_dataset.csv')

def introduce_mar(df, missing_fraction):
    # Define missingness parameters
    obs_rate = 0.3  # 30% of variables are fully observed
    missing_rate = 0.2  # 20% missing values in remaining variables
    dataframe = df
    df = df.to_numpy()

    # Create random missing values (MAR case) with a logistic model. 
    df_mar = mar_logistic(df, obs_rate, missing_rate)
    df_mar = pd.DataFrame(df_mar, columns=dataframe.columns)

    return df_mar

# Function to introduce MCAR missing values
def introduce_mcar(df, missing_fraction):
    # Define missingness probability
    p = 0.2  # 20% probability of missing values

    #Create completely random missing values (MCAR case).
    # Apply MCAR missing value generation with 20% missing data
    dataframe = df
    df = df.to_numpy()

    df_mcar = mcar(df, p)
    df_mcar = pd.DataFrame(df_mcar, columns=dataframe.columns)

    return df_mcar

# Function to introduce MNAR missing values
def introduce_mnar(df, missing_fraction):
    num_samples, n_features = df.shape
    dataframe = df
    df = df.to_numpy()
    df = df.reshape(num_samples, 1, -1)
    df_mnar = mnar_t(df, cycle=20, pos=10, scale=3)
    df_mnar = df_mnar.reshape(num_samples, -1)
    df_mnar = pd.DataFrame(df_mnar, columns=dataframe.columns)
    return df_mnar


# # Function to introduce MCAR missing values
# def introduce_mcar(df, missing_fraction):
#     df_mcar = df.copy()
#     # Randomly select a fraction of the data to be NaN
#     mask = np.random.rand(*df.shape) < missing_fraction
#     df_mcar[mask] = np.nan
#     return df_mcar

# # Function to introduce MAR missing values
# def introduce_mar(df, missing_fraction):
#     df_mar = df.copy()
#     # Introduce NaNs based on multiple variables (using a logistic function)
#     for column in df.columns:
#         if df[column].isnull().sum() == 0:  # Ensure the column has no missing values
#             # Use multiple variables to determine probability of missingness
#             prob_missing = (
#                 missing_fraction * 
#                 ((df['lactate_max'] > df['lactate_max'].median()).astype(int) +
#                  (df['ph_min'] < df['ph_min'].median()).astype(int) +
#                  (df['platelets_min'] < df['platelets_min'].median()).astype(int))
#                 / 3)  # Average of binary indicators
            
#             random_effect = np.random.rand(len(df))
#             mask = random_effect < prob_missing
#             df_mar.loc[mask, column] = np.nan
#     return df_mar

# # Function to introduce MNAR missing values
# def introduce_mnar(df, missing_fraction):
#     df_mnar = df.copy()
#     # Introduce NaNs based on the values themselves with a nonlinear relationship
#     for column in df.columns:
#         if df[column].isnull().sum() == 0:  # Ensure the column has no missing values
#             # Nonlinear probability of missingness based on column values
#             prob_missing = np.clip((df[column] / df[column].max())**2 * missing_fraction, 0, 1)
#             random_effect = np.random.rand(len(df))
#             mask = random_effect < prob_missing
#             df_mnar.loc[mask, column] = np.nan
#     return df_mnar

# # Example usage
# # Set the fraction of missing data to introduce
missing_fraction = 0.2  # 20% missing data

# Simulate each missing data mechanism
aki_data_mcar = introduce_mcar(aki_data, missing_fraction)
aki_data_mar = introduce_mar(aki_data, missing_fraction)
aki_data_mnar = introduce_mnar(aki_data, missing_fraction)

# Save the datasets with missing values
aki_data_mcar.to_csv('aki_data_mcar.csv', index=False)
aki_data_mar.to_csv('aki_data_mar.csv', index=False)
aki_data_mnar.to_csv('aki_data_mnar.csv', index=False)

print("Simulated datasets with missing values saved.")
