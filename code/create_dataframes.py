#%%
import pandas as pd
import numpy as np
import os
import glob

# Function to find data files automatically
def find_data_files():
    """Find the 3D pose and fiber photometry data files automatically."""
    # Try multiple possible data directory locations
    possible_data_dirs = [
        'data',  # Current directory
        '../data',  # Parent directory
        './data',  # Explicit current directory
        os.path.join(os.path.dirname(__file__), 'data'),  # Relative to script location
        os.path.join(os.path.dirname(__file__), '..', 'data')  # Parent of script location
    ]
    
    data_dir = None
    for dir_path in possible_data_dirs:
        if os.path.exists(dir_path):
            data_dir = dir_path
            print(f"Found data directory: {data_dir}")
            break
    
    if data_dir is None:
        raise FileNotFoundError(f"Data directory not found. Tried: {possible_data_dirs}")
    
    # Search for 3D pose files
    pose_pattern = os.path.join(data_dir, '*3dpose*.pkl')
    pose_files = glob.glob(pose_pattern)
    
    # Search for fiber photometry files
    fip_pattern = os.path.join(data_dir, '*fip*.pkl')
    fip_files = glob.glob(fip_pattern)
    
    if not pose_files:
        raise FileNotFoundError(f"No 3D pose files found matching pattern: {pose_pattern}")
    
    if not fip_files:
        raise FileNotFoundError(f"No fiber photometry files found matching pattern: {fip_pattern}")
    
    # Use the first file found for each type
    pose_file = pose_files[0]
    fip_file = fip_files[0]
    
    print(f"Found 3D pose file: {pose_file}")
    print(f"Found fiber photometry file: {fip_file}")
    
    return pose_file, fip_file

# Load the dataframes
print("Searching for data files...")
try:
    pose_file, fip_file = find_data_files()
    
    print("Loading dataframes...")
    df_3dpose = pd.read_pickle(pose_file)
    df_fip = pd.read_pickle(fip_file)
    
except Exception as e:
    print(f"Error finding or loading data files: {e}")
    print("Falling back to hardcoded filenames...")
    
    # Try to find the data directory for fallback
    possible_data_dirs = [
        'data', '../data', './data',
        os.path.join(os.path.dirname(__file__), 'data'),
        os.path.join(os.path.dirname(__file__), '..', 'data')
    ]
    
    data_dir = None
    for dir_path in possible_data_dirs:
        if os.path.exists(dir_path):
            data_dir = dir_path
            break
    
    if data_dir:
        df_3dpose = pd.read_pickle(os.path.join(data_dir, 'df_3dpose_718494_2024-12-03_10-29-27.pkl'))
        df_fip = pd.read_pickle(os.path.join(data_dir, 'df_fip_718494_718494_2024-12-03_10-29-27.pkl'))
    else:
        # Last resort - try direct paths
        df_3dpose = pd.read_pickle('data/df_3dpose_718494_2024-12-03_10-29-27.pkl')
        df_fip = pd.read_pickle('data/df_fip_718494_718494_2024-12-03_10-29-27.pkl')

print("Dataframes loaded successfully!")
print(f"df_3dpose shape: {df_3dpose.shape}")
print(f"df_fip shape: {df_fip.shape}")

# Display basic information about the dataframes
print("\n" + "="*50)
print("DF_3DPOSE INFO")
print("="*50)
print(f"Columns: {list(df_3dpose.columns)}")
print(f"Data types:\n{df_3dpose.dtypes}")
print(f"First few rows:\n{df_3dpose.head()}")

print("\n" + "="*50)
print("DF_FIP INFO")
print("="*50)
print(f"Columns: {list(df_fip.columns)}")
print(f"Data types:\n{df_fip.dtypes}")
print(f"First few rows:\n{df_fip.head()}")

# Examine time columns specifically
print("\n" + "="*50)
print("TIME COLUMNS ANALYSIS")
print("="*50)

# Check if time columns exist in both dataframes
time_cols_3dpose = [col for col in df_3dpose.columns if 'time' in col.lower()]
time_cols_fip = [col for col in df_fip.columns if 'time' in col.lower()]

print(f"Time columns in df_3dpose: {time_cols_3dpose}")
print(f"Time columns in df_fip: {time_cols_fip}")

# If time columns exist, examine them
if time_cols_3dpose:
    print(f"\ndf_3dpose time column info:")
    for col in time_cols_3dpose:
        print(f"  {col}: {df_3dpose[col].dtype}")
        print(f"    Range: {df_3dpose[col].min()} to {df_3dpose[col].max()}")
        print(f"    Sample values: {df_3dpose[col].head().tolist()}")

if time_cols_fip:
    print(f"\ndf_fip time column info:")
    for col in time_cols_fip:
        print(f"  {col}: {df_fip[col].dtype}")
        print(f"    Range: {df_fip[col].min()} to {df_fip[col].max()}")
        print(f"    Sample values: {df_fip[col].head().tolist()}")

# Filter df_3dpose to retain only body part coordinate columns
print("\n" + "="*50)
print("FILTERING 3D POSE DATA")
print("="*50)

# Identify body part coordinate columns (x, y, z coordinates)
body_part_columns = []
metadata_columns = []

for col in df_3dpose.columns:
    if col.endswith('_x') or col.endswith('_y') or col.endswith('_z'):
        body_part_columns.append(col)
    else:
        metadata_columns.append(col)

print(f"Body part coordinate columns: {len(body_part_columns)}")
print(f"Metadata columns: {len(metadata_columns)}")
print(f"Metadata columns include: {metadata_columns[:10]}...")  # Show first 10 metadata columns

# Create filtered dataframe with only body part coordinates and essential metadata
essential_columns = ['time', 'fnum', 'filename', 'project']  # Keep essential metadata
filtered_columns = body_part_columns + [col for col in essential_columns if col in df_3dpose.columns]

df_3dpose_filtered = df_3dpose[filtered_columns].copy()

print(f"\nOriginal df_3dpose shape: {df_3dpose.shape}")
print(f"Filtered df_3dpose shape: {df_3dpose_filtered.shape}")
print(f"Columns retained: {len(filtered_columns)}")

# Show some examples of the body part columns
print(f"\nSample body part columns:")
for i, col in enumerate(body_part_columns[:15]):  # Show first 15
    print(f"  {col}")
if len(body_part_columns) > 15:
    print(f"  ... and {len(body_part_columns) - 15} more")

print(f"\nEssential metadata columns retained:")
for col in essential_columns:
    if col in df_3dpose_filtered.columns:
        print(f"  {col}")

# Display first few rows of filtered dataframe
print(f"\nFirst few rows of filtered dataframe:")
print(df_3dpose_filtered.head())

# PCA Analysis of Body Movements
print("\n" + "="*50)
print("PCA ANALYSIS OF BODY MOVEMENTS")
print("="*50)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Prepare data for PCA - use only the coordinate columns (x, y, z)
coordinate_columns = [col for col in df_3dpose_filtered.columns if col.endswith('_x') or col.endswith('_y') or col.endswith('_z')]
print(f"Using {len(coordinate_columns)} coordinate columns for PCA")

# Extract coordinate data
pose_coords = df_3dpose_filtered[coordinate_columns].values
print(f"Pose coordinate data shape: {pose_coords.shape}")

# Standardize the data (important for PCA)
scaler = StandardScaler()
pose_coords_scaled = scaler.fit_transform(pose_coords)
print("Data standardized for PCA")

# Perform PCA with 5 components
pca = PCA(n_components=10)
pca_result = pca.fit_transform(pose_coords_scaled)

print(f"\nPCA Results:")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {pca.explained_variance_ratio_.cumsum()}")

# Create a dataframe with PCA results and time information
df_pca = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(10)])
df_pca['time'] = df_3dpose_filtered['time'].values
df_pca['fnum'] = df_3dpose_filtered['fnum'].values

print(f"\nPCA dataframe shape: {df_pca.shape}")
print(f"PCA dataframe columns: {list(df_pca.columns)}")

# Align with fiber photometry data
print("\n" + "="*50)
print("ALIGNING PCA WITH FIBER PHOTOMETRY DATA")
print("="*50)

# Use time_fip from df_fip as the reference time (it seems to be in the same format as df_3dpose time)
# Interpolate PCA data to match fiber photometry timestamps
from scipy.interpolate import interp1d

# Sort both dataframes by time
df_pca_sorted = df_pca.sort_values('time')
df_fip_sorted = df_fip.sort_values('time_fip')

print(f"PCA time range: {df_pca_sorted['time'].min():.2f} to {df_pca_sorted['time'].max():.2f}")
print(f"FIP time range: {df_fip_sorted['time_fip'].min():.2f} to {df_fip_sorted['time_fip'].max():.2f}")

# Find overlapping time range
time_start = max(df_pca_sorted['time'].min(), df_fip_sorted['time_fip'].min())
time_end = min(df_pca_sorted['time'].max(), df_fip_sorted['time_fip'].max())
print(f"Overlapping time range: {time_start:.2f} to {time_end:.2f}")

# Filter data to overlapping time range
df_pca_overlap = df_pca_sorted[(df_pca_sorted['time'] >= time_start) & (df_pca_sorted['time'] <= time_end)]
df_fip_overlap = df_fip_sorted[(df_fip_sorted['time_fip'] >= time_start) & (df_fip_sorted['time_fip'] <= time_end)]

print(f"Overlapping PCA data points: {len(df_pca_overlap)}")
print(f"Overlapping FIP data points: {len(df_fip_overlap)}")

# Interpolate PCA components to FIP timestamps
pca_interpolated = pd.DataFrame()
pca_interpolated['time'] = df_fip_overlap['time_fip'].values

for pc_col in [f'PC{i+1}' for i in range(10)]:
    interp_func = interp1d(df_pca_overlap['time'], df_pca_overlap[pc_col], 
                          kind='linear', bounds_error=False, fill_value='extrapolate')
    pca_interpolated[pc_col] = interp_func(df_fip_overlap['time_fip'])

# Add fiber photometry signal
pca_interpolated['fip_signal'] = df_fip_overlap['signal'].values

print(f"Interpolated PCA data shape: {pca_interpolated.shape}")

# Note: Plotting code removed as requested

# Print summary statistics
print(f"\nSummary Statistics:")
print(f"PCA Components:")
for i, pc_col in enumerate([f'PC{i+1}' for i in range(10)]):
    print(f"  {pc_col}: mean={pca_interpolated[pc_col].mean():.3f}, std={pca_interpolated[pc_col].std():.3f}")

print(f"FIP Signal: mean={pca_interpolated['fip_signal'].mean():.3f}, std={pca_interpolated['fip_signal'].std():.3f}")

# Calculate correlations between PCs and FIP signal
print(f"\nCorrelations between PCs and FIP signal:")
for pc_col in [f'PC{i+1}' for i in range(10)]:
    corr = pca_interpolated[pc_col].corr(pca_interpolated['fip_signal'])
    print(f"  {pc_col} vs FIP: {corr:.3f}")

# Process df_fip to extract different channels and excitations
print("\n" + "="*50)
print("PROCESSING FIBER PHOTOMETRY DATA")
print("="*50)

# Analyze unique values in channel and excitation columns
unique_channels = df_fip['channel'].unique()
unique_excitations = df_fip['excitation'].unique()
unique_fibers = df_fip['fiber_number'].unique()

print(f"Unique channels: {unique_channels}")
print(f"Unique excitations: {unique_excitations}")
print(f"Unique fiber numbers: {unique_fibers}")

# Create a new dataframe with channel-excitation combinations as columns
print(f"\nCreating new dataframe with channel-excitation combinations as columns...")

# Get unique combinations of channel, excitation, and fiber
combinations = df_fip[['channel', 'excitation', 'fiber_number']].drop_duplicates()
print(f"Found {len(combinations)} unique combinations:")
for idx, row in combinations.iterrows():
    print(f"  Channel: {row['channel']}, Excitation: {row['excitation']}, Fiber: {row['fiber_number']}")

# Create a new dataframe with properly aligned signals
# Since different channels are sampled sequentially at fast rate, we need to align them properly
print(f"\nAligning signals to remove missing values...")

# Use pivot_table with proper aggregation to handle sequential sampling
print(f"Using pivot_table approach for proper alignment...")

# Create the aligned dataframe using pivot_table
df_fip_aligned = df_fip.pivot_table(
    index=['time_fip', 'time', 'frame_number'], 
    columns=['channel', 'excitation', 'fiber_number'], 
    values='signal', 
    aggfunc='first'  # Take first value if duplicates
).reset_index()

# Flatten the multi-level column names
df_fip_aligned.columns = ['time_fip', 'time', 'frame_number'] + [f'signal_{col[0]}_{col[1]}_fiber{col[2]}' for col in df_fip_aligned.columns[3:]]

print(f"Pivot table approach - aligned dataframe shape: {df_fip_aligned.shape}")
print(f"Columns: {list(df_fip_aligned.columns)}")

# Check if we still have missing values and try alternative approach if needed
signal_cols = [col for col in df_fip_aligned.columns if col.startswith('signal_')]
missing_counts = [df_fip_aligned[col].isna().sum() for col in signal_cols]

if any(count > 0 for count in missing_counts):
    print(f"Still have missing values. Trying alternative approach...")
    
    # Alternative approach: Group by time_fip and aggregate
    df_fip_grouped = df_fip.groupby(['time_fip', 'time', 'frame_number']).agg({
        'signal': lambda x: x.tolist(),  # Collect all signals for this time point
        'channel': lambda x: x.tolist(),
        'excitation': lambda x: x.tolist(), 
        'fiber_number': lambda x: x.tolist()
    }).reset_index()
    
    # Create a new dataframe with proper structure
    df_fip_aligned = pd.DataFrame()
    df_fip_aligned['time_fip'] = df_fip_grouped['time_fip']
    df_fip_aligned['time'] = df_fip_grouped['time']
    df_fip_aligned['frame_number'] = df_fip_grouped['frame_number']
    
    # For each combination, extract the signal values
    for idx, row in combinations.iterrows():
        channel = row['channel']
        excitation = row['excitation']
        fiber = row['fiber_number']
        
        col_name = f'signal_{channel}_{excitation}_fiber{fiber}'
        
        # Extract signals for this combination
        signals = []
        for i, (ch_list, ex_list, fib_list, sig_list) in enumerate(zip(
            df_fip_grouped['channel'], 
            df_fip_grouped['excitation'], 
            df_fip_grouped['fiber_number'], 
            df_fip_grouped['signal']
        )):
            # Find matching signal
            signal_val = None
            for j, (ch, ex, fib, sig) in enumerate(zip(ch_list, ex_list, fib_list, sig_list)):
                if ch == channel and ex == excitation and fib == fiber:
                    signal_val = sig
                    break
            signals.append(signal_val)
        
        df_fip_aligned[col_name] = signals

print(f"Aligned dataframe shape: {df_fip_aligned.shape}")
print(f"Columns: {list(df_fip_aligned.columns)}")

# Display summary statistics for each signal column
print(f"\nSummary statistics for each signal column:")
signal_cols = [col for col in df_fip_aligned.columns if col.startswith('signal_')]
for col in signal_cols:
    non_null_count = df_fip_aligned[col].count()
    if non_null_count > 0:
        mean_val = df_fip_aligned[col].mean()
        std_val = df_fip_aligned[col].std()
        print(f"  {col}: {non_null_count} values, mean={mean_val:.3f}, std={std_val:.3f}")
    else:
        print(f"  {col}: No data")

# Display first few rows
print(f"\nFirst few rows of aligned signals:")
print(df_fip_aligned.head())

# Check for any missing values in the signal columns
print(f"\nMissing values in signal columns:")
for col in signal_cols:
    missing_count = df_fip_aligned[col].isna().sum()
    total_count = len(df_fip_aligned)
    print(f"  {col}: {missing_count}/{total_count} missing ({missing_count/total_count*100:.1f}%)")

# Store the new dataframe for further analysis
df_fip_signals = df_fip_aligned.copy()
print(f"\nFinal aligned signals dataframe shape: {df_fip_signals.shape}")

# Fix the signal alignment issue - signals are sampled every 3 time points
print("\n" + "="*50)
print("FIXING SIGNAL ALIGNMENT - CREATING SEPARATE TIME INDICES")
print("="*50)

# Analyze the pattern of non-NaN values
signal_cols = [col for col in df_fip_signals.columns if col.startswith('signal_')]
print(f"Signal columns: {signal_cols}")

# Check the pattern of non-NaN values
for col in signal_cols:
    non_nan_indices = df_fip_signals[col].dropna().index
    if len(non_nan_indices) > 0:
        print(f"{col}: {len(non_nan_indices)} non-NaN values, first few indices: {non_nan_indices[:10].tolist()}")

# Create a new dataframe with separate time indices for each signal
print(f"\nCreating separate time indices for each signal...")

# Get the base time information
base_times = df_fip_signals[['time_fip', 'time', 'frame_number']].copy()

# Create a new dataframe starting with the base time information
df_fip_clean = base_times.copy()

# For each signal column, create a separate time index
for i, col in enumerate(signal_cols):
    # Get non-NaN values and their indices
    signal_data = df_fip_signals[col].dropna()
    signal_indices = signal_data.index
    
    # Create a new time column for this signal
    time_col_name = f'time_{col.replace("signal_", "")}'
    
    # Create time values for this signal (assuming sequential sampling every 3 points)
    # We'll create a continuous time series for each signal
    signal_times = []
    for j, idx in enumerate(signal_indices):
        # Use the original time_fip as base and add small increments for sequential sampling
        base_time = df_fip_signals.loc[idx, 'time_fip']
        # Add small time increment based on signal index (assuming ~1ms between signals)
        signal_time = base_time + (j * 0.001)  # 1ms increments
        signal_times.append(signal_time)
    
    # Create a temporary dataframe for this signal
    temp_df = pd.DataFrame({
        'time_fip': signal_times,
        col: signal_data.values
    })
    
    # Merge with the main dataframe
    df_fip_clean = df_fip_clean.merge(temp_df, on='time_fip', how='left', suffixes=('', f'_{col}'))

# Alternative approach: Create a completely new dataframe with proper time alignment
print(f"\nUsing alternative approach - creating clean dataframe with proper time alignment...")

# Create a new dataframe with continuous time series for each signal
df_fip_clean = pd.DataFrame()

# For each signal, create a separate time series with proper indexing
for i, col in enumerate(signal_cols):
    # Get the signal data (non-NaN values)
    signal_mask = df_fip_signals[col].notna()
    signal_data = df_fip_signals[signal_mask][['time_fip', 'time', 'frame_number', col]].copy()
    
    if len(signal_data) > 0:
        # Create a new time column for this signal with proper indexing
        signal_time_col = f'time_{col.replace("signal_", "")}'
        
        # Create continuous time series for this signal
        signal_times = []
        signal_values = []
        base_times = []
        frame_numbers = []
        
        for j, (_, row) in enumerate(signal_data.iterrows()):
            # Create a continuous time series
            base_time = row['time_fip']
            signal_time = base_time + (j * 0.001)  # 1ms increments
            signal_times.append(signal_time)
            signal_values.append(row[col])
            base_times.append(row['time_fip'])
            frame_numbers.append(row['frame_number'])
        
        # Create a temporary dataframe for this signal
        temp_df = pd.DataFrame({
            signal_time_col: signal_times,
            'time_fip': base_times,
            'time': [row['time'] for _, row in signal_data.iterrows()],
            'frame_number': frame_numbers,
            col: signal_values
        })
        
        # Merge with the main dataframe
        if i == 0:
            df_fip_clean = temp_df.copy()
        else:
            df_fip_clean = df_fip_clean.merge(temp_df, on='time_fip', how='outer', suffixes=('', f'_{col}'))

print(f"Clean dataframe shape: {df_fip_clean.shape}")
print(f"Columns: {list(df_fip_clean.columns)}")

# Check for missing values in the clean dataframe
print(f"\nMissing values in clean dataframe:")
for col in df_fip_clean.columns:
    if col.startswith('signal_'):
        missing_count = df_fip_clean[col].isna().sum()
        total_count = len(df_fip_clean)
        print(f"  {col}: {missing_count}/{total_count} missing ({missing_count/total_count*100:.1f}%)")

# Update the final dataframe
df_fip_signals = df_fip_clean.copy()
print(f"\nFinal clean signals dataframe shape: {df_fip_signals.shape}")

# Recreate df_fip_signals with general frame number and individual channel times
print("\n" + "="*50)
print("RECREATING DF_FIP_SIGNALS WITH INDIVIDUAL CHANNEL TIMING")
print("="*50)

# Create a new dataframe that preserves individual timing for each channel
print("Creating new dataframe with individual channel timing...")

# Use pivot_table to create signal columns and time columns separately
print("Creating signal columns...")
df_signals = df_fip.pivot_table(
    index=['frame_number'], 
    columns=['channel', 'excitation', 'fiber_number'], 
    values='signal', 
    aggfunc='first'
).reset_index()

# Flatten signal column names
signal_columns = ['frame_number']
for col in df_signals.columns[1:]:
    signal_columns.append(f'signal_{col[0]}_{col[1]}_fiber{col[2]}')
df_signals.columns = signal_columns

print("Creating time columns...")
df_times = df_fip.pivot_table(
    index=['frame_number'], 
    columns=['channel', 'excitation', 'fiber_number'], 
    values='time_fip', 
    aggfunc='first'
).reset_index()

# Flatten time column names
time_columns = ['frame_number']
for col in df_times.columns[1:]:
    time_columns.append(f'time_{col[0]}_{col[1]}_fiber{col[2]}')
df_times.columns = time_columns

# Merge signals and times
df_fip_signals_new = df_signals.merge(df_times, on='frame_number', how='left')

# Add general time information (from the original time column)
time_info = df_fip[['frame_number', 'time']].drop_duplicates().sort_values('frame_number')
df_fip_signals_new = df_fip_signals_new.merge(time_info, on='frame_number', how='left')

print(f"New df_fip_signals shape: {df_fip_signals_new.shape}")
print(f"Columns: {list(df_fip_signals_new.columns)}")

# Check for missing values in the new dataframe
signal_cols = [col for col in df_fip_signals_new.columns if col.startswith('signal_')]
print(f"\nMissing values in new signal columns:")
for col in signal_cols:
    missing_count = df_fip_signals_new[col].isna().sum()
    total_count = len(df_fip_signals_new)
    print(f"  {col}: {missing_count}/{total_count} missing ({missing_count/total_count*100:.1f}%)")

# Display summary statistics for each signal column
print(f"\nSummary statistics for each signal column:")
for col in signal_cols:
    non_null_count = df_fip_signals_new[col].count()
    if non_null_count > 0:
        mean_val = df_fip_signals_new[col].mean()
        std_val = df_fip_signals_new[col].std()
        print(f"  {col}: {non_null_count} values, mean={mean_val:.3f}, std={std_val:.3f}")
    else:
        print(f"  {col}: No data")

# Display first few rows
print(f"\nFirst few rows of new df_fip_signals:")
print(df_fip_signals_new.head())

# Update the final dataframe
df_fip_signals = df_fip_signals_new.copy()
print(f"\nFinal recreated df_fip_signals shape: {df_fip_signals.shape}")

# Extract session_id from filenames and save dataframes
print("\n" + "="*50)
print("SAVING DATAFRAMES TO RESULTS FOLDER")
print("="*50)

# Extract session_id from the original dataframes
session_id_3dpose = df_3dpose['filename'].iloc[0] if 'filename' in df_3dpose.columns else 'unknown'
session_id_fip = df_fip['session'].iloc[0] if 'session' in df_fip.columns else 'unknown'

print(f"Session ID from 3D pose data: {session_id_3dpose}")
print(f"Session ID from FIP data: {session_id_fip}")

# Use the FIP session_id as it's more complete
session_id = session_id_fip
print(f"Using session ID: {session_id}")

# Create data directory if it doesn't exist
import os
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Created data directory: {data_dir}")

# Save df_fip_signals
fip_signals_filename = f'df_fip_signals_{session_id}.pkl'
fip_signals_path = os.path.join(data_dir, fip_signals_filename)
df_fip_signals.to_pickle(fip_signals_path)
print(f"Saved df_fip_signals to: {fip_signals_path}")
print(f"  Shape: {df_fip_signals.shape}")
print(f"  Columns: {list(df_fip_signals.columns)}")

# Save cleaned behavioral traces (PCA data)
behavioral_traces_filename = f'df_behavioral_traces_{session_id}.pkl'
behavioral_traces_path = os.path.join(data_dir, behavioral_traces_filename)
df_pca.to_pickle(behavioral_traces_path)
print(f"Saved behavioral traces (PCA data) to: {behavioral_traces_path}")
print(f"  Shape: {df_pca.shape}")
print(f"  Columns: {list(df_pca.columns)}")

# Also save the filtered 3D pose data for reference
pose_filtered_filename = f'df_3dpose_filtered_{session_id}.pkl'
pose_filtered_path = os.path.join(data_dir, pose_filtered_filename)
df_3dpose_filtered.to_pickle(pose_filtered_path)
print(f"Saved filtered 3D pose data to: {pose_filtered_path}")
print(f"  Shape: {df_3dpose_filtered.shape}")

print(f"\nAll dataframes saved successfully to {data_dir}/ folder!")

#%%