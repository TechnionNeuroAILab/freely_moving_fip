"""
Analysis script for fiber photometry and behavioral data correlation.
Loads processed data and creates plots of PCA components with FIP channels.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import os
import glob

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def find_processed_data_files():
    """Find the processed data files in the data directory."""
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
    
    if data_dir is None:
        raise FileNotFoundError(f"Data directory not found. Tried: {possible_data_dirs}")
    
    # Find the processed files
    fip_pattern = os.path.join(data_dir, 'df_fip_signals_*.pkl')
    behavioral_pattern = os.path.join(data_dir, 'df_behavioral_traces_*.pkl')
    
    fip_files = glob.glob(fip_pattern)
    behavioral_files = glob.glob(behavioral_pattern)
    
    if not fip_files:
        raise FileNotFoundError(f"No FIP signals files found matching pattern: {fip_pattern}")
    if not behavioral_files:
        raise FileNotFoundError(f"No behavioral traces files found matching pattern: {behavioral_pattern}")
    
    # Use the first (and should be only) file found
    fip_file = fip_files[0]
    behavioral_file = behavioral_files[0]
    
    print(f"Found data directory: {data_dir}")
    print(f"Found FIP signals file: {fip_file}")
    print(f"Found behavioral traces file: {behavioral_file}")
    
    return fip_file, behavioral_file

def load_data():
    """Load the processed data files."""
    print("="*60)
    print("LOADING PROCESSED DATA")
    print("="*60)
    
    fip_file, behavioral_file = find_processed_data_files()
    
    # Load the data
    df_fip_signals = pd.read_pickle(fip_file)
    df_behavioral = pd.read_pickle(behavioral_file)
    
    print(f"FIP signals data shape: {df_fip_signals.shape}")
    print(f"FIP signals columns: {list(df_fip_signals.columns)}")
    print(f"\nBehavioral traces data shape: {df_behavioral.shape}")
    print(f"Behavioral traces columns: {list(df_behavioral.columns)}")
    
    return df_fip_signals, df_behavioral

def align_data_for_plotting(df_fip_signals, df_behavioral):
    """Align FIP signals and behavioral data for plotting, filtering noisy signals."""
    print("\n" + "="*60)
    print("ALIGNING DATA FOR PLOTTING")
    print("="*60)
    
    # Get signal columns and time columns
    signal_cols = [col for col in df_fip_signals.columns if col.startswith('signal_')]
    time_cols = [col for col in df_fip_signals.columns if col.startswith('time_')]
    
    print(f"Signal columns: {signal_cols}")
    print(f"Time columns: {time_cols}")
    
    # Analyze signal quality and filter noisy signals
    signal_quality = {}
    aligned_data = {}
    
    for i, signal_col in enumerate(signal_cols):
        time_col = time_cols[i] if i < len(time_cols) else 'time'
        
        # Get non-null data for this signal
        mask = df_fip_signals[signal_col].notna()
        signal_data = df_fip_signals[mask].copy()
        
        if len(signal_data) > 0:
            # Use the specific time column for this signal
            if time_col in signal_data.columns:
                signal_data = signal_data.sort_values(time_col)
                signal_values = signal_data[signal_col].values
                signal_times = signal_data[time_col].values
            else:
                # Fallback to general time
                signal_data = signal_data.sort_values('time')
                signal_values = signal_data[signal_col].values
                signal_times = signal_data['time'].values
            
            # Calculate signal quality metrics
            signal_std = np.std(signal_values)
            signal_mean = np.mean(signal_values)
            signal_range = np.max(signal_values) - np.min(signal_values)
            cv = signal_std / signal_mean if signal_mean != 0 else 0  # Coefficient of variation
            
            signal_quality[signal_col] = {
                'std': signal_std,
                'mean': signal_mean,
                'range': signal_range,
                'cv': cv,
                'n_points': len(signal_values)
            }
            
            aligned_data[signal_col] = {
                'time': signal_times,
                'signal': signal_values
            }
    
    # Print signal quality analysis
    print("\nSignal Quality Analysis:")
    for signal_name, quality in signal_quality.items():
        print(f"  {signal_name}:")
        print(f"    Mean: {quality['mean']:.2f}, Std: {quality['std']:.2f}")
        print(f"    Range: {quality['range']:.2f}, CV: {quality['cv']:.3f}")
        print(f"    Data points: {quality['n_points']}")
    
    # Filter signals based on quality (keep signals with reasonable signal-to-noise)
    # Keep signals with CV > 0.01 (at least 1% variation) and reasonable range
    # Also keep all G (green) channel signals regardless of quality
    filtered_signals = {}
    for signal_name, quality in signal_quality.items():
        # Keep G channel signals or high-quality signals
        if 'G_' in signal_name or (quality['cv'] > 0.01 and quality['range'] > 10):
            filtered_signals[signal_name] = aligned_data[signal_name]
            if 'G_' in signal_name:
                print(f"  ✓ Keeping {signal_name} (G channel - CV: {quality['cv']:.3f}, Range: {quality['range']:.2f})")
            else:
                print(f"  ✓ Keeping {signal_name} (CV: {quality['cv']:.3f}, Range: {quality['range']:.2f})")
        else:
            print(f"  ✗ Filtering out {signal_name} (CV: {quality['cv']:.3f}, Range: {quality['range']:.2f})")
    
    # Align behavioral data
    behavioral_data = df_behavioral.sort_values('time')
    
    print(f"\nCreated aligned data for {len(filtered_signals)} high-quality signals")
    print(f"Behavioral data points: {len(behavioral_data)}")
    
    return filtered_signals, behavioral_data

def plot_pca_fip_correlation(aligned_data, behavioral_data, save_path='results'):
    """Create single panel plot with PCA components and FIP signals aligned on same time axis."""
    print("\n" + "="*60)
    print("CREATING PCA-FIP CORRELATION PLOTS")
    print("="*60)
    
    # Create results directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created results directory: {save_path}")
    
    # Get PCA components (first 10)
    pca_cols = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
    
    # Create single panel figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    fig.suptitle('PCA Components and Fiber Photometry Signals Over Time', fontsize=16, fontweight='bold')
    
    # Find common time range for alignment
    pca_time_range = (behavioral_data['time'].min(), behavioral_data['time'].max())
    fip_time_ranges = [(data['time'].min(), data['time'].max()) for data in aligned_data.values()]
    
    # Use PCA time range as reference (it has more data points)
    time_min, time_max = pca_time_range
    print(f"Using time range: {time_min:.2f} to {time_max:.2f}")
    
    # Plot PCA components first (as reference)
    colors_pca = plt.cm.viridis(np.linspace(0, 1, len(pca_cols)))
    offset_counter = 0
    all_plot_data = []  # Store all plot data for y-axis limits
    
    for i, pc_col in enumerate(pca_cols):
        # Normalize and offset each PCA component
        pca_norm = (behavioral_data[pc_col] - np.mean(behavioral_data[pc_col])) / np.std(behavioral_data[pc_col])
        pca_offset = pca_norm + offset_counter
        
        # Clip extreme outliers (beyond 3 standard deviations)
        pca_clipped = np.clip(pca_offset, 
                             np.percentile(pca_offset, 1), 
                             np.percentile(pca_offset, 99))
        
        ax.plot(behavioral_data['time'], pca_clipped, 
                linewidth=2, alpha=0.8, color=colors_pca[i], label=f'{pc_col}')
        
        all_plot_data.extend(pca_clipped)
        offset_counter += 4  # Space PCA components 4 units apart
    
    # Plot FIP signals with proper time alignment - use bright, distinct colors
    colors_fip = ['red', 'orange', 'purple', 'brown', 'pink', 'cyan']
    
    for i, (signal_name, signal_data) in enumerate(aligned_data.items()):
        # Normalize FIP signal
        signal_norm = (signal_data['signal'] - np.mean(signal_data['signal'])) / np.std(signal_data['signal'])
        
        # Interpolate FIP signal to PCA time points for proper alignment
        fip_interp = interp1d(signal_data['time'], signal_norm, 
                             kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # Create time points that match PCA time range
        fip_time_aligned = behavioral_data['time']
        fip_signal_aligned = fip_interp(fip_time_aligned)
        
        # Offset FIP signal
        fip_offset = fip_signal_aligned + offset_counter
        
        # Clip extreme outliers (beyond 3 standard deviations)
        fip_clipped = np.clip(fip_offset, 
                             np.percentile(fip_offset, 1), 
                             np.percentile(fip_offset, 99))
        
        ax.plot(fip_time_aligned, fip_clipped, 
                linewidth=2, alpha=0.8, color=colors_fip[i],
                label=f'{signal_name.replace("signal_", "")}')
        
        all_plot_data.extend(fip_clipped)
        offset_counter += 4  # Space FIP signals 4 units apart
    
    # Set y-axis limits based on robust statistics (excluding extreme outliers)
    if all_plot_data:
        y_min = np.percentile(all_plot_data, 1.0)  # Use 1st percentile
        y_max = np.percentile(all_plot_data, 99.0)  # Use 99th percentile
        # Add some padding to the range
        y_range = y_max - y_min
        y_min = y_min - 0.1 * y_range
        y_max = y_max + 0.1 * y_range
        ax.set_ylim(y_min, y_max)
        print(f"Y-axis limits set to: {y_min:.2f} to {y_max:.2f} (robust range with padding)")
    
    # Customize plot
    ax.set_ylabel('Normalized Signals (Offset for Clarity)', fontweight='bold')
    ax.set_xlabel('Time', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Remove top and right spines for cleaner appearance
    sns.despine()
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_path, 'pca_fip_correlation_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved correlation plot to: {plot_path}")
    
    plt.show()
    
    return plot_path



def main():
    """Main analysis function."""
    print("FIBER PHOTOMETRY AND BEHAVIORAL DATA ANALYSIS")
    print("="*60)
    
    try:
        # Load data
        df_fip_signals, df_behavioral = load_data()
        
        # Align data for plotting
        aligned_data, behavioral_data = align_data_for_plotting(df_fip_signals, df_behavioral)
        
        # Create plot
        plot_pca_fip_correlation(aligned_data, behavioral_data)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Generated plot:")
        print("- PCA-FIP time series with 10 PCs and filtered FIP signals")
        print(f"Results saved in: results/")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
