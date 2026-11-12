import json
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd

def parse_results(file_pattern):
    data = []
    files = glob.glob(file_pattern)
    for file_path in files:
        filename = os.path.basename(file_path)
        with open(file_path, 'r') as f:
            try:
                content = json.load(f)
                if 'solutions' in content:
                    for length_str, solution in content['solutions'].items():
                        if 'total_clock_time' in solution:
                            length = int(length_str)
                            clock_time = solution['total_clock_time']['secs'] + solution['total_clock_time']['nanos'] / 1e9
                            data.append({
                                'file': filename,
                                'length': length,
                                'clock_time': clock_time
                            })
            except json.JSONDecodeError:
                print(f"Error decoding {file_path}")
    return pd.DataFrame(data)

def main():
    df = parse_results('results/v094*.json')
    if df.empty:
        print("No data found for v094 files.")
        return

    plt.figure(figsize=(12, 8))
    
    # Plot each file's data points
    for filename in df['file'].unique():
        file_df = df[df['file'] == filename].sort_values('length')
        plt.scatter(file_df['length'], file_df['clock_time'], label=filename, alpha=0.6, s=10)
    
    # Also plot a trend line (mean per length)
    df_mean = df.groupby('length')['clock_time'].mean().reset_index()
    plt.plot(df_mean['length'], df_mean['clock_time'], 'k--', label='Mean', linewidth=1, alpha=0.5)

    plt.yscale('log')
    plt.xlabel('Length (L)')
    plt.ylabel('Clock Time (seconds) - Log Scale')
    plt.title('Clock Time vs Length for v094 Results')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    output_path = 'visualizations/v094_clock_time_vs_length.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
