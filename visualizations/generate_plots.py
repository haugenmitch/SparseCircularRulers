import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import numpy as np
from collections import Counter
from math import comb

def apply_style():
    """Apply the custom 'mch' style if available, otherwise fallback."""
    style_name = 'mch'
    # Possible paths for the style file
    home = os.path.expanduser("~")
    custom_style_path = os.path.join(home, "repos/plotting_style/mch.mplstyle")
    
    if os.path.exists(custom_style_path):
        plt.style.use(custom_style_path)
    else:
        try:
            plt.style.use(style_name)
        except:
            # Fallback to a clean built-in style
            plt.style.use('seaborn-v0_8-darkgrid')
            plt.rcParams['figure.facecolor'] = '#404040'
            plt.rcParams['axes.facecolor'] = '#404040'
            plt.rcParams['text.color'] = '#e0e0e0'

def parse_args():
    parser = argparse.ArgumentParser(description='Generate all plots for Sparse Rulers data.')
    parser.add_argument('filename', type=str, help='Path to the JSON dataset')
    parser.add_argument('--outdir', type=str, default='plots', help='Output directory for plots')
    return parser.parse_args()

def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def prepare_dataframe(data):
    solutions = data.get('solutions', {})
    total_evaluated_global = data.get('total_rulers_evaluated', 1)
    
    df_list = []
    for length_str, sol in solutions.items():
        length = int(length_str)
        rulers = sol.get('rulers', [])
        num_segments = sol.get('num_segments', 0)
        lower_bound = sol.get('lower_bound_num_segments', 0)
        
        # CPU Time
        cpu_time = sol.get('total_cpu_time', {})
        cpu_time_secs = cpu_time.get('secs', 0) + (cpu_time.get('nanos', 0) / 1e9)
        
        # Evaluations
        evaluated_per_length = sol.get('total_rulers_evaluated', 0)
        
        # Segment stats
        avg_seg = 0
        max_seg = 0
        min_max_seg = 0
        if rulers:
            max_segments = [max(r) for r in rulers]
            all_segs = [s for r in rulers for s in r]
            avg_seg = sum(all_segs) / len(all_segs) if all_segs else 0
            max_seg = max(max_segments)
            min_max_seg = min(max_segments)

        # Theoretical complexity: (L-1) choose (k-1)
        # where k is lower bound num segments
        theoretical_space = 0
        if length > 0 and lower_bound > 0:
            try:
                theoretical_space = comb(length - 1, lower_bound - 1)
            except OverflowError:
                theoretical_space = float('inf')

        df_list.append({
            'length': length,
            'num_segments': num_segments,
            'lower_bound': lower_bound,
            'diff_segments': num_segments - lower_bound,
            'num_rulers': len(rulers),
            'total_cpu_time': cpu_time_secs,
            'evaluated_per_length': evaluated_per_length,
            'cost_per_eval': cpu_time_secs / evaluated_per_length if evaluated_per_length > 0 else 0,
            'evals_per_sec': evaluated_per_length / cpu_time_secs if cpu_time_secs > 0 else 0,
            'global_ratio': len(rulers) / total_evaluated_global if total_evaluated_global > 0 else 0,
            'local_ratio': len(rulers) / evaluated_per_length if evaluated_per_length > 0 else 0,
            'avg_segment_value': avg_seg,
            'max_segment_value': max_seg,
            'min_max_segment_value': min_max_seg,
            'theoretical_space': theoretical_space,
            'rulers': rulers
        })
    
    return pd.DataFrame(df_list).sort_values('length')

def plot_structural_analysis(df, outdir):
    # New: Segment Adjacency Heatmap (Transition Matrix)
    # What segment length follows another?
    transitions = []
    for rulers in df['rulers']:
        for r in rulers:
            for i in range(len(r) - 1):
                transitions.append({'current': r[i], 'next': r[i+1]})
    
    if transitions:
        df_trans = pd.DataFrame(transitions)
        # Normalize to probabilities
        pivot = pd.crosstab(df_trans['current'], df_trans['next'], normalize='index')
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot, cmap='magma')
        plt.title('Segment Adjacency Heatmap (P(Next | Current))')
        plt.xlabel('Next Segment Length')
        plt.ylabel('Current Segment Length')
        plt.savefig(os.path.join(outdir, 'segment_adjacency_heatmap.png'))
        plt.close()

def plot_segment_metrics(df, outdir):
    # Plot 1: Lower bound vs Actual vs Diff
    plt.figure(figsize=(10, 6))
    plt.scatter(df['length'], df['num_segments'], label='Actual Segments', alpha=0.7)
    plt.scatter(df['length'], df['lower_bound'], label='Lower Bound', alpha=0.7)
    plt.scatter(df['length'], df['diff_segments'], label='Difference', marker='x', alpha=0.5)
    plt.title('Number of Segments vs Length')
    plt.xlabel('Ruler Length')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(outdir, 'segments_vs_length.png'))
    plt.close()

    # Plot 6 & 7: Segment values and count
    plt.figure(figsize=(10, 6))
    plt.scatter(df['length'], df['avg_segment_value'], label='Avg Segment Value', alpha=0.6)
    plt.scatter(df['length'], df['max_segment_value'], label='Max Segment Value', alpha=0.6)
    plt.scatter(df['length'], df['num_segments'], label='Num Segments', marker='s', alpha=0.4)
    plt.title('Segment Statistics vs Length')
    plt.xlabel('Ruler Length')
    plt.ylabel('Value / Count')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(outdir, 'segment_stats_vs_length.png'))
    plt.close()

    # Plot 12: Max vs Min-Max
    plt.figure(figsize=(10, 6))
    plt.scatter(df['length'], df['max_segment_value'], label='Max of Max Segments', alpha=0.7)
    plt.scatter(df['length'], df['min_max_segment_value'], label='Min of Max Segments', alpha=0.7)
    plt.title('Max vs Min-Max Segment Length')
    plt.xlabel('Ruler Length')
    plt.ylabel('Segment Length')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(outdir, 'max_vs_minmax_segments.png'))
    plt.close()

def plot_ruler_counts(df, outdir):
    # Plot 2 & 8: Num Rulers Log
    plt.figure(figsize=(10, 6))
    plt.scatter(df['length'], df['num_rulers'], alpha=0.7)
    plt.yscale('log')
    plt.title('Number of Solution Rulers vs Length (Log Scale)')
    plt.xlabel('Ruler Length')
    plt.ylabel('Number of Rulers')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(outdir, 'num_rulers_log.png'))
    plt.close()

    # Plot 17: Solutions colored by num_segments
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(data=df, x='length', y='num_rulers', hue='num_segments', palette='viridis', s=60, alpha=0.8)
    plt.yscale('log')
    plt.title('Number of Solutions (Colored by Num Segments)')
    plt.xlabel('Ruler Length')
    plt.ylabel('Number of Solutions (Log Scale)')
    plt.legend(title='Segments')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(outdir, 'solutions_vs_length_colored.png'))
    plt.close()

def plot_ratios(df, outdir):
    # Plot 13 & 14: Ratios
    plt.figure(figsize=(10, 6))
    plt.scatter(df['length'], df['global_ratio'], label='Global Ratio', alpha=0.6)
    plt.scatter(df['length'], df['local_ratio'], label='Per-Length Ratio', alpha=0.6)
    plt.yscale('log')
    plt.title('Solution Ratios (Log Scale)')
    plt.xlabel('Ruler Length')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(outdir, 'solution_ratios_log.png'))
    plt.close()

def plot_performance(df, outdir):
    # Plot 15: Cost per eval
    plt.figure(figsize=(10, 6))
    plt.scatter(df['length'], df['cost_per_eval'], alpha=0.7)
    plt.yscale('log')
    plt.title('CPU Time per Ruler Evaluated')
    plt.xlabel('Ruler Length')
    plt.ylabel('Seconds per Evaluation')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(outdir, 'cost_per_eval_log.png'))
    plt.close()

    # New: Clock time vs Length
    plt.figure(figsize=(10, 6))
    plt.scatter(df['length'], df['total_cpu_time'], alpha=0.7)
    plt.yscale('log')
    plt.title('Total CPU Time vs Length')
    plt.xlabel('Ruler Length')
    plt.ylabel('Seconds')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(outdir, 'cpu_time_vs_length.png'))
    plt.close()

    # New: Throughput
    plt.figure(figsize=(10, 6))
    plt.scatter(df['length'], df['evals_per_sec'], alpha=0.7)
    plt.yscale('log')
    plt.title('Evaluations per Second (Throughput)')
    plt.xlabel('Ruler Length')
    plt.ylabel('Evals / Sec')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(outdir, 'throughput_log.png'))
    plt.close()

def plot_theoretical(df, outdir):
    # New: Theoretical complexity
    plt.figure(figsize=(10, 6))
    plt.scatter(df['length'], df['theoretical_space'], alpha=0.7)
    plt.yscale('log')
    plt.title('Theoretical Search Space Size: (L-1) choose (k-1)')
    plt.xlabel('Ruler Length')
    plt.ylabel('Combinations')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(outdir, 'theoretical_complexity_log.png'))
    plt.close()

def plot_heatmap(df, outdir):
    # Plot 11: Heatmap flipped, normalized
    heatmap_data = []
    for _, row in df.iterrows():
        length = row['length']
        rulers = row['rulers']
        if not rulers: continue
        
        counts = Counter([s for r in rulers for s in r])
        total_segs = sum(counts.values())
        
        for segment_val, count in counts.items():
            heatmap_data.append({
                'length': length,
                'segment_val': segment_val,
                'percentage': (count / total_segs) * 100
            })

    if not heatmap_data: return
    
    df_hm = pd.DataFrame(heatmap_data)
    pivot = df_hm.pivot(index='segment_val', columns='length', values='percentage').fillna(0)
    
    plt.figure(figsize=(14, 10))
    # Flip Y axis: we can just sort index descending
    sns.heatmap(pivot.sort_index(ascending=False), cmap='viridis')
    plt.title('Segment Frequency Heatmap (% per Length)')
    plt.xlabel('Ruler Length')
    plt.ylabel('Segment Value')
    plt.savefig(os.path.join(outdir, 'heatmap_segment_freq.png'))
    plt.close()

def plot_polar(df, outdir):
    # Plot 16: Polar rulers (first one)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='polar')
    
    cmap = plt.get_cmap('tab20')
    
    for _, row in df.iterrows():
        length = row['length']
        rulers = row['rulers']
        if not rulers: continue
        
        ruler = rulers[0] # Use first ruler
        
        total_len = sum(ruler)
        # Use simple equal angular spacing for segments as a placeholder 
        # or calculate based on segment relative size
        # Let's use relative size for better visualization
        angles = [0]
        curr = 0
        for s in ruler:
            curr += (s / total_len) * 2 * np.pi
            angles.append(curr)
            
        for i in range(len(ruler)):
            ax.bar(
                (angles[i] + angles[i+1]) / 2, 
                1, 
                width=(angles[i+1] - angles[i]), 
                bottom=length,
                color=cmap(ruler[i] % 20),
                edgecolor='black',
                linewidth=0.1
            )

    ax.set_title('Polar Visualization of First Ruler per Length')
    ax.set_yticklabels([])
    plt.savefig(os.path.join(outdir, 'polar_rulers.png'))
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    apply_style()
    
    print(f"Loading data from {args.filename}...")
    data = load_data(args.filename)
    
    print("Preparing data...")
    df = prepare_dataframe(data)
    
    print("Generating plots...")
    plot_structural_analysis(df, args.outdir)
    plot_segment_metrics(df, args.outdir)
    plot_ruler_counts(df, args.outdir)
    plot_ratios(df, args.outdir)
    plot_performance(df, args.outdir)
    plot_theoretical(df, args.outdir)
    plot_heatmap(df, args.outdir)
    plot_polar(df, args.outdir)
    
    print(f"Done! All plots saved to {args.outdir}/")

if __name__ == "__main__":
    main()
