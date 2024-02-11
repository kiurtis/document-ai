import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm


def plot_metrics_from_csv(csv_dict):
    metrics = ['accuracy', 'recall', 'precision', 'TP', 'FP', 'TN', 'FN']

    # Use a colormap to generate distinct colors
    colormap = cm.get_cmap('tab20')  # This colormap has 20 distinct colors
    color_indices = np.linspace(0, 1, len(csv_dict))

    # Prepare a figure for plotting with higher DPI for better resolution
    fig, axes = plt.subplots(len(metrics), figsize=(14, 25), dpi=120)
    fig.tight_layout(rect=[0, 0, 0.75, 1], h_pad=8.0, pad=4.0)

    # Initialize a list to keep track of legend handles
    legend_handles = []

    for csv_index, (legend_name, csv_path) in enumerate(csv_dict.items()):
        # Read each CSV file
        df = pd.read_csv(csv_path)

        # Generate a numeric range for the x-axis positions
        x_positions = np.arange(len(df))

        # Use colormap to assign color
        color = colormap(color_indices[csv_index])

        for metric_index, metric in enumerate(metrics):
            if metric not in df.columns:
                continue  # Skip metrics not present in the CSV

            ax = axes[metric_index]
            # Offset for each CSV to avoid bar overlap
            offset = csv_index * 0.3 - (0.05 * len(csv_dict))

            # Plot bars with adjusted positions for each CSV
            bars = ax.bar(x_positions + offset, df[metric], width=0.3, color=color)

            ax.set_ylabel(metric)
            ax.set_title(metric.capitalize())
            ax.set_xticks(x_positions)
            ax.set_xticklabels(df['cause'], rotation=30, ha="right", fontsize='small')


        # Add a legend item for this CSV file
        legend_handles.append(bars[0])  # Use the first bar object as a proxy for the legend

    # Place legend outside the plot area on the right side
    fig.legend(legend_handles, list(csv_dict.keys()), loc='center right',
               #bbox_to_anchor=(1.05, 0.5),
               title='Models')
    csv_dict_str = '_'.join(csv_dict.keys())
    plt.savefig(f'plots/error_analysis_invalid_{csv_dict_str}.png', dpi=120)
    plt.show()


# Example usage
csv_dict = {
    'gemini-vision-pro': 'results/error_analysis_invalid_GEMINI_20240211_124106/summary.csv',
    #'gpt-4-vision': 'results/error_analysis_invalid_20231227_105231/summary.csv',
    'gpt-4-vision': 'results/error_analysis_invalid_GPT_base/summary.csv',
    #'llava-4bit-not-tuned': 'results/error_analysis_valid_2024-02-10 08_25_46.164134_4_bit_llava/summary.csv',
    #'llava-4bit-not-tuned-2': 'results/error_analysis_valid_2024-02-10 19_20_12.592687_4_bit_llava/summary.csv',
    #'llava-8bit-not-tuned': 'results/error_analysis_valid_2024-02-10 18_24_21.475152_8_bit_llava/summary.csv',
    # Add more CSV files as needed
}
plot_metrics_from_csv(csv_dict)
