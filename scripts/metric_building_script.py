from performances import build_metrics_dfs, load_ground_truth_data

filename_suffix = 'invalid_GPT_base'
predictions_path = 'results/full_result_analysis_invalid_GPT_base.csv'
ground_truth_data = load_ground_truth_data('invalid')
build_metrics_dfs(predictions_path, ground_truth_data, filename_suffix)