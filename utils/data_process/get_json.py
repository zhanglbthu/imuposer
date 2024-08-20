import json
import pandas as pd
import matplotlib.pyplot as plt

# Load the JSON file
file_path = '/root/autodl-tmp/imuposer/checkpoints/IMUPoserGlobalModel_global-08102024-040836/results/model.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract results into a DataFrame
results = [
    {**combo['results'], 'combo_id': combo['combo_id']} for combo in data['combos']
]

df = pd.DataFrame(results)

# 计算results中的指标的平均值

# Calculate the mean of each metric
mean_metrics = df.mean(numeric_only=True)
print(mean_metrics)

# # 绘制并保存每个指标的图
# metrics = ['jre', 'jpe', 've']
# for metric in metrics:
#     plt.figure(figsize=(10, 6))
#     df.plot(x='combo_id', y=metric, kind='bar', legend=False)
#     plt.title(f'Values of {metric} for Each Combo ID')
#     plt.xlabel('Combo ID')
#     plt.ylabel(f'{metric} Value')
#     plt.xticks(rotation=45, ha="right")
#     plt.tight_layout()
#     plt.savefig(f'{metric}_values.png')
#     plt.close()  # 关闭图形避免在 Jupyter Notebook 中显示
