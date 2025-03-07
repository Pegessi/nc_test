import matplotlib.pyplot as plt

# 数据
budget_ratio = ["100%", "80%", "70%", "60%", "50%", "40%", "30%"]
# DTR = [14.0273, 16.9178, 17.4995, 18.949, 21.902, 23.6092, 27.1921]
DTR = [14.3113, 16.0468, 16.9178, 17.4995, 20.2838, 23.5911, 29.0501]
# Megatron_LM = [13.7601, 13.7639, 14.9881, 15.5822, 16.3968, 18.3816, 21.3592]
Megatron_LM = [14.3113, 15.4288, 15.9725, 17.0442, 17.6347, 18.7095, 19.8337]

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 创建画布
plt.figure(figsize=(10, 6))

# 绘制 DTR 折线图
plt.plot(budget_ratio, DTR, marker='o', label='DTR', color='orange')

# 绘制 Megatron-LM 折线图
plt.plot(budget_ratio, Megatron_LM, marker='s', label='Megatron-LM', color='green')

# 添加标题和标签
plt.title('Performance Comparison')
plt.xlabel('Budget Ratio')
plt.ylabel('Time (s)')

# 添加图例
plt.legend()

# 显示网格线
plt.grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)
plt.tight_layout()

# 显示图形
plt.savefig('plot_intro.png')