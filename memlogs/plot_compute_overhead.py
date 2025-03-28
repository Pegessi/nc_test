import matplotlib.pyplot as plt
import numpy as np

# 新增参数
border_width = 5
title_font_size = 20
label_font_size = 18
tick_font_size = 14
font_weight = 'bold'

plt.rcParams['font.weight'] = font_weight

### col 1
training_time = np.array([
    [2.019, 2.274, 2.725, 2.874, 3.361, 4.273, 5.127],
    [2.02, 2.342, 2.680, 2.856, 3.424, 3.923, 5.643],
    # [2.188, 21.22, 7.112, 21.483, 8.659, 5.671, 0],
    [2.188, 2.42, 2.514, 3.483, 4.659, 5.671, 0],
    [2.039, 2.147, 2.213, 2.287, 2.360, 2.532, 2.748],
    [2.030, 2.136, 2.183, 2.297, 2.433, 2.627, 2.944]
])
# 对training_time的所有数据都*1000
training_time = training_time * 1000
memory_budget_ratios = np.array([0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
memory_budget_ratios = [str(i) for i in memory_budget_ratios]

methods = ["DTR", "DTE", "DTR+GMLake", "Nebula-Chain w/o mem", "Nebula-Chain"]

### col 2
# reserve部分数据
training_time_2 = np.array([
    [4473.248, 4593.271, 4727.406, 5082.458, 7641.244, 8747.982, 10185.342],
    [4586.56, 8688.4, 12494.33, 9056.86, 9891.5, 0, 0],
    [4644.92, 4706.42, 4950.78, 7048.35, 8598.72, 8962.1, 9998.61],
    [4544.34, 4621.38, 4769.29, 4813.65, 4863.4, 4927.55, 4990.65],
    [4572.13, 4656.96, 4921.54, 4935.38, 5014.23, 5310.08, 0]
])
memory_budget_ratios_2 = np.array([0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
memory_budget_ratios_2 = [str(i) for i in memory_budget_ratios_2]

methods_2 = ["DTR", "DTE", "DTR+GMLake", "Nebula-Chain w/o mem", "Nebula-Chain"]

### col 3
training_time_3 = np.array([
    [10987.00, 12109.6, 13328.3, 14055.4, 15027.4],
    [11021.00, 11578.8, 12242, 13310.3, 13011.4],
    [11054.00, 12012.0, 12258.2, 12881.2, 13196.5]
])
memory_budget_ratios_3 = np.array([0, 0.8, 0.7, 0.6, 0.5])
memory_budget_ratios_3 = [str(i) for i in memory_budget_ratios_3]
methods_3 = ["Megatron-LM", "Nebula-Chain w/o mem", "Nebula-Chain"]
colors_3 = ['#906cb9', '#2ba02d', '#c85862']



# 创建子图
fig, axs = plt.subplots(1, 3, figsize=(16, 6))
# 定义颜色数组，方便自定义颜色
# colors = ['#396aa7', '#e97c2a', '#459232', '#f6d568',  '#b15559']
colors = ['#ffd966', '#FF9b9b', '#5898d5', '#2ba02d', '#c85862']
# 定义mark数组设置折线图点的样式
marks = ['o', 's', '^', 'D', 'v']

width = 0.15
x = np.arange(len(memory_budget_ratios))
x_3 = np.arange(len(memory_budget_ratios_3))

# 第四个子图：reserve，Llama2-7B Lora 8GPU
for i in range(len(methods)):
    # 修改索引为一维
    axs[0].bar(x + i * width, training_time[i], width=width, edgecolor='black', label=methods[i], color=colors[i],  zorder=10)
# axs[0].set_xlabel('Memory Budget Ratios', fontsize=label_font_size, fontweight=font_weight)
axs[0].set_ylabel('Training time per iter (ms)', fontsize=label_font_size, fontweight=font_weight)
axs[0].set_title('Llama2-7B Lora 8GPU', fontsize=title_font_size, fontweight=font_weight)
axs[0].set_xticks(x + width * (len(methods) - 1) / 2)
axs[0].set_xticklabels(memory_budget_ratios, fontsize=tick_font_size)
axs[0].tick_params(axis='y', which='major', labelsize=tick_font_size)
for spine in axs[0].spines.values():
    spine.set_linewidth(border_width)
axs[0].grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)

# 第五个子图：reserve，假设为AlphaFold 8GPU
for i in range(len(methods)):
    # 修改索引为一维
    axs[1].bar(x + i * width, training_time_2[i], width=width, edgecolor='black', label=methods[i], color=colors[i],  zorder=10)
axs[1].set_title('AlphaFold 8GPU', fontsize=title_font_size, fontweight=font_weight)
axs[1].set_xlabel('Memory Budget Ratios', fontsize=label_font_size, fontweight=font_weight)
axs[1].set_xticks(x + width * (len(methods) - 1) / 2)
axs[1].set_xticklabels(memory_budget_ratios, fontsize=tick_font_size)
axs[1].tick_params(axis='y', which='major', labelsize=tick_font_size)
for spine in axs[1].spines.values():
    spine.set_linewidth(border_width)
axs[1].grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)

# 第六个子图：reserve，假设为GPT3-7.5B 64GPU
for i in range(len(methods_3)):
    axs[2].bar(x_3 + i * width, training_time_3[i], width=width, edgecolor='black', label=methods_3[i], color=colors_3[i],  zorder=10)
axs[2].set_title('GPT3-7.5B 64GPU', fontsize=title_font_size, fontweight=font_weight)
axs[2].set_xticks(x_3 + width * (len(methods) - 1) / 2)
axs[2].set_xticklabels(memory_budget_ratios_3, fontsize=tick_font_size)
axs[2].tick_params(axis='y', which='major', labelsize=tick_font_size)
for spine in axs[2].spines.values():
    spine.set_linewidth(border_width)
axs[2].grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)

# 集中显示图例在所有图表的最上方
handles, labels = axs[0].get_legend_handles_labels()
handles_, labels_ = axs[2].get_legend_handles_labels()
handles += [handles_[0]]
labels += [labels_[0]]
fig.legend(handles, labels, loc='upper center', ncol=len(methods)+1, fontsize=label_font_size-4)

plt.tight_layout()
# 调整布局以适应上方的图例
plt.subplots_adjust(top=0.85)
plt.savefig('compute_overhead.pdf', dpi=300, format='pdf', facecolor='white')