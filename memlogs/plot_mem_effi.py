import matplotlib.pyplot as plt
import numpy as np
import os 

SAVE_PREFIX = os.environ.get('SAVE_PREFIX', "./")
OURS = os.environ.get('OURS', "DT-Control")

# 新增参数
border_width = 5
title_font_size = 20
label_font_size = 18
tick_font_size = 14
font_weight = 'bold'

# plt.rcParams['font.weight'] = font_weight

### col 1
fragmentation_ratios = np.array([
    [0.034, 0.189, 0.184, 0.193, 0.231, 0.217, 0.144],
    [0.034, 0.142, 0.17, 0.177, 0.184, 0.169, 0.156],
    [0.013, 0.09, 0.129, 0.185, 0.136, 0.149, None],
    [0.036, 0.115, 0.107, 0.103, 0.102, 0.138, 0.149],
    [0.038, 0.033, 0.037, 0.051, 0.066, 0.072, 0.0908]
])
# reserve部分数据
reserve_ratios = np.array([
    [64060, 64608, 56424, 49000, 43142, 34182, 23688],
    [64060, 60512, 54880, 47456, 40020, 31614, 23426],
    [62700, 57606, 52832, 48502, 38388, 31472, 0],
    [64988, 58056, 50400, 42932, 35776, 29876, 22624],
    [65128, 53132, 46732, 40844, 34956, 28300, 23062]
])
memory_budget_ratios = np.array([1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
memory_budget_ratios = [str(i) for i in memory_budget_ratios]

methods = ["DTR", "DTE", "DTR+GMLake", OURS+" w/o mem", OURS]

### col 2
# fragmentation部分数据
fragmentation_ratios_2 = np.array([
    [0.042, 0.17, 0.211, 0.236, 0.345, 0.35, 0.37],
    [0.042, 0.12, 0.161, 0.224, 0.224, None, None],
    [0.037, 0.163, 0.182, 0.247, 0.261, 0.282, 0.349],
    [0.05, 0.138, 0.225, 0.22, 0.238, 0.238, 0.249],
    [0.05, 0.061, 0.098, 0.079, 0.092, 0.115, None]
])
# reserve部分数据
reserve_ratios_2 = np.array([
    [76062, 71214, 65662, 58158, 56788, 45978, 35784],
    [76062, 67160, 61658, 57204, 47774, None, None],
    [76822, 70602, 63246, 59142, 50338, 41890, 34596],
    [74718, 68138, 66286, 56470, 48194, 38542, 29346],
    [74718, 62538, 57212, 50176, 41622, 36800, None]
])
memory_budget_ratios_2 = np.array([1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
memory_budget_ratios_2 = [str(i) for i in memory_budget_ratios_2]

methods_2 = ["DTR", "DTE", "DTR+GMLake", OURS+" w/o mem", OURS]

### col 3
# fragmentation部分数据
fragmentation_ratios_3 = np.array([
    [0.011413, 0.073947, 0.076371, 0.109187, 0.079744],
    [0.011413, 0.06628533, 0.098532028, 0.126338688, 0.081313348],
    [0.011413, 0.057890805, 0.039683598, 0.034470665, 0.052670439],
])
# reserve部分数据
reserve_ratios_3 = np.array([
    [36720, 29148, 25988, 23208, 20288],
    [36818, 29630, 26746, 23738, 20482],
    [36818, 29452, 26724, 22791, 19327],
])
memory_budget_ratios_3 = np.array([1, 0.8, 0.7, 0.6, 0.5])
memory_budget_ratios_3 = [str(i) for i in memory_budget_ratios_3]
methods_3 = ["Megatron-LM", OURS+" w/o mem", OURS]
colors_3 = ['#906cb9', '#2ba02d', '#c85862']



# 创建子图
fig, axs = plt.subplots(2, 3, figsize=(16, 9))
# 定义颜色数组，方便自定义颜色
colors = ['#ffd966', '#FF9b9b', '#5898d5', '#2ba02d', '#c85862']
# 定义mark数组设置折线图点的样式
marks = ['o', 's', '^', 'D', 'v']

# 第一个子图：fragmentation，Llama2-7B Lora 8GPU
for i in range(len(methods)):
    axs[0, 0].plot(memory_budget_ratios, fragmentation_ratios[i], label=methods[i], color=colors[i], marker=marks[i],  zorder=10)
axs[0, 0].set_ylabel('Fragment Ratio', fontsize=label_font_size, fontweight=font_weight)
axs[0, 0].set_title('Llama2-7B Lora 8GPU', fontsize=title_font_size)
axs[0, 0].tick_params(axis='both', which='major', labelsize=tick_font_size)
# 设置x轴和y轴刻度标签字体为bold
for tick in axs[0, 0].get_xticklabels():
    tick.set_fontweight('bold')
for tick in axs[0, 0].get_yticklabels():
    tick.set_fontweight('bold')
for spine in axs[0, 0].spines.values():
    spine.set_linewidth(border_width)
axs[0, 0].grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)

# 第二个子图：fragmentation，假设为AlphaFold 8GPU
for i in range(len(methods)):
    axs[0, 1].plot(memory_budget_ratios, fragmentation_ratios_2[i], label=methods[i], color=colors[i], marker=marks[i],  zorder=10)
axs[0, 1].set_title('AlphaFold 8GPU', fontsize=title_font_size)
axs[0, 1].tick_params(axis='both', which='major', labelsize=tick_font_size)
# 设置x轴和y轴刻度标签字体为bold
for tick in axs[0, 1].get_xticklabels():
    tick.set_fontweight('bold')
for tick in axs[0, 1].get_yticklabels():
    tick.set_fontweight('bold')
for spine in axs[0, 1].spines.values():
    spine.set_linewidth(border_width)
axs[0, 1].grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)

x_3 = np.arange(len(memory_budget_ratios_3))

# 第三个子图：fragmentation，假设为GPT3-7.5B 64GPU
for i in range(len(methods_3)):
    axs[0, 2].plot(memory_budget_ratios_3, fragmentation_ratios_3[i], label=methods_3[i], color=colors_3[i], marker=marks[i],  zorder=10)
axs[0, 2].set_title('GPT3-7.5B 64GPU', fontsize=title_font_size)
axs[0, 2].tick_params(axis='both', which='major', labelsize=tick_font_size)
# 设置x轴和y轴刻度标签字体为bold
for tick in axs[0, 2].get_xticklabels():
    tick.set_fontweight('bold')
for tick in axs[0, 2].get_yticklabels():
    tick.set_fontweight('bold')
for spine in axs[0, 2].spines.values():
    spine.set_linewidth(border_width)
axs[0, 2].grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)

width = 0.15
x = np.arange(len(memory_budget_ratios))

# 第四个子图：reserve，Llama2-7B Lora 8GPU
for i in range(len(methods)):
    axs[1, 0].bar(x + i * width, reserve_ratios[i], width=width, edgecolor='black', label=methods[i], color=colors[i],  zorder=10)
axs[1, 0].set_ylabel('Peak Reserve Memory (MB)', fontsize=label_font_size, fontweight=font_weight)
axs[1, 0].set_xticks(x + width * (len(methods) - 1) / 2)
axs[1, 0].set_xticklabels(memory_budget_ratios, fontsize=tick_font_size)
axs[1, 0].tick_params(axis='y', which='major', labelsize=tick_font_size)
for tick in axs[1, 0].get_xticklabels():
    tick.set_fontweight('bold')
for tick in axs[1, 0].get_yticklabels():
    tick.set_fontweight('bold')
for spine in axs[1, 0].spines.values():
    spine.set_linewidth(border_width)
axs[1, 0].grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)

# 第五个子图：reserve，假设为AlphaFold 8GPU
for i in range(len(methods)):
    axs[1, 1].bar(x + i * width, reserve_ratios[i], width=width, edgecolor='black', label=methods[i], color=colors[i],  zorder=10)
axs[1, 1].set_xlabel('Memory Budget Ratios', fontsize=label_font_size, fontweight=font_weight)
axs[1, 1].set_xticks(x + width * (len(methods) - 1) / 2)
axs[1, 1].set_xticklabels(memory_budget_ratios, fontsize=tick_font_size)
axs[1, 1].tick_params(axis='y', which='major', labelsize=tick_font_size)
for tick in axs[1, 1].get_xticklabels():
    tick.set_fontweight('bold')
for tick in axs[1, 1].get_yticklabels():
    tick.set_fontweight('bold')
for spine in axs[1, 1].spines.values():
    spine.set_linewidth(border_width)
axs[1, 1].grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)

# 第六个子图：reserve，假设为GPT3-7.5B 64GPU
for i in range(len(methods_3)):
    axs[1, 2].bar(x_3 + i * width, reserve_ratios_3[i], width=width, edgecolor='black', label=methods_3[i], color=colors_3[i],  zorder=10)
axs[1, 2].set_xticks(x_3 + width * (len(methods) - 1) / 2)
axs[1, 2].set_xticklabels(memory_budget_ratios_3, fontsize=tick_font_size)
axs[1, 2].tick_params(axis='y', which='major', labelsize=tick_font_size)
for tick in axs[1, 2].get_xticklabels():
    tick.set_fontweight('bold')
for tick in axs[1, 2].get_yticklabels():
    tick.set_fontweight('bold')
for spine in axs[1, 2].spines.values():
    spine.set_linewidth(border_width)
axs[1, 2].grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)

# 集中显示图例在所有图表的最上方
handles, labels = axs[0, 0].get_legend_handles_labels()
handles_, labels_ = axs[0, 2].get_legend_handles_labels()
# import ipdb; ipdb.set_trace()
handles += [handles_[0]]
labels += [labels_[0]]
fig.legend(handles, labels, loc='upper center', ncol=len(methods)+1, fontsize=label_font_size-4, frameon=False)

plt.tight_layout()
# 调整布局以适应上方的图例
plt.subplots_adjust(top=0.9)
plt.savefig(SAVE_PREFIX+'exp_mem_efficiency.pdf', dpi=300, backend='cairo', facecolor='white')