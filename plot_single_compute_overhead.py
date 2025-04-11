import matplotlib.pyplot as plt
import numpy as np
import os

SAVE_PREFIX = os.environ.get('SAVE_PREFIX', "./figure/exp/")
OURS = os.environ.get('OURS', "DT-Control")

# 新增参数
border_width = 5
title_font_size = 20
label_font_size = 18
tick_font_size = 18
font_weight = 'bold'

plt.rcParams['font.family'] = 'Arial'

### col 1
training_time = np.array([
    [1.05, 1.070389321, 1.107970662, 1.155163833, 1.191516533, 1.261844661, 1.689451613],
    [1.04, 1.386802964, 1.126946607, 1.158280868, 1.204244105, 1.263288042, 1.570894377],
    [1.1, 3.584988877, 4.213971079, 2.757323749, 2.712492214, 3.083695773, 3.164922136],
    [1.002, 1.015016685, 1.063403782, 1.096662959, 1.1256396, 1.152614016, 1.191101224],
    [1.002, 1.019688543, 1.056952169, 1.095216908, 1.131256952, 1.186651835, 1.275472747]
])
base_time = 179.8
training_time = training_time * base_time

# training_time = [item[1:] for item in training_time]
memory_budget_ratios = np.array([1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
memory_budget_ratios = [str(i) for i in memory_budget_ratios]
# memory_budget_ratios = memory_budget_ratios[1:]


methods = ["DTR", "DTE", "DTR+GMLake", OURS+" w/o mem", OURS]

### col 2
# reserve部分数据
# training_time_2 = np.array([  ### bert
#     [1, 1.102171077, 1.132743043, 1.133152395, 1.178723522, 1.280363213, 1.376130394],
#     [1, 1.101074978, 1.126079513, 1.136262983, 1.160880538, 1.279961393, 1.384795937],
#     [1, 2.615887688, 2.485723226, 2.296095788, 2.432997686, 2.088519071, 6.764910237],
#     [1, 1.034435434, 1.05678811, 1.087076034, 1.112953162, 1.176436514, 1.236485899],
#     [1, 1.061938425, 1.070566802, 1.086623222, 1.092145813, 1.125883941, 1.211546477]
# ])
training_time_2 = np.array([
    [1.06, 1.089675242, 1.121496484, 1.205726519, 1.812754877, 2.075309601, 2.416298758],
    [1.08, 1.11651807, 1.17448728, 1.672100408, 2.039899024, 2.126105683, 2.371999039],
    [1.1, 2.061174398, 2.964066775, 2.148585404, 2.346589753, 0, 0],
    [1.07, 1.096342437, 1.131431801, 1.141955597, 1.153758743, 1.168976525, 1.183946802],
    [1.08, 1.10478366, 1.167552415, 1.170835245, 1.189539403, 1.259724918, 0]
])
base_time = 4215.266
training_time_2 = training_time_2 * base_time
memory_budget_ratios_2 = np.array([1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
memory_budget_ratios_2 = [str(i) for i in memory_budget_ratios_2]
# training_time_2 = [item[1:] for item in training_time_2]
# memory_budget_ratios_2 = memory_budget_ratios_2[1:]

methods_2 = ["DTR", "DTE", "DTR+GMLake", OURS+" w/o mem", OURS]

### col 3
training_time_3 = np.array([
    [1.03, 1.162528912, 1.232265808, 1.310478598, 1.41444962, 1.568742084, 1.715860888, 2.063724185],
    [1.06, 1.070670422, 1.174388205, 1.247757133, 1.360342533, 1.487871833, 1.650953433, 1.967535433],
    [1.1, 2.429619505, 2.467108598, 2.499378055, 2.591231313, 2.722803372, 2.888778651, 3.210497958],
    [1.03, 1.061386644, 1.091859291, 1.114744116, 1.150900154, 1.180918196, 1.254335651, 1.299495311],
    [1.04, 1.068357857, 1.090592393, 1.111364606, 1.140198154, 1.180063976, 1.241361518, 1.28563083]
])
base_time = 530.336
training_time_3 = training_time_3 * base_time
memory_budget_ratios_3 = np.array([1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
memory_budget_ratios_3 = [str(i) for i in memory_budget_ratios_3]
# memory_budget_ratios_3 = memory_budget_ratios_3[1:]
# training_time_3 = [item[1:] for item in training_time_3]
# methods_3 = ["Megatron-LM", "DT-Control w/o mem", "DT-Control"]
methods_3 = ["DTR", "DTE", "DTR+GMLake", OURS+" w/o mem", OURS]

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
    # axs[0].bar(x + i * width, training_time[i], width=width, edgecolor='black', label=methods[i], color=colors[i],  zorder=10)
    for j, value in enumerate(training_time[i]):
        if value == 0:
            axs[0].text(x[j] + i * width, 0, 'Zero', rotation=90, ha='center', va='bottom', fontsize=10, color='red')
        else:
            axs[0].bar(x[j] + i * width, value, width=width, edgecolor='black', label=methods[i] if j == 0 else "", color=colors[i], zorder=10)
# axs[0].set_xlabel('Memory Budget Ratios', fontsize=label_font_size, fontweight=font_weight)
# axs[0].set_ylabel('Compute Overhead', fontsize=label_font_size, fontweight=font_weight)
axs[0].set_ylabel('Training Time ms/step', fontsize=label_font_size, fontweight=font_weight)
axs[0].set_title('Resnet50', fontsize=title_font_size)
axs[0].set_xticks(x + width * (len(methods) - 1) / 2)
axs[0].set_xticklabels(memory_budget_ratios, fontsize=tick_font_size, fontweight=font_weight)
axs[0].tick_params(axis='y', which='major', labelsize=tick_font_size)
axs[0].tick_params(axis='y', which='major', labelsize=tick_font_size)
# axs[0].set_yticklabels(['1.0','1.2','1.4','1.6','1.8','2.0'],fontsize=tick_font_size, fontweight=font_weight)
plt.setp(axs[0].get_yticklabels(), fontweight=font_weight, fontsize=tick_font_size)
# axs[0].set_ylim(1, 2)
for spine in axs[0].spines.values():
    spine.set_linewidth(border_width)
axs[0].grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)

# 第五个子图：reserve，假设为AlphaFold 8GPU
for i in range(len(methods)):
    # 修改索引为一维
    # axs[1].bar(x + i * width, training_time_2[i], width=width, edgecolor='black', label=methods[i], color=colors[i],  zorder=10)
    for j, value in enumerate(training_time_2[i]):
        if value == 0:
            axs[1].text(x[j] + i * width, 1.01, 'FAIL', rotation=90, ha='center', va='bottom', fontsize=8, color='red')
        else:
            axs[1].bar(x[j] + i * width, value, width=width, edgecolor='black', label=methods[i], color=colors[i], zorder=10)
axs[1].set_title('AlphaFold', fontsize=title_font_size)
axs[1].set_xlabel('Memory Budget Ratios', fontsize=label_font_size, fontweight=font_weight)
axs[1].set_xticks(x + width * (len(methods) - 1) / 2)
axs[1].set_xticklabels(memory_budget_ratios, fontsize=tick_font_size, fontweight=font_weight)
axs[1].tick_params(axis='y', which='major', labelsize=tick_font_size)
# axs[1].set_yticklabels(['1.0','1.2','1.4','1.6','1.8','2.0'],fontsize=tick_font_size, fontweight=font_weight)
plt.setp(axs[1].get_yticklabels(), fontweight=font_weight, fontsize=tick_font_size)

# axs[1].set_ylim(1, 2)
for spine in axs[1].spines.values():
    spine.set_linewidth(border_width)
axs[1].grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)

# 第六个子图：reserve，假设为GPT3-7.5B 64GPU
for i in range(len(methods_3)):
    # axs[2].bar(x_3 + i * width, training_time_3[i], width=width, edgecolor='black', label=methods_3[i], color=colors[i], zorder=10)
    for j, value in enumerate(training_time_3[i]):
        if value == 0:
            axs[2].text(x_3[j] + i * width, 1.01, 'FAIL', rotation=90, ha='center', va='bottom', fontsize=8, color='red')
        else:
            axs[2].bar(x_3[j] + i * width, value, width=width, edgecolor='black', label=methods_3[i], color=colors[i], zorder=10)
axs[2].set_title('ViT', fontsize=title_font_size)
axs[2].set_xticks(x_3 + width * (len(methods) - 1) / 2)
axs[2].set_xticklabels(memory_budget_ratios_3, fontsize=tick_font_size, fontweight=font_weight)
axs[2].tick_params(axis='y', which='major', labelsize=tick_font_size)
# axs[2].set_yticklabels([str(round(i*0.1, 1)) for i in range(10, 26, 2)],fontsize=tick_font_size, fontweight=font_weight)
plt.setp(axs[2].get_yticklabels(), fontweight=font_weight, fontsize=tick_font_size)
# axs[2].set_ylim(1, 2.5)
for spine in axs[2].spines.values():
    spine.set_linewidth(border_width)
axs[2].grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)

# 集中显示图例在所有图表的最上方
handles, labels = axs[0].get_legend_handles_labels()
# handles_, labels_ = axs[2].get_legend_handles_labels()
# handles += [handles_[0]]
# labels += [labels_[0]]
fig.legend(handles, labels, loc='upper center', ncol=len(methods)+1, fontsize=label_font_size, frameon=False)

plt.tight_layout()
# 调整布局以适应上方的图例
plt.subplots_adjust(top=0.85)
plt.savefig(SAVE_PREFIX+'exp_dp_compute_overhead.pdf', dpi=300, format='pdf', facecolor='white')