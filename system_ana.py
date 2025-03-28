import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


# 修改函数定义，添加边框粗细和字体大小参数
def plot_actions(log_file_path, img_name, 
                 border_width=5, title_font_size=32, label_font_size=20, tick_font_size=16, fontweight='bold'):
    plt.rcParams['font.weight'] = fontweight
    # 初始化用于存储每个 [action] 下不同数值的列表
    action_value_list = {}
    # 初始化用于存储每个 [action] 的出现频次
    action_counts = {}

    # 打开日志文件并逐行读取
    with open(log_file_path, 'r') as file:
        for line in file:
            # 使用正则表达式查找 [action] 和数值
            actions = re.findall(r'\[(\w+)\](\d+)us', line)
            for action, value in actions:
                # 如果 action 不在字典中，初始化一个新的列表来存储该 action 下的数值
                if action not in action_value_list:
                    action_value_list[action] = []
                # 添加该 action 下的数值
                action_value_list[action].append(int(value))
                # 更新该 action 的出现频次
                if action in action_counts:
                    action_counts[action] += 1
                else:
                    action_counts[action] = 1

    # 可视化部分
    actions = list(action_value_list.keys())
    num_actions = len(actions)

    # 创建一个包含两个子图的画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

    # 绘制箱线图展示每个操作对应的时间区间
    data = [action_value_list[action] for action in actions]
    boxplot = ax1.boxplot(data, labels=actions, patch_artist=True)

    # 设置箱线图的颜色
    colors = ['#ffd966', '#FF9b9b', '#5898d5', '#c85862', '#2ba02d']
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)

    # 将箱线图的 y 轴设置为对数坐标
    ax1.set_yscale('log')

    # 修改边框粗细
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(border_width)

    # 修改标题字体大小
    ax1.set_title('Time Interval for Each Action', fontsize=title_font_size, fontweight=fontweight)
    ax2.set_title('Frequency of Each Action', fontsize=title_font_size, fontweight=fontweight)

    # 修改标签字体大小
    ax1.set_ylabel('Time (us)', fontsize=label_font_size, fontweight=fontweight)
    ax2.set_ylabel('Frequency', fontsize=label_font_size, fontweight=fontweight)

    # 修改刻度字体大小
    ax1.tick_params(axis='both', which='major', labelsize=tick_font_size)
    ax2.tick_params(axis='both', which='major', labelsize=tick_font_size)
    

    # 绘制柱状图展示每个操作的发生频次
    index = np.arange(num_actions)
    bar_width = 0.4
    bars = ax2.bar(index, [action_counts[action] for action in actions], bar_width, edgecolor='black',
                   color=[tuple(list(mpl.colors.to_rgb(c)) + [1]) for c in colors[:num_actions]], zorder=10)

    ax2.grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)

    ax2.set_ylabel('Frequency', fontsize=label_font_size)
    ax2.set_xticks(index)
    ax1.set_xticklabels(actions, fontsize=tick_font_size, rotation=-10)
    ax2.set_xticklabels(actions, fontsize=tick_font_size, rotation=-10)
    ax2.tick_params(axis='both', which='major', labelsize=tick_font_size)

    # 添加数据标签到柱状图
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 1),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=label_font_size-6)

    ax1.grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)

    plt.tight_layout()
    plt.savefig(img_name, dpi=300)


# 修改函数定义，添加边框粗细和字体大小参数
def plot_overhead(border_width=5, title_font_size=32, label_font_size=20, tick_font_size=16, fontweight='bold'):
    plt.rcParams['font.weight'] = fontweight
    models = ['GPT3 1.7B PP4', 'Llama2 7B', 'AlphaFold', 'Resnet32']
    baseline_times = [9414.8, 1965.11, 4503.792763, 180.319744]
    with_dag_times = [9552.4, 2026.81666, 4839.603901, 180.34521]
    with_segman_times = [9449.6, 2021.646667, 4715.163708, 180.224479]
    all_times = [9583.9, 2023.596667, 4978.488684, 180.4388346]
    overhead_values = [[1.014615287, 1.031401123, 1.074561854, 1.000141227],
                       [1.003696308, 1.02877023, 1.046931765, 0.999471688],
                       [1.017961083, 1.029762541, 1.105399148, 1.000660441]]

    x = np.arange(len(models))
    width = 0.2


    bar_labels = ['PyTorch', 'w/ DSTP', 'w/ Mem optim', 'w/ DSTP & Mem optim']
    # 提取配色数组，支持自定义
    bar_colors = ['#ffd966', '#FF9b9b', '#5898d5', '#c85862']
    line_colors = ['#FF3333', '#007BFF', '#993333']

    fig, ax1 = plt.subplots(figsize=(16, 10))
    ax2 = ax1.twinx()

    rects1 = ax1.bar(x - width * 1.5, baseline_times, width, edgecolor='black', label=bar_labels[0], color=bar_colors[0], zorder=10)
    rects2 = ax1.bar(x - width / 2, with_dag_times, width, edgecolor='black', label=bar_labels[1], color=bar_colors[1], zorder=10)
    rects3 = ax1.bar(x + width / 2, with_segman_times, width, edgecolor='black', label=bar_labels[2], color=bar_colors[2], zorder=10)
    rects4 = ax1.bar(x + width * 1.5, all_times, width, edgecolor='black', label=bar_labels[3], color=bar_colors[3], zorder=10)

    # 修改标题字体大小
    ax1.set_title('Time and Overhead Comparison of Different Models', fontsize=title_font_size, fontweight=fontweight)
    # 修改标签字体大小
    ax1.set_ylabel('Time (ms)', fontsize=label_font_size, fontweight=fontweight)
    ax1.set_xticks(x)
    # 修改刻度字体大小
    ax1.set_xticklabels(models, fontsize=tick_font_size)

    markers = ['o','v','s']
    # 还有什么符号

    for i, ov in enumerate(overhead_values):
        ax2.plot(x, ov, label=f'Overhead {bar_labels[1+i]}', marker=markers[i], color=line_colors[i])

    # 修改标签字体大小
    ax2.set_ylabel('Runtime Overhead', fontsize=label_font_size, fontweight=fontweight)

    # 修改刻度字体大小
    ax1.tick_params(axis='both', which='major', labelsize=tick_font_size)
    ax2.tick_params(axis='both', which='major', labelsize=tick_font_size)

    # 获取两个轴的图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # 合并图例
    ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=label_font_size-12)

    # 修改边框粗细
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(border_width)

    plt.tight_layout()
    plt.savefig('overhead.pdf', dpi=300)

font_config = (5, 30, 28, 20)

log_file_path = './logs/system_overhead.log'
plot_actions(log_file_path, 'segman_overhead.pdf', *font_config)
log_file_path = './logs/dag_overhead.log'
plot_actions(log_file_path, 'dag_overhead.pdf', *font_config)

plot_overhead(*font_config)
