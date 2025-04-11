import matplotlib.pyplot as plt
from memlogs.plot_mem_timeline import get_mem_events
from matplotlib.ticker import FuncFormatter
import numpy as np
import os

SAVE_PREFIX = os.environ.get('SAVE_PREFIX', "./figure/intro/")
FONT_WEIGHT = 'bold'
FONT_SIZE = 24
TICK_FONT_SIZE = 18
BORDER_WIDTH = 5
# 设置font family
plt.rcParams['font.family'] = 'Arial'

def plot_x_2y_data(x, y1, y2, title, xlabel, ylabel, label1, label2, save_name, figsize=(10, 6)):
    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300
    # 创建画布
    plt.figure(figsize=figsize)
    # 绘制第一个折线图
    plt.plot(x, y1, marker='o', label=label1, color='orange')
    # 绘制第二个折线图
    plt.plot(x, y2, marker='s', label=label2, color='green')
    # 添加标题和标签
    # plt.title(title)
    plt.xlabel(xlabel, fontsize=FONT_SIZE + 4, fontweight=FONT_WEIGHT)
    plt.ylabel(ylabel, fontsize=FONT_SIZE + 4, fontweight=FONT_WEIGHT)

    plt.xticks(fontsize=TICK_FONT_SIZE, fontweight=FONT_WEIGHT)
    plt.yticks(fontsize=TICK_FONT_SIZE, fontweight=FONT_WEIGHT)
    # 添加图例
    plt.legend(fontsize=TICK_FONT_SIZE)
    # 显示网格线
    plt.grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)
    # 设置图像边框大小
    for spine in plt.gca().spines.values():
        spine.set_linewidth(BORDER_WIDTH)
    # 调整布局
    plt.tight_layout()
    # 保存图片
    plt.savefig(SAVE_PREFIX + save_name)


def plot_x_multiple_y_data(x, y_list, title, xlabel, ylabel, label_list, save_name, right_y_list=None, right_label_list=None, right_ylabel=None):
    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300
    # 创建画布
    plt.figure(figsize=(10, 6)) 

    # 遍历每个 y 数据列表并绘制左 y 轴数据
    for i, y in enumerate(y_list):
        plt.plot(x, y, marker=['o', 's', '^', 'v', 'D'][i % 5], label=label_list[i])
        # 为每个数据点添加文本标注
        # for j, (x_val, y_val) in enumerate(zip(x, y)):
        #     plt.text(x_val, y_val, str(y_val), ha='center', va='bottom', fontsize=8)

    # 添加标题和标签
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 添加左 y 轴图例
    plt.legend(loc='upper left')

    # 如果传入了右 y 轴数据，则绘制右 y 轴数据
    if right_y_list and right_label_list and right_ylabel:
        ax2 = plt.twinx()  # 创建第二个 y 轴
        for i, y in enumerate(right_y_list):
            ax2.plot(x, y, marker=['o', 's', '^', 'v', 'D'][(i + len(y_list)) % 5], label=right_label_list[i], color=['r', 'g', 'b', 'c', 'm'][(i + len(y_list)) % 5])
            # 为每个数据点添加文本标注
            for j, (x_val, y_val) in enumerate(zip(x, y)):
                ax2.text(x_val, y_val + 0.05, str(y_val), ha='center', va='baseline', fontsize=8, ) # color=['r', 'g', 'b', 'c', 'm'][(i + len(y_list)) % 5]
        ax2.set_ylabel(right_ylabel)
        # 添加右 y 轴图例
        ax2.legend(loc='upper right')

    # 显示网格线
    plt.grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)
    # 调整布局
    plt.tight_layout()
    # 保存图片
    plt.savefig(SAVE_PREFIX + save_name)


def plot_x_6y_data(x, y1, y2, y3, y4, y5, y6, title, xlabel, ylabel, label1, label2, label3, label4, label5, label6, save_name):
    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300
    # 创建一个1行3列的画布
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # plt.rcParams.update({'font.weight': FONT_WEIGHT, 'font.size': FONT_SIZE})
    x_labels = [f'{xlabel}\n(a)', f'{xlabel}\n(b)', f'{xlabel}\n(c)']
    # x_labels = ['(a)Training time', f'(b)recompute counts\n{xlabel}', '(c)recurisive counts']

    # 定义一个函数，将刻度值转换为以 K 为单位的字符串
    def k_formatter(x, pos):
        return '{:.0f}K'.format(x / 1000)

    # 绘制第一个子图
    axes[0].plot(x, y1, marker='o', label=label1, color='orange')
    axes[0].plot(x, y2, marker='s', label=label2, color='green')
    axes[0].set_xlabel(x_labels[0], fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
    axes[0].set_ylabel('Time/s', fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
    axes[0].tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    plt.setp(axes[0].get_xticklabels(), fontweight=FONT_WEIGHT)
    plt.setp(axes[0].get_yticklabels(), fontweight=FONT_WEIGHT)
    axes[0].legend(fontsize=TICK_FONT_SIZE, frameon=False)
    axes[0].grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)

    # 绘制第二个子图
    axes[1].plot(x, y3, marker='o', label=label3, color='orange')
    axes[1].plot(x, y4, marker='s', label=label4, color='green')
    axes[1].set_xlabel(x_labels[1], fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
    axes[1].set_ylabel('Remat Counts', fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
    axes[1].tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    plt.setp(axes[1].get_xticklabels(), fontweight=FONT_WEIGHT)
    plt.setp(axes[1].get_yticklabels(), fontweight=FONT_WEIGHT)
    # 使用自定义的格式化函数
    axes[1].yaxis.set_major_formatter(FuncFormatter(k_formatter))
    # axes[1].legend(fontsize=TICK_FONT_SIZE)
    axes[1].grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)

    # 绘制第三个子图
    axes[2].plot(x, y5, marker='o', label=label5, color='orange')
    axes[2].plot(x, y6, marker='s', label=label6, color='green')
    axes[2].set_xlabel(x_labels[2], fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
    axes[2].set_ylabel('Recursive Counts', fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
    axes[2].tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    plt.setp(axes[2].get_xticklabels(), fontweight=FONT_WEIGHT)
    plt.setp(axes[2].get_yticklabels(), fontweight=FONT_WEIGHT)
    # 使用自定义的格式化函数
    axes[2].yaxis.set_major_formatter(FuncFormatter(k_formatter))
    # axes[2].legend(fontsize=TICK_FONT_SIZE)
    axes[2].grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)

    # 设置图像边框大小
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(BORDER_WIDTH)
    # plt.xticks(fontsize=TICK_FONT_SIZE, fontweight=FONT_WEIGHT)
    # plt.yticks(fontsize=TICK_FONT_SIZE, fontweight=FONT_WEIGHT)
    # 调整布局
    plt.tight_layout()
    # 保存图片
    plt.savefig(SAVE_PREFIX + save_name)

def plot_training_time():
    # 数据
    # budget_ratio = ["100%", "80%", "70%", "60%", "50%", "40%", "30%"]
    budget_ratio = ["1", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3"]
    # DTR = [14.3113, 14.4345, 15.5789, 16.0468, 20.2838, 23.5911, 29.0501]
    DTR = [14.3113, 16.0468, 16.9178, 17.4995, 20.2838, 23.5911, 29.0501]
    Megatron_LM = [14.3113, 15.4288, 15.9725, 17.0442, 17.6347, 18.7095, 19.8337]

    plot_x_2y_data(budget_ratio, DTR, Megatron_LM, 'Training Time Comparison', 'Budget Ratio', 'Time (s)', 'DTR', 'Megatron-LM', 'training_time.pdf')


def plot_remat_counts():
    budget_ratio = ["100%", "80%", "70%", "60%", "50%", "40%", "30%"]
    DTR = [0, 8961, 60496, 168853, 337652, 443568, 563662]
    Megatron_LM = [0, 117120, 156160, 234240, 273280, 351360, 390400]

    plot_x_2y_data(budget_ratio, DTR, Megatron_LM, 'Remat Counts Comparison', 'Budget Ratio', 'Remat Counts', 'DTR', 'Megatron-LM', 'remat_counts.png')

def plot_recurisve_counts():
    budget_ratio = ["100%", "80%", "70%", "60%", "50%", "40%", "30%"]
    DTR = [0, 8809, 60344, 168573, 336580, 442094, 561034]
    Megatron_LM = [0, 132480, 176640, 264960, 309120, 397440, 441600]

    plot_x_2y_data(budget_ratio, DTR, Megatron_LM, 'Recursive Counts Comparison', 'Budget Ratio', 'Recursive Counts', 'DTR', 'Megatron-LM', 'recursive_counts.png')

def plot_centraility():
    x_datas = ['betweeness centrailty', 'closeness centrailty', 'degree centrailty', 'Cut Point points']
    evict_counts = [1255, 1279, 3975, 1413]
    remat_counts = [2973, 4349, 17795, 4786]
    normalize_time = [1, 1.738336714, 3.612576065, 1.863529412]
    normalize_time = [round(i, 2) for i in normalize_time]
    plot_x_multiple_y_data(x_datas, [evict_counts, remat_counts], 'Centraility Comparison', 'Metrics', 'Counts', ['evict counts', 'remat counts'], 'centraility.png',
                           right_y_list=[normalize_time], right_label_list=['normalize time per iter'], right_ylabel='Time (x)')

def plot_fragmentation():
    budget_ratio = ["1", "0.8", "0.7", "0.6", "0.5", "0.4"]
    y1 = [0.051413, 0.20383, 0.23629, 0.251684806, 0.255173, 0.23967]
    y2 = [0.053683, 0.073947, 0.089294, 0.10502, 0.129744, 0.157866]

    plot_x_2y_data(budget_ratio, y1, y2, 'Fragmentation Comparison', 'Budget Ratio', 'Fragmentation', 'DTR', 'Megatron-LM', 
                   'back_fragmentation.pdf', figsize=(6, 6.5))

def plot_time_remat_recurisive():
    budget_ratio = ["0.8", "0.7", "0.6", "0.5", "0.4"]
    # DTR_y1 = [14.3113, 16.0468, 16.9178, 17.4995, 20.2838, 23.5911, 29.0501]
    DTR_y1 = [18.8065, 20.9372, 22.5164, 24.9875, 29.3384]
    Megatron_LM_y1 = [15.4288, 15.9725, 17.0442, 17.6347, 18.7095]
    DTR_y2 = [292481, 358065, 417885, 516485, 563662]
    Megatron_LM_y2 = [117120, 156160, 234240, 273280, 351360]
    DTR_y3 = [347626, 361011, 442094, 514736, 561034]
    Megatron_LM_y3 = [132480, 176640, 264960, 309120, 397440]

    plot_x_6y_data(budget_ratio, DTR_y1, Megatron_LM_y1, DTR_y2, Megatron_LM_y2, DTR_y3, Megatron_LM_y3, 'Time, Remat Counts, Recursive Counts Comparison',
                    'Memory Budget Ratio', 'Counts', 'DTR', 'Megatron-LM', 'DTR', 'Megatron-LM', 'DTR', 'Megatron-LM', 'training_time_remat_recursive.pdf')


def plot_frag_and_timeline():
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import FuncFormatter

    # 创建一个包含左右两栏的布局，比例为2:8
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 10)

    # 左栏：绘制 plot_fragmentation 的内容
    ax1 = plt.subplot(gs[:, :3])

    budget_ratio = ["1", "0.8", "0.7", "0.6", "0.5", "0.4"]
    y1 = [0.051413, 0.20383, 0.23629, 0.251684806, 0.255173, 0.23967]
    y2 = [0.053683, 0.073947, 0.089294, 0.10502, 0.129744, 0.157866]

    # 直接在 ax1 上绘图
    # 设置图片清晰度
    ax1.figure.set_dpi(300)
    # 绘制第一个折线图
    ax1.plot(budget_ratio, y1, marker='o', label='DTR', color='orange')
    # 绘制第二个折线图
    ax1.plot(budget_ratio, y2, marker='s', label='Megatron-LM', color='green')
    # 添加标题和标签
    ax1.set_xlabel('Budget Ratio\n(a)', fontsize=FONT_SIZE + 4, fontweight=FONT_WEIGHT)
    ax1.set_ylabel('Fragmentation', fontsize=FONT_SIZE + 4, fontweight=FONT_WEIGHT)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    plt.setp(ax1.get_xticklabels(), fontweight=FONT_WEIGHT)
    plt.setp(ax1.get_yticklabels(), fontweight=FONT_WEIGHT)
    # 添加图例
    ax1.legend(fontsize=TICK_FONT_SIZE)
    # 显示网格线
    ax1.grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)
    # 设置图像边框大小
    for spine in ax1.spines.values():
        spine.set_linewidth(BORDER_WIDTH)

    # 右栏：绘制原有的内存时间线内容
    ax2 = plt.subplot(gs[:, 3:])

    log_files = [
        '/data/wangzehua/Megatron-LM/plot/memlogs/GMLAKE-B7.8-4step-2025-2-7-15-34-59-2480311-default.log',
        '/data/wangzehua/Megatron-LM/plot/memlogs/DTR-B6.5-4step-2025-2-7-15-22-32-2450911-default.log',
        '/data/wangzehua/Megatron-LM/plot/memlogs/TORCH-2025-2-1-11-31-35-3009862-default.log'
    ]

    labels = [
        ['GMLAKE_alloc', 'GMLAKE_reserve'],
        ['DTR_alloc', 'DTR_reserve'],
        ['TORCH_alloc', 'TORCH_reserve']
    ]

    datas = []
    for idx, file_path in enumerate(log_files):
        ax, y1, y2 = get_mem_events(file_path)
        ax = [x / 1000 for x in ax]
        ax2.plot(ax, y1, label=labels[idx][0])
        ax2.plot(ax, y2, label=labels[idx][1])
        # datas.append(ax, y1, y2)

    def k_formatter(x, pos):
        return '{:.0f}K'.format(x / 1000)

    # 设置x轴y轴刻度字体大小
    # ax2.set_xticks(fontsize=TICK_FONT_SIZE, fontweight=FONT_WEIGHT)
    # ax2.set_yticks(fontsize=TICK_FONT_SIZE, fontweight=FONT_WEIGHT)
    plt.setp(ax2.get_xticklabels(), fontsize=TICK_FONT_SIZE, fontweight=FONT_WEIGHT)
    plt.setp(ax2.get_yticklabels(), fontsize=TICK_FONT_SIZE, fontweight=FONT_WEIGHT)
    ax2.set_xlabel('Time/s\n(b)', fontsize=FONT_SIZE + 4, fontweight=FONT_WEIGHT)
    ax2.set_ylabel('Memory Usage/MB', fontsize=FONT_SIZE + 4, fontweight=FONT_WEIGHT)
    ax2.yaxis.set_major_formatter(FuncFormatter(k_formatter))
    # ax2.set_title('Memory Timeline', fontsize=FONT_SIZE + 4, fontweight=FONT_WEIGHT)
    # 设置图像边框大小
    for spine in ax2.spines.values():
        spine.set_linewidth(BORDER_WIDTH)
    ax2.grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)
    # 设置图例位置在左上角
    ax2.legend(fontsize=TICK_FONT_SIZE)

    plt.tight_layout()
    plt.savefig(SAVE_PREFIX+'back_frag_and_timeline.pdf', dpi=400)


def plot_all_centrality():
    colors = ['#FF9b9b', '#5898d5', '#2ba02d', '#c85862']
    colors.reverse()
    marks = ['o', 's', '^', 'v', 'd']
    data1 = {
        'Betweenness Centrality': [2.0165, 2.0716, 2.2133, 2.2899, 2.4037, 2.6171],
        'Closeness Centrality': [2.0991, 2.1396, 2.6942, 2.753, 5.1557, 6.8639],
        'Degree Centrality': [2.0261, 2.2029, 2.7548, 3.0501, 2.9872, 3.4194],
        'Cut Point': [1.9206, 2.0485, 2.3231, 2.8827, 3.0316, 3.4527]
    }

    # 第二组数据
    data2 = {
        'Betweenness Centrality': [248, 336, 1273, 3680, 5536, 7743],
        'Closeness Centrality': [298, 352, 4665, 6428, 11118, 30112],
        'Degree Centrality': [293, 402, 4282, 5855, 6629, 10094],
        'Cut Point': [153, 330, 2269, 4217, 7757, 9980]
    }

    # 指标值
    categories = [str(i) for i in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]]

    # 创建一个包含两个子图的画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 第一个子图：绘制第二组数据的柱状图，添加边框
    index = np.arange(len(categories))
    bar_width = 0.1
    for i, (key, values) in enumerate(data2.items()):
        ax1.bar(index + i * bar_width, values, color=colors[i], width=bar_width, label=key, edgecolor='black')

    # 设置第一个子图的X轴刻度
    ax1.set_xticks(index + bar_width * (len(data2) - 1) / 2)
    ax1.set_xticklabels(categories)

    # 设置第一个子图的轴标签和标题
    ax1.set_xlabel('Memory Budget Ratio', fontsize=FONT_SIZE + 4, fontweight=FONT_WEIGHT)
    ax1.set_ylabel('Remat conuts per step', fontsize=FONT_SIZE + 4, fontweight=FONT_WEIGHT)

    # 设置第一个子图的刻度字体大小和权重
    ax1.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE+2)
    plt.setp(ax1.get_xticklabels(), fontweight=FONT_WEIGHT)
    plt.setp(ax1.get_yticklabels(), fontweight=FONT_WEIGHT)

    # 显示第一个子图的图例
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='upper left', fontsize=TICK_FONT_SIZE+2, frameon=False)

    # 显示第一个子图的网格线
    ax1.grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)

    # 设置第一个子图的图像边框大小
    for spine in ax1.spines.values():
        spine.set_linewidth(BORDER_WIDTH)

    # 第二个子图：绘制第一组数据的折线图，并添加阴影
    for i, (key, values) in enumerate(data1.items()):
        line, = ax2.plot(categories, values, color=colors[i], marker=marks[i], label=key)
        # ax2.fill_between(categories, [v - 0.01 for v in values], [v + 0.01 for v in values], color='black', alpha=0.8)

    # 设置第二个子图的轴标签和标题
    ax2.set_xlabel('Memory Budget Ratio', fontsize=FONT_SIZE + 4, fontweight=FONT_WEIGHT)
    ax2.set_ylabel('Training time s/step', fontsize=FONT_SIZE + 4, fontweight=FONT_WEIGHT)

    # 设置第二个子图的刻度字体大小和权重
    ax2.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE+2)
    plt.setp(ax2.get_xticklabels(), fontweight=FONT_WEIGHT)
    plt.setp(ax2.get_yticklabels(), fontweight=FONT_WEIGHT)

    # 显示第二个子图的图例
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines2, labels2, loc='upper left', fontsize=TICK_FONT_SIZE+2, frameon=False)

    # 显示第二个子图的网格线
    ax2.grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)

    # 设置第二个子图的图像边框大小
    for spine in ax2.spines.values():
        spine.set_linewidth(BORDER_WIDTH)

    # 调整布局
    plt.tight_layout()

    plt.savefig(SAVE_PREFIX+'back_different_centrality.pdf')


if __name__ == "__main__":
    plot_time_remat_recurisive()
    plot_all_centrality()
    # plot_frag_and_timeline()

    # plot_fragmentation()
    # plot_training_time()
    # plot_remat_counts()
    # plot_recurisve_counts()
    # plot_centraility()