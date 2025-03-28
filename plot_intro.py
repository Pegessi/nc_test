import matplotlib.pyplot as plt

SAVE_PREFIX = "./figure/intro/"

def plot_x_2y_data(x, y1, y2, title, xlabel, ylabel, label1, label2, save_name):
    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300
    # 创建画布
    plt.figure(figsize=(10, 6)) 
    # 绘制第一个折线图
    plt.plot(x, y1, marker='o', label=label1, color='orange')
    # 绘制第二个折线图
    plt.plot(x, y2, marker='s', label=label2, color='green')
    # 添加标题和标签
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # 添加图例
    plt.legend()
    # 显示网格线
    plt.grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)
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


def plot_training_time():
    # 数据
    budget_ratio = ["100%", "80%", "70%", "60%", "50%", "40%", "30%"]
    DTR = [14.3113, 14.4345, 15.5789, 16.0468, 20.2838, 23.5911, 29.0501]
    # DTR = [14.3113, 16.0468, 16.9178, 17.4995, 20.2838, 23.5911, 29.0501]
    Megatron_LM = [14.3113, 15.4288, 15.9725, 17.0442, 17.6347, 18.7095, 19.8337]

    plot_x_2y_data(budget_ratio, DTR, Megatron_LM, 'Training Time Comparison', 'Budget Ratio', 'Time (s)', 'DTR', 'Megatron-LM', 'training_time.png')


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
    x_datas = ['betweeness centrailty', 'closeness centrailty', 'degree centrailty', 'cut points']
    evict_counts = [1255, 1279, 3975, 1413]
    remat_counts = [2973, 4349, 17795, 4786]
    normalize_time = [1, 1.738336714, 3.612576065, 1.863529412]
    normalize_time = [round(i, 2) for i in normalize_time]
    plot_x_multiple_y_data(x_datas, [evict_counts, remat_counts], 'Centraility Comparison', 'Metrics', 'Counts', ['evict counts', 'remat counts'], 'centraility.png',
                           right_y_list=[normalize_time], right_label_list=['normalize time per iter'], right_ylabel='Time (x)')


if __name__ == "__main__":
    plot_training_time()
    # plot_remat_counts()
    # plot_recurisve_counts()
    # plot_centraility()