import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os

SAVE_PREFIX = os.environ.get('SAVE_PREFIX', "./figure/")
OURS = os.environ.get('OURS', "DT-Control")
BORDER_WIDTH = 2
FONT_SIZE = 10
TICK_FONT_SIZE = 8
FONT_WEIGHT = 'bold'

def read_data(files, model='gpt'):
    datas = []
    if model == 'gpt':
        for file_name in files:
            f_data = []
            with open(file_name, 'r') as f:
                data = f.readlines()
                data = [row.replace('\n', '') for row in data]
                for row in data:
                    parts = row.split('|')
                    consumed_samples = None
                    lm_loss = None
                    for part in parts:
                        if "consumed samples:" in part:
                            consumed_samples = part.split(':')[1].strip()
                        elif "lm loss:" in part:
                            lm_loss = part.split(':')[1].strip()
                    print("Consumed samples:", consumed_samples)
                    print("LM loss:", lm_loss)
                    if consumed_samples and lm_loss:
                        f_data.append((consumed_samples, lm_loss))
            datas.append(f_data[:200])
    else:
        for file_name in files:
            f_data = []
            with open(file_name, 'r') as f:
                data = f.readlines()
                data = [row.replace('\n', '') for row in data]
                for row in data:
                    item = eval(row)
                    loss = item['loss']
                    f_data.append(loss)
            datas.append(f_data)
    return datas

def plot_training_loss(x, y1, y2, label1, label2, xlabel, ylabel, save_name, sub=False):
    plt.figure(figsize=(5, 3))

    # x = [512*2048*i for i in range(1, len(y1)+1)]  # batch size=512, seqlen=2048
    # y1 = [eval(ele[1]) for ele in datas[0]]
    # y2 = [eval(ele[1]) for ele in datas[1]]

    plt.plot(x, y1, label=label1)
    plt.plot(x, y2, label=label2)

    # plt.title('training convergence')  # 添加标题
    plt.xlabel(xlabel, fontsize=FONT_SIZE + 4, fontweight=FONT_WEIGHT)  # 添加X轴标签
    plt.ylabel(ylabel, fontsize=FONT_SIZE + 4, fontweight=FONT_WEIGHT)  # 添加Y轴标签

    plt.xticks(fontsize=TICK_FONT_SIZE, fontweight=FONT_WEIGHT)
    plt.yticks(fontsize=TICK_FONT_SIZE, fontweight=FONT_WEIGHT)

    # plt.xticks(rotation=90)  # 旋转X轴标签以提高可读性
    # plt.grid(axis='y')  # 添加Y轴网格线
    for spine in plt.gca().spines.values():
        spine.set_linewidth(BORDER_WIDTH)

    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True)) 

    plt.legend(loc='upper right', fontsize=FONT_SIZE)

    if sub:
        # 确定局部放大的范围，这里假设放大 x 轴从 2 到 4 的部分
        x_min, x_max = 188, 195
        # 找到对应放大范围的索引
        x_indices = [i for i, val in enumerate(x) if x_min <= val <= x_max]
        if x_indices:
            x_zoom = [x[i] for i in x_indices]
            y1_zoom = [y1[i] for i in x_indices]
            y2_zoom = [y2[i] for i in x_indices]

            # 创建局部放大子图
            axins = ax.inset_axes([0.55, 0.05, 0.3, 0.3])
            axins.plot(x_zoom, y1_zoom, linestyle='-')
            axins.plot(x_zoom, y2_zoom, linestyle='--')

            # import ipdb; ipdb.set_trace()
            # for i in range(len(x_zoom)):
            #     if i not in [2]:
            #         continue
            #     axins.text(x_zoom[i], y1_zoom[i], f'{y1_zoom[i]:.4f}', fontsize=6, fontweight=FONT_WEIGHT, ha='center', va='top')
            #     axins.text(x_zoom[i], y2_zoom[i], f'{y2_zoom[i]:.4f}', fontsize=6, fontweight=FONT_WEIGHT, ha='center', va='bottom')

            # 设置子图的坐标轴范围
            # axins.set_xlim(x_min, x_max)
            # axins.set_ylim(min(min(y1_zoom), min(y2_zoom)), max(max(y1_zoom), max(y2_zoom)))
            axins.set_xticks([])
            axins.set_yticks([])


            # 绘制方框和连接线
            ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.tight_layout()  # 自动调整子图参数，以充分利用图表空间
    plt.savefig(SAVE_PREFIX+save_name)

if __name__ == "__main__":
    gpt_files = ['./logs/GPT-1.7B_train_log_ml.log', './logs/GPT-1.7B_train_log_nc.log']
    llama_files = ['./logs/Llama2-7B-lora-base-loss.log', './logs/Llama2-7B-lora-ours-loss.log']
    
    gpt_y1, gpt_y2 = read_data(gpt_files, model='gpt')
    llama_y1, llama_y2 = read_data(llama_files, model='llama')
    x1 = [i for i in range(1, len(gpt_y1)+1)]
    x2 = [i for i in range(1, len(llama_y1)+1)]
    gpt_y1 = [float(ele[1]) for ele in gpt_y1]
    gpt_y2 = [float(ele[1]) for ele in gpt_y2]
    llama_y1 = [float(ele) for ele in llama_y1]
    llama_y2 = [float(ele) for ele in llama_y2]
    
    plot_training_loss(x1, gpt_y1, gpt_y2, 'Megatron-LM', OURS, 'Training Step', 'Training Loss', 'exp_training_loss_gpt.pdf')
    plot_training_loss(x2, llama_y1, llama_y2, 'Accelerate', OURS, 'Training Step', 'Training Loss', 'exp_training_loss_llama.pdf', True)