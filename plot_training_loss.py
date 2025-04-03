import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

SAVE_PREFIX = './figure/'
BORDER_WIDTH = 2
FONT_SIZE = 8
TICK_FONT_SIZE = 8
FONT_WEIGHT = 'bold'


def plot_training_loss(files):
    datas = []
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
        datas.append(f_data[:1300])
    
    plt.figure(figsize=(5, 3))

    x = [512*2048*i for i in range(1, len(datas[0])+1)]  # batch size=512, seqlen=2048
    y1 = [eval(ele[1]) for ele in datas[0]]
    y2 = [eval(ele[1]) for ele in datas[1]]

    plt.plot(x, y1, label='Megatron-LM')
    plt.plot(x, y2, label='Nebula-Chain')

    # plt.title('training convergence')  # 添加标题
    plt.xlabel('tokens', fontsize=FONT_SIZE + 4, fontweight=FONT_WEIGHT)  # 添加X轴标签
    plt.ylabel('training loss', fontsize=FONT_SIZE + 4, fontweight=FONT_WEIGHT)  # 添加Y轴标签

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
    ax.xaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))  # Add this line

    plt.legend(loc='upper right', fontsize=FONT_SIZE)
    plt.tight_layout()  # 自动调整子图参数，以充分利用图表空间
    plt.savefig(SAVE_PREFIX+'loss.pdf')

if __name__ == "__main__":
    plot_training_loss(['./logs/GPT-1.7B_train_log_ml.log', './logs/GPT-1.7B_train_log_nc.log'])