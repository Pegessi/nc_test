import matplotlib.pyplot as plt
# import networkx as nx
import igraph as ig
import numpy as np
from matplotlib import cm
import matplotlib.colors as mcolors
import os

SAVE_PREFIX = './figure/test/llama27B/'
CUSTOM_NODES_PATH = '/data/wangzehua/Megatron-LM/nc_test/forkmerge/test_ancestor_nodes.txt'
OP_LOG_PATH = './logs/llama2_7B.log' # resnet50_once.log gpt3_350M_forward_once.log llama2_7B_once.log
SCALE_UP = 1000
ONLY_CUSTOM = int(os.environ.get('ONLY_CUSTOM', 0)) == 1
ENABLE_CUSTOM = int(os.environ.get('ENABLE_CUSTOM', 0)) == 1

SAVE_PREFIX = os.environ.get('SAVE_PREFIX', SAVE_PREFIX)
CUSTOM_NODES_PATH = os.environ.get('CUSTOM_NODES_PATH', CUSTOM_NODES_PATH)
OP_LOG_PATH = os.environ.get('OP_LOG_PATH', OP_LOG_PATH)
SCALE_UP = int(os.environ.get('SCALE_UP', SCALE_UP))


def rgba_to_hex(r, g, b, a):
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)
    a = int(a * 255)
    hex_r = hex(r)[2:].zfill(2).upper()
    hex_g = hex(g)[2:].zfill(2).upper()
    hex_b = hex(b)[2:].zfill(2).upper()
    hex_a = hex(a)[2:].zfill(2).upper()
    return "#{}{}{}{}".format(hex_r, hex_g, hex_b, hex_a)

def float_to_rgb_coolwarm(x, vmin, vmax):
    x = x+1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('coolwarm')
    rgba = cmap(norm(x))
    return rgba_to_hex(*rgba)
    
def float_to_rgb_lognorm_coolwarm(x, vmin, vmax):
    x = x+1
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('coolwarm')
    rgba = cmap(norm(x))
    return rgba_to_hex(*rgba)
    


def plot_compute_graph(filename, color_mode=None, plot_comm=False):
    r"""
    读取计算过程中的op日志，解析出对应的计算图并绘制
    """
    with open(filename, 'r') as f:
        data = f.readlines()
        data = [eval(row.replace('\n', '')) for row in data]
    nodes = set()
    edges = []
    nodes_times = {}
    SHOW_CUSTOM_CLUSTER = False
    CHECK_DEGREE = 6 # gpt 6
    SPLIT_INDEX = int(os.environ.get('SPLIT_INDEX', len(data)))
    data = data[:SPLIT_INDEX] #  1500 [467:2100]
    # 解析日志得到无向图
    for row in data:
        if row['INSTRUCTION'] != 'INSTRUCTION':
            continue
        # if row['name']=='add_' and row['inputs'][0] == row['outputs'][0]:
        #     continue
        for iid in row['inputs']:
            input_id = int(iid[1:])
            nodes.add(input_id)
            if input_id not in nodes_times.keys():
                nodes_times[input_id] = 0
            for oid in row['outputs']:
                output_id = int(oid[1:])
                nodes_times[input_id] += 1
                edges.append((input_id, output_id, {'op': row['name'], 'cost': row['compute_cost'], 'mem': row['mem_cost']}))
        for oid in row['outputs']:
            output_id = int(oid[1:])
            if output_id not in nodes_times.keys():
                nodes_times[output_id] = 0
            for iid in row['inputs']:
                nodes_times[output_id] += 1
            nodes.add(output_id)
    # print(nodes_times)

    counts = 0
    print(f"Nodes id with degree=={CHECK_DEGREE}")
    for k,v in nodes_times.items():
        if v == CHECK_DEGREE:
            counts+=1
            print(k, end=', ')     
    print('\ntotal', counts)

    # 构造图对象
    # g = ig.Graph(directed=ONLY_CUSTOM)
    g = ig.Graph(directed=True)
    g.add_vertices(max(list(nodes))+1)
    for ed in edges:
        g.add_edge(ed[0], ed[1])

    ##### custom clusters #####
    if SHOW_CUSTOM_CLUSTER:
        with open('./dynamo_community.txt', 'r') as f:
            data = f.readlines()
            data = [int(row.replace('\n','')) for row in data]
        n2c = data[:g.vcount()+1]
        import colorsys
        def distinct_colors(k):
            colors = []
            for i in range(k):
                # 色调分布在0到1之间
                hue = i / k
                # 选择饱和度和亮度为1，以获取鲜艳的颜色
                saturation = 1.0
                value = 1.0
                # 转换为RGB格式
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                # 将RGB从0-1范围转换为0-255范围，并取整
                rgb = tuple(int(x * 255) for x in rgb)
                colors.append(rgb)
            return colors
        g.vs["color"] = distinct_colors(max(n2c))
    
    ### alg 1
    # with open('/data/wangzehua/Megatron-LM/nc_test/forkmerge/critical_nodes.txt', 'r') as f:
    #     data = f.readlines()
    #     data = [eval(row) for row in data]
    # custom_colors = [1 for _ in range(g.vcount()+1)]
    # min_bc = min([row[2] for row in data])
    # max_bc = max([row[2] for row in data])
    # for row in data:
    #     custom_colors[row[0]] = float_to_rgb_coolwarm(row[2], 1+min_bc, 1+max_bc)
    # critical_nodes_colors = custom_colors


    ### alg 2
    if ENABLE_CUSTOM:
        file_path = CUSTOM_NODES_PATH
        # file_path = '/data/wangzehua/Megatron-LM/nc_test/forkmerge/main_path_nodes_restnet32.txt'
        with open(file_path, 'r') as f:
            data = f.readlines()
            data = [eval(row) for row in data]
        custom_colors = [1 for _ in range(g.vcount()+1)]
        for idx in data[0]:
            custom_colors[idx] = float_to_rgb_coolwarm(10000, 1, 10000)
        path_colors = custom_colors
        g.vs['color'] = path_colors

    # 删除孤立节点 [WARNING] - 这一步后要再重新生成布局才能去正常绘图
    isolated_vertices = [v.index for v in g.vs if g.degree(v.index) == 0]
    g.delete_vertices(isolated_vertices)

    scale_factor = int(g.vcount()/SCALE_UP)
    layout = g.layout_kamada_kawai()          # 力学布局
    
    if not ONLY_CUSTOM:
        # 度中心性
        degrees = g.degree()
        max_degree = max(degrees)
        degree_static = { i:0 for i in range(0, max_degree+1) }
        for i in range(len(degrees)):
            degree_static[degrees[i]] += 1
        print(degree_static)
        deg_colors = [float_to_rgb_coolwarm(b, min(degrees), max(degrees)) for b in degrees]
        # deg_colors = ["red" if degree == CHECK_DEGREE else "blue" for degree in degrees]
        print('max degree:', max(degrees))

        # 介数中心性
        betweenness = g.betweenness()
        bet_dict = { vertex:b for vertex, b in enumerate(betweenness)}
        bets = list(bet_dict.values())
        bet_colors = [float_to_rgb_lognorm_coolwarm(b, 1+min(bets), 1+max(bets)) for b in bets]
        print('betweeness(min, max, mean, median): ', np.min(bets), np.max(bets), np.mean(bets), np.median(bets))

        # 接近中心性
        closeness = g.closeness()
        closeness = [ 0 if np.isnan(val) else val  for val in closeness]
        clo_colors = [float_to_rgb_lognorm_coolwarm(b, 1+min(closeness), 1+max(closeness)) for b in closeness]
        print('closeness(min, max, mean, median): ', np.min(closeness), np.max(closeness), np.mean(closeness), np.median(closeness))


        # 求解割点
        cut_points = g.articulation_points()
        cut_points = [int(v) for v in cut_points]
        cut_colors = ['red' if i in cut_points else 'blue' for i in range(1, g.vcount()+1)]


        # community = g.community_infomap()
        # community = g.community_leiden(objective_function='modularity', n_iterations=100)   
        # 使用边介数的层次聚类方法
        # community = g.community_edge_betweenness()
        # 得到实际的社区划分
        # clusters = community.as_clustering()

    if ENABLE_CUSTOM:
        # # layout = g.layout_reingold_tilford()        # tree layout
        # if SHOW_CUSTOM_CLUSTER:
        #     ig.plot(g, SAVE_PREFIX+'graph_manual_comm.png', layout=layout, bbox=(1980*scale_factor, 1600*scale_factor), vertex_size=5*scale_factor, edge_arrow_size=0.5*scale_factor)
    
        ## custom colors
        custom_export_filename = os.environ.get('CUSTOM_EXPORT_FILENAME', 'graph_custom.png')
        ig.plot(g, SAVE_PREFIX+custom_export_filename, layout=layout, bbox=(1980*scale_factor, 1600*scale_factor), vertex_size=5*scale_factor, edge_arrow_size=0.25*scale_factor)
        
    if not ONLY_CUSTOM:
        vcolor_dict = {
            'degree': deg_colors,
            'betweeness': bet_colors,
            'closeness': clo_colors,
            'cut': cut_colors,
            # 'critical': critical_nodes_colors,
            # 'path': path_colors
        }
        if color_mode:
            g.vs["color"] = vcolor_dict[color_mode]
            ig.plot(g, SAVE_PREFIX+f'graph_{color_mode}.png', layout=layout, bbox=(1980*scale_factor, 1600*scale_factor), vertex_size=5*scale_factor, edge_arrow_size=0.5*scale_factor)
        else:
            for k,v in vcolor_dict.items():
                g.vs["color"] = v
                ig.plot(g, SAVE_PREFIX+f'graph_{k}.png', layout=layout, bbox=(1980*scale_factor, 1600*scale_factor), vertex_size=5*scale_factor, edge_arrow_size=0.5*scale_factor)
        
        if plot_comm:
            ig.plot(community, SAVE_PREFIX+'./graph_with_comm.png', layout=layout, bbox=(1980*scale_factor, 1600*scale_factor), vertex_size=5*scale_factor, edge_arrow_size=0.5*scale_factor, mark_groups=True)

            layout = community.cluster_graph().layout_kamada_kawai()
            ig.plot(community.cluster_graph(), SAVE_PREFIX+'./comm_graph.png', layout=layout, bbox=(1980*scale_factor, 1600*scale_factor), vertex_size=5*scale_factor, edge_arrow_size=0.5*scale_factor, mark_groups=True)
    
    # colors = ["red" if degree == CHECK_DEGREE else "blue" for degree in degrees]
    
    # clusters = ig.VertexClustering(g)
    # clusters._membership = n2c
        
def plot_cumulative_remat_counts(filenames, labels, attr_name, title, visual_percent):
    ay_list = []
    max_range = 0
    for filename in filenames:
        with open(filename, 'r') as f:
            data = f.readlines()
            data = [eval(row.replace('\n', '')) for row in data]
        # {"INSTRUCTION":"INSTRUCTION","cumulative_remat_counts":"1","name":"aten::add.Tensor","rid":"-4661336787295776820"}
        # {"INSTRUCTION":"CONSTANT","NAME":"dependecy counts","VALUE":0}
        ay = []
        for row in data:
            ay.append(int(row[attr_name]))
        # ay_list.append(sorted(ay)[:int(visual_percent * len(ay))])
        ay_list.append(ay)
        print(np.mean(ay_list[-1]), np.min(ay_list[-1]), np.max(ay_list[-1]), np.median(ay_list[-1]))
        max_range = max(max_range, len(ay_list[-1]))
    ax = [i for i in range(1, max_range+1)]
    for i in range(len(ay_list)):
        ay = ay_list[i]
        while len(ay) < max_range:
            ay.append(0)
        plt.bar(ax, ay, label=labels[i])  # 绘制柱状图，调整柱子宽度为0.8
    for li in ay_list:
        for _ in li:
            print(_, end=' ')
        print()
    plt.title(title)  # 添加标题
    plt.xlabel('events index')  # 添加X轴标签
    plt.ylabel('counts')  # 添加Y轴标签

    plt.xticks(rotation=90)  # 旋转X轴标签以提高可读性
    plt.grid(axis='y')  # 添加Y轴网格线

    plt.legend(loc='upper left')
    plt.tight_layout()  # 自动调整子图参数，以充分利用图表空间
    plt.savefig(SAVE_PREFIX+'cunum.jpg', dpi=600)


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
    
    x = [512*2048*i for i in range(1, len(datas[0])+1)]  # batch size=512, seqlen=2048
    y1 = [eval(ele[1]) for ele in datas[0]]
    y2 = [eval(ele[1]) for ele in datas[1]]

    plt.plot(x, y1, label='Megatron-LM')
    plt.plot(x, y2, label='Nebula-Chain')

    plt.title('training convergence')  # 添加标题
    plt.xlabel('tokens')  # 添加X轴标签
    plt.ylabel('training loss')  # 添加Y轴标签

    # plt.xticks(rotation=90)  # 旋转X轴标签以提高可读性
    # plt.grid(axis='y')  # 添加Y轴网格线

    plt.legend(loc='upper right')
    plt.tight_layout()  # 自动调整子图参数，以充分利用图表空间
    plt.savefig(SAVE_PREFIX+'loss.jpg', dpi=600)


    # return datas

    

if __name__ == '__main__':
    ### 画计算图的 degree betweeness closeness cut
    plot_compute_graph(OP_LOG_PATH) # resnet50_once.log gpt3_350M_forward_once.log llama2_7B_once.log


    # fig_range = [3, 4, 5]
    # fns = [ './logs/remat/remat_counts_' + str(i) + '0%.log' for i in fig_range]
    # labels = [ str(i) + r"0% budget" for i in fig_range]
    # attr_name = 'cumulative_remat_counts' # 'cumulative_remat_counts'
    # title = 'Cumulatvie Remat Counts' # 'Cumulatvie Remat Counts'
    # visual_percent = 1

    # plot_cumulative_remat_counts(fns, labels, attr_name, title, visual_percent)


    ### 实验说明递归改善的数据
    # fns = [ './logs/remat/remat_counts_30%.log', './logs/remat/remat_nc_counts_30%.log' ]
    # labels = [ r"30% budget dtr", r"30% budget NC" ]
    # attr_name = 'cumulative_remat_counts' # 'cumulative_remat_counts'
    # title = 'Cumulatvie Remat Counts Comparsion' # 'Cumulatvie Remat Counts'
    # visual_percent = 1

    # plot_cumulative_remat_counts(fns, labels, attr_name, title, visual_percent)

    # plot_training_loss(['./logs/GPT-1.7B_train_log_ml.log', './logs/GPT-1.7B_train_log_nc.log'])
