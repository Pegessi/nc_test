import matplotlib.pyplot as plt
# import networkx as nx
import igraph as ig
import numpy as np

SAVE_PREFIX = './figure/'


def plot_compute_graph(filename):
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
    SCALE_UP = 1000
    data = data[:] # [467:2100]
    # 解析日志得到无向图
    for row in data:
        if row['INSTRUCTION'] != 'INSTRUCTION':
            continue
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
    g = ig.Graph()
    # g = ig.Graph(directed=True)
    g.add_vertices(max(list(nodes))+1)
    # for node in list(nodes):
    #     g.add_vertex(node)
    
    for ed in edges:
        g.add_edge(ed[0], ed[1])

    # 度中心性
    degrees = g.degree()
    max_degree = max(degrees)
    degree_static = { i:0 for i in range(0, max_degree+1) }
    for i in range(len(degrees)):
        degree_static[degrees[i]] += 1
    print(degree_static)
    # colors = ["red" if degree == CHECK_DEGREE else "blue" for degree in degrees]
    print('max degree:', max(degrees))

    # 介数中心性
    # betweenness = g.betweenness()
    # bet_dict = { vertex:b for vertex, b in enumerate(betweenness)}
    # # print(bet_dict)
    # bets = list(bet_dict.values())
    # colors = ["red" if b > 70911 else "blue" for b in bets]
    # print(np.min(bets), np.max(bets), np.mean(bets), np.median(bets))

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

    # 删除孤立节点 [WARNING] - 这一步后要再重新生成布局才能去正常绘图
    isolated_vertices = [v.index for v in g.vs if g.degree(v.index) == 0]
    g.delete_vertices(isolated_vertices)

    # community = g.community_infomap()
    community = g.community_leiden(objective_function='modularity', n_iterations=100)   
    # 使用边介数的层次聚类方法
    # community = g.community_edge_betweenness()
    # 得到实际的社区划分
    # clusters = community.as_clustering()

    layout = g.layout_kamada_kawai()          # 力学布局
    # layout = g.layout_reingold_tilford()        # tree layout
    scale_factor = int(g.vcount()/SCALE_UP)
    if SHOW_CUSTOM_CLUSTER:
        ig.plot(g, SAVE_PREFIX+'./graph_manual_comm.png', layout=layout, bbox=(1980*scale_factor, 1600*scale_factor), vertex_size=5*scale_factor, edge_arrow_size=0.5*scale_factor)
    
    degrees = g.degree()
    g.vs["color"] = ["red" if degree == CHECK_DEGREE else "blue" for degree in degrees]
    ig.plot(g, SAVE_PREFIX+'./graph_degree.png', layout=layout, bbox=(1980*scale_factor, 1600*scale_factor), vertex_size=5*scale_factor, edge_arrow_size=0.5*scale_factor)
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


if __name__ == '__main__':
    ### 画计算图的
    plot_compute_graph('./logs/gpt3_350M_forward_once.log') # resnet50_once.log pp4_ml_gpt.log llama_op_once.log


    fig_range = [3, 4, 5]
    fns = [ './logs/remat/remat_counts_' + str(i) + '0%.log' for i in fig_range]
    labels = [ str(i) + r"0% budget" for i in fig_range]
    attr_name = 'cumulative_remat_counts' # 'cumulative_remat_counts'
    title = 'Cumulatvie Remat Counts' # 'Cumulatvie Remat Counts'
    visual_percent = 1

    plot_cumulative_remat_counts(fns, labels, attr_name, title, visual_percent)
