from collections import deque, Counter
import time
import os


# 定义节点类
class Node:
    def __init__(self, name):
        self.name = name
        self.distance = float('inf')  # 到起始节点的最短距离
        self.neighbors = []  # 邻接表，存储邻居节点和边的权重
        self.in_edges = []  # 存储指向该节点的边的起始节点
        self.in_degree = 0  # 节点的入度
        self.out_degree = 0  # 节点的出度
        self.level = 0  # 节点的层级

    def add_neighbor(self, neighbor, weight):
        self.neighbors.append((neighbor, weight))
        neighbor.in_edges.append(self)
        self.out_degree += 1
        neighbor.in_degree += 1

    def __repr__(self):
        return f"{self.name}"

# 动态有向无环图单源最短路径类
class DynamicDAGShortestPath:
    def __init__(self, start_node_name):
        self.start = Node(start_node_name)
        self.start.distance = 0
        self.start.level = 0
        self.nodes = {start_node_name: self.start}
        self.queue = deque()
        self.sorted_nodes = [self.start]  # 按距离和层级排序的节点序列
        self.distance_to_max_level_node = {0: self.start}  # 记录每个距离对应的最大层级节点
        self.distance_to_last_change_time = {0: time.time()}
        self.REMOVE_COUNTS = 0
        self.ADD_COUNTS = 0
        self.last_change_indexs = {}
        self.changed_last_indexs = []
        self.last_change_index = 0
        self.debug_ids = []
        self.window_size = 10

        self.last_time_order_nodes = []
        self.last_same_idx = 0

    def add_node(self, node_name):
        if node_name not in self.nodes:
            new_node = Node(node_name)
            self.nodes[node_name] = new_node

    def _update_last_changes(self, index):
        self.changed_last_indexs.append(index)
        if index not in self.last_change_indexs.keys():
            self.last_change_indexs[index] = 0
        self.last_change_indexs[index] += 1
        # self.last_change_index = max(self.last_change_indexs, key=self.last_change_indexs.get)
        # 找到 self.changed_last_indexs[:-10] 中出现最多的数字
        if len(self.changed_last_indexs) > self.window_size:
            counter = Counter(self.changed_last_indexs[-self.window_size:])
            self.last_change_index = counter.most_common(1)[0][0]

    def _insert_sorted(self, node):
        # 插入节点到按距离和层级排序的序列中
        index = 0
        while index < len(self.sorted_nodes):
            current = self.sorted_nodes[index]
            if node.distance < current.distance:  # 出现了distance更小的节点，直接插入
                break
            elif node.distance == current.distance: # 此时只会是同一个节点，skip
                return
            index += 1

        # 插入新节点
        self.sorted_nodes.insert(index, node)
        self._update_last_changes(index)
        self.ADD_COUNTS += 1
        if self.ADD_COUNTS % 100 == 0:
            # print(f"ADD_COUNTS: {self.ADD_COUNTS}, REMOVE_COUNTS: {self.REMOVE_COUNTS}, last_change_index: {self.last_change_index}")
            time_order_distance = sorted(self.distance_to_last_change_time, key=lambda k: self.distance_to_last_change_time[k])
            # 获取对应这些distance的node
            time_order_nodes = [self.distance_to_max_level_node[distance] for distance in time_order_distance]
            if len(self.last_time_order_nodes) > 0:
                # 寻找第一个不同的node的index
                gap_idx = 0 
                for i in range(len(self.last_time_order_nodes)):
                    if self.last_time_order_nodes[i] != time_order_nodes[i]:
                        gap_idx = i
                        break
                if gap_idx >= self.last_same_idx:
                    same_nodes = time_order_nodes[:gap_idx]
                    print(f"len:{len(same_nodes)}, time_order_nodes: {[n.name for n in same_nodes]}")
                self.last_same_idx = gap_idx
                
            self.last_time_order_nodes = time_order_nodes

        self.distance_to_max_level_node[node.distance] = node
        self.distance_to_last_change_time[node.distance] = time.time()

    def _update_sorted_nodes(self, node):
        # 从序列中移除旧的同距离节点，这里相当于一定会移除旧节点，如果是同一个节点反而会保留
        # 旧节点的level一定是小于等于新节点的level的
        if node.distance in self.distance_to_max_level_node:
            old_node = self.distance_to_max_level_node[node.distance]
            if old_node != node and old_node.level <= node.level and old_node in self.sorted_nodes:
            # if old_node != node and old_node in self.sorted_nodes:
                # self._update_last_changes(index)
                # index = self.sorted_nodes.index(old_node)
                self.sorted_nodes.remove(old_node)

                self.REMOVE_COUNTS += 1
        # 重新插入节点到合适位置
        self._insert_sorted(node)


    def add_edge(self, u_name, v_name, weight):
        self.add_node(u_name)
        self.add_node(v_name)
        u = self.nodes[u_name]
        v = self.nodes[v_name]
        u.add_neighbor(v, weight)
        
        self.relax(u, v, weight)


    def relax(self, u, v, weight):
        # 更新节点层级
        v.level = max([predecessor.level for predecessor in v.in_edges], default=-1) + 1
        if v.distance > u.distance + weight:
            v.distance = u.distance + weight
            # 如果v的距离被更新，将其后继加入队列
            # if prev_dist != v.distance:
            self._update_sorted_nodes(v)
            self.queue.append(v)


    def process_queue(self):
        while self.queue:
            u = self.queue.popleft()
            for (v, weight) in u.neighbors:
                self.relax(u, v, weight)
    

    def get_shortest_distance(self, node_name):
        node = self.nodes.get(node_name)
        return node.distance if node else float('inf')

    def get_sorted_nodes(self):
        return self.sorted_nodes


# 处理多个不连通子图的类
class MultiDAGShortestPaths:
    def __init__(self):
        self.subgraphs = []
        self.node_to_subgraph = {}

    def add_node(self, node_name):
        if node_name not in self.node_to_subgraph:
            # 创建一个新的子图
            new_subgraph = DynamicDAGShortestPath(node_name)
            self.subgraphs.append(new_subgraph)
            self.node_to_subgraph[node_name] = new_subgraph

    def add_edge(self, u_name, v_name, weight):
        self.add_node(u_name)
        self.add_node(v_name)
        u_subgraph = self.node_to_subgraph[u_name]
        v_subgraph = self.node_to_subgraph[v_name]

        if u_subgraph != v_subgraph:
            # 合并两个子图
            if len(u_subgraph.nodes) < len(v_subgraph.nodes):
                u_subgraph, v_subgraph = v_subgraph, u_subgraph

            # 较小子图的所有节点与边添加到较大子图
            for node_name, node in v_subgraph.nodes.items():
                # 添加节点到较大子图
                u_subgraph.add_node(node_name)
                # 更新节点到子图的映射
                self.node_to_subgraph[node_name] = u_subgraph
                # 添加边到较大子图
                for neighbor, edge_weight in node.neighbors:
                    neighbor_name = neighbor.name
                    u_subgraph.add_edge(node_name, neighbor_name, edge_weight)

            self.subgraphs.remove(v_subgraph)

        u_subgraph.add_edge(u_name, v_name, weight)
        u_subgraph.process_queue()

    def get_shortest_distance(self, node_name):
        subgraph = self.node_to_subgraph.get(node_name)
        return subgraph.get_shortest_distance(node_name) if subgraph else float('inf')

    def get_sorted_nodes_for_subgraph_containing(self, node_name):
        subgraph = self.node_to_subgraph.get(node_name)
        return subgraph.get_sorted_nodes() if subgraph else []



# 示例使用
def test_mock_multi():
    multi_dag = MultiDAGShortestPaths()

    # 逐步添加节点和边
    multi_dag.add_edge('A', 'B', 2)
    multi_dag.add_edge('B', 'C', 3)
    multi_dag.add_edge('D', 'E', 1)  # 一个新的不连通子图

    print("Shortest distance from A to C:", multi_dag.get_shortest_distance('C'))
    print("Shortest distance from A to B:", multi_dag.get_shortest_distance('B'))
    print("Shortest distance from D to E:", multi_dag.get_shortest_distance('E'))

    # 输出包含节点 A 的子图的排序节点序列
    sorted_nodes = multi_dag.get_sorted_nodes_for_subgraph_containing('A')
    print("Nodes sorted by distance and level from the start node in the subgraph containing A:")
    for node in sorted_nodes:
        print(f"Node: {node.name}, Distance: {node.distance}, Level: {node.level}, In-degree: {node.in_degree}, Out-degree: {node.out_degree}")

    # 输出包含节点 D 的子图的排序节点序列
    sorted_nodes = multi_dag.get_sorted_nodes_for_subgraph_containing('D')
    print("Nodes sorted by distance and level from the start node in the subgraph containing D:")
    for node in sorted_nodes:
        print(f"Node: {node.name}, Distance: {node.distance}, Level: {node.level}, In-degree: {node.in_degree}, Out-degree: {node.out_degree}")

    # 连接两个子图
    multi_dag.add_edge('C', 'D', 4)

    print("Shortest distance from A to E after connecting subgraphs:", multi_dag.get_shortest_distance('E'))

    # 输出包含节点 A 的子图（现在是合并后的大子图）的排序节点序列
    sorted_nodes = multi_dag.get_sorted_nodes_for_subgraph_containing('A')
    print("Nodes sorted by distance and level from the start node in the subgraph containing A after connection:")
    for node in sorted_nodes:
        print(f"Node: {node.name}, Distance: {node.distance}, Level: {node.level}, In-degree: {node.in_degree}, Out-degree: {node.out_degree}")

def load_log_data(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
        data = [eval(row.replace('\n', '')) for row in data]
    nodes = set()
    edges = []
    nodes_times = {}
    data = data[:] #  1500 [467:2100]
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
    return edges


def test_log(filename):
    edges = load_log_data(filename)

    g = DynamicDAGShortestPath(edges[0][0])
    s = time.time()
    for _, ed in enumerate(edges):
        g.add_edge(ed[0], ed[1], 1)
        if _ % 60:
            g.process_queue()
    g.process_queue()
    e = time.time()
    print(f"total cost time: {(e-s)*1000} ms")

    sorted_nodes = g.get_sorted_nodes()
    print(f"Nodes sorted by distance and level from the start node in the subgraph with start {g.start.name}:")
    for node in sorted_nodes:
        print(f"Node: {node.name}, Distance: {node.distance}, Level: {node.level}, In-degree: {node.in_degree}, Out-degree: {node.out_degree}")
    # sorted_ids = [n.name for n in sorted_nodes]
    # with open('test_sp_nodes.txt', 'w') as f:
    #     f.write(f"{list(sorted_ids)[:100]}")

def test_log_multi_dag(filename):
    edges = load_log_data(filename)
    mdag = MultiDAGShortestPaths()
    s = time.time()
    for _, ed in enumerate(edges):
        mdag.add_edge(ed[0], ed[1], 1) # int(ed[2]['cost'])，这里如果使用计算代价会导致distance过于离散，节点过多
    e = time.time()
    print(f"total cost time: {(e-s)*1000} ms")
    sorted_ids = []
    for dag in mdag.subgraphs:
        sorted_nodes = dag.get_sorted_nodes()
        if len(dag.nodes) < 100:
            continue
        # import ipdb; ipdb.set_trace()
        sorted_ids.extend([n.name for n in sorted_nodes])
        # distance_ids = [n.name for n in dag.distance_to_max_level_node.values()]
        # sorted_ids.extend(dag.debug_ids)
        print(dag.REMOVE_COUNTS, dag.ADD_COUNTS)
        print(f"Nodes sorted by distance and level from the start node in the subgraph with start {dag.start.name}:")
        for node in sorted_nodes:
            print(f"Node: {node.name}, Distance: {node.distance}, Level: {node.level}, In-degree: {node.in_degree}, Out-degree: {node.out_degree}")

    with open('test_sp_nodes.txt', 'w') as f:
        # f.write(f"{list(sorted_ids)[:-1*int(len(sorted_ids)*0.1)]}")
        f.write(f"{list(sorted_ids)}")

def test_mock():
    dag = DynamicDAGShortestPath(1)

    test_edges = [
        (1,2,1), (2,3,1), (3,4,1), (4,5,1), (5,6,1), (6,7,1), (2,9,1), (7,9,1), (8,9,1),
        (9,11,1), (10,11,1), (11,12,1), (12,13,1), (11,14,1)
    ]

    for u, v, w in test_edges:
        dag.add_edge(u, v, w)
    dag.process_queue()  # 处理所有待更新的节点
    print([(n.name, n.distance, n.level) for n in dag.nodes.values()])

    dag.add_edge(14, 15, 1)
    dag.add_edge(2, 14, 1)
    dag.process_queue()  # 处理所有待更新的节点
    print([(n.name, n.distance, n.level) for n in dag.nodes.values()])

    sorted_nodes = dag.get_sorted_nodes()
    print([(n.name, n.distance, n.level) for n in sorted_nodes])
    print(dag.distance_to_max_level_node)

# 示例使用
if __name__ == "__main__":
    # test_mock()
    # test_mock_multi()

    # file_path = '/data/wangzehua/Megatron-LM/nc_test/logs/gpt3_350M_forward_once.log'
    # file_path = os.environ.get('OP_LOG_PATH', file_path)
    # test_log(file_path)

    file_path = '/data/wangzehua/Megatron-LM/nc_test/logs/llama2_7B_layers.log'
    file_path = os.environ.get('OP_LOG_PATH', file_path)
    test_log_multi_dag(file_path)
