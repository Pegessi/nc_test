from typing import DefaultDict
from collections import defaultdict
import queue
import time
import os

INF_COMPUTE_COST = 1E10

class TNode:
    def __init__(self, name, compute_cost, layer_id=-1):
        self.name = name
        self.parents = []     # 父节点列表
        self.children = []   
        self.pred = []        # 最近前驱
        self.upper_pred = []
        self.in_degree = 0    # 入度
        self.out_degree = 0   # 出度
        self.compute_cost = compute_cost # 计算代价
        self.layer_id = layer_id     # 所属的层级
        self.calculate_temp = {
            'm': 0, 'n': 0, 'eta': 0, 'theta': 0,
            'bc': 0
        }
        # self.level = 0        # 拓扑层级
        # self.distance = 0     # 距离最近祖先的距离
        # self.anc_forks = set()    # 上游分叉点缓存

    # def __str__(self):
    #     parents_info = ', '.join([str(p.name) for p in self.parents])
    #     return f"TNode(name={self.name}, out_degree={self.out_degree}, level={self.level}, parents={parents_info})"
    
    # def __repr__(self):
    #     parents_info = ', '.join([str(p.name) for p in self.parents])
    #     return f"TNode(name={self.name}, degree={self.in_degree, self.out_degree}, level={self.level}, parents={parents_info})"

class TLayer:
    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.nodes = set()
        self.starts = []
        self.ends = []
        self.bc = 0
        self.distance = {} # 层内的距离 TNode: dist
        self.shortest_paths = {}
        self.dependency = {}
        self.dependency_reverse = {}
        self.layer_length = 0
    
    def _reset_stats(self):
        for w in self.nodes:
            w.pred = []
            self.distance[w] = INF_COMPUTE_COST
            self.shortest_paths[w] = 0
            self.dependency[w] = 0
            self.dependency_reverse[w] = 0



    def add_node(self, node):
        self.nodes.add(node)
        self.distance[node] = INF_COMPUTE_COST
        self.shortest_paths[node] = 0


    def calculate_bc_for_new_layer(self):
        for s in self.nodes: # starts or all nodes?
            ### init
            self._reset_stats()
            self.distance[s] = 0
            self.shortest_paths[s] = 1
            Q = queue.Queue()
            Q.put(s)
            S = []

            ### BFT
            self._bf_traverse(s, S, 'forward')
            while len(S) > 0:
                w = S.pop()
                self._accumulation(w, 'forward')
                w.calculate_temp['bc'] += self.dependency[w]
                if s in self.starts and w in self.ends:
                    self.layer_length = max(self.layer_length, self.distance[w])

        for t in self.ends:
            S = []
            self._reset_stats()
            self._bf_traverse(t, S, 'backward')
            while len(S) > 0:
                w = S.pop()
                self._accumulation(w, 'backward')

    def calculate_spvb_for_new_layer(self):
        for s in self.nodes: # starts or all nodes?
            ### init
            self._reset_stats()
            self.distance[s] = 0
            self.shortest_paths[s] = 1
            Q = queue.Queue()
            Q.put(s)
            S = []

            ### single-source shortest-paths
            while not Q.empty():
                v = Q.get()
                S.append(v)
                for w in v.children:
                    ### path discovery
                    if self.distance[w] == INF_COMPUTE_COST:
                        Q.put(w)
                        self.distance[w] = self.distance[v] + v.compute_cost
                    
                    ### path counting
                    if self.distance[w] == self.distance[v] + v.compute_cost:
                        self.shortest_paths[w] += self.shortest_paths[v]
                        w.pred.append(v)
            
            ### accumulation
            while len(S) > 0:
                w = S.pop()
                for v in w.pred:
                    self.dependency[v] += self.shortest_paths[v] / self.shortest_paths[w] * (1 + self.dependency[w])
                if w not in self.starts:
                    w.calculate_temp['bc'] += self.dependency[w]
            import ipdb; ipdb.set_trace()
            print()



    def _bf_traverse(self, s: TNode, S: list, direction='forward'):
        ### forward aims to find single-source shortest-paths
        Q = queue.Queue()
        Q.put(s)
        self.distance[s] = 0
        self.shortest_paths[s] = 1
        while not Q.empty():
            v = Q.get()
            S.append(v)
            if direction == 'forward':
                for w in v.children:
                    self._geodesic_searching(v, w, Q)
            else:
                for w in v.parents:
                    if w in self.starts: # skip layer starts
                        continue
                    self._geodesic_searching(v, w, Q)
        
    
    def _geodesic_searching(self, v: TNode, w: TNode, Q: queue.Queue):
        if self.distance[w] == INF_COMPUTE_COST:
            Q.put(w)

        if self.distance[w] > self.distance[v] + v.compute_cost:
            self.distance[w] = self.distance[v] + v.compute_cost
            self.shortest_paths[w] = 0
            w.pred = []
        if self.distance[w] == self.distance[v] + v.compute_cost:
            self.shortest_paths[w] += self.shortest_paths[v]
            w.pred.append(v)
    

    def _accumulation(self, w: TNode, direction='forward'):
        for v in w.pred:
            self.dependency[v] += self.shortest_paths[v] / self.shortest_paths[w] * (1 + self.dependency[w])
            # print(f"res = {v.calculate_temp['d']} + {self.shortest_paths[v]} / {self.shortest_paths[w]} * (1 + {w.calculate_temp['d']}) \
            #       |{v.name}-{v.calculate_temp}| --> |{w.name}-{w.calculate_temp}|")
        # if w not in self.starts:
        if direction == 'forward':
            w.calculate_temp['m'] = self.shortest_paths[w]
            w.calculate_temp['eta'] = self.dependency[w]
        else:
            w.calculate_temp['n'] = self.shortest_paths[w]
            w.calculate_temp['theta'] = self.dependency[w]
        

        



class Graph:
    def __init__(self):
        self.nodes: DefaultDict[str, TNode] = {}       # 节点名到节点的映射
        self.betweenness = defaultdict(int)
        self.layers = []   # [ TLayer, ...]
        self.current_layer_id = -1
        self.current_begins = []
        self.current_ends = []

        self.upper_nodes = set()
        self.upper_dependency = {}
        self.upper_shortest_paths = {}
        self.upper_cost = {}
        self.upper_distance = {}

        self.adj = defaultdict(list)
        self.reverse_adj = defaultdict(list)
        self.in_degree = defaultdict(int)
        self.out_degree = defaultdict(int)

        self.target_ids = set()
        self.target_nodes = set()
        self.time_costs = []
    
    def _add_single_edge(self, u_name, v_name, compute_cost=1):
        # 确保节点存在
        if u_name not in self.nodes:
            u = TNode(u_name, 0, self.current_layer_id)
            self.nodes[u_name] = u
            if self.current_layer_id > -1: # 存在一些正式layer之前的张量
                self.layers[self.current_layer_id].add_node(u)
        else:
            u = self.nodes[u_name]
        if v_name not in self.nodes:
            v = TNode(v_name, compute_cost, self.current_layer_id)
            self.nodes[v_name] = v
            if self.current_layer_id > -1 and self.current_layer_id < len(self.layers): # 存在一些layer之后的张量
                self.layers[self.current_layer_id].add_node(v)
        else:
            v = self.nodes[v_name]
        v.compute_cost = max(compute_cost, v.compute_cost)

        # 维护邻接表 (optional)
        self.adj[u].append(v)
        self.reverse_adj[v].append(u)
        self.out_degree[u] += 1
        self.in_degree[v] += 1
        
        # 添加父节点关系并更新出度
        v.parents.append(u)
        u.children.append(v)
        u.out_degree += 1
        v.in_degree += 1

    def add_edges(self, ins, outs, extra_info):
        mark_starts = False
        if self.current_layer_id != -1 and len(self.layers[self.current_layer_id].nodes) == 0:
            mark_starts = True

        for u in ins:
            for v in outs:
                # cost = int(extra_info['compute_cost'])
                cost = 1
                cost = cost if cost > 0 else 1
                self._add_single_edge(u, v, )
        self.current_begins = [self.nodes[u] for u in ins]
        self.current_ends = [self.nodes[v] for v in outs]
        
        # 标记层的起始点
        if mark_starts:
            self.layers[self.current_layer_id].starts = self.current_begins
            for w in self.current_begins:
                if w.layer_id != self.current_layer_id: # 输入是上一层的输出，手动添加一下
                    self.layers[self.current_layer_id].add_node(w)

    def _handle_begin_new_layer(self):
        self.current_layer_id += 1
        new_layer = TLayer(self.current_layer_id)
        self.layers.append(new_layer)
    
    def _handle_end_of_layer(self):
        self.layers[self.current_layer_id].ends = self.current_ends
        self.layers[self.current_layer_id].calculate_bc_for_new_layer()
        self._update_bcs()
        self.upper_nodes.update(self.layers[self.current_layer_id].starts)
        self.upper_nodes.update(self.layers[self.current_layer_id].ends)
        # self.layers[self.current_layer_id].calculate_spvb_for_new_layer()
    
    def _update_bcs(self):
        layer_nodes_nums = [len(layer.nodes) for layer in self.layers]
        for layer_id, tl in enumerate(self.layers):

            for v in tl.nodes:
                lamb = v.calculate_temp['bc']
                # alpha = v.calculate_temp['m']*v.calculate_temp['n']
                alpha = 0
                ahead_nodes_num = sum(layer_nodes_nums[:layer_id])
                beta = v.calculate_temp['eta'] * ahead_nodes_num if layer_id == self.current_layer_id else 0
                gamma = v.calculate_temp['theta'] * (sum(layer_nodes_nums[layer_id+1:]))
                v.calculate_temp['bc'] = lamb + alpha + beta + gamma
    
    def _compute_upper_level(self):
        ### init
        for layer_id, tl in enumerate(self.layers):
            for s in tl.starts:
                if s not in self.upper_dependency.keys():
                    self.upper_cost[s] = {}
                for t in tl.ends:
                    if t in s.children:
                        self.upper_cost[s][t] = t.compute_cost
                    else:
                        self.upper_cost[s][t] = tl.layer_length
        
        for s in self.upper_nodes:
            ### init
            self.upper_distance[s] = 0
            self.upper_shortest_paths[s] = 1
            Q = queue.Queue()
            Q.put(s)
            S = []

            ### single-source shortest-paths
            while not Q.empty():
                v = Q.get()
                S.append(v)
                for w in v.children:
                    if w not in self.upper_nodes:
                        continue
                    ### path discovery
                    if self.upper_distance[w] == INF_COMPUTE_COST:
                        Q.put(w)
                        self.upper_distance[w] = self.upper_distance[v] + self.upper_cost[v][w] 
                    
                    ### path counting
                    if self.upper_distance[w] == self.upper_distance[v] + self.upper_cost[v][w] :
                        self.upper_shortest_paths[w] += self.upper_shortest_paths[v]
                        w.upper_pred.append(v)
            
            ### accumulation
            while len(S) > 0:
                w = S.pop()
                for v in w.upper_pred:
                    self.upper_dependency[v] += self.upper_shortest_paths[v] / self.upper_shortest_paths[w] * (1 + self.upper_dependency[w])
                if w not in self.starts:
                    w.calculate_temp['bc'] += self.upper_dependency[w]
    
    def export_bcs_topk(self, k):
        # 对节点按照 calculate_temp['bc'] 进行排序
        sorted_nodes = sorted(self.nodes.values(), key=lambda node: node.calculate_temp['bc'], reverse=True)
        # 取前 k 个节点
        top_k_nodes = sorted_nodes[:k]
        # 返回前 k 个节点的 name
        return [node.name for node in top_k_nodes]
        
    

def test_mock():
    g = Graph()
    edges = [
        ("1", "2"), ("2", "3"),
        ("3", "4"), ("4", "5"),  # 3成为分叉点
        ('5', '6'), ('5', '7'), ('5', '8'),
        ('6', '9'), ('7', '10'),
        ("8", "11"), ('9', '11'), ("10", "11"),
        ("11", "12"), ("12", "13"), ("13", "14"),
        ("3", "15"), ("14", "15"), ("15", "16"),
        # 11的父节点3和10
    ]
    
    for u, v in edges:
        g.add_edge(u, v)
    print(g.target_ids)

def test_log(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
        data = [eval(row.replace('\n', '')) for row in data]
    edges = []
    data = data[:] #  1500 [467:2100]
    # 解析日志得到无向图
    for row in data:
        if row['INSTRUCTION'] == 'ANNOTATION':
            note_type = row['NAME']
            if note_type == 'begin_layer':
                edges.append({'instruction': 'begin_layer'})
            if note_type == 'end_layer':
                edges.append({'instruction': 'end_layer'})
            continue
        if row['INSTRUCTION'] != 'INSTRUCTION':
            continue
        # if row['name']=='add_' and row['inputs'][0] == row['outputs'][0]: # why?
        #     continue
        input_ids = [int(tid[1:]) for tid in row['inputs']]
        output_ids = [int(tid[1:]) for tid in row['outputs']]
        extra_info = {'op': row['name'], 'compute_cost': row['compute_cost'], 'mem': row['mem_cost']}
        edges.append((input_ids, output_ids, extra_info))

    g = Graph()
    s = time.time()
    for ed in edges:
        if type(ed) is dict:
            if ed['instruction'] == 'begin_layer':
                g._handle_begin_new_layer()
            if ed['instruction'] == 'end_layer':
                g._handle_end_of_layer()
            continue
        # g.add_edge(ed[0], ed[1], ed[2]['compute_cost'])
        g.add_edges(ed[0], ed[1], ed[2])
    e = time.time()
    print(f"total cost time: {(e-s)*1000} ms")

    K = 50
    all_bcs = g.export_bcs_topk(K)
    with open('test_bc_topk.txt', 'w') as f:
        f.write(f"{list(all_bcs)}")

if __name__ == '__main__':
    # test_mock()
    file_path = '/data/wangzehua/Megatron-LM/nc_test/logs/llama2_7B_layers.log'
    file_path = os.environ.get('OP_LOG_PATH', file_path)
    test_log(file_path)