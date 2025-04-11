from typing import DefaultDict
from collections import defaultdict
import time
import os

# resnet32 2, 3
# llama2 5, 10
# gpt350m 5, 10

LEVEL_THRESHOLD = 1
TAG_THRESHOLD = 15

class TNode:
    def __init__(self, name):
        self.name = name
        self.parents = []     # 父节点列表
        self.children = []
        self.in_degree = 0    # 入度
        self.out_degree = 0   # 出度
        self.level = 0        # 拓扑层级
        self.distance = 0     # 距离最近祖先的距离
        self.anc_forks = set()    # 上游分叉点缓存

    def __str__(self):
        parents_info = ', '.join([str(p.name) for p in self.parents])
        return f"TNode(name={self.name}, out_degree={self.out_degree}, level={self.level}, parents={parents_info})"
    
    def __repr__(self):
        parents_info = ', '.join([str(p.name) for p in self.parents])
        return f"TNode(name={self.name}, degree={self.in_degree, self.out_degree}, level={self.level}, parents={parents_info})"

class Graph:
    def __init__(self):
        self.nodes: DefaultDict[str, TNode] = {}       # 节点名到节点的映射
        self.betweenness = defaultdict(int)

        self.adj = defaultdict(list)
        self.reverse_adj = defaultdict(list)
        self.in_degree = defaultdict(int)
        self.out_degree = defaultdict(int)

        self.target_ids = set()
        self.target_nodes = set()
        self.time_costs = []
    
    def add_edge(self, u_name, v_name):
        # 确保节点存在
        if u_name not in self.nodes:
            self.nodes[u_name] = TNode(u_name)
        u = self.nodes[u_name]
        if v_name not in self.nodes:
            self.nodes[v_name] = TNode(v_name)
        v = self.nodes[v_name]

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

        self._update_dist(u, v)

        if len(u.children) >=2 :
            self._tag_if_fork(u)
        
        # 当节点入度≥2时触发检测
        if len(v.parents) >= 2:
            s = time.time()
            self._tag_if_merge(v)
            e = time.time()
            self.time_costs.append((e-s)*1000)
    
    def _update_dist(self, u: TNode, v: TNode):
        # if u.out_degree > 1:
        #     v.anc_forks.add(u)
        #     v.anc_forks.update(u.anc_forks) # 继承祖先

        # 更新dist与level
        dist = v.distance
        level = v.level
        for p in v.parents:
            dist = max(dist, p.distance + 1)
            level = max(level, p.level + 1)
        v.distance = dist
        v.level = level
    
    def _tag_if_merge(self, v: TNode):
        min_level = v.level
        max_dist = 0
        min_dist = v.distance
        ancestor = None
        for p in v.parents:
            min_level = min(min_level, p.level)
            max_dist = max(max_dist, p.distance)
            if min_dist > p.distance:
                ancestor = p
                min_dist = p.distance
        if min_level <= LEVEL_THRESHOLD:
            return
        if max_dist - min_dist > TAG_THRESHOLD:
            # self.target_ids.add(ancestor.name)
            self.target_ids.add(v.name)
            # self.target_nodes.add(ancestor)
            self.target_nodes.add(v)
            v.distance = 0 # reset 
    
    def _tag_if_fork(self, u: TNode):
        max_level = u.level
        max_dist = 0
        # min_dist = v.distance
        farest = None
        for p in u.children:
            if p.level > max_level:
                farest = p
                max_level = p.level
            # max_level = max(max_level, p.level)
            # max_dist = max(max_dist, p.distance)
            # if min_dist > p.distance:
            #     ancestor = p
            #     min_dist = p.distance
        if u.level <= LEVEL_THRESHOLD:
            return
        if max_level - u.level > TAG_THRESHOLD:
            # self.target_ids.add(ancestor.name)
            self.target_ids.add(u.name)
            # self.target_nodes.add(ancestor)
            self.target_nodes.add(u)
            farest.distance = 0 # reset 

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
    nodes = set()
    edges = []
    nodes_times = {}
    CHECK_DEGREE = 6 # gpt 6
    data = data[:] #  1500 [467:2100]
    # 解析日志得到无向图
    for row in data:
        if row['INSTRUCTION'] != 'INSTRUCTION':
            continue
        if row['name']=='add_' and row['inputs'][0] == row['outputs'][0]:
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

    g = Graph()
    s = time.time()
    for ed in edges:
        g.add_edge(ed[0], ed[1])
    e = time.time()
    print(f"total cost time: {(e-s)*1000} ms")
    print(len(g.target_ids), g.target_ids)
    with open('test_fm_nodes.txt', 'w') as f:
        f.write(f"{list(g.target_ids)}")

if __name__ == '__main__':
    # test_mock()
    file_path = '/data/wangzehua/Megatron-LM/nc_test/logs/gpt3_350M_forward_once.log'
    file_path = os.environ.get('OP_LOG_PATH', file_path)
    test_log(file_path)