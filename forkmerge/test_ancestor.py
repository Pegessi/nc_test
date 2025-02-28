from typing import DefaultDict
from collections import defaultdict
import time
import os

class Node:
    def __init__(self, name):
        self.name = name
        self.parents = []     # 父节点列表
        self.ancestors = []
        self.out_degree = 0   # 出度
        self.level = 0        # 拓扑层级
        self.forks = set()    # 上游分叉点缓存

    def __str__(self):
        parents_info = ', '.join([p.name for p in self.parents])
        return f"Node(name={self.name}, out_degree={self.out_degree}, level={self.level}, parents={parents_info})"

class Graph:
    def __init__(self):
        self.nodes: DefaultDict[str, Node] = {}       # 节点名到节点的映射
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
            self.nodes[u_name] = Node(u_name)
        u = self.nodes[u_name]
        if v_name not in self.nodes:
            self.nodes[v_name] = Node(v_name)
        v = self.nodes[v_name]

        # 维护邻接表 (optional)
        self.adj[u].append(v)
        self.reverse_adj[v].append(u)
        self.out_degree[u] += 1
        self.in_degree[v] += 1
        
        # 添加父节点关系并更新出度
        v.parents.append(u)
        u.out_degree += 1
        
        # 更新拓扑层级
        v.level = max(v.level, u.level + 1)
        # 更新分叉点缓存
        if u.out_degree >= 2:  # 如果u是分叉点
            v.forks.add(u.name)
        v.forks.update(u.forks)  # 继承u的缓存
        
        # 当节点入度≥2时触发检测
        if len(v.parents) >= 2:
            s = time.time()
            self._detect_jump(v)
            e = time.time()
            self.time_costs.append((e-s)*1000)
    
    def _detect_jump(self, v):
        # 遍历所有父节点对
        for i in range(len(v.parents)):
            for j in range(i+1, len(v.parents)):
                p1, p2 = v.parents[i], v.parents[j]
                # 查找最近公共分叉点
                lca = self._find_lca_fork(p1, p2)
                if lca is not None:
                    self.target_ids.add(lca.name)
                    self.target_ids.add(v.name)
                    self.target_nodes.add(lca)
                    self.target_nodes.add(v)
                    # print(f"[CHECK {v.name}] ({p1.name}, {p2.name}) 跳接结构: 起点 {lca.name} -> 终点 {v.name}")

    def _find_lca_fork(self, a, b):
        # 获取所有分叉点祖先并按层级降序排列

        # s = time.time()
        forks_a = self._get_fork_ancestors(a)
        # e = time.time()
        # self.time_costs.append((e-s)*1000)
        # s = time.time()
        forks_b = self._get_fork_ancestors(b)
        # e = time.time()
        # self.time_costs.append((e-s)*1000)
        
        # 寻找公共分叉点中层级最大的
        common = []
        for node in forks_a:
            if node in forks_b:
                common.append(node)
        if not common:
            return None
        return max(common, key=lambda x: x.level)

    def _get_fork_ancestors(self, node):
        # 收集所有出度≥2的祖先节点
        forks = []
        visited = set()
        stack = [node]
        
        while stack:
            current = stack.pop()
            if current.name in visited:
                continue
            visited.add(current.name)
            
            # 记录分叉点
            if current.out_degree >= 2:
                forks.append(current)
            
            # 继续遍历父节点
            for parent in current.parents:
                stack.append(parent)
        
        return forks

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
    print(f"_get_fork_ancestors cost time: {g.time_costs}")
    
    with open('test_ancestor_nodes.txt', 'w') as f:
        f.write(f"{list(g.target_ids)}")

# 测试示例
if __name__ == "__main__":
    # test_mock()

    file_path = '/data/wangzehua/Megatron-LM/nc_test/logs/llama2_7B_once.log'
    file_path = '/data/wangzehua/Megatron-LM/nc_test/logs/resnet32_once.log'
    file_path = '/data/wangzehua/Megatron-LM/nc_test/logs/gpt3_350M_forward_once.log'
    file_path = '/data/wangzehua/Megatron-LM/nc_test/logs/llama2_7B.log'
    file_path = os.environ.get('OP_LOG_PATH', file_path)
    test_log(file_path)
