from collections import defaultdict, deque
import os

class EnhancedDynamicDAG:
    def __init__(self):
        self.adj = defaultdict(list)
        self.reverse_adj = defaultdict(list)
        self.in_degree = defaultdict(int)
        self.out_degree = defaultdict(int)
        self.max_in = defaultdict(int)    # 最长入路径长度
        self.max_out = defaultdict(int)    # 最长出路径长度
        self.bc = defaultdict(float)      # 介数中心性（基于最长路径）
        self.num_paths = defaultdict(int)  # 总最长路径数
        self.node_contrib = defaultdict(lambda: defaultdict(int)) # 节点对路径的贡献

    def add_edge(self, u, v):
        # 更新图结构
        self.adj[u].append(v)
        self.reverse_adj[v].append(u)
        self.out_degree[u] += 1
        self.in_degree[v] += 1

        # 更新最长入路径
        new_in = self.max_in[u] + 1
        if new_in > self.max_in[v]:
            self._update_max_in(v, new_in)

        # 更新最长出路径
        new_out = self.max_out[v] + 1
        if new_out > self.max_out[u]:
            self._update_max_out(u, new_out)

        # 更新介数中心性
        self._update_betweenness(u, v)

    def _update_max_in(self, node, new_length):
        self.max_in[node] = new_length
        for succ in self.adj[node]:
            if self.max_in[node] + 1 > self.max_in[succ]:
                self._update_max_in(succ, self.max_in[node] + 1)

    def _update_max_out(self, node, new_length):
        self.max_out[node] = new_length
        for pred in self.reverse_adj[node]:
            if self.max_out[node] + 1 > self.max_out[pred]:
                self._update_max_out(pred, self.max_out[node] + 1)

    def _update_betweenness(self, u, v):
        # 获取u的前驱最长路径贡献
        pred_contrib = sum(self.node_contrib[p][u] for p in self.reverse_adj[u]) 
        if not self.reverse_adj[u]:
            pred_contrib = 1  # 源节点
        
        # 计算通过u->v的新路径贡献
        new_contrib = pred_contrib
        
        # 向后传播贡献到v的后继
        queue = deque([(v, new_contrib)])
        while queue:
            node, contrib = queue.popleft()
            self.node_contrib[u][node] += contrib
            self.bc[node] += contrib
            
            for succ in self.adj[node]:
                queue.append((succ, contrib * len(self.adj[node])))
        
        # 更新总路径数（示例简化，实际需更精确计算）
        self.num_paths[v] += new_contrib

    def get_critical_nodes(self):
        # 综合分叉/交汇点与介数中心性
        forks = [(n, self.max_out[n], self.bc[n]) 
                for n in self.adj if self.out_degree[n] >= 2]
        forks.sort(key=lambda x: (-x[1], -x[2]))
        
        merges = [(n, self.max_in[n], self.bc[n]) 
                 for n in self.reverse_adj if self.in_degree[n] >= 2]
        merges.sort(key=lambda x: (-x[1], -x[2]))
        
        total = forks + merges
        total.sort(key=lambda x: -x[2])
        # 将total内容写入文件
        with open('critical_nodes.txt', 'w') as f:
            for node_info in total:
                f.write(f"{node_info}\n")
        return (forks[0][0] if forks else None, 
                merges[0][0] if merges else None)


from collections import defaultdict

class TestDynamicDAG:
    def __init__(self):
        self.adj = defaultdict(list)
        self.reverse_adj = defaultdict(list)
        self.in_degree = defaultdict(int)
        self.out_degree = defaultdict(int)
        
        # 新增祖先相关属性
        self.primary_ancestor = defaultdict(lambda: None)  # 主祖先
        self.ancestor_counts = defaultdict(dict)     # 祖先出现次数统计
        self.all_ancestors = defaultdict(set)        # 所有可能的祖先集合
        self.target_points = set()                   # 存储规则4的目标点

    def add_edge(self, u, v):
        # 更新图结构
        self.adj[u].append(v)
        self.reverse_adj[v].append(u)
        self.out_degree[u] += 1
        self.in_degree[v] += 1

        # 规则3处理：出度>2的节点设自己为直接祖先，同时更新祖先集合
        if self.out_degree[u] >= 2:
            self._update_ancestor(u, u)
            
        # 规则2处理：入度0节点初始化
        if self.in_degree[u] == 0:
            self._update_ancestor(u, u)
            
        # 处理v的祖先
        if self.in_degree[v] == 1:  # 规则1
            self._inherit_ancestor(v, u)
        elif self.in_degree[v] >= 2: # 规则4-5
            self._resolve_conflict_ancestors(v)

    def _update_ancestor(self, node, ancestor):
        """更新节点的主祖先和祖先集合"""
        self.primary_ancestor[node] = ancestor
        self.all_ancestors[node].add(ancestor)

    def _inherit_ancestor(self, child, parent):
        """继承父节点的祖先（规则1）"""
        self.primary_ancestor[child] = self.primary_ancestor[parent]
        self.all_ancestors[child] = self.all_ancestors[parent].copy()

    def _resolve_conflict_ancestors(self, node):
        """处理多前驱的祖先冲突（规则4-5）"""
        predecessors = self.reverse_adj[node]
        
        # 收集所有前驱的祖先集合
        # ancestor_sets = [self.all_ancestors[p] for p in predecessors]
        # primary_sets = [self.primary_ancestor[p] for p in predecessors]

        ancestor_sets = []
        primary_dict = dict()
        for p in predecessors:
            ancestor_sets.append(self.all_ancestors[p])
            if self.primary_ancestor[p] not in primary_dict.keys():
                primary_dict[self.primary_ancestor[p]] = 1
            else:
                primary_dict[self.primary_ancestor[p]] += 1
        
        print(f'[CHECK ancestor] {ancestor_sets}')
        # 寻找公共祖先
        common = set.intersection(*map(set, ancestor_sets))
        if common:
            # import ipdb; ipdb.set_trace()
            for ans in primary_dict.keys():
                if ans in common:
                    self.target_points.add(ans)
                    for p in predecessors:
                        self.all_ancestors[p].discard(ans) # 移除合并的祖宗
        
        # 共同直接祖先，继承
        if len(primary_dict.keys()) == 1:
            self._update_ancestor(node, list(primary_dict.keys())[0])
        else:
            self._update_ancestor(node, node)


        # if common:
        #     selected = min(common)  # 假设选择最小的公共祖先
        #     self._update_ancestor(node, selected)
        #     self.target_points.add((node, selected))
        # else:
        #     # 统计所有祖先出现的次数
        #     counter = defaultdict(int)
        #     for s in ancestor_sets:
        #         for anc in s:
        #             counter[anc] += 1
            
        #     # 找到最大出现次数
        #     max_count = max(counter.values(), default=0)
        #     candidates = [k for k, v in counter.items() if v == max_count]
            
        #     if len(candidates) == 1:
        #         self._update_ancestor(node, candidates[0])
        #     else:
        #         # 保留所有候选祖先
        #         self.primary_ancestor[node] = tuple(candidates)
        #         self.all_ancestors[node] = set(candidates)

class PathDynamicDAG:
    def __init__(self):
        self.adj = defaultdict(list)
        self.reverse_adj = defaultdict(list)
        self.in_degree = defaultdict(int)
        self.out_degree = defaultdict(int)
        self.max_in = defaultdict(int)     # 最长入路径长度（替代depth）
        self.parent = defaultdict(lambda: None)  # 主路径前驱
        self.main_path_end = None          # 当前主路径末端
        self.main_path_set = set()         # 主路径节点集合

    def _update_main_path(self):
        """通过parent指针回溯生成主路径集合"""
        ### 不回溯的情况下，是不能稳定获得最长路径的，需要有其他方法来判断最长路径
        # current = self.main_path_end
        # self.main_path_set.add(current)

        self.main_path_set.clear()
        current = self.main_path_end
        while current is not None:
            self.main_path_set.add(current)
            current = self.parent[current]

    def add_edge(self, u, v):
        # 更新图结构
        self.adj[u].append(v)
        self.reverse_adj[v].append(u)
        self.out_degree[u] += 1
        self.in_degree[v] += 1

        # 处理输入节点（入度为0时初始化）
        if self.in_degree[u] == 0 and self.max_in[u] == 0:
            self.parent[u] = None  # 明确标记输入节点
            
        # 更新最长入路径
        new_depth = self.max_in[u] + 1
        if new_depth > self.max_in[v]:
            self.max_in[v] = new_depth
            self.parent[v] = u

            # 更新主路径末端（类似广度优先的贪心策略）
            if (self.main_path_end is None) or (self.max_in[v] > self.max_in[self.main_path_end]):
                self.main_path_end = v
                self._update_main_path()

        # 动态检测分叉点和交汇点
        fork_cond = self.out_degree[u] >= 2 and u in self.main_path_set
        merge_cond = self.in_degree[v] >= 2 and v in self.main_path_set
        
        if fork_cond:
            print(f"分叉点检测: {u} (出度={self.out_degree[u]})")
        if merge_cond:
            print(f"交汇点检测: {v} (入度={self.in_degree[v]})")

    def get_main_path(self):
        """获取当前主路径节点列表（输入节点到末端）"""
        # with open('main_path_nodes.txt', 'w') as f:
        #     f.write(f"{list(self.main_path_set)}")
        path = []
        current = self.main_path_end
        while current is not None:
            path.append(current)
            current = self.parent[current]
        with open('main_path_nodes.txt', 'w') as f:
            f.write(f"{path[::-1]}")
        return path[::-1]  # 反向得到正向路径




# 单元测试
def test_enhanced():
    dag = EnhancedDynamicDAG()
    # edges = [(1,2), (2,3), (2,4), (3,5), (10,5), (5,6), (6,7), (7,8), (4,8), (8,9)]
    edges = [(1,2), (2,3), (3,4), (4,5), (4,6), (4,7), (5,8), (6,9), (7,10), (11, 10), (9,12), (10,12), (8,13), (12,13), (13,14), (14,15)]
    for u, v in edges:
        dag.add_edge(u, v)
    
    fork, merge = dag.get_critical_nodes()
    # assert fork == 2, f"Expected fork 2, got {fork}"
    # assert merge == 8, f"Expected merge 8, got {merge}"
    print(f"forks: {fork}")
    print(f"merges: {merge}")

def test_with_log(filename):
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


    # dag = EnhancedDynamicDAG()
    # for ed in edges:
    #     dag.add_edge(ed[0], ed[1])
    # fork, merge = dag.get_critical_nodes()
    # print(f"forks: {fork}")
    # print(f"merges: {merge}")

    dag = PathDynamicDAG()
    for ed in edges:
        dag.add_edge(ed[0], ed[1])
    dag.get_main_path()

    """
    2-26 
    这样的做法适合简单的计算图，当节点的入度出度都很多的时候，直接祖先会由于频繁分叉交汇导致丢失信息，无法识别长路径跳接的case
    """
    # dag = TestDynamicDAG()
    # for ed in edges:
    #     dag.add_edge(ed[0], ed[1])
    # nodes = set()
    # print(dag.target_points)
    # with open('main_path_nodes_test.txt', 'w') as f:
    #     f.write(f"{list(dag.target_points)}")
    # for p in dag.target_points:
    #     nodes.add(p[0])
    #     nodes.add(p[1])
    # print(list(nodes))
    # with open('main_path_nodes_test.txt', 'w') as f:
    #     f.write(f"{list(nodes)}")


def test_alg():
    dag = TestDynamicDAG()
    # edges = [(1,2), (2,3), (3,4), (4,5), (4,6), (4,7), (5,8), (6,9), (7,10), (11, 10), (9,12), (10,12), (8,13), (12,13), (13,14), (14,15)]
    edges = [(1,2), (2,3), (3,4), (4,5), (4,6), (4,7), (5,8), (6,9), (7,10), (11, 10), (9,12), (10,12), (8,13), (12,13), (13,14), (14,15)]
    for u, v in edges:
        dag.add_edge(u, v)
    
    print(dag.target_points)

# test_enhanced()
# test_alg()

# file_path = '/data/wangzehua/Megatron-LM/nc_test/logs/gpt3_350M_forward_once.log'
file_path = '/data/wangzehua/Megatron-LM/nc_test/logs/llama2_7B_once.log'
file_path = os.environ.get('OP_LOG_PATH', file_path)
# file_path = '/data/wangzehua/Megatron-LM/nc_test/logs/resnet32_once.log'
test_with_log(file_path)