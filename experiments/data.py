"""
Synthetic tree 生成器 (toy, 用于 anchor_2/anchor_4 smoke run)。
README §6.1: binary / 4-ary / 8-ary / variable branching; nodes 1K/5K/10K。

toy smoke: small binary tree。输出图上两两最短路距离矩阵, 用于双曲 embedding。
"""

import torch


def gen_balanced_tree(branching=2, depth=8):
    """生成平衡树的边列表 (BFS 编号, root=0)。

    返回:
      n_nodes, edges (list of (parent, child)), parent array
    """
    edges = []
    parent = [-1]
    frontier = [0]
    n = 1
    for _ in range(depth):
        new_frontier = []
        for p in frontier:
            for _b in range(branching):
                c = n
                edges.append((p, c))
                parent.append(p)
                new_frontier.append(c)
                n += 1
        frontier = new_frontier
    return n, edges, parent


def graph_distance_matrix(n, edges):
    """无权树上两两最短路距离 (BFS from each node)。树上距离 = 边数。"""
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    D = torch.full((n, n), -1, dtype=torch.float32)
    for src in range(n):
        D[src, src] = 0
        q = [src]
        head = 0
        while head < len(q):
            u = q[head]; head += 1
            for w in adj[u]:
                if D[src, w] < 0:
                    D[src, w] = D[src, u] + 1
                    q.append(w)
    return D


def make_toy_binary_tree(depth=8):
    """小 binary tree: depth=8 -> 2^9-1 = 511 nodes。返回 n, D (距离矩阵)。"""
    n, edges, parent = gen_balanced_tree(branching=2, depth=depth)
    D = graph_distance_matrix(n, edges)
    return n, edges, parent, D
