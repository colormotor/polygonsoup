'''
  _   _   _   _   _   _   _   _   _   _   _
 / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \
( P | O | L | Y | G | O | N | S | O | U | P )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/

Plotter-friendly graphics utilities
© Daniel Berio (@colormotor) 2021 - ...

graph - networkx-based graph manipulation utils
'''

import numpy as np
import networkx as nx
import itertools

def connected_component_subgraphs(G, copy=False):
    ''' Backwards compatibility with older NetworkX < 2.4'''
    try:
        return nx.connected_component_subgraphs(G)
    except AttributeError:
        if not copy:
            return (G.subgraph(c) for c in nx.connected_components(G))
        else:
            return (G.subgraph(c).copy() for c in nx.connected_components(G))

def remove_edge_safe(G, a, b):
    if G.has_edge(a, b):
        G.remove_edge(a, b)

def remove_node_safe(G, n):
    if G.has_node(n):
        G.remove_node(n)

def neighbor_list(G, n):
    return list(G.neighbors(n))


def graph_branches(G, discard_single=False, stop_nodes=set()):
    ''' Find all 'branches' of a graph,
        that is sequences of vertices that connect vertices of degree  1 or >2'''

    comps = connected_component_subgraphs(G)
    branches = []

    # mark visited edges
    pair = frozenset
    visited = set()

    degree = {n:G.degree(n) for n in G.nodes()}

    def traverse_branch(e):
        branch = [e[0], e[1]]
        visited.add(pair(e))
        start = e[0] # to handle loop cases
        prev, next = e
        while degree[next]==2 and next != start and next not in stop_nodes:
            prev, next = next, [bor for bor in G.neighbors(next) if bor != prev][0]
            branch.append(next)
        # also visit end of branch
        visited.add(pair(branch[-2:]))
        return branch

    for comp in comps:
        start_edges = []
        nodes = list(comp.nodes())

        if len(nodes)==1:
            if not discard_single:
                branches.append([nodes[0]])
            continue

        # select potential branch starts
        for n in nodes:
            if degree[n] == 1 or degree[n] > 2 or n in stop_nodes:
                for bor in G.neighbors(n):
                    start_edges.append((n, bor))

        # in case there are none we have a loop
        # so just select one direction along the loop
        if not start_edges and G.degree(n)==2:
            start_edges = [(n, next(G.neighbors(n)))]

        # compute branch once for each edge
        for e in start_edges:
            if not pair(e) in visited:
                branches.append(traverse_branch(e))

    # discard eventual corrupted branches
    if discard_single:
        branches = [b for b in branches if len(b)>1]

    return branches

def traverse_until_fork_or_terminal(G, a, b):
    path = [a]
    prev = a
    while G.degree(b)==2 and b != a:
        path.append(b)
        bors = [v for v in G.neighbors(b) if v != prev]
        prev = b
        b = bors[0]
    path.append(b)
    return path

def incident_branches(G, n, branches=[]):
    ''' Returns branches incident to node n, ordered so n is the first node'''
    try:
        if G.degree(n) < 3:
            branches = []
            for b in G.neighbors(n):
                branches.append(traverse_until_fork_or_terminal(G, n, b))
            return branches
    except nx.NetworkXError:
        print('incident_branches: Could not parse non-fork')
        #pdb.set_trace()
        return []

    branches = [branch for branch in branches if branch[0] == n or branch[-1]==n]
    return [branch if branch[0] == n else branch[::-1] for branch in branches]

def branch_contour(branch, pos):
    return np.array([pos[n] for n in branch])
