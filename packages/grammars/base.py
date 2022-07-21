from __future__ import annotations
import enum
import dataclasses
from copy import deepcopy
from typing import Optional, Sequence
import random
from xml_parser import XMLModel, Body, geoms, joints
from utils import Transform
from grammars import transforms
import pydot
import numpy as np
from scipy.spatial.transform import Rotation


@dataclasses.dataclass
class NodeData():
    symbol: int
    geom: Sequence[Optional[geoms.Geom]] = None


@dataclasses.dataclass
class EdgeData():
    symbol: int
    transform: Optional[callable] = None
    joint: Sequence[Optional[joint.Joint]] = None
    mirror_x: bool = False
    mirror_y: bool = False
    mirror_z: bool = False
    mirror_first: bool = True

    def update(self, data: EdgeData):
        self.symbol = data.symbol
        if data.transform is not None:
            self.transform = transforms.compose(data.transform, self.transform)
        if data.joint is not None:
            self.joint = data.joint


class Tree():
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.parents = {}
        self._n = 0
        self.root = 0

    def add_node(self, node: NodeData):
        self.nodes[self._n] = node
        self.edges[self._n] = {}
        self._n += 1
        return self._n - 1

    def add_edge(self, parent: int, child: int, edge: EdgeData):
        self.edges[parent][child] = edge
        self.parents[child] = parent

    def insert_subtree(self, node: int, tree: Tree):
        if len(tree.nodes) == 0:  # Delete node (i.e. replace it with an empty subtree)
            if node == 0:
                raise ValueError("The root node cannot be deleted.")
            del self.nodes[node]
            del self.edges[node]
            del self.edges[self.parents[node]][node]

        else:                     # Replace node with subtree
            self.nodes[node] = tree.nodes[tree.root]

            index_map = {}
            for ind, data in tree.nodes.items():
                if ind == 0:
                    self.nodes[node] = tree.nodes[0]
                    index_map[0] = node
                else:
                    index_map[ind] = self.add_node(data)

            for parent, edges in tree.edges.items():
                for child, edge in edges.items():
                    self.add_edge(index_map[parent], index_map[child], edge)

    def get_node_data(self, ind: int):
        return self.nodes[ind]

    def get_children(self, node_ind: int):
        return self.edge_map[node_ind]

    def to_networkx(self):
        import networkx as nx
        g = nx.DiGraph()
        for i, node in self.nodes.items():
            g.add_node(i, symbol=node.symbol)
            for child in self.edges[i]:
                g.add_edge(i, child)
        return g

    def visualize(self):
        from matplotlib import pyplot as plt
        import networkx as nx
        plt.clf()
        graph = self.to_networkx()
        labels = {i: str(n.symbol.name) for i, n in self.nodes.items()}
        nx.draw(graph, with_labels=True, font_weight='bold', labels=labels)
        plt.show()


class Rule():
    pass

@dataclasses.dataclass
class NodeExpansion(Rule):
    symbol: str
    graph: Tree
    description: str

    def apply(self, graph: Tree, node_ind: int):
        node = graph.nodes[node_ind]
        if node.symbol != self.symbol:
            raise ValueError(f'Tried to apply a rule acting on symbol {self.symbol.name} to a '
                             f'node with symbol {node.symbol.name}')
        graph.insert_subtree(node_ind, deepcopy(self.graph))

    def __repr__(self):
        return f'Rule({self.description})'


@dataclasses.dataclass
class EdgeExpansion(Rule):
    symbol: str
    edge: EdgeData
    description: str

    def apply(self, graph: Tree, parent: int, child: int):
        edge = graph.edges[parent][child]
        if edge.symbol != self.symbol:
            raise ValueError(f'Tried to apply a rule acting on {self.symbol} to a node with symbol'
                             f' {edge.symbol}')

        edge.update(self.edge)

    def __repr__(self):
        return f'Rule({self.description})'


class SymbolSet(enum.IntEnum):
    pass


class RuleSet():
    def __init__(self):
        self._rules_by_symbol = {}
        self._rules_by_id = {}
        self._id_by_symbol = {}
        self.non_terminals = set([])

    def add_rule(self, rule: Rule):
        if rule.symbol not in self._rules_by_symbol:
            self.non_terminals.add(rule.symbol)
            self._rules_by_symbol[rule.symbol] = []
            self._id_by_symbol[rule.symbol] = []
        self._rules_by_symbol[rule.symbol].append(rule)
        count = self.nrules
        self._rules_by_id[count] = rule
        self._id_by_symbol[rule.symbol].append(count)

    def get_rule_by_id(self, ind: int):
        return self._rules_by_id[ind]

    def get_rules_by_symbol(self, symbol: int):
        if symbol in self._rules_by_symbol:
            return self._rules_by_symbol[symbol]
        else:
            return []

    def get_ids_by_symbol(self, symbol: int):
        if symbol in self._id_by_symbol:
            return self._id_by_symbol[symbol]
        else:
            return []

    @property
    def nrules(self):
        return len(self._rules_by_id)

    def __getitem__(self, symbol: int):
        return self.get_rules_by_symbol(symbol)


class Grammar():
    def __init__(self, symbols: SymbolSet, rules: RuleSet, initial_graph: Tree):
        self._initial_graph = initial_graph
        self.symbols = symbols
        self.rules = rules

    def initialize_graph(self):
        return deepcopy(self._initial_graph)

    def get_valid_expansions(self, graph: Tree):
        valid_rules = {}
        for i, node in graph.nodes.items():
            valid_rules[i] = self.rules[node.symbol]
        for parent, children in graph.edges.items():
            for child, edge in children.items():
                valid_rules[(parent, child)] = self.rules[edge.symbol]
        return valid_rules

    def contains_non_terminal_symbols(self, graph: Tree):
        for node in graph.nodes.values():
            if node.symbol in self.rules.non_terminals:
                return True
        for children in graph.edges.values():
            for edge in children.values():
                if edge.symbol in self.rules.non_terminals:
                    return True
        return False

    def get_non_terminal_nodes(self, graph: Tree):
        inds = []
        for i, node in graph.nodes.items():
            if node.symbol in self.rules.non_terminals:
                inds.append(i)
        return inds

    def get_non_terminal_edges(self, graph: Tree):
        edges = []
        for parent, children in graph.edges.items():
            for child, edge in children.items():
                if edge.symbol in self.rules.non_terminals:
                    edges.append((parent, child))
        return edges

    def get_non_terminals(self, graph: Tree):
        return self.get_non_terminal_nodes(graph) + self.get_non_terminal_edges(graph)

    def sample(self):
        g = self.initialize_graph()
        while self.contains_non_terminal_symbols(g):
            valid_rules = self.get_valid_expansions(g)
            ind = random.choice(self.get_non_terminals(g))
            rule = random.choice(valid_rules[ind])
            if isinstance(ind, tuple):
                rule.apply(g, *ind)
            else:
                rule.apply(g, ind)
        return g

    def to_xml(self, graph: Tree, filename: str = None):
        if self.contains_non_terminal_symbols(graph):
            raise ValueError('Input graph must contain only terminal symbols in order '
                             'to generate an xml file.')
        xml = to_xml(graph)
        if filename is not None:
            xml.write(filename)
        return xml


def to_xml(graph: Tree):
    xml = XMLModel()
    root = xml.root

    default_transform = transforms.apply(Transform())  # identity

    class MirrorContext():
        def __init__(self):
            self.mirror_axes = []
            self.is_first = False

        def add_mirror(self, axis: int, mirror_first=True):
            self.mirror_axes.append(axis)
            self.is_first = True
            self.mirror_first = mirror_first
            return self

        def adjust_geom(self, geom: Sequence[geoms.Geom]):
            out = []
            for g in geom:
                for ax in reversed(self.mirror_axes):
                    if self.is_first and not self.mirror_first:
                        continue
                    else:
                        g = g.mirror(ax)
                out.append(g)
            self.is_first = False
            return out

        def adjust_joint(self, joint: Sequence[joints.Joint]):
            out = []
            for j in joint:
                for ax in reversed(self.mirror_axes):
                    j = j.mirror(ax)
                out.append(j)
            return out

        def adjust_transform(self, t: Transform):
            axis_angle = Rotation.from_quat(t.quat).as_rotvec()
            if np.allclose(axis_angle, 0.):
                return t
            for ax in reversed(self.mirror_axes):
                for ind in range(3):
                    if ind != ax:
                        axis_angle[ind] *= -1
            return Transform(t.pos, Rotation.from_rotvec(axis_angle).as_quat())

        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            self.mirror_axes.pop()

    mirror = MirrorContext()


    def add_child_body(body, child_ind, geom, edge):
        transform = default_transform if edge.transform is None else edge.transform
        t = transform(mirror.adjust_geom(geom))
        t = mirror.adjust_transform(t)
        child_body = body.add_body(pos=t.pos, quat=t.quat)
        for j in mirror.adjust_joint(edge.joint):
            if not isinstance(j, joints.RigidJoint):
                child_body.add_joint(j)
        depth_first_traversal(child_ind, child_body)


    def depth_first_traversal(node_ind: int, body: Body):
        node = graph.nodes[node_ind]
        gs = mirror.adjust_geom(node.geom)
        for g in gs:
            body.add_geom(g)
        for child_ind, edge in graph.edges[node_ind].items():
            if edge.joint is None:
                raise ValueError("All edges must specify a joint to connect bodies.")

            add_child_body(body, child_ind, node.geom, edge)
            if edge.mirror_x:
                with mirror.add_mirror(0, edge.mirror_first):
                    add_child_body(body, child_ind, node.geom, edge)
            if edge.mirror_y:
                with mirror.add_mirror(1, edge.mirror_first):
                    add_child_body(body, child_ind, node.geom, edge)
            if edge.mirror_z:
                with mirror.add_mirror(2, edge.mirror_first):
                    add_child_body(body, child_ind, node.geom, edge)

    depth_first_traversal(0, root)
    xml.adjust_root()
    return xml
