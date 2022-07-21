from nlimb.designs import Design
from xml_parser import XMLModel
from grammars.quadruped_grammar import QUADRUPED_GRAMMAR, HEXAPOD_GRAMMAR
from grammars.base import Tree
import torch
import torch.nn.functional as F
import random


def depth_first_traversal(graph: Tree):
    """Order nodes with a depth first traversal."""

    def _dfs(node: int, node_order: list, edge_order: list):
        node_order.append(node)
        children = sorted(list(graph.edges[node].keys()))
        for child in children:
            edge_order.append((node, child))
            _dfs(child, node_order, edge_order)
        return node_order, edge_order

    return _dfs(graph.root, [], [])



def generate_grammar_based_design(grammar, representation_fn):

    class GrammarBasedDesign(Design):
        """String format: 'rule_id node1 [node2]:rule_id ...'
        The representation_fn takes in a graph and outputs a nest of torch tensors.
        """
        design_grammar = grammar
        def __init__(self, graph: Tree = None, rule_sequence: list = None):
            if graph is None:
                self.graph = grammar.initialize_graph()
            else:
                self.graph = graph
            self.rules = [] if rule_sequence is None else rule_sequence

        @staticmethod
        def from_str(rules_str):
            graph = grammar.initialize_graph()
            rule_sequence = []
            for rule in rules_str.split(':'):
                rule_id, *nodes = [int(x) for x in rule.split()]
                rule_sequence.append((rule_id, *nodes))
                grammar.rules.get_rule_by_id(rule_id).apply(graph, *nodes)
            return GrammarBasedDesign(graph, rule_sequence)

        @staticmethod
        def from_torch(param):
            raise NotImplementedError

        @staticmethod
        def from_random():
            graph = grammar.sample()
            return GrammarBasedDesign(graph)

        def to_str(self):
            rule_strings = []
            for rule in self.rules:
                rule_strings.append(' '.join([str(x) for x in rule]))
            return ':'.join(rule_strings)

        def to_torch(self):
            return representation_fn(self.graph)

        def to_xml(self, path: str = None):
            xml = grammar.to_xml(self.graph)
            if path is not None:
                xml.write(path)
            return xml

        def is_complete(self):
            return not grammar.contains_non_terminal_symbols(self.graph)

        def apply(self, rule_id: int, *nodes):
            grammar.rules.get_rule_by_id(rule_id).apply(self.graph, *nodes)
            self.rules.append((rule_id, *nodes))

        def generate_random_graph(self):
            self.graph = grammar.sample()

    return GrammarBasedDesign


def flat_nodes_and_edges(graph):
    node_order, edge_order = depth_first_traversal(graph)
    node_dict = {n: i for i, n in enumerate(node_order)}
    node_types = torch.tensor([graph.nodes[ind].symbol for ind in node_order], dtype=torch.long)
    edge_types = torch.tensor([graph.edges[p][c].symbol for p, c in edge_order], dtype=torch.long)
    if len(edge_order) == 0:
        parents, children = [], []
    else:
        parents, children  = zip(*edge_order)
    return {'nodes': node_types, 'edges': edge_types,
            'node_order': node_order,
            'parents': torch.tensor(parents, dtype=torch.long),
            'children': torch.tensor(children, dtype=torch.long)}


QuadrupedGrammarDesign = generate_grammar_based_design(QUADRUPED_GRAMMAR, flat_nodes_and_edges)
HexapodGrammarDesign = generate_grammar_based_design(HEXAPOD_GRAMMAR, flat_nodes_and_edges)
