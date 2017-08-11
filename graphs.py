import torch
import networkx as nx
from graphviz import Digraph
from splitting_layers import Conv1dSplittable, ExpandableParameter


def find_dependencies(model, out_var):
    g = make_graph(out_var)
    module_dict = dict()

    for module in model.modules():
        if not isinstance(module, Conv1dSplittable):
            continue
        module_dict[id(module)] = module

    module_dict[id("out")] = "out"
    #module_dict.get()

    #print(module_dict, "\n")

    for n, m in module_dict.items():
        if m == "out":
            continue
        #print("module:", m)
        m.input_tied_modules = [module_dict[i] for i in g.successors(n) if i != id("out")]
        #print("input_tied:", m.input_tied_modules)
        output_tied = set()
        for s in g.successors(n):
            output_tied.update(g.predecessors_iter(s))
        output_tied.discard(n)
        output_tied.discard(id("all"))
        m.output_tied_modules = [module_dict[i] for i in output_tied]
        #print("output_tied:", m.output_tied_modules)
    pass


def make_graph(var):
    g = nx.DiGraph()
    seen = set()

    def add_nodes(var, connection=None):
        if var not in seen:
            # seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if hasattr(u[0], 'variable'):
                        v = u[0].variable
                        if isinstance(v, ExpandableParameter):
                            m = v.expandable_module
                            g.add_node(id(m))
                            if connection is not None and id(m) != connection:
                                g.add_edge(id(m), connection)
                            connection = id(m)
                for u in var.next_functions:
                    if u[0] is not None:
                        add_nodes(u[0], connection)

    add_nodes(var.grad_fn, id("out"))
    return g


def make_dot(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            #if torch.is_tensor(var):
            #    dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            if hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map[id(u)], size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__).replace('Backward', ''))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot