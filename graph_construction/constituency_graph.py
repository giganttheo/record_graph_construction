import jraph
import jax.numpy as jnp
import spacy
import gradio as gr
import en_core_web_trf
import numpy as np
import benepar
import re

nlp = en_core_web_trf.load()

benepar.download('benepar_en3')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

def dependency_parser(sentences):
  return [nlp(sentence) for sentence in sentences]

def parse_tree(sentence):
    stack = []  # or a `collections.deque()` object, which is a little faster
    top = items = []
    for token in filter(None, re.compile(r'(?:([()])|\s+)').split(sentence)):
        if token == '(':
            stack.append(items)
            items.append([])
            items = items[-1]
        elif token == ')':
            if not stack:
                raise ValueError("Unbalanced parentheses")
            items = stack.pop()  
        else:
            items.append(token)
    if stack:
        raise ValueError("Unbalanced parentheses")    
    return top

class Tree():
  def __init__(self, name, children):
    self.children = children
    self.name = name
    self.id = None
  def set_id_rec(self, id=0):
    self.id = id
    last_id=id
    for child in self.children:
      last_id = child.set_id_rec(id=last_id+1)
    return last_id
  def set_all_ids(self):
    self.set_id_rec(0)
  def print_tree(self, level=0):
    to_print = f'|{"-" * level} {self.name} ({self.id})'
    for child in self.children:
      to_print += f"\n{child.print_tree(level + 1)}"
    return to_print
  def __str__(self):
    return self.print_tree(0)
  def get_list_nodes(self):
    return [self.name] + [_ for child in self.children for _ in child.get_list_nodes()]

def rec_const_parsing(list_nodes):
  if isinstance(list_nodes, list):
    name, children = list_nodes[0], list_nodes[1:]
  else:
    name, children = list_nodes, []
  return Tree(name, [rec_const_parsing(child) for i, child in enumerate(children)])

def tree_to_graph(t):
  senders = []
  receivers = []
  for child in t.children:
    senders.append(t.id)
    receivers.append(child.id)
    s_rec, r_rec = tree_to_graph(child)
    senders.extend(s_rec)
    receivers.extend(r_rec)
  return senders, receivers

def construct_constituency_graph(docs):
  doc = docs[0]
  sent = list(doc.sents)[0]
  print(sent._.parse_string)
  t = rec_const_parsing(parse_tree(sent._.parse_string)[0])
  t.set_all_ids()
  senders, receivers = tree_to_graph(t)
  nodes = t.get_list_nodes()
  graphs = [{"nodes": nodes, "senders": senders, "receivers": receivers, "edge_labels": {}}]
  return graphs

def to_jraph(graph):
  nodes = graph["nodes"]
  s = graph["senders"]
  r = graph["receivers"]

  # Define a three node graph, each node has an integer as its feature.
  node_features = jnp.array([0]*len(nodes))

  # We will construct a graph for which there is a directed edge between each node
  # and its successor. We define this with `senders` (source nodes) and `receivers`
  # (destination nodes).
  senders = jnp.array(s)
  receivers = jnp.array(r)

  # We then save the number of nodes and the number of edges.
  # This information is used to make running GNNs over multiple graphs
  # in a GraphsTuple possible.
  n_node = jnp.array([len(nodes)])
  n_edge = jnp.array([len(s)])

  return jraph.GraphsTuple(nodes=node_features, senders=senders, receivers=receivers,
  edges=None, n_node=n_node, n_edge=n_edge, globals=None)

def get_adjacency_matrix(jraph_graph: jraph.GraphsTuple):
  nodes, edges, receivers, senders, _, _, _ = jraph_graph
  adj_mat = jnp.zeros((len(nodes), len(nodes)))
  for i in range(len(receivers)):
    adj_mat = adj_mat.at[senders[i], receivers[i]].set(1)
  return adj_mat

if __name__ == "__main__":
  sentence="This is a test sentence."
  docs = dependency_parser([sentence])
  graphs = construct_constituency_graph(docs)
  g = to_jraph(graphs[0])
  adj_mat = get_adjacency_matrix(g)