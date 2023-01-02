import jraph
import jax.numpy as jnp
import spacy
import gradio as gr
import en_core_web_trf
import numpy as np
import re
from ..node_encoding.word_encoding import w2v_embedder
from gensim.models import KeyedVectors
from ..node_encoding.sentence_encoding import get_sentence_embeddings

w2v = KeyedVectors.load_word2vec_format('./models/custom_w2v_100d.txt')
get_embedding = w2v_embedder(w2v)

nlp = en_core_web_trf.load()

def dependency_parser(sentences):
  """
  sentences is a list of sentences from an input document
  """
  return [nlp(sentence) for sentence in sentences]

def dependency_parser(sentences):
  return [nlp(sentence) for sentence in sentences]

def construct_both_graph(docs):
  """
  docs is a list of outputs of the SpaCy dependency parser
  """
  graphs = []
  for doc in docs:
    nodes = [token.text for token in doc]
    nodes.append("Sentence")
    node_features = [get_embedding(token.text) for token in doc]
    node_features.append(get_sentence_embeddings([doc])[0])
    senders = [token.i for token in doc][:-1]
    receivers = [token.i for token in doc][1:]
    edge_labels = {(token.i, token.i + 1): "next" for token in doc[:-1]}
    for node in range(len(nodes) - 1):
      senders.append(node)
      receivers.append(len(nodes) - 1)
      edge_labels[(node, len(nodes) - 1)] = "in"
    for token in doc:
        for child in token.children:
            senders.append(child.i)
            receivers.append(token.i)
            edge_labels[(token.i, child.i)] = child.dep_
    graphs.append({"node_features": node_features, "nodes": nodes, "senders": senders, "receivers": receivers, "edge_labels": edge_labels})
  return graphs

def to_jraph(graph):
  nodes = graph["nodes"]
  s = graph["senders"]
  r = graph["receivers"]

  # Define a three node graph, each node has an integer as its feature.
  node_features = graph["node_features"]#jnp.array([0]*len(nodes))

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

if __name__ == "__main__":
    sentence="This is a test sentence."
    docs = dependency_parser([sentence])
    graphs = construct_both_graph(docs)
    g = to_jraph(graphs[0])
    print([n.shape for n in g.nodes])