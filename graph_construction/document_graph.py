from .utils import split_sentences
from .dependency_graph import dependency_parser
from .dependency_sequ_graph import construct_both_graph, to_jraph, get_adjacency_matrix

def doc_graph(docs):
  nodes = []
  senders = []
  receivers = []
  edge_labels = {}
  graphs = construct_both_graph(docs)
  offset = 0
  sentences_ids = []
  for graph in graphs:
    nodes.extend(graph["nodes"][:-1])
    nodes.append(f"S_{len(sentences_ids) + 1}")
    senders.extend([offset + s for s in graph["senders"]])
    receivers.extend([offset + r for r in graph["receivers"]])
    for (s, r) in graph["edge_labels"].keys():
      edge_labels[(s+offset, r+offset)] = graph["edge_labels"][(s, r)]
    offset += len(graph["nodes"])
    sentences_ids.append(offset - 1)
  #bag of sentences?
  for s1 in sentences_ids:
    for s2 in sentences_ids:
      senders.append(s1)
      receivers.append(s2)
      edge_labels[(s1, s2)] = "sentence_sentence"
  return {"nodes": nodes, "senders": senders, "receivers": receivers, "edge_labels": edge_labels}



def get_adj_from_sentences(sentences):
  docs = dependency_parser(sentences)
  graph = doc_graph(docs)
  g = to_jraph(graph)
  adj_mat = get_adjacency_matrix(g)
  return g, adj_mat


def construct_document_graph(document):
    """
    document: text document without any processing
    """
    sentences = split_sentences(document)
    graph, mat_adj = get_adj_from_sentences(sentences)
    return graph, mat_adj

if __name__ == "__main__":
    document = "This sentence comes first. This one follows? Yes, there's a last one!!"
    graph, mat_adj = construct_document_graph(document)
    print(mat_adj)