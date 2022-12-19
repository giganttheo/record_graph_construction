import spacy

nlp = spacy.load("en_core_web_sm")

def dependency_parser(sentences):
  """
  sentences is a list of sentences from an input document
  """
  return [nlp(sentence) for sentence in sentences]

def construct_dependency_graph(docs):
  """
  docs is a list of outputs of the SpaCy dependency parser
  """
  graphs = []
  for doc in docs:
    nodes = [token.text for token in doc]
    senders = []
    receivers = []
    for token in doc:
        for child in token.children:
            senders.append(token.i)
            receivers.append(child.i)
    graphs.append({"nodes": nodes, "senders": senders, "receivers": receivers})
  return graphs

if __name__ == "__main__":
    # not tested
    from datasets import load_dataset
    dataset = load_dataset("gigant/tib_transcripts")
    sentences = dataset["train"][42]["transcript"].split(".")
    docs = dependency_parser(sentences)
    graphs = construct_dependency_graph(docs)
    print(graphs[0])