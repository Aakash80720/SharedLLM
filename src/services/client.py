import re
import grpc
import sentence_transformer_pb2_grpc
import sentence_transformer_pb2
import asyncio
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastmcp import FastMCP
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict
mcp = FastMCP("RemoteClient")

class client:
   def __init__(self):
         self.channel = grpc.aio.insecure_channel('localhost:50051')
         self.document = None
class EmbeddingClient(client):
    def __init__(self):
        self.stub = sentence_transformer_pb2_grpc.SentenceEncoderStub(self.channel)

    @mcp.tool("document_embedding", description="Get the embedding of a document")
    async def encode_document(self, document):
        request = sentence_transformer_pb2.EncodeRequest(document=document)
        response = await self.stub.EncodeDocument(request)
        return response.embedding
    
class ClusteringClient(client):
    def __init__(self):
        self.stub = sentence_transformer_pb2_grpc.SentenceEncoderStub(self.channel)

    @mcp.tool("getting_linkage", description="Get the dendrogram of a document by embedding")
    def get_linkage(self, embeddings):
       embeddings = np.array(embeddings).astype('float32')
       return linkage(embeddings, method='ward')
    
    @mcp.tool("getting_clusters", description="Get the clusters of a document with linkage matrix")
    def get_clusters(self, linkage, documents, threshold=0.7):
        clusters = fcluster(linkage, threshold, criterion='distance')
        cluster_dict = defaultdict(list)
        for sent, label in zip(documents, clusters):
           cluster_dict[label].append(sent)
        return cluster_dict

index = faiss.IndexFlatL2(384) 

@mcp.tool("vectorize_document", description="Get the embedding of a document")
def store_embeddings(embeddings):
    # Initialize FAISS index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension) 
    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)
    print("Embeddings stored in FAISS index.")
    return index
    
def sample_document():
    file = open(r"sample.txt", "r")
    text = file.read()
    print("text length: ", len(text))
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=80,
        length_function=len
    )
    texts = text_splitter.split_text(text)
    return texts

@mcp.tool("find_similarity", description="Find the most similar sentences")
def find_similarity(index, query_embedding, k=30):
    # Perform a search in the FAISS index
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    print("test: ", index.ntotal)
    return distances, indices

async def run():
    client = EmbeddingClient()
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = sentence_transformer_pb2_grpc.SentenceEncoderStub(channel)
        sentences =  sample_document()
        print("Sending request...")
        tasks = []
        for sentence in sentences:
            tasks.append(client.encode_document(stub, sentence))
        result = await asyncio.gather(*tasks)
        print("Received response.")
        index = store_embeddings(result)

        print("Query section started")
        try:
            while True:
                query = input("Enter your query: ")
                query_embedding = await client.encode_document(stub, query)
                print("Query embedding received.", query_embedding)
                distances, doc_index = find_similarity(index, query_embedding, k=5)
                print("Top 5 similar sentences:\n")
                for idx, distance in zip(doc_index[0], distances[0]):
                    print(f"{sentences[idx]}, Distance: {distance}")
                    print()
        except KeyboardInterrupt:
            print("Exiting query section.")
        print("Client stopped.")

if __name__ == '__main__':
    asyncio.run(run())