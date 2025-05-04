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
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("application.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class client:
   def __init__(self):
         self.channel = grpc.aio.insecure_channel('localhost:50051')
         self.document = None
class EmbeddingClient(client):
    def __init__(self):
        super().__init__()
        self.stub = sentence_transformer_pb2_grpc.SentenceEncoderStub(self.channel)

    async def encode_document(self, document : str):
        logger.info("Encoding document: %s", document)
        request = sentence_transformer_pb2.EncodeRequest(document=document)
        response = await self.stub.EncodeDocument(request)
        logger.info("Received response: %s", response.embedding)
        if not response.embedding:
            logger.error("Received empty embedding from server.")
            raise ValueError("Received empty embedding from server.")
        return response.embedding

class DocumentSplitter:
    def __init__(self, chunk_size=512, chunk_overlap=80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        return text_splitter.split_text(text)

class ClusteringClient(client):
    def __init__(self):
        super().__init__()
        self.stub = sentence_transformer_pb2_grpc.SentenceEncoderStub(self.channel)

    def get_linkage(self, embeddings):
       embeddings = np.array(embeddings).astype('float32')
       return linkage(embeddings, method='ward')
    
    def get_clusters(self, linkage, documents, threshold=0.7):
        clusters = fcluster(linkage, threshold, criterion='distance')
        cluster_dict = defaultdict(list)
        for sent, label in zip(documents, clusters):
           cluster_dict[label].append(sent)
        return cluster_dict

index = faiss.IndexFlatL2(384) 

async def store_embeddings(embeddings):
    logger.info("Storing embeddings in FAISS index.")
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension) 
    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)
    logger.info("Embeddings stored in FAISS index.")
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

async def find_similarity(index, query_embedding, k=30):
    logger.info("Performing similarity search in FAISS index.")
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    logger.info("Similarity search completed. Found %d results.", len(indices[0]))
    return distances, indices


async def run():
    logger.info("Client started.")
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = sentence_transformer_pb2_grpc.SentenceEncoderStub(channel)
        sentences = sample_document()
        logger.info("Sending requests for %d sentences.", len(sentences))
        tasks = []
        for sentence in sentences:
            tasks.append(EmbeddingClient().encode_document(sentence))
        result = await asyncio.gather(*tasks)
        logger.info("Received responses for all sentences.")
        index = await store_embeddings(result)

        logger.info("Query section started.")
        try:
            while True:
                query = input("Enter your query: ")
                query_embedding = await EmbeddingClient().encode_document(query)
                logger.info("Query embedding received.")
                distances, doc_index = await find_similarity(index, query_embedding, k=5)
                logger.info("Top 5 similar sentences retrieved.")
                for idx, distance in zip(doc_index[0], distances[0]):
                    print(f"{sentences[idx]}, Distance: {distance}")
                    print()
        except KeyboardInterrupt:
            logger.info("Exiting query section.")
        logger.info("Client stopped.")

if __name__ == "__main__":
    # for testing purposes, we can run the client directly
    asyncio.run(run())