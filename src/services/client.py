import grpc
import sentence_transformer_pb2_grpc
import sentence_transformer_pb2
import asyncio
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

index = faiss.IndexFlatL2(384) 

async def send_request(stub, sentence):
    request = sentence_transformer_pb2.EncodeRequest(document=sentence)
    response = await stub.EncodeDocument(request)
    return response.embedding

def store_embeddings(embeddings):
    # Initialize FAISS index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension) 

    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)
    
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

def find_similarity(query_embedding, k=5):
    # Perform a search in the FAISS index
    query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    return distances, indices

async def run():
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = sentence_transformer_pb2_grpc.SentenceEncoderStub(channel)
        sentences =  sample_document()
        print("Sending request...")
        tasks = []
        for sentence in sentences:
            tasks.append(send_request(stub, sentence))
        result = await asyncio.gather(*tasks)
        print("Received response.")
        store_embeddings(result)

        print("Query section started")
        try:
            while True:
                query = input("Enter your query: ")
                query_embedding = await send_request(stub, query)
                distances, index = find_similarity(query_embedding, k=5)
                print("Top 5 similar sentences:")
                for idx, distance in zip(index[0], distances[0]):
                    print(f"{sentences[idx]}, Distance: {distance}")
        except KeyboardInterrupt:
            print("Exiting query section.")
        print("Client stopped.")

if __name__ == '__main__':
    asyncio.run(run())