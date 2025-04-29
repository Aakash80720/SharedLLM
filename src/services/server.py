import grpc
from concurrent import futures
import time
import sentence_transformer_pb2_grpc
import sentence_transformer_pb2
from sentence_transformers import SentenceTransformer
from transformers import pipeline

print("Loading SentenceTransformer model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
summerizer = pipeline("summarization")
print("Model loaded successfully.")

class SentenceEncoderServicer(sentence_transformer_pb2_grpc.SentenceEncoderServicer):
    def EncodeDocument(self, request, context):
        sentences = request.document
        print(f"Received {len(sentences)} sentences for encoding.")
        embeddings = model.encode(sentences)
        print("Testing in Server : ",embeddings)
        response = sentence_transformer_pb2.EncodeResponse(
            embedding = embeddings.tolist()  # Convert to list for serialization
        )
        
        return response
    

def serve():
    print("Starting gRPC server...")
    print("Listening on port 50051...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    sentence_transformer_pb2_grpc.add_SentenceEncoderServicer_to_server(SentenceEncoderServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(60 * 60 * 24) 
    except KeyboardInterrupt:
        server.stop(0)
    print("Server stopped.")

if __name__ == '__main__':
    serve()