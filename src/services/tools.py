from fastmcp import FastMCP
from services import EmbeddingClient, ClusteringClient, store_embeddings, sample_document
mcp = FastMCP("RemoteService")

@mcp.tool("getting_linkage", description="Get the dendrogram of a document by embedding")
def get_linkage(embeddings):
    response = ClusteringClient().get_linkage(embeddings)
    return response

@mcp.tool("getting_clusters", description="Get the clusters of a document with linkage matrix")
def get_clusters(linkage, documents, threshold=0.7):
    response = ClusteringClient().get_clusters(linkage, documents, threshold)
    return response

@mcp.tool("store_embeddings", description="Store embeddings in FAISS index")
def store_embeddings(embeddings):
    response = store_embeddings(embeddings)
    return response

@mcp.tool("get_embeddings", description="Get embeddings of a document")
def get_embeddings(document):
    response = EmbeddingClient().encode_document(document)
    return response

