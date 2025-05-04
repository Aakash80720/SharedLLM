from fastmcp import FastMCP
from dotenv import load_dotenv
import client
import json
import logging
import model

logger = logging.getLogger(__name__)

mcp = FastMCP("RemoteService")

load_dotenv()

@mcp.tool("getting_linkage", description="Get the dendrogram of a document by embedding")
async def get_linkage(embeddings : model.Embedding) -> model.Linkage:
    response = await client.ClusteringClient().get_linkage(embeddings)
    linkage_result = model.Linkage(
        matrix=response
    )
    return linkage_result

@mcp.tool("getting_clusters", description="Get the clusters of a document with linkage matrix")
async def get_clusters(input : model.Cluster) -> model.ClusterResult:
    response = await client.ClusteringClient().get_clusters(input.linkage, input.documents, input.threshold)
    cluster_result = model.ClisterResult(
        cluster=response,
        metadata={"threshold": input.threshold}
    )
    return cluster_result

@mcp.tool("store_embeddings", description="Store embeddings in FAISS index")
async def store_embeddings(embeddings):
    response = await store_embeddings(embeddings)
    return response

@mcp.tool("get_embeddings", description="Get embeddings of a document")
async def get_embeddings(document: model.Document) -> model.EmbeddingResult:
    try:
        logger.info("Serializing document for embedding.")
        document_json = json.dumps({"document": document})
        logger.debug("Serialized Document: %s", document_json)
        response = await client.EmbeddingClient().encode_document(document)
        embedding_result = model.EmbeddingResult(
            embedding=response.embedding,
            metadata= {"document": document_json}
        )
        return embedding_result
    except Exception as e:
        logger.error("Error serializing document: %s", e)
        raise

@mcp.tool("split_text", description="Split text into chunks for embedding. Kindly add overlap if needed, by default is 80 and chunk size is 512")
async def split_text(text, chunk_size=512, chunk_overlap=80) -> model.DocumentSplit:
    try:
        logger.info("Splitting text into chunks.")
        splitter = client.DocumentSplitter(chunk_size, chunk_overlap)
        chunks = splitter.split_text(text)
        logger.debug("Split text into %d chunks.", len(chunks))
        result = model.DocumentSplit(
            chunks=chunks,
            metadata={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
        )
        return result
    except Exception as e:
        logger.error("Error splitting text: %s", e)
        raise

if __name__ == "__main__":
    logger.info("Starting MCP tools service.")
    mcp.run()

