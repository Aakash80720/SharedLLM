from pydantic import BaseModel, Field
from typing import DefaultDict, Optional, List
from collections import defaultdict

class Embedding(BaseModel):
    """Class representing an embedding."""
    vector: List[float] = Field(..., description="The embedding vector")
    metadata: Optional[dict] = Field(None, description="Optional metadata associated with the embedding")

class Document(BaseModel):
    """Class representing a document."""
    text: str = Field(..., description="The text of the document")
    metadata: Optional[dict] = Field(None, description="Optional metadata associated with the document")
    embeddings: Optional[List[Embedding]] = Field(None, description="List of embeddings associated with the document")

class Linkage(BaseModel):
    """Class representing a linkage matrix."""
    matrix: List[List[float]] = Field(..., description="The linkage matrix")
    metadata: Optional[dict] = Field(None, description="Optional metadata associated with the linkage matrix")

class Cluster(BaseModel):
    """Class representing a cluster."""
    documents: List[Document] = Field(..., description="List of documents in the cluster")
    linkage : Linkage = Field(..., description="Linkage matrix for the cluster")
    metadata: Optional[dict] = Field(None, description="Optional metadata associated with the cluster")
    threshold: Optional[float] = Field(0.7, description="Threshold for clustering")


class EmbeddingResult(BaseModel):
    """Class representing the result of an embedding operation."""
    embedding: Embedding = Field(..., description="The embedding result")
    metadata: Optional[dict] = Field(None, description="Optional metadata associated with the embedding result")

class DocumentSplit(BaseModel):
    """Class representing a split document."""
    chunks: List[str] = Field(..., description="List of text chunks")
    metadata: Optional[dict] = Field(None, description="Optional metadata associated with the split document")

class DocumentEmbeddingRequest(BaseModel):
    """Class representing a request for document embedding."""
    document: Document = Field(..., description="The document to be embedded")
    metadata: Optional[dict] = Field(None, description="Optional metadata associated with the request")

class ClusterResult(BaseModel):
    """Class representing a request for clustering."""
    cluster: DefaultDict[str, List[str]] = Field(
        default_factory=lambda: defaultdict(list),
        description="The cluster to be processed"
    )
    metadata: Optional[dict] = Field(None, description="Optional metadata associated with the clustering result")