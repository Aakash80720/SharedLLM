# SharedLLM Application

## Overview
The SharedLLM application is designed to demonstrate the use of a gRPC-based service for encoding documents into embeddings using a Sentence Transformer model. It also provides functionality to store these embeddings in a FAISS index and perform similarity searches. This application is modular and scalable, making it suitable for various AI-driven tasks such as document retrieval, similarity search, and more.

## Features
- **gRPC Server**: A server that encodes documents into embeddings using the Sentence Transformer model.
- **gRPC Client**: A client that sends documents to the server for encoding and stores the embeddings in a FAISS index.
- **FAISS Integration**: Efficient similarity search using FAISS.
- **Text Splitting**: Splits large documents into smaller chunks for processing using LangChain's text splitter.
- **Asynchronous Communication**: Uses Python's asyncio for efficient gRPC communication.
- **Decoupled LLM Service**: The LLM service is decoupled from the main application, allowing for separate contexts and easier scalability.

## Technologies Used
- **Python 3.10**: The programming language used for the application.
- **gRPC**: For client-server communication.
- **Sentence Transformers**: For generating embeddings from text.
- **FAISS**: For efficient similarity search.
- **LangChain**: For splitting large documents into smaller chunks.
- **Protobuf**: For defining the gRPC service and message formats.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd SharedLLM
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, use the `environment.yml` file to create a Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate sharedllm
   ```

## Usage
### Starting the Server
1. Navigate to the `src/services` directory.
2. Run the server:
   ```bash
   python server.py
   ```

### Running the Client
1. Navigate to the `src/services` directory.
2. Run the client:
   ```bash
   python client.py
   ```
3. Follow the prompts to input queries and find similar sentences.

## File Structure
- `src/proto`: Contains the `.proto` file defining the gRPC service.
- `src/services`: Contains the server and client implementations.
- `src/models`: Contains the Sentence Transformer model wrapper.
- `src/utils`: Utility functions (currently empty).
- `environment.yml`: Conda environment configuration file.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [gRPC](https://grpc.io/)