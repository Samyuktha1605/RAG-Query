Abstract

This project proposes a novel approach to exploring and querying storybooks, focusing specifically on the Hindu mythological epic, the Ramayana. Leveraging advanced Natural Language Processing (NLP) techniques, including Retrieval-Augmented Generation (RAG) modeling, which comprises of : FAISS similarity search based retriever,  and LLAMA-2  LLM for query understanding, the objective is to create an efficient and intuitive system for answering questions and exploring narratives within the Ramayana. The system aims to provide accurate and contextually relevant responses to user queries. The project's ultimate goal is to offer an immersive and insightful experience for readers.

Significance of RAG:

RAG optimizes the output of an LLM by referencing (accessing) an external knowledge base outside of the information on which it was trained. In other words, RAG enables genAI to find and use relevant external information, often from an organization’s proprietary data sources or other content to which it’s directed.

Project pipeline:

Load the storybook:
The storybook saved as a .pdf file is loaded using the PyPDF loader available under the langchain community loaders.

Splitting the document into chunks:
The entire document is split into chunks using RecursiveCharacterTextSplitter.
Size of each chunk is 1000 characters and there is a 20-character overlap between consecutive chunks.

Embeddings:
These chunks must be converted into vector embeddings in order to store them in the Vector Store. We use the OllamaEmbeddings(), which is an application to run llama locally.

FAISS (Facebook AI Similarity Search) retriever:
FAISS is a vector store that is built to efficiently perform similarity search among vectors.
The vector space is split into segments using some clustering algorithm (usually k-means)
Each cluster has a corresponding centroid.
Given a user query, find its vector representation using the embedding method
Instead of comparing the similarity of the user query with each other vector in the DB, we find the closest centroid to the user query vector (nprobe=1).
 We find the top-k similar document chunks from the selected segment.

Generator:
The generator Q and A chain comprises of two parts: LLM and prompt

Prompt:
We define a prompt to be supplied to the LLM which includes:
System prompt, to define the role of LLM
Enhanced context (Similar documents retrieved from DB)
User query

LLM:

We use the llama-2 LLM as the generator component of our RAG pipeline.
The key features of llama-2 architecture include:

The tokenizer uses a byte pair encoding (BPE) algorithm.
It uses the standard transformer architecture:
Applies normalization using RMSNorm
Rotary positional embedding
Grouped query attention using KV cache
Uses the SwiGLU activation function
The key difference includes increased context length.

Langchain:
LangChain enables the architecting of RAG systems with numerous tools to transform, store, search, and retrieve information that refine language model responses. It helps to put together the individual blocks in the pipeline to build a RAG-chain.





