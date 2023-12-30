# Fully_Open_Source_RAG_LlamaIndex

Business Problem:

Many users struggle to find accurate and contextually relevant information from vast amounts of unstructured data. Traditional search engines may not capture the nuance or context of a user's query, leading to less-than-optimal search results. Additionally, generating detailed and coherent responses to complex queries often requires a deep understanding of the context and the ability to retrieve relevant information. Privacy-conscious users may be concerned about sharing their queries or data with proprietary platforms.

Project Objective:
The primary objective of this open-source project is to develop a Privacy-Focused Retrieval-Augmented Generation (RAG) system using open-source embeddings and a Large Language Model (LLM). The system aims to enhance the accuracy and contextuality of responses to user queries while addressing privacy concerns associated with data sharing on proprietary platforms.

## Table of Contents

- [Project Overview](#project-overview)
- [Files](#files)
- [Data](#data)
- [Workflow](#workflow)
- [Results](#results)
- [More ideas](#More-ideas)
- [Dependencies](#dependencies)
- [Contact](#contact)
- [References](#references)

## Project Overview

The goal of this project is to develop a complete open source Retrieval Augmentated Generation customizable solution. Finally, we even compare the RAG with the current Open AI's ChatGPT RAG solution as well.

Key Features
* Open Source Embeddings
* Open Source LLM
* Custom Document Object ->Node Object splitting  

## Files

```bash
Fully_Open_Source_RAG_LlamaIndex.ipynb contains the end-to-end code for RAG with LlamaIndex and Huggingface
```

Check Dependencies for further details.

## Data

Since we wanted to compare with the OpenAI's RAG built using Chat GPT, we used the Paul Graham Essay text [1] which is used even in the OpenAI cookbook for RAG solution [2].  

## Workflow


**1st Part: Make NodeObejcts or NodeIndex as called in LlamaIndex.**

1. Parse the source documents using various data loaders into LlamaIndex's Document objects. By default, for each page in the document, you get one Document object. In our case, since we parse from one webpage, we get just one Docuemnt object.
2. Now, you can either: 
  * Submitting the complete Document object to the index is a fitting method for treating the entire document as a cohesive entity. This proves advantageous when dealing with relatively brief documents or when preserving the context     
    between different segments of the document is crucial.

  * Transforming the Document into Node objects prior to indexing is a viable approach for lengthy documents, where the objective is to segment them into smaller units (nodes) before incorporation into the index. This strategy proves 
    beneficial when the intention is to retrieve particular sections of a document instead of the document in its entirety.

Since we could encounter longer docuemnts, we decide to split the Document Object to Node Objects.

3. After splitting to Node Objects, using SimpleNodeParser with a SentenceWindowNodeParser, we further split each paragraphs (identified by newline characters) and capture a window of surrounding sentences for each node to give us sub-nodes.

4. We then convert the base nodes and their corresponding sub-nodes into IndexNode instances.

An IndexNode, derived from a TextNode, primarily encapsulates textual content. Its distinctive attribute, the index_id, serves as a unique identifier or reference, establishing links to other entities in the system. This referencing capability enhances connectivity and relational information, extending beyond textual content. In scenarios involving recursive retrieval and node references, smaller chunks (embodied as IndexNode objects) can point to larger parent chunks. While smaller chunks are retrieved during query time, references to more substantial chunks are pursued, providing richer context for synthesis.

Below is the Diagram of all the above steps:

![Image 1](https://github.com/krunalgedia/Fully_Open_Source_RAG_LlamaIndex/blob/main/images_readme/ip.png)

**2nd Part: Embedding, Recursive Retrieval, and LLM Generative AI answer.**

We use 
* Small BAAI general embedding from Huggingface [3]
* LLM MistralAI from Huggingface [4]

5. We then store all the indexes using VectorStoreIndex. A VectorStoreIndex in LlamaIndex is a type of index that uses vector representations of text for efficient retrieval of relevant context.It is built on top of a VectorStore, which is a data structure that stores vectors and allows for quick nearest neighbor search. The VectorStoreIndex takes in IndexNode objects, which represent chunks of the original documents.

6. We then use the RecursiveRetriever to fetch relevant nodes. It recursively explore links from nodes to other retrievers or query engines. Thus, if any of those nodes point to another retriever or query engine, the RecursiveRetriever will follow that link and query the linked retriever or engine as well.
       

## Results

The question tested by Open AI (What did the author do growing up?) is tested by our solution as well.

Following are the first two nodes retrieved by the query:

![Image 1](https://github.com/krunalgedia/Fully_Open_Source_RAG_LlamaIndex/blob/main/images_readme/samplequesmine.png) | ![Image 2](https://github.com/krunalgedia/Fully_Open_Source_RAG_LlamaIndex/blob/main/images_readme/samplequesopenai.png)
--- | --- 
First two nodes retrieved by our RAG | First two nodes retrieved by OpenAI GPT4

As seen, the first node has the same content in both.
The final answer given by Open AI and our solution is:

![Image 1](https://github.com/krunalgedia/Fully_Open_Source_RAG_LlamaIndex/blob/main/images_readme/ansmine.png) | ![Image 2](https://github.com/krunalgedia/Fully_Open_Source_RAG_LlamaIndex/blob/main/images_readme/ansopenai.png)
--- | --- 
Answer by our RAG | Answer by OpenAI GPT4

Further Response Evaluation (as given and used by Open AI):

* FaithfulnessEvaluator: Measures if the response from a query engine matches any source nodes which is useful for measuring if the response is hallucinated.

* Relevancy Evaluator: Measures if the response + source nodes match the query.

We get a Faithfulness score of 0.4 and a Relevancy score of 0.9. Open AI gets both as 1.0. However, it is important to note that we generated 466/*2 questions while OpenAI 28/*2. This is because we worked on a GPU with 16 GB RAM and thus had to keep the Node size small while OpenAI had a much larger node size, thus reducing the probability of nodes not containing the answer easily.


* Hit Rate:

Hit rate calculates the fraction of queries where the correct answer is found within the top-k retrieved documents. In simpler terms, it’s about how often our system gets it right within the top few guesses.

* Mean Reciprocal Rank (MRR):

For each query, MRR evaluates the system’s accuracy by looking at the rank of the highest-placed relevant document. Specifically, it’s the average of the reciprocals of these ranks across all the queries. So, if the first relevant document is the top result, the reciprocal rank is 1; if it’s second, the reciprocal rank is 1/2, and so on.

They get both 1.0. We get

|    | Retriever Name                    |   Hit Rate |      MRR |
|---:|:----------------------------------|-----------:|---------:|
|  0 | Our open source RAG solution      |   0.253912 | 0.226885 |

Again, this can't be apple to apple comparison given it was tested only on 28/*2 questions by Open AI, thus they had way larger chunks compared to ours 466/*2 questions on every small chunk.


## More ideas

With LLM and embeddings becoming more powerful and lightweight, it shows great promise for future RAGs given open source solutions and can be cost-effective and also avoid privacy concerns and data leaks.

## Dependencies

This project uses the following dependencies:

- transformers: 4.35.0
- openai: 1.6.1
- llama_index: 0.9.22
- pypdf: 3.17.4
- accelerate: 0.25.0
- sentence_transformers: 2.2.2
- pydantic: 1.10.13
- accelerate: 0.25.0

Given the pace of LLM development, it is likely other versions may encounter issues.
  
## Contact

Feel free to reach out if you have any questions, suggestions, or feedback related to this project. I'd love to hear from you!

- **LinkedIn:** [Krunal Gedia](https://www.linkedin.com/in/krunal-gedia-00188899/)

## References
[1]: [Paul Graham Essay text](https://www.paulgraham.com/worked.html) 

[2]: [Open AI cookbook for RAG](https://cookbook.openai.com/examples/evaluation/evaluate_rag_with_llamaindex)

[3]: [Small general Embedding, BAAI general Embedding from Huggingface](https://huggingface.co/BAAI/bge-small-en-v1.5)

[4]: [LLM: mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)

