# Fully_Open_Source_RAG_LlamaIndex

Business Problem:

Many users struggle to find accurate and contextually relevant information from vast amounts of unstructured data. Traditional search engines may not capture the nuance or context of a user's query, leading to less-than-optimal search results. Additionally, generating detailed and coherent responses to complex queries often requires a deep understanding of the context and the ability to retrieve relevant information. Privacy-conscious users may be concerned about sharing their queries or data with proprietary platforms.

Project Objective:
The primary objective of this open-source project is to develop a Privacy-Focused Retrieval-Augmented Generation (RAG) system using open-source embeddings and a Large Language Model (LLM). The system aims to enhance the accuracy and contextuality of responses to user queries while addressing privacy concerns associated with data sharing on proprietary platforms.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Workflow](#workflow)
- [Results](#results)
- [More ideas](#More-ideas)
- [Dependencies](#dependencies)
- [License](#license)
- [Contact](#contact)
- [References](#references)

## Project Overview

The goal of this project is to develop a complete open source Retrieval Augmentated Generation customizable solution. Finally we even compare the RAG with the current Open AI's ChatGPT RAG solution as well.

Key Features
* Open Source Embeddings
* Open Source LLM
* Custom Document Object ->Node Object splitting  

## Installation

```bash
# Example installation command
pip install -r requirements.txt

# Run Web Application
streamlit run app.py
```

## Data

Since we wanted to compare with the OpenAI's RAG built using Chat GPT, we use the Paul Graham Essay text [1] which is used even in the OpenAI cookbook for RAG solution [2].  

## Workflow


**1st Part: Make NodeObejcts or NodeIndex as called in LlamaIndex.**

1. Parse the source docuemnts using various data loaders into LlamaIndex's Docuement objects. By default, for each page in the document, you get one Docuemnt object. In our case, since we parse form one webpage, we get just one Docuemnt object.
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
![Image 1](https://github.com/krunalgedia/SBB_TrainTicketParser/blob/main/images_app/sample.gif)

5. We then store all the indexes using VectorStoreIndex. A VectorStoreIndex in LlamaIndex is a type of index that uses vector representations of text for efficient retrieval of relevant context.It is built on top of a VectorStore, which is a data structure that stores vectors and allows for quick nearest neighbor search. The VectorStoreIndex takes in IndexNode objects, which represent chunks of the original documents.

6. We then use the RecursiveRetriever to fetch relevant nodes. It recursively explore links from nodes to other retrievers or query engines. Thus, if any of those nodes point to another retriever or query engine, the RecursiveRetriever will follow that link and query the linked retriever or engine as well.
       
We use 
* Small BAAI general embedding [3]
* LLM MistralAI [4]

* .ipynb contains the end-to-end code for RAG with LlamaIndex and Huggingface

## Results



We fine-tuned using Facebook/Meta's LayoutLM (which utilizes BERT as the backbone and adds two new input embeddings: 2-D position embedding and image embedding) [3]. The model was imported from the Hugging Face library [4] with end-to-end code implemented in PyTorch. We leveraged the tokenizer provided by the library itself. For the test case, we perform the OCR using Pytesseract.

With just 4 SBB train tickets we can achieve an average F1 score of 0.81.   

| Epoch | Average Precision | Average Recall | Average F1 | Average Accuracy |
|--------:|------------:|---------:|-----:|-----------:|
|     145 |        0.89 |     0.77 | 0.82 |       0.9  |
|     146 |        0.9  |     0.79 | 0.84 |       0.9  |
|     147 |        0.86 |     0.77 | 0.81 |       0.89 |
|     148 |        0.87 |     0.78 | 0.82 |       0.9  |
|     149 |        0.86 |     0.77 | 0.81 |       0.89 |

The web application serves demo:
![Image 1]() | ![Image 2](https://github.com/krunalgedia/SBB_TrainTicketParser/blob/main/images_app/test1.gif)
--- | --- 
Opening page | Testing ... 

Once the user uploads the image, the document gets parsed and the information from the document gets updated in the relational database which can be used to verify the traveler's info and also to automate the travel cost-processing task.


## More ideas

Instead of using OCR from the UBIAI tool, it best is to use pyteserract or same OCR tool for train and test set. Further, with Document AI being developed at a rapid pace, it would be worthwhile to test newer multimodal models which hopefully either provide a new solution for not using OCR or inbuilt OCR since it is important to be consistent in preprocessing train and test set for best results.

Also, train on at least >50 tickets, since this was just a small test case to see how well the model can work.

## Dependencies

This project uses the following dependencies:

- **Python:** 3.10.12/3.9.18 
- **PyTorch:** 2.1.0+cu121/2.1.1+cpu
- **Streamlit:** 1.28.2 

- [SBB ticket parser model on Hugging Face](https://huggingface.co/KgModel/sbb_ticket_parser_LayoutLM)
  
## Contact

Feel free to reach out if you have any questions, suggestions, or feedback related to this project. I'd love to hear from you!

- **LinkedIn:** [Krunal Gedia](https://www.linkedin.com/in/krunal-gedia-00188899/)

## References
[1]: [Paul Graham Essay text](https://www.paulgraham.com/worked.html) 

[2]: [Open AI cookbook for RAG](https://cookbook.openai.com/examples/evaluation/evaluate_rag_with_llamaindex)

[3]: [Small general Embedding, BAAI general Embedding from Huggingface](https://huggingface.co/BAAI/bge-small-en-v1.5)

[4]: [LLM: mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)

