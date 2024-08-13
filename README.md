# RAG-With-Langchain

This repository contains the code for using a large language model to answer questions based on PDF documents, based on the insights shared by [AI VIETNAM](https://www.facebook.com/aivietnam.edu.vn/posts/778244334418287?rdid=T8Lv2BzNXM8If0u6). The project is organized into several components to handle different aspects of the workflow, including PDF parsing, query processing, and response generation. Each component is designed to work together to enable efficient and accurate question-answering from PDF content.

## Getting Started

### Prerequisites

- Python 3.4+
- Anaconda
- pip (Python package installer)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/dauvannam1804/RAG-With-Langchain.git
    ```

2. Create a virtual environment:
    ```sh
    conda create -n rag python=3.10
    ```

3. Activate a virtual environment:
    ```sh
    conda activate rag
    ```
    
4. Activate a virtual environment:
    ```sh
    pip install -r requirements.txt
    ```
    
### Running the API

1. Navigate to the `Rag-With-Langchain` directory and use the command:
    ```sh
    uvicorn src.app:app --host "0.0.0.0" --port 5000 --reload
    ```
    Note: Use a different port if port 5000 is already in use.
    
2. API after successful deployment:
   
   ![image](https://github.com/user-attachments/assets/687f86ee-00fb-4eb1-b6a1-99165e6cb91b)

3. The model results accessible through the built API:
   
   ![image](https://github.com/user-attachments/assets/fa6053d7-bd98-43f5-95f0-d22f7e2e478a)

