# RAG_DSPy_Langchain
A RAG chatbot made with FastAPI, Ollama, ChromaDB, DSPy, and Langchain.

## *Update* ##
*Now with the recent updates to the OpenAI platform, the structured output feature with pydantic models is preferred over DSPy. Additionally, the OpenAI Agents SDK may also be preferred over LangGraph and its predecessor, LangChain.*

## To Install ##
Open a terminal in a new directory
`git clone https://github.com/StephenSand/RAG_DSPy_Langchain.git`
`cd RAG_DSPy_Langchain`
`docker compose up`
Open your browser and go to http://0.0.0.0:80 or http://127.0.0.1/

## To Remove ##
Make sure you are in the RAG_DSPy_Langchain directory with the docker-compose.yml and enter this in your terminal:
`docker compose down --rmi all -v --remove-orphans`

### Description ###
This is a lightweight RAG web application for local development.
    1. We render HTML with FastAPI so you can interact with the app in your browser.
    2. We scrape the link you enter with requests_html and parse it with BS4.
    3. We embed the longest 30 sentences using a small, local Ollama model (qwen2.5:0.5b).
    4. We save the embeddings in a vector store with ChromaDB.
    5. We embed the question you type using the same model.
    6. Your question embedding is passed to the DSPy agent to answer, the vector store is queried, and the agent generates a JSON response.
    7. Your question embedding is passed to the LangChain agent to answer, the vector store is queried, and the agent generates a markdown response.
