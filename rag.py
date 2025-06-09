from requests_html import HTMLSession
from bs4 import BeautifulSoup
from ollama import Client
import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import asyncio


## Scrape website
async def scrape_website(url):
    session = HTMLSession()
    r = session.get(url)
    html = r.text
    soup = BeautifulSoup(r.text, 'lxml')
    text = soup.text.split("\n")
    texts = list(set(text))
    sentences = [x for x in texts if " " in x]
    thirty = sorted(sentences, key=len, reverse=True)[:20] ### take out for openai version
    #return sentences ### change for openai version
    return thirty


## Create chromadb client and create collection
async def main(url, question):
    ## Initiate ollama client
    ollama = Client(host='http://ollama:11434/')
    ollama.pull("qwen2.5:0.5b")
    ## Initialize chromadb client
    client = chromadb.PersistentClient(path="./chromadb/db.chroma", settings=Settings(allow_reset=True))
    collection = client.get_or_create_collection(name="my_collection")
    if url:
        ## Scrape website
        sentences = await scrape_website(url)
        ## Add documents to the chromadb to be embedded
        for i, d in enumerate(sentences):
            response = ollama.embed(model="qwen2.5:0.5b", input=d)
            embeddings = response["embeddings"]
            collection.add(
                ids=[str(i)],
                embeddings=embeddings,
                documents=[d]
            )
    ## DSPy
    ## Initialize llm for dspy
    lm = dspy.LM('ollama_chat/qwen2.5:0.5b', api_base='http://ollama:11434', api_key='')
    dspy.configure(lm=lm)
    ## Create embedding function with Ollama
    embedding_function = OllamaEmbeddingFunction(
        url="http://ollama:11434",
        model_name="qwen2.5:0.5b"
    )
    ## Create DSPy retriever model with chromadb
    retriever_model = ChromadbRM(
        client=client,
        collection_name="my_collection",
        persist_directory="./chromadb/db.chroma",
        embedding_function=embedding_function,
        k=5
    )
    ## Create DSPy function calling the chroma retriever
    def default_rag(query: str) -> list[str]:
        results = retriever_model(query, k=5)
        final_list =[]
        for x in results:
            if "text" in list(x.keys()):
                final_list.append(x["text"])
            elif "long_text" in list(x.keys()):
                final_list.append(x["long_text"])
        return final_list
    ## Create the DSPy signature for RAG
    class GenerateAnswer(dspy.Signature):
        """Think and list Answers to question based on the context provided."""
        context: list[str] = dspy.InputField(desc="May contain relevant facts about user query")
        question: str = dspy.InputField(desc="User query")
        answers: dict[int, str] = dspy.OutputField(desc="Iterate through answers")
    ## Create the DSPy rag function with the Signature specifying response data type
    rag = dspy.ChainOfThought(GenerateAnswer)
    ## Ask a question using the DSPy rag function
    dspy_prediction = rag(context=default_rag(question), question=question)
    ## Langchain
    embeddings = OllamaEmbeddings(model="qwen2.5:0.5b")
    q_embedded = ollama.embed(model="qwen2.5:0.5b", input=question)
    q_embeddings = q_embedded["embeddings"]
    results = collection.query(
        query_embeddings=q_embeddings,
        n_results=5
    )
    docs = [Document(page_content=x) for x in results["documents"][0]]
    model = ChatOllama(
        model="qwen2.5:0.5b",
    )
    prompt = ChatPromptTemplate.from_template(
        "Summarize the main themes in these retrieved docs: {docs} and respond in a list of key, value pairs."
    )
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    chain = {"docs": format_docs} | prompt | model | StrOutputParser()
    lc_response = chain.invoke(docs)
    return (dspy_prediction.answers, lc_response)


async def reset_chroma():
    client = chromadb.PersistentClient(path="./chromadb/db.chroma", settings=Settings(allow_reset=True))
    ## Empty / Reset chromadb
    client.reset()


