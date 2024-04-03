from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter 

def rag(urls,question):
    #  Defining the local model and loading the urls
    model = Ollama(model="mistral")
    urls_list = urls.split("\n")
    docs = [WebBaseLoader(url).load() for url in urls_list]
    docs_list = [item for sublist in docs for item in sublist]
    
    # Splitting the texts into chunks
    text_splitter = RecursiveCharacterTextSplitter( chunk_size=5000, 
                                                    chunk_overlap=100,
                                                    length_function=len)
    all_splits = text_splitter.split_documents(docs_list)
    
    vectorstore = Chroma.from_documents(documents=all_splits,
                                        collection_name="rag-chroma",
                                        embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
                                        )
    
    retriever = vectorstore.as_retriever()
    
    #RAG

    prompt_template = """Answer the question based only on the following context :
                        {context}
                        Question : {question}"""

    after_rag_prompt = ChatPromptTemplate.from_template(prompt_template)

    post_rag_chain = ChatPromptTemplate.from_template(
        {"context":retriever,"question":RunnablePassthrough()}
        | after_rag_prompt
        | model
        | StrOutputParser()
    )
    return post_rag_chain.invoke(question)