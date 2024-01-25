# %%writefile appy.py

import streamlit as st
from streamlit_chat import message
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.llms import HuggingFaceHub
from langchain.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

st.session_state.clicked=True
def process_data_sample(context,query):

    processed_example="""<|system|>>
      You are an AI Assistant that follows instructions extremely well.
      Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT

      CONTEXT: {context}
      </s>
      <|user|>
      {query}
      </s>
      <|assistant|>"""
    return processed_example
@st.cache_resource(show_spinner=True)
def create_bot():
  dataset_folder_path='dataset_news/'
  documents=[]
  for file in os.listdir(dataset_folder_path):
    loader=TextLoader(dataset_folder_path+file)
    documents.extend(loader.load())

  text_splitter=RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=50)
  text_splits=text_splitter.split_documents(documents)
  HF_token='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'
  os.environ['HUGGINGFACEHUB_API_TOKEN'] = HF_token
  embeddings=HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_token,
    model_name='BAAI/bge-base-en-v1.5'
)
  vectorstore = FAISS.from_documents(text_splits, embeddings)
  retriever_vectordb = vectorstore.as_retriever(search_kwargs={"k": 5})
  keyword_retriever = BM25Retriever.from_documents(text_splits)
  keyword_retriever.k =  5
  ensemble_retriever = EnsembleRetriever(retrievers=[retriever_vectordb,keyword_retriever],
                                       weights=[0.5, 0.5])
  model=HuggingFaceHub(repo_id='HuggingFaceH4/zephyr-7b-alpha',
                     model_kwargs={"temperature":0.5,"max_new_tokens":512,"max_length":64}
)
  Cohere_API_token='yJJo9CD1qw1wels1SGeexeCPeBKKbLvD0RFXnRmY'
  os.environ["COHERE_API_KEY"] =Cohere_API_token
  compressor = CohereRerank()
  compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensemble_retriever
)

  output_parser = StrOutputParser()


  return ensemble_retriever,model,compression_retriever





ensemble_retriever,model,compression_retriever = create_bot()

bot = create_bot()

def infer_bot(query):
  output_parser = StrOutputParser()
  processed_prompt = process_data_sample(compression_retriever, query)
  prompt = ChatPromptTemplate.from_template(processed_prompt)


  chain = (
    {"context": compression_retriever, "query": RunnablePassthrough()}
    | prompt
    | model
    | output_parser
)
  response = chain.invoke(query)
  return response





def display_conversation(history):
    for i in range(len(history["assistant"])):
        message(history["user"][i], is_user=True, key=str(i) + "_user")
        message(history["assistant"][i],key=str(i))

def main():
    st.title(" Tech-News Chatbot ")
    st.subheader("Chatbot showing the implementation of RAG with Zephyr LLM. ")
     # Initialize session state variables
    if "user" not in st.session_state:
        st.session_state.user = []
    if "assistant" not in st.session_state:
        st.session_state.assistant = []
    user_input = st.text_input("Enter your query")
    if st.session_state.clicked:
        if st.button("Answer"):

            answer = infer_bot(user_input)
            st.session_state["user"].append(user_input)
            st.session_state["assistant"].append(answer)

            if st.session_state["assistant"]:
                display_conversation(st.session_state)

if __name__ == "__main__":
    main()

