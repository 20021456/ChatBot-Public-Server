import streamlit as st

from dotenv import load_dotenv
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import re


load_dotenv()

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja")
persist_directory = 'chroma/'

# ### Save db
# folder_path="./documents"
# loader = DirectoryLoader(folder_path, glob="*.docx", loader_cls=UnstructuredWordDocumentLoader, use_multithreading=True)
# # Create a vector store based on the crawled data
# index = VectorstoreIndexCreator().from_loaders([loader])

# ### Use Chroma
# # load the document and split it into chunks
# documents = loader.load()

# # split it into chunks
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

# db = Chroma.from_documents(docs, embedding_function, persist_directory=persist_directory)

### Load db
db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name='gpt-3.5-turbo-16k',temperature=0.1), db.as_retriever())

def is_valid_input(input_text):
    # Kiểm tra xem input_text có phải là một câu hỏi hợp lệ hay không
    if not input_text.endswith("?"):
        return False

    # Kiểm tra xem input_text có liên quan đến thủ tục hành chính về đất đai hay không
    keywords = ["thủ tục", "hành chính"]
    for keyword in keywords:
        if re.search(r"\b" + re.escape(keyword) + r"\b", input_text, re.IGNORECASE):
            return True

    return False

def chat_actions():
    history = []
    text = st.session_state["chat_input"]
    st.session_state["chat_history"].append(
        {"role": "user", "content": text},
    )

    if not is_valid_input(text):
        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "content": "Đây là một câu hỏi không hợp lệ. Xin vui lòng đặt câu hỏi khác.",
            },
        )
        return

    res = qa({"question": text, "chat_history": history})
    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": res["answer"],
        },
    )

def main():

    st.title("Hỏi đáp dịch vụ công trực tuyến")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    text = st.chat_input("Câu hỏi của bạn", on_submit=chat_actions, key="chat_input")

    for i in st.session_state["chat_history"]:
        with st.chat_message(name=i["role"]):
            st.write(i["content"])

if __name__ == '__main__':
    main()