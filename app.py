# import PyPDF2
# from pathlib import Path
# from langchain_ollama import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain.chains import ConversationalRetrievalChain
# from langchain_ollama import ChatOllama
# from langchain.memory import ChatMessageHistory, ConversationBufferMemory
# import chainlit as cl

# llm_local = ChatOllama(model="llama3.2:1b", base_url="http://127.0.0.1:11434")
# #llm_local = ChatOllama(model_name='llama3.2:1b', base_url="http://localhost:8000")

# @cl.on_chat_start
# async def on_chat_start():
    
#     language = "vi-VN"

#     root_path  = Path(__file__).parent
    
#     translated_chainlit_md_path = root_path / f"chainlit_{language}.md"
#     default_chainlit_md_path = root_path / "chainlit.md"
#     if translated_chainlit_md_path.exists():
#         message = translated_chainlit_md_path.read_text()
#     else:
#         message = default_chainlit_md_path.read_text()
#     startup_message = cl.Message(content=message)
#     await startup_message.send()
    
#     files = None #Initialize variable to store uploaded files

#     # Wait for the user to upload a file
#     while files is None:
#         files = await cl.AskFileMessage(
#             content="Please upload a pdf file to begin!",
#             accept={"application/pdf" : [".pdf"]},
#             max_size_mb=100,
#             timeout=180, 
#         ).send()

#     file = files[0] # Get the first uploaded file
    
#     # Inform the user that processing has started
#     msg = cl.Message(content=f"Processing `{file.name}`...")
#     await msg.send()

#     # Read the PDF file
#     pdf = PyPDF2.PdfReader(file.path)
#     pdf_text = ""
#     for page in pdf.pages:
#         pdf_text += page.extract_text()

#     # Split the text into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     texts = text_splitter.split_text(pdf_text)

#     # Create a metadata for each chunk
#     metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

#     # Create a Chroma vector store
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     docsearch = await cl.make_async(Chroma.from_texts)(
#         texts, embeddings, metadatas=metadatas
#     )
    
#     # Initialize message history for conversation
#     message_history = ChatMessageHistory()
    
#     # Memory for conversational context
#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         output_key="answer",
#         chat_memory=message_history,
#         return_messages=True,
#     )

#     # Create a chain that uses the Chroma vector store
#     chain = ConversationalRetrievalChain.from_llm(
#         llm = llm_local,
#         chain_type="stuff",
#         retriever=docsearch.as_retriever(),
#         memory=memory,
#         return_source_documents=True,
#     )

#     # Let the user know that the system is ready
#     msg.content = f"Processing `{file.name}` done. You can now ask questions!"
#     await msg.update()
#     #store the chain in user session
#     cl.user_session.set("chain", chain)


# @cl.on_message
# async def main(message: cl.Message):
        
#      # Retrieve the chain from user session
#     chain = cl.user_session.get("chain") 
#     #call backs happens asynchronously/parallel 
#     cb = cl.AsyncLangchainCallbackHandler()
    
#     # call the chain with user's message content
#     res = await chain.ainvoke(message.content, callbacks=[cb])
#     answer = res["answer"]
#     source_documents = res["source_documents"] 

#     text_elements = [] # Initialize list to store text elements
    
#     # Process source documents if available
#     if source_documents:
#         for source_idx, source_doc in enumerate(source_documents):
#             source_name = f"source_{source_idx}"
#             # Create the text element referenced in the message
#             text_elements.append(
#                 cl.Text(content=source_doc.page_content, name=source_name)
#             )
#         source_names = [text_el.name for text_el in text_elements]
        
#          # Add source references to the answer
#         if source_names:
#             answer += f"\nSources: {', '.join(source_names)}"
#         else:
#             answer += "\nNo sources found"
#     #return results
#     await cl.Message(content=answer, elements=text_elements).send()
    
# @cl.on_stop
# def on_stop():
#     print("The user wants to stop the task!")

# @cl.on_chat_end
# def on_chat_end():
#     print("The user disconnected!")

import os
import hashlib
import PyPDF2
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import ChatOllama
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
import json

# Base paths
BASE_DIR = Path(__file__).parent / ".files"
PDF_DIR = BASE_DIR / "pdf"
CHUNKS_DIR = BASE_DIR / "chunks"
SOURCE_DIR = BASE_DIR / "source"
METADATA_FILE = BASE_DIR / "metadata.json"

# Ensure necessary directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize metadata
if not METADATA_FILE.exists():
    METADATA_FILE.write_text(json.dumps({"files": {}, "questions": {}}, indent=4))

# Load LLM
llm_local = ChatOllama(model="llama3.2:1b", base_url="http://127.0.0.1:11434")


def calculate_file_hash(file_path: Path) -> str:
    """Calculate a unique hash for a file."""
    hasher = hashlib.md5()
    with file_path.open("rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def save_metadata(metadata: dict):
    """Save metadata to a JSON file."""
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)


def load_metadata() -> dict:
    """Load metadata from a JSON file."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {"files": {}, "questions": {}}


@cl.on_chat_start
async def on_chat_start():
    metadata = load_metadata()

    # Wait for the user to upload a file
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept={"application/pdf": [".pdf"]},
            max_size_mb=100,
            timeout=180,
        ).send()

    file = files[0]
    uploaded_file_path = PDF_DIR / file.name

    # Move the uploaded file to the PDF directory
    os.rename(file.path, uploaded_file_path)

    file_hash = calculate_file_hash(uploaded_file_path)

    if file_hash in metadata["files"]:
        # File already processed
        msg = cl.Message(content=f"File `{file.name}` is already processed. Reusing data...")
        await msg.send()
        setup_chain(metadata["files"][file_hash]["chunks"])
    else:
        # Process new file
        msg = cl.Message(content=f"Processing `{file.name}`...")
        await msg.send()

        # Read the PDF file
        pdf = PyPDF2.PdfReader(uploaded_file_path)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(pdf_text)

        # Create chunk files
        chunk_files = []
        for idx, chunk in enumerate(texts):
            chunk_file = CHUNKS_DIR / f"chunk_{file_hash}_{idx}.txt"
            chunk_file.write_text(chunk)
            chunk_files.append(str(chunk_file))

        # Update metadata
        metadata["files"][file_hash] = {
            "file_name": file.name,
            "chunks": chunk_files,
        }
        save_metadata(metadata)

        msg.content = f"Processing `{file.name}` completed. You can now ask questions!"
        await msg.update()
        setup_chain(chunk_files)


def setup_chain(chunk_files: list):
    """Setup the conversational chain using chunk files."""
    chunk_texts = [Path(chunk).read_text() for chunk in chunk_files]
    metadatas = [{"source": f"{idx}"} for idx in range(len(chunk_texts))]

    # Create Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma.from_texts(chunk_texts, embeddings, metadatas=metadatas)

    # Memory for conversational context
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Conversational chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_local,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    # Save sources for the question
    metadata = load_metadata()
    question_index = len(metadata["questions"]) + 1
    source_files = []

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_file = SOURCE_DIR / f"source_{question_index}_{source_idx}.txt"
            source_file.write_text(source_doc.page_content)
            source_files.append(str(source_file))

    metadata["questions"][question_index] = {"question": message.content, "sources": source_files}
    save_metadata(metadata)

    # Format response
    text_elements = [cl.Text(content=Path(src).read_text(), name=f"source_{idx}") for idx, src in enumerate(source_files)]
    source_names = [text_el.name for text_el in text_elements]

    if source_names:
        answer += f"\nSources: {', '.join(source_names)}"
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()