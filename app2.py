import os
import json
import hashlib
from pathlib import Path
import PyPDF2
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
import chainlit as cl

# Base paths
BASE_DIR = Path(__file__).parent / ".files"
PDF_DIR = BASE_DIR / "pdf"
CHUNKS_DIR = BASE_DIR / "chunks"
METADATA_FILE = BASE_DIR / "metadata.json"

# Ensure directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize metadata
if not METADATA_FILE.exists():
    METADATA_FILE.write_text(json.dumps({"files": {}, "chats": {}}, indent=4))

# Load LLM
llm_local = ChatOllama(model="llama3.2:1b", base_url="http://127.0.0.1:11434")

def load_metadata():
    """Load metadata from file."""
    with METADATA_FILE.open("r") as f:
        return json.load(f)

def save_metadata(metadata):
    """Save metadata to file."""
    with METADATA_FILE.open("w") as f:
        json.dump(metadata, f, indent=4)

def extract_text_from_pdf(file_path):
    """Extract text from PDF."""
    pdf = PyPDF2.PdfReader(file_path)
    return "".join(page.extract_text() for page in pdf.pages)

def create_chunks(filename, text):
    """Split text into chunks and save them."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)

    chunk_files = []
    for idx, chunk in enumerate(texts):
        chunk_file = CHUNKS_DIR / f"chunk_{filename}_{idx}.txt"
        chunk_file.write_text(chunk)
        chunk_files.append(str(chunk_file))
    return chunk_files

def setup_chain(chunk_files):
    """Set up the retrieval chain."""
    chunk_texts = [Path(chunk).read_text() for chunk in chunk_files]
    metadatas = [{"source": chunk} for chunk in chunk_files]

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma.from_texts(chunk_texts, embeddings, metadatas=metadatas)

    # Set the retriever to return up to 4 documents
    retriever = docsearch.as_retriever(search_kwargs={"k": 4})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm_local,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

@cl.on_chat_start
async def on_chat_start():
    metadata = load_metadata()

    # Determine the next chat index
    chat_index = len(metadata["chats"]) + 1
    chat_folder_name = f"chat_{chat_index}"
    chat_folder = BASE_DIR / chat_folder_name
    chat_folder.mkdir(parents=True, exist_ok=True)

    # Add chat entry to metadata
    metadata["chats"][chat_index] = {"folder": str(chat_folder), "questions": [], "chunks": []}
    save_metadata(metadata)

    # Store in user session
    cl.user_session.set("chat_index", chat_index)
    cl.user_session.set("metadata", metadata)

    await cl.Message(content=f"New chat started! Chat folder: `{chat_folder_name}`.").send()

    if not metadata["files"]:
        await cl.Message(content="No files found. Please upload a PDF to start.").send()
        await prompt_for_file(chat_index)
    else:
        # Use chunks from all previously uploaded files for the first chain setup
        chat_chunks = metadata["chats"][chat_index]["chunks"]
        if not chat_chunks:
            all_chunks = [chunk for file in metadata["files"].values() for chunk in file["chunks"]]
            metadata["chats"][chat_index]["chunks"].extend(all_chunks)
            save_metadata(metadata)
        cl.user_session.set("chain", setup_chain(metadata["chats"][chat_index]["chunks"]))
        await cl.Message(content="You can start asking questions based on uploaded files.").send()

async def prompt_for_file(chat_index):
    """Prompt the user to upload a file and process it."""
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Upload a PDF file:",
            accept={"application/pdf": [".pdf"]},
            max_size_mb=100,
            timeout=180,
        ).send()

    file = files[0]
    uploaded_file_path = PDF_DIR / file.name
    os.rename(file.path, uploaded_file_path)

    # Process the uploaded file
    pdf_text = extract_text_from_pdf(uploaded_file_path)
    chunk_files = create_chunks(file.name, pdf_text)

    metadata = cl.user_session.get("metadata")
    metadata["files"][file.name] = {"file_name": file.name, "chunks": chunk_files}
    metadata["chats"][chat_index]["chunks"].extend(chunk_files)
    save_metadata(metadata)

    # Update chain with new chunks
    cl.user_session.set("chain", setup_chain(metadata["chats"][chat_index]["chunks"]))

    await cl.Message(content=f"File `{file.name}` processed successfully. You can now ask questions!").send()

@cl.on_message
async def main(message: cl.Message):
    metadata = cl.user_session.get("metadata")
    chat_index = cl.user_session.get("chat_index")

    # Validate chat index
    if not chat_index or chat_index not in metadata["chats"]:
        await cl.Message(content="Error: Chat session not properly initialized.").send()
        return

    chain = cl.user_session.get("chain")
    if not chain:
        await cl.Message(content="No chain found. Please upload a file to continue.").send()
        return

    # Process user query
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    # Specific check for the "upload another file" trigger
    if "Tôi không biết" in answer:
        await cl.Message(
            content=(
                "It seems I don't have enough information to answer your question. "
                "Please upload another file to provide more context."
            )
        ).send()
        await prompt_for_file(chat_index)
        return

    # Save question/answer and sources in chat metadata
    question_index = len(metadata["chats"][chat_index]["questions"]) + 1
    sources = []

    # Collect content from all sources
    for doc in source_documents:
        source_text = Path(doc.metadata["source"]).read_text()
        sources.append(source_text)

    # Save sources to a file
    sources_file = Path(metadata["chats"][chat_index]["folder"]) / f"sources_{question_index}.txt"
    sources_file.write_text("\n".join(sources))

    metadata["chats"][chat_index]["questions"].append({
        "question": message.content,
        "answer": answer,
        "sources_file": str(sources_file),
    })
    save_metadata(metadata)

    # Respond with answer and sources
    sources_text = "\n".join(f"- {src}" for src in sources)
    await cl.Message(content=f"{answer}\n\nSources:\n{sources_text}").send()

@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")

@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")
