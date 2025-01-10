import os
import json
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
CHATS_DIR = BASE_DIR / "chats"
PDF_DIR = BASE_DIR / "pdf"
CHUNKS_DIR = BASE_DIR / "chunks"
METADATA_FILE = BASE_DIR / "metadata.json"

# Ensure directories exist
CHATS_DIR.mkdir(parents=True, exist_ok=True)
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


def ensure_chat_folder(metadata):
    """Ensure the `chats` directory exists and create a new chat folder."""
    chat_index = len(metadata["chats"]) + 1
    chat_folder_name = f"chat_{chat_index}"
    chat_folder = CHATS_DIR / chat_folder_name
    chat_folder.mkdir(parents=True, exist_ok=True)

    # Add chat entry to metadata
    metadata["chats"][chat_index] = {"folder": str(chat_folder), "questions": []}
    save_metadata(metadata)

    return chat_index, chat_folder


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


def setup_chain():
    """Set up the retrieval chain using all available chunks."""
    chunk_files = list(CHUNKS_DIR.glob("chunk_*.txt"))
    chunk_texts = [chunk.read_text() for chunk in chunk_files]
    metadatas = [{"source": str(chunk)} for chunk in chunk_files]

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

    if uploaded_file_path.exists():
        await cl.Message(content=f"File `{file.name}` already exists. No need to re-upload.").send()
        return

    os.rename(file.path, uploaded_file_path)

    # Process the uploaded file
    pdf_text = extract_text_from_pdf(uploaded_file_path)
    create_chunks(file.name, pdf_text)

    metadata = cl.user_session.get("metadata")
    metadata["files"][file.name] = {"file_name": file.name}
    save_metadata(metadata)

    # Update chain with new chunks
    cl.user_session.set("chain", setup_chain())

    await cl.Message(content=f"File `{file.name}` processed successfully. You can now ask questions!").send()


@cl.on_chat_start
async def on_chat_start():
    """Handle the start of a new chat session."""
    metadata = load_metadata()

    # Ensure chat folder structure and create a new chat folder
    chat_index, chat_folder = ensure_chat_folder(metadata)

    # Store in user session
    cl.user_session.set("chat_index", chat_index)
    cl.user_session.set("metadata", metadata)

    await cl.Message(content=f"New chat started! Chat folder: `{chat_folder.name}`.").send()

    if not metadata["files"]:
        await cl.Message(content="No files found. Please upload a PDF to start.").send()
        await prompt_for_file(chat_index)
    else:
        # Set up the chain with all available chunks
        cl.user_session.set("chain", setup_chain())
        await cl.Message(content="You can start asking questions based on uploaded files.").send()


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

    # Check for file upload triggers
    if message.content.lower() in ["add a file", "thêm file"]:
        await prompt_for_file(chat_index)
        return

    # Process user query
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    # Trigger file upload if answer is not known
    if "I don't know" in answer or "Tôi không biết" in answer:
        await cl.Message(
            content="It seems I don't have enough information to answer your question. Please upload another file."
        ).send()
        await prompt_for_file(chat_index)
        return

    # Save the answer to a file
    chat_folder = Path(metadata["chats"][chat_index]["folder"])
    answer_file = chat_folder / f"answer_{len(metadata['chats'][chat_index]['questions']) + 1}.txt"
    answer_file.write_text(answer)

    metadata["chats"][chat_index]["questions"].append({
        "question": message.content,
        "answer_file": str(answer_file),
    })
    save_metadata(metadata)

    # Respond with the answer and sources
    sources_text = "\n".join(f"- {Path(doc.metadata['source']).name}" for doc in source_documents)
    await cl.Message(content=f"{answer}\n\nSources:\n{sources_text}").send()


@cl.on_chat_end
async def on_chat_end():
    """Handle the end of a chat session."""
    metadata = cl.user_session.get("metadata")

    # Restart the chat with a new folder
    chat_index, chat_folder = ensure_chat_folder(metadata)
    cl.user_session.set("chat_index", chat_index)
    cl.user_session.set("metadata", metadata)

    await cl.Message(content=f"Chat ended. New chat started in folder `{chat_folder.name}`.").send()
