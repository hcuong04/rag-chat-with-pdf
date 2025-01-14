import os
import json
from pathlib import Path
import PyPDF2
from PIL import Image  # For image processing
import pytesseract  # For OCR (Optical Character Recognition)
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
IMAGES_DIR = BASE_DIR / "images"
METADATA_FILE = BASE_DIR / "metadata.json"

# Ensure directories exist
CHATS_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Initialize metadata
if not METADATA_FILE.exists():
    METADATA_FILE.write_text(json.dumps({"files": {}, "chats": {}, "images": {}}, indent=4))

# Load LLM
llm_local = ChatOllama(model="llama3.2:1b", base_url="http://127.0.0.1:11434")


# Metadata utility functions
def load_metadata():
    """Load metadata from file."""
    with METADATA_FILE.open("r") as f:
        return json.load(f)


def save_metadata(metadata):
    """Save metadata to file."""
    with METADATA_FILE.open("w") as f:
        json.dump(metadata, f, indent=4)


# Create chat folder
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


# Extract text from PDFs
def extract_text_from_pdf(file_path):
    """Extract text from PDF."""
    pdf = PyPDF2.PdfReader(file_path)
    return "".join(page.extract_text() for page in pdf.pages)


# Extract text from images
def extract_text_from_image(image_path):
    """Extract text from an image using OCR."""
    return pytesseract.image_to_string(Image.open(image_path))


# Create text chunks
def create_chunks(filename, text):
    """Split text into chunks and save them."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)

    chunk_files = []
    for idx, chunk in enumerate(texts):
        chunk_file_name = f"chunk_{Path(filename).stem}_{idx}.txt"
        chunk_file = CHUNKS_DIR / chunk_file_name
        chunk_file.write_text(chunk)
        chunk_files.append(str(chunk_file))
    return chunk_files


# Set up retrieval chain
def setup_chain():
    """Set up the retrieval chain using all available chunks."""
    chunk_files = list(CHUNKS_DIR.glob("chunk_*.txt"))
    chunk_texts = [chunk.read_text() for chunk in chunk_files]
    metadatas = [{"source": str(chunk)} for chunk in chunk_files]

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma.from_texts(chunk_texts, embeddings, metadatas=metadatas)

    # Set the retriever to return up to x documents
    retriever = docsearch.as_retriever(search_kwargs={"k": 20})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )
    
    # cl.Message(content="You can start asking questions based on uploaded files.").send()

    return ConversationalRetrievalChain.from_llm(
        llm=llm_local,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )


@cl.on_chat_start
async def on_chat_start():
    language = "vi-VN"

    root_path  = Path(__file__).parent
    
    translated_chainlit_md_path = root_path / f"chainlit_{language}.md"
    default_chainlit_md_path = root_path / "chainlit.md"
    if translated_chainlit_md_path.exists():
        message = translated_chainlit_md_path.read_text()
    else:
        message = default_chainlit_md_path.read_text()
    startup_message = cl.Message(content=message)
    await startup_message.send()
    
    """Handle the start of a new chat session."""
    metadata = load_metadata()

    # Ensure chat folder structure and create a new chat folder
    chat_index, chat_folder = ensure_chat_folder(metadata)

    # Store chat info in user session
    cl.user_session.set("chat_index", chat_index)
    cl.user_session.set("metadata", metadata)

    await cl.Message(content=f"New chat started! Chat folder: `{chat_folder.name}`.").send()

    # Prompt for file if no files are available
    if not metadata["files"]:
        await cl.Message(content="No files found. Please upload a PDF to start.").send()
        await prompt_for_file(chat_index)
    else:
        # Set up the chain with all available chunks
        cl.user_session.set("chain", setup_chain())
        await cl.Message(content="You can start asking questions based on uploaded files.").send()


async def prompt_for_file(chat_index):
    """Prompt the user to upload a file (PDF or image) and process it."""
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Upload a file (PDF or image):",
            accept={
                "application/pdf": [".pdf"],
                "image/jpeg": [".jpg", ".jpeg"],
                "image/png": [".png"],
            },
            max_size_mb=100,
            timeout=180,
        ).send()

    metadata = cl.user_session.get("metadata")

    for file in files:
        file_extension = Path(file.name).suffix.lower()

        if file_extension == ".pdf":
            uploaded_file_path = PDF_DIR / file.name
            if not uploaded_file_path.exists():
                os.rename(file.path, uploaded_file_path)
                pdf_text = extract_text_from_pdf(uploaded_file_path)
                create_chunks(file.name, pdf_text)

                metadata["files"][file.name] = {"file_name": file.name}
                save_metadata(metadata)

        elif file_extension in [".jpg", ".jpeg", ".png"]:
            uploaded_image_path = IMAGES_DIR / file.name
            if not uploaded_image_path.exists():
                os.rename(file.path, uploaded_image_path)
                extracted_text = extract_text_from_image(uploaded_image_path)
                create_chunks(file.name, extracted_text)

                metadata["images"][file.name] = {"file_name": file.name}
                save_metadata(metadata)

    await cl.Message(content="Files uploaded and processed successfully.").send() 
    cl.user_session.set("chain", setup_chain())
    
from PIL import Image
import pytesseract  # Ensure Tesseract OCR is installed on your system


def extract_text_from_image(image_path):
    """Extract text from an image using Tesseract OCR."""
    try:
        return pytesseract.image_to_string(Image.open(image_path))
    except Exception as e:
        return f"Error processing image: {str(e)}"


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

    # Handle attached files (PDFs and Images)
    if message.elements:
        for file in message.elements:
            file_extension = Path(file.name).suffix.lower()

            # Handle PDFs
            if file_extension == ".pdf":
                uploaded_file_path = PDF_DIR / file.name
                os.rename(file.path, uploaded_file_path)

                pdf_text = extract_text_from_pdf(uploaded_file_path)
                create_chunks(file.name, pdf_text)

                metadata["files"][file.name] = {"file_name": file.name}
                save_metadata(metadata)
                await cl.Message(content=f"PDF `{file.name}` processed successfully.").send()

            # Handle Images
            elif file_extension in [".jpg", ".jpeg", ".png"]:
                uploaded_image_path = IMAGES_DIR / file.name
                os.rename(file.path, uploaded_image_path)

                extracted_text = extract_text_from_image(uploaded_image_path)
                if extracted_text.strip():
                    create_chunks(file.name, extracted_text)
                    metadata["images"][file.name] = {"file_name": file.name}
                    save_metadata(metadata)
                    await cl.Message(content=f"Text extracted from image `{file.name}`:\n\n{extracted_text}").send()
                else:
                    await cl.Message(content=f"No text found in image `{file.name}`.").send()

        cl.user_session.set("chain", setup_chain())
        return

    # Process user queries if no files are attached
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.ainvoke(message.content, callbacks=[cb])

    answer = res["answer"]
    source_documents = res["source_documents"]

    # Save the answer to a file
    chat_folder = Path(metadata["chats"][chat_index]["folder"])
    answer_file = chat_folder / f"answer_{len(metadata['chats'][chat_index]['questions']) + 1}.txt"
    answer_file.write_text(answer)

    metadata["chats"][chat_index]["questions"].append({
        "question": message.content,
        "answer_file": str(answer_file),
    })
    save_metadata(metadata)

    # Send the answer to the user
    sources_text = "\n".join(f"- {Path(doc.metadata['source']).name}" for doc in source_documents)
    await cl.Message(content=f"{answer}\n\nSources:\n{sources_text}").send()
    
@cl.on_chat_end
async def on_chat_end():
    """Handle the end of a chat session."""
    print("The user disconnected!")
    await cl.Message(content="Chat ended. You can start a new chat by typing a message or reloading the page.").send()