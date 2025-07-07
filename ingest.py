import os
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define paths
BLOGS_PATH = "blogs"
DB_PATH = "db"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

def ingest_blogs():
    """
    Ingests blog posts from the specified directory, processes them,
    and stores them in a Chroma vector database.
    """
    print("Starting blog ingestion...")

    # Find all story.md files
    markdown_files = []
    for root, _, files in os.walk(BLOGS_PATH):
        for file in files:
            if file == "story.md":
                markdown_files.append(os.path.join(root, file))

    if not markdown_files:
        print(f"No 'story.md' files found in the '{BLOGS_PATH}' directory. Please add your blogs to proceed.")
        return

    print(f"Found {len(markdown_files)} blog(s) to process.")

    # Load documents
    documents = []
    for md_file in markdown_files:
        try:
            loader = UnstructuredMarkdownLoader(md_file)
            docs = loader.load()
            documents.extend(docs)
            print(f"Successfully loaded: {md_file}")
        except Exception as e:
            print(f"Error loading {md_file}: {e}")
            continue
    
    if not documents:
        print("Could not load any documents. Aborting.")
        return

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Create embeddings
    print("Creating embeddings... (This may take a while for the first run)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Create and persist the vector store
    print("Creating vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print(f"\nIngestion complete! Vector store created at '{DB_PATH}'.")
    print(f"You can now run the main application to interact with your AI writing assistant.")

if __name__ == "__main__":
    ingest_blogs()