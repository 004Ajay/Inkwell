

import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
DB_PATH = "db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Prompt Templates ---
# We now have different templates based on the desired style.

CREATIVE_PROMPT_TEMPLATE = """
**Instructions:** You are a creative writing assistant. Your primary goal is to write a new, original story based on the user's request.
It is crucial that you adopt the unique writing style, tone, and narrative voice of the author whose work is provided below as examples.
Do not copy the examples, but use them as a stylistic guide for your own generation. Emphasize narrative flow, character, and evocative descriptions.

**Author's Style Examples (Context):**
---
{context}
---

**User's Request (Question):**
{question}

**Formatting Instructions:**
{format_instructions}

**Your Story:**
"""

TECHNICAL_PROMPT_TEMPLATE = """
**Instructions:** You are a technical writer. Your primary goal is to create a clear, informative, and well-structured technical article based on the user's request.
Adopt the author's writing style from the examples below, focusing on how they explain complex topics.
Prioritize clarity, accuracy, and logical structure. Use the provided context for stylistic guidance only.

**Author's Style Examples (Context):**
---
{context}
---

**User's Request (Question):**
{question}

**Formatting Instructions:**
{format_instructions}

**Your Story:**
"""

def main(prompt: str, model: str, style: str, output_format: str):
    """
    Main function to run the style-augmented generation.
    """
    print(f"\nInitializing {model} model...")
    print(f"Style: {style} | Format: {output_format}")

    # --- Select Prompt Template based on Style ---
    if style == 'creative':
        template = CREATIVE_PROMPT_TEMPLATE
    else: # Default to technical
        template = TECHNICAL_PROMPT_TEMPLATE

    # --- Define Formatting Instructions ---
    if output_format == 'markdown':
        format_instructions = "Use Markdown for formatting, including headings, lists, and bold/italic text where appropriate."
    else: # Default to plaintext
        format_instructions = "Use plain text only, with no special formatting."

    # --- Load Components ---
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    try:
        llm = OllamaLLM(model=model)
        llm.invoke("Hello", stop=["Hello"])
    except Exception as e:
        print(f"\n--- Ollama Connection Error ---")
        print(f"Could not connect to Ollama. Please ensure Ollama is running and the model '{model}' is installed.")
        print(f"You can install the model by running: ollama pull {model}")
        print(f"Error details: {e}")
        return

    custom_prompt = PromptTemplate(
        input_variables=["context", "question", "format_instructions"],
        template=template
    )

    # --- Build the Generation Chain ---
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough(), "format_instructions": lambda x: format_instructions}
        | custom_prompt
        | llm
        | StrOutputParser()
    )

    print(f"\nPrompt: '{prompt}'")
    print("This may take a moment...\n")
    print(f"--- Your AI-Generated Story (Style: {style.capitalize()}) ---")

    # --- Stream the Output ---
    full_response = ""
    for chunk in rag_chain.stream(prompt):
        print(chunk, end="", flush=True)
        full_response += chunk
    
    print("\n\n--- End of Story ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Writing Assistant.\n\nSample Usage: python main.py -p 'Write a story about computer networks' -m gemma3:4b -s technical -f markdown",
        formatter_class=argparse.RawTextHelpFormatter # better formatting of help text
    )
    parser.add_argument(
        "-p", "--prompt",
        required=True,
        default="Ask user for a prompt",
        type=str, 
        help="The prompt or idea for the story you want to write."
    )
    parser.add_argument(
        "-m", "--model", 
        type=str, 
        default="gemma3:4b", 
        help="The Ollama model to use (e.g., 'gemma3:12b'). Must be installed in Ollama."
    )
    parser.add_argument(
        "-s", "--style", 
        type=str, 
        default="technical", 
        choices=['technical', 'creative'], 
        help="The writing style to adopt."
    )
    parser.add_argument(
        "-f", "--format", 
        type=str, 
        default="markdown", 
        choices=['markdown', 'plaintext'], 
        help="The output format."
    )
    
    args = parser.parse_args()
    
    main(args.prompt, args.model, args.style, args.format)
