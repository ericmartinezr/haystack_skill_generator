from haystack.tools import tool
from haystack.components.converters import MarkdownToDocument
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack.components.preprocessors import DocumentCleaner
from constants import EMBEDDING_MODEL
from dotenv import load_dotenv

load_dotenv()

store = PgvectorDocumentStore(
    embedding_dimension=768,
    vector_function="cosine_similarity",
    recreate_table=True,
    search_strategy="exact_nearest_neighbor"
)

doc_embedder = OllamaDocumentEmbedder(
    model=EMBEDDING_MODEL,
    url="http://localhost:11434",
    progress_bar=True,
    generation_kwargs={
        "temperature": 0.1
    }
)

text_embedder = OllamaTextEmbedder(
    model=EMBEDDING_MODEL,
    url="http://localhost:11434",
    generation_kwargs={
        "temperature": 0.1
    }
)

cleaner = DocumentCleaner()

converter = MarkdownToDocument(
    progress_bar=True,
    store_full_path=True
)


@tool
def read_skill(sources: list[str], query: str) -> str:
    """
    Parses a markdown file and retrieves the most relevant snippets based on a query.

    Arguments:
    - sources: List of file paths to read.
    - query: The query to search for.
    """
    try:

        docs = converter.run(sources=sources)["documents"]
        cleaned_docs = cleaner.run(documents=docs)["documents"]
        embedded_docs = doc_embedder.run(documents=cleaned_docs)["documents"]
        embedded_query = text_embedder.run(text=query)["embedding"]

        writer = DocumentWriter(document_store=store)
        writer.run(documents=embedded_docs)

        # Retorna los mas relevantes
        retriever = PgvectorEmbeddingRetriever(document_store=store)
        result = retriever.run(query_embedding=embedded_query)

        # Formatea la salida
        found_docs = result["documents"]
        output = ""
        for d in found_docs:
            output += f"--- Snippet from {d.meta.get('file_path', 'unknown')} ---\n"
            output += d.content + "\n\n"

        return output if output else "No relevant content found."
    except Exception as e:
        return f"Error reading skill: {str(e)}"
