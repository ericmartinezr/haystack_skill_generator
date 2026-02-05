from haystack.tools import tool
from haystack.components.converters import MarkdownToDocument
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


@tool
def read_skill(sources: list[str], query: str) -> str:
    """
    Parses a markdown file and retrieves the most relevant snippets based on a query.

    Arguments:
    - sources: List of file paths to read.
    - query: The query to search for.
    """
    try:
        store = InMemoryDocumentStore()

        # Indexa los documentos
        converter = MarkdownToDocument(
            progress_bar=False,
            store_full_path=True
        )
        docs = converter.run(sources=sources)["documents"]
        writer = DocumentWriter(document_store=store)
        writer.run(documents=docs)

        # Retorna los mas relevantes
        retriever = InMemoryBM25Retriever(document_store=store, top_k=2)
        result = retriever.run(query=query)

        # Formatea la salida
        found_docs = result["documents"]
        output = ""
        for d in found_docs:
            output += f"--- Snippet from {d.meta.get('file_path', 'unknown')} ---\n"
            output += d.content + "\n\n"

        return output if output else "No relevant content found."
    except Exception as e:
        return f"Error reading skill: {str(e)}"
