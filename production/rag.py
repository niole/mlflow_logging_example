from db import collection

def query_docs(question: str) -> str:
    results = collection.query(
        query_texts=[question],
        n_results=5
    )

    return '.'.join(results['documents'][0])
