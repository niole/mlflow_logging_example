import chromadb
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")
collection.add(documents=["this week I have to do DOM-1234", "the president in 20205 is trump"], ids=["domino1234", "whopres2025"])
