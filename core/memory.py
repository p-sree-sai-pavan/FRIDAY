import chromadb
from sentence_transformers import SentenceTransformer
import uuid,threading,sys,os,time,config

client = chromadb.PersistentClient(path=config.MEMORY_DB_PATH)
model = SentenceTransformer(config.EMBEDDING_MODEL)
collection = client.get_or_create_collection(name=config.MEMORY_COLLECTION)