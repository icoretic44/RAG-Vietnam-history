from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_core.documents import Document
import torch
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
def split(docs):
   """
   Split all the documents into many chunks

   Args:
        docs: documents
   """
   text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 1404,
      chunk_overlap = 200,
      add_start_index = True,
   )
   all_split = text_splitter.split_documents(documents=docs)
   return all_split


def extract_data(folder_path:str):
  """
  Processes PDF files within a specified folder path.

  Args:
    folder_path: The path to the folder containing PDF files.
  """
  all_docs = []
  for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
      file_path = os.path.join(folder_path,filename)
      loader = PyPDFLoader(file_path)
      docs = loader.load()
      all_docs.extend(docs)
  return all_docs


def get_or_create_vector_db(persist_directory, all_splits,embeddings):
  """
  Gets an existing vector database or creates a new one if it doesn't exist.
  Saves the database to the specified directory.

  Args:
    persist_directory: The directory to persist the vector database.
    all_splits: The documents to embed and store in the vector database.
  """

  if os.path.exists(persist_directory):
    print(f"Loading existing vector database from {persist_directory}")
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

  else:
    print(f"Creating new vector database at {persist_directory}")
    vector_db = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory=persist_directory)
    vector_db.persist()
    print(f"Vector database created and saved to {persist_directory}")

  return vector_db

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
load_dotenv()

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
os.environ["HUGGINGFACE_TOKEN"] = huggingface_token
print(huggingface_token)
#embeddings from huggingface
embeddings = HuggingFaceEmbeddings(model_name="AITeamVN/Vietnamese_Embedding")

#folder path for extracted information from pypdf
path_folder = r"C:\Users\ADmin\Documents\ML\PDF"
docs = extract_data(path_folder)
all_splits = split(docs)

#create_database
db_path = r"C:\Users\ADmin\Documents\ML\vector_database"
vector_database = get_or_create_vector_db(db_path,all_splits,embeddings=embeddings)
