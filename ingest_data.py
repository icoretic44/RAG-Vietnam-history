import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
load_dotenv()

PDF_FOLDER_PATH = "PDF"
VECTORSTORE_PATH = "faiss_index" 

# === CHOOSE YOUR EMBEDDING MODEL ===
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEVICE_SETTING = 'cpu'

def ingest_documents():
    """
    Loads PDF documents, splits them, creates embeddings, and saves to FAISS.
    Deletes and recreates the vector store if it already exists.
    """
    if not os.path.exists(PDF_FOLDER_PATH):
        print(f"Lỗi: Thư mục PDF '{PDF_FOLDER_PATH}' không tồn tại.")
        print("Vui lòng tạo thư mục và đặt các tệp PDF của bạn vào đó.")
        return False
    if not os.listdir(PDF_FOLDER_PATH):
        print(f"Lỗi: Thư mục PDF '{PDF_FOLDER_PATH}' đang trống.")
        print("Vui lòng đặt các tệp PDF của bạn vào đó.")
        return False

    if os.path.exists(VECTORSTORE_PATH):
        print(f"Đang xóa vector store cũ tại: {VECTORSTORE_PATH}...")
        try:
            shutil.rmtree(VECTORSTORE_PATH)
            print("Đã xóa vector store cũ thành công.")
        except OSError as e:
            print(f"Lỗi khi xóa vector store cũ: {e}. Vui lòng xóa thủ công và thử lại.")
            return False

    print(f"Đang tải tài liệu từ: {PDF_FOLDER_PATH}...")
    try:
        loader = DirectoryLoader(
            PDF_FOLDER_PATH,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
            silent_errors=True
        )
        documents = loader.load()
        if not documents:
            print(f"Không tìm thấy hoặc không thể tải tệp PDF nào từ '{PDF_FOLDER_PATH}'.")
            return False
        print(f"Đã tải {len(documents)} trang từ các tệp PDF.")
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi tải tài liệu PDF: {e}")
        return False

    print("Đang chia nhỏ tài liệu...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    splits = text_splitter.split_documents(documents)
    print(f"Đã chia thành {len(splits)} đoạn văn bản.")

    if not splits:
        print("Không có đoạn văn bản nào được tạo sau khi chia nhỏ.")
        return False

    print(f"Đang khởi tạo mô hình nhúng: {EMBEDDING_MODEL_NAME}...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': DEVICE_SETTING}
        )
        print(f"Mô hình nhúng đã được khởi tạo trên {DEVICE_SETTING.upper()}.")
    except Exception as e:
        print(f"Lỗi khi khởi tạo mô hình nhúng: {e}")
        return False

    print("Đang tạo vector store FAISS...")
    try:
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        print("Đang lưu vector store...")
        vectorstore.save_local(VECTORSTORE_PATH)
        print(f"Vector store đã được tạo và lưu tại: {VECTORSTORE_PATH}.")
        print("Quá trình ingest dữ liệu hoàn tất!")
        return True
    except Exception as e:
        print(f"Lỗi khi tạo hoặc lưu vector store FAISS: {e}")
        return False

if __name__ == "__main__":
    if not os.path.exists(PDF_FOLDER_PATH):
        os.makedirs(PDF_FOLDER_PATH)
        print(f"Đã tạo thư mục '{PDF_FOLDER_PATH}'. Vui lòng đặt các tệp PDF của bạn vào đó và chạy lại script.")
    else:
        ingest_documents()
