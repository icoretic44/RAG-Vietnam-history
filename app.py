import os
import streamlit as st
from dotenv import load_dotenv
import requests

# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.tools.render import render_text_description_and_args 

st.set_page_config(page_title="Hỏi Đáp Lịch Sử VN", layout="wide") #st config should be the first place

load_dotenv()
VECTORSTORE_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
DEVICE_SETTING_EMBEDDINGS = 'cpu'
USE_OLLAMA = False #False to use Gemini
OLLAMA_MODEL = "gemma:2b"
GEMINI_MODEL = "gemini-2.0-flash"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not TAVILY_API_KEY:
    st.error("Khóa API Tavily chưa được đặt...")
    st.stop()
if not USE_OLLAMA and not GOOGLE_API_KEY:
    st.error("Khóa API Google chưa được đặt...")
    st.stop()

# --- Load Resources 
@st.cache_resource
def load_embeddings_app(model_name):
    print(f"App: Đang tải mô hình nhúng: {model_name}")
    try:
        embeddings_instance = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': DEVICE_SETTING_EMBEDDINGS}
        )
        print(f"App: Mô hình nhúng đã được tải trên {DEVICE_SETTING_EMBEDDINGS.upper()}.")
        return embeddings_instance
    except Exception as e:
        st.error(f"App: Lỗi khi tải mô hình nhúng '{model_name}': {e}")
        st.stop()

@st.cache_resource
def load_vectorstore_app(vectorstore_path, _embeddings_instance):
    if os.path.exists(vectorstore_path):
        print(f"App: Đang tải vector store từ: {vectorstore_path}")
        try:
            vectorstore = FAISS.load_local(
                vectorstore_path,
                _embeddings_instance,
                allow_dangerous_deserialization=True
            )
            print("App: Vector store đã được tải.")
            return vectorstore
        except Exception as e:
            st.error(f"App: Lỗi khi tải vector store: {e}")
            return None
    else:
        st.error(f"App: Vector store không tìm thấy tại '{vectorstore_path}'. Vui lòng chạy ingest_data.py trước.")
        return None

@st.cache_resource
def get_llm_app():
    if USE_OLLAMA:
        try:
            llm_instance = ChatOllama(model=OLLAMA_MODEL, temperature=0.0)
            llm_instance.invoke("Xin chào")
            print(f"App: Mô hình Ollama '{OLLAMA_MODEL}' đã được khởi tạo.")
            return llm_instance
        except Exception as e:
            st.error(f"App: Không thể kết nối với mô hình Ollama '{OLLAMA_MODEL}'. Lỗi: {e}")
            st.stop()
    else:
        try:
            llm_instance = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.0
            )
            llm_instance.invoke("Xin chào")
            print(f"App: Mô hình Gemini '{GEMINI_MODEL}' đã được khởi tạo.")
            return llm_instance
        except Exception as e:
            st.error(f"App: Không thể khởi tạo mô hình Gemini '{GEMINI_MODEL}'. Lỗi: {e}")
            st.stop()

embeddings_app = load_embeddings_app(EMBEDDING_MODEL_NAME)
vectorstore_app = load_vectorstore_app(VECTORSTORE_PATH, embeddings_app)
llm_app = get_llm_app()

# --- Streamlit UI ---

st.title("💬 Hỏi Đáp Lịch Sử Việt Nam (RAG + Tavily Search)")
st.caption("Dựa trên tài liệu PDF và tìm kiếm web với Tavily")

if vectorstore_app and llm_app:
    retriever = vectorstore_app.as_retriever(search_kwargs={'k': 3})

    retriever_tool = create_retriever_tool(
        retriever,
        "tim_kiem_tai_lieu_lich_su_viet_nam_pdf",
        ("Công cụ này tìm kiếm trong một cơ sở dữ liệu PDF cục bộ chứa thông tin chi tiết về lịch sử Việt Nam, "
         "bao gồm các sự kiện, trận đánh (ví dụ: đồi Him Lam, Ấp Bắc), nhân vật và diễn biến liên quan đến "
         "Chiến dịch Điện Biên Phủ và Chiến dịch Hồ Chí Minh. "
         "Hãy SỬ DỤNG CÔNG CỤ NÀY ĐẦU TIÊN VÀ ƯU TIÊN NHẤT cho các câu hỏi liên quan đến các chủ đề này. "
         "Chỉ dùng công cụ khác nếu công cụ này không tìm thấy thông tin đầy đủ hoặc trả về 'không có thông tin'."),
    )
    tavily_tool = TavilySearchResults(
        max_results=2,
        api_key=TAVILY_API_KEY,
        name="tim_kiem_web_tavily", # Tool name for the agent
        description="Một công cụ tìm kiếm trên web. Sử dụng công cụ này để tìm thông tin cập nhật hoặc các chủ đề không có trong tài liệu lịch sử PDF, hoặc khi công cụ tìm kiếm PDF không cung cấp đủ thông tin."
    )
    tools = [retriever_tool, tavily_tool]

    # 4. Get the Agent Prompt Template from LangChain Hub
    try:
        prompt = hub.pull("hwchase17/react")
        if isinstance(prompt, ChatPromptTemplate):
            pass 
        elif isinstance(prompt, PromptTemplate) and isinstance(prompt.template, str):
             if "\nHãy luôn trả lời bằng tiếng Việt." not in prompt.template:
                prompt.template = prompt.template + "\nHãy luôn trả lời bằng tiếng Việt. Ưu tiên sử dụng 'tim_kiem_tai_lieu_lich_su_viet_nam_pdf' trước cho các câu hỏi lịch sử Việt Nam."
        print("Prompt from LangChain Hub loaded.")

    except Exception as e:
        st.error(f"Không thể tải prompt từ LangChain Hub: {e}. Sử dụng prompt dự phòng.")
        # Fallback prompt remains the same
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Bạn là một trợ lý AI hữu ích. Hãy trả lời câu hỏi của người dùng một cách tốt nhất có thể. Bạn có quyền truy cập vào các công cụ sau:\n{tools}\nSử dụng định dạng sau:\nCâu hỏi: câu hỏi đầu vào bạn phải trả lời\nSuy nghĩ: bạn nên luôn suy nghĩ về những việc cần làm\nHành động: hành động cần thực hiện, phải là một trong [{tool_names}]\nĐầu vào hành động: đầu vào cho hành động\nQuan sát: kết quả của hành động\n... (Suy nghĩ/Hành động/Đầu vào hành động/Quan sát này có thể lặp lại N lần)\nSuy nghĩ: Bây giờ tôi đã biết câu trả lời cuối cùng\nCâu trả lời cuối cùng: câu trả lời cuối cùng cho câu hỏi đầu vào ban đầu của người dùng. Luôn trả lời bằng tiếng Việt."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"), # Correct for ReAct
        ])


    if "agent_executor" not in st.session_state:
        try:
            agent = create_react_agent(llm_app, tools, prompt)
            st.session_state.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10
            )
            print("Agent Executor được tạo.")
        except Exception as e:
            st.error(f"Lỗi khi tạo Agent Executor: {e}")
            st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Câu hỏi của bạn:"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            thinking_process = "" # To capture agent's thoughts
            with st.spinner("Agent đang xử lý..."):
                try:
                    result = st.session_state.agent_executor.invoke(
                        {"input": user_query}
                    )
                    full_response = result.get("output", "Xin lỗi, tôi không thể tìm thấy câu trả lời.")

                except Exception as e:
                    full_response = f"Lỗi agent: {e}\n\nSuy nghĩ của Agent (nếu có):\n{thinking_process}"
                    st.error(full_response)

            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.error("Không thể tải Vector Store hoặc LLM. Hãy đảm bảo bạn đã chạy `ingest_data.py` và cấu hình đúng.")
