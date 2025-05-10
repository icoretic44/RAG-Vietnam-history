import os # Assuming you're using Chroma for vector storage
import torch  # Import PyTorch for CUDA support

from dotenv import load_dotenv,find_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import Client
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
# Load environment variables
load_dotenv(find_dotenv())

os.environ["HUGGINGFACE_API_KEY"] = str(os.getenv("HUGGINGFACE_API_KEY"))
os.environ["LANGSMITH_API_KEY"] = str(os.getenv("LANGSMITH_API_KEY"))
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
client = Client()


# Check if vector_database is already existed then
# using again, otherwise create them
def get_vector_database(persistent_directory, embeddings):
    # Check if the database already exists
    if os.path.exists(persistent_directory):
        print("Using existing vector database.")
        try:
            # Load the existing database
            vector_database = Chroma(
                persist_directory=persistent_directory,
                embedding_function=embeddings
            )
        except KeyError as e:
            print(f"KeyError encountered: {e}. It may indicate a corrupted database.")
            # Optionally, you can delete the existing database here
            # os.rmdir(persistent_directory)  # Uncomment to delete the directory
            raise
    else:
        print("Creating a new vector database.")
        # Create a new database
        vector_database = Chroma.from_documents(
            documents=[],  # You may need to provide actual documents here
            persist_directory=persistent_directory,
            embedding_function=embeddings
        )
    return vector_database

# Get the Hugging Face token from the environment variable
# Initialize embeddings with GPU support
device = "cuda" if torch.cuda.is_available() else "cpu"  # Check if CUDA is available
embeddings = HuggingFaceEmbeddings(model_name="AITeamVN/Vietnamese_Embedding")

db_path = r"C:\Users\ADmin\Documents\ML\vector_database"  # Specify your database path
vector_database = get_vector_database(db_path, embeddings)

retriever = vector_database.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":5},
)
llm = OllamaLLM(model = "gemma3:4b",device=device)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm , retriever, contextualize_q_prompt
)

qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

def rag_tool_wrapper(input, chat_history=None):
    try:
        response = rag_chain.invoke({
            "input": input, 
            "chat_history": chat_history or []
        })
        # Ensure we return a string
        return response.get('answer', 'No answer found')
    except Exception as e:
        print(f"RAG Tool Error: {e}")
        return f"Error in retrieving answer: {str(e)}"

# Create tools with more explicit description
tools = [
    Tool.from_function(
        func=rag_tool_wrapper,
        name="Contextual_QA",
        description="Answers questions based on the provided context. Use when you need to find information within the document context."
    )
]

react_docstore_prompt = PromptTemplate(
    template="""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: what to input to the tool
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}""",
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt= react_docstore_prompt,
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent = agent,
    tools= tools,
    handle_parsing_errors = True,
    verbose = True,
)
chat_history = []
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    response = agent_executor.invoke(
        {"input": query, "chat_history": chat_history})
    print(f"AI: {response['output']}")

    # Update history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))

