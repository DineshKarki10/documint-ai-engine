import os
from groq import Groq
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def initialize_vector_db():
    """Initializes local ChromaDB using open-source Hugging Face sentence-transformers."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    db = Chroma(
        persist_directory=CHROMA_PERSIST_DIRECTORY, 
        embedding_function=embeddings
    )
    return db

def generate_llm_response(prompt: str, context: str) -> tuple[str, str]:
    """
    Attempts generation via Groq primary API. 
    If Groq fails or hits rate limits, gracefully falls back to Hugging Face Serverless Inference API.
    """
    system_prompt = (
        "You are an expert enterprise AI assistant. Answer the user's query strictly "
        "using the provided context. If the answer cannot be found in the context, "
        "say 'I cannot find the answer in the provided documentation.'"
    )
    user_content = f"Context:\n{context}\n\nQuery: {prompt}"

    # 1. TRY PRIMARY: GROQ
    try:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if groq_api_key:
            client = Groq(api_key=groq_api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
                max_tokens=512
            )
            return response.choices[0].message.content, "Groq (Primary - Llama 3.3 70B)"
    except Exception as e:
        print(f"Groq API primary call failed: {e}. Switching to Hugging Face fallback...")

    # 2. FALLBACK: HUGGING FACE SERVERLESS INFERENCE API 
    try:
        hf_token = os.environ.get("HF_TOKEN")
        hf_client = InferenceClient(
            token=hf_token, 
            model="meta-llama/Llama-3.1-8B-Instruct"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        completion = hf_client.chat_completion(messages=messages, max_tokens=512)
        return completion.choices[0].message.content, "Hugging Face Serverless (Fallback - Llama 3.1 8B)"
    except Exception as hf_err:
        return f"Critical Error: Both primary (Groq) and fallback (Hugging Face) inference failed. Details: {str(hf_err)}", "None"

def query_rag_pipeline(query_text: str, db: Chroma):
    """Executes vector similarity search and passes context to the resilient LLM runner."""
    docs = db.similarity_search(query_text, k=3)
    
    if not docs:
        return "No relevant documents found in the vector database. Please ingest documents first.", []
        
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("source", "Unknown PDF") for doc in docs]))
    
    answer, provider_used = generate_llm_response(query_text, context)
    
    formatted_answer = f"{answer}\n\n*Inference Provider Used: `{provider_used}`*"
    return formatted_answer, sources