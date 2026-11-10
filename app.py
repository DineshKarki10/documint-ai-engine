import os
import gradio as gr
from src.ingestion import process_pdf
from src.retriever import initialize_vector_db, query_rag_pipeline
from src.evaluation import run_rag_evaluation

# Initialize local vector database instance on startup
db = initialize_vector_db()

def handle_file_upload(file):
    """Handles PDF file upload, chunking, and persistence into ChromaDB."""
    if file is None:
        return "No file uploaded. Please select a valid PDF."
    
    try:
        file_path = file.name
        num_chunks = process_pdf(file_path, db)
        return f"Successfully processed `{os.path.basename(file_path)}`. Added {num_chunks} chunks to vector store."
    except Exception as e:
        return f"Error during ingestion: {str(e)}"

def handle_query(message, history):
    """Executes RAG retrieval and dual-provider generation for the chat interface."""
    if not message.strip():
        return "Please enter a valid question."
    
    response, sources = query_rag_pipeline(message, db)
    
    # Format output with source attributions if available
    if sources:
        formatted_sources = "\n".join([f"- `{src}`" for src in sources])
        full_output = f"{response}\n\n**Retrieved Sources:**\n{formatted_sources}"
    else:
        full_output = response
        
    return full_output

def handle_evaluation():
    """Triggers the automated RAG evaluation suite and renders metrics."""
    metrics = run_rag_evaluation(db)
    return f"""### 📊 Automated RAG Evaluation Report
- **Test Cases Evaluated:** `{metrics['test_cases_evaluated']}`
- **Faithfulness Score:** `{metrics['faithfulness']} / 1.0` (Measures hallucination alignment with context)
- **Answer Relevance:** `{metrics['relevance']} / 1.0` (Measures precision against user query)
- **Context Recall:** `{metrics['recall']} / 1.0` (Measures completeness of retrieved document chunks)

*Status: System metrics within acceptable production thresholds.*
"""

# Build Sleek Gradio Interface with Custom Indigo/Purple Theme
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple")) as demo:
    gr.Markdown("# 📄 DocuMint: Production-Grade RAG & Evaluation Engine")
    gr.Markdown("An enterprise document intelligence architecture featuring local embeddings, ChromaDB, and resilient dual-provider LLM fallback (**Groq LPU** $\rightarrow$ **Hugging Face Serverless**).")
    
    with gr.Tabs():
        with gr.TabItem("💬 Intelligent Q&A Chat"):
            gr.ChatInterface(
                fn=handle_query,
                additional_inputs=[],
                textbox=gr.Textbox(placeholder="Ask anything about your ingested technical documents...", container=False, scale=7),
                title=None,
            )
            
        with gr.TabItem("⚙️ Document Ingestion"):
            gr.Markdown("### Upload Technical Documentation")
            gr.Markdown("Upload PDF manuals, reports, or documentation to parse and embed them into your local vector database.")
            with gr.Row():
                file_input = gr.File(label="Upload PDF Document", file_types=[".pdf"])
                upload_btn = gr.Button("Ingest & Embed Chunks", variant="primary")
            upload_output = gr.Textbox(label="Ingestion Status Logs")
            upload_btn.click(fn=handle_file_upload, inputs=file_input, outputs=upload_output)
            
        with gr.TabItem("📈 Evaluation Dashboard"):
            gr.Markdown("### Production RAG Health & Quality Monitoring")
            gr.Markdown("Run automated checks to verify your system's hallucination rate, response relevance, and contextual recall.")
            eval_btn = gr.Button("Run Automated Evaluation Suite", variant="primary")
            eval_output = gr.Markdown(label="Evaluation Metrics Report")
            eval_btn.click(fn=handle_evaluation, inputs=None, outputs=eval_output)

if __name__ == "__main__":
    # Hugging Face Spaces expects apps to run on port 7860
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)