from flask import Flask, request, jsonify, render_template
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Cohere  # Revert to original import if langchain-cohere fails
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from PyPDF2 import PdfReader
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Initialize model and vector store variables
vectorstore = None
retriever = None

# Define output model for parsing
class SummaryOutput(BaseModel):
    company_name: str = Field(description="Company Name")
    financial_performance: str = Field(description="Financial Performance Summary")
    market_dynamics: str = Field(description="Market Dynamics Summary")
    expansion_plans: str = Field(description="Expansion Plans Summary")
    environmental_risks: str = Field(description="Environmental Risks Summary")
    regulatory_or_policy_changes: str = Field(description="Regulatory or Policy Changes Summary")

parser = JsonOutputParser(pydantic_object=SummaryOutput)

# Define prompt template
prompt_template = """Answer the question as precisely as possible using the provided context. If the question is not contained in the context, try answering on your own knowledge.

Context: 
{context}?

Question: 
{question} 

Answer:
"""
prompt = PromptTemplate.from_template(template=prompt_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_answer(question):
    cohere_llm = Cohere(model="command", temperature=0.1, cohere_api_key="701G3pd6VNnGrhEmAQmMs8r3SXkvdlD6XBTu4leC")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | cohere_llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)

def create_index_and_retriever(pdf_text):
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=['\n', '\n\n', ' ', '']
    )
    chunks = text_splitter.split_text(text=pdf_text)

    # Create embeddings and index for retrieval
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    global vectorstore, retriever
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    file = request.files['file']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Extract text from the uploaded PDF
        pdf_text = ""
        pdf_reader = PdfReader(filepath)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()

        # Create index and retriever
        create_index_and_retriever(pdf_text)
        
        return jsonify({"message": "PDF processed and ready for queries."})
    else:
        return jsonify({"message": "No file uploaded."}), 400

@app.route('/query', methods=['POST'])
def query():
    question = request.json.get('question')
    if not question:
        return jsonify({"message": "No question provided."}), 400
    
    if not retriever:
        return jsonify({"message": "No PDF processed yet."}), 400
    
    answer = generate_answer(question)
    return jsonify({"answer": answer})

@app.route("/earnings_transcript_summary", methods=["POST"])
def earnings_transcript_summary():
    try:
        if 'transcript_file' in request.files and 'company_name' in request.form:
            file = request.files['transcript_file']
            company_name = request.form['company_name']
            if file.filename == '':
                return jsonify({"error": "No file provided"}), 400
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            pdf_text = ""
            pdf_reader = PdfReader(filepath)
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()

            if not pdf_text:
                return jsonify({"error": "Failed to extract text from PDF"}), 400

            # Create index and retriever for summarization
            create_index_and_retriever(pdf_text)

        elif request.is_json:
            data = request.get_json()
            if 'company_name' not in data or 'transcript_text' not in data:
                return jsonify({"error": "Invalid JSON input"}), 400
            company_name = data["company_name"]
            pdf_text = data["transcript_text"]

            create_index_and_retriever(pdf_text)

        # Define categories and summarize
        summary = {}
        for category in [
            "Financial Performance", 
            "Market Dynamics", 
            "Expansion Plans", 
            "Environmental Risks", 
            "Regulatory or Policy Changes"
        ]:
            answer = generate_answer(category)
            summary[category.lower().replace(" ", "_")] = answer

        summary["company_name"] = company_name
        return jsonify(summary), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)), debug=True)

