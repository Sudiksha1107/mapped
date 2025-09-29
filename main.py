from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import FileResponse
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
from fpdf import FPDF
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="🌍 YatraBot API", description="AI-powered travel planning API", version="1.0")

# ==== Text-to-Speech (Optional) ====
try:
    import pyttsx3
    engine = pyttsx3.init()

    def speak_text(text: str):
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
except ImportError:
    print("pyttsx3 not installed. Text-to-speech functionality will be unavailable.")

# ==== Data Models ====
class UserProfile(BaseModel):
    budget: str
    interests: List[str]
    duration: str
    style: str
    city: str

class ChatRequest(BaseModel):
    message: str

# ==== Load data once ====
def load_data():
    # Load from local PDF
    loader = PyPDFLoader("India Travel Guide.pdf")
    raw_docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    docs = splitter.split_documents(raw_docs)

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory="./tour_chroma_db"
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    qa_nlp = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    return qa_chain, qa_nlp, retriever, llm

# Global LangChain objects
qa_chain, qa_nlp, retriever, llm = load_data()

# ==== Prompt Templates ====
filter_prompt = PromptTemplate.from_template("""
Act as a professional tour planner. Based on the user's profile, plan the top 5 travel destinations in India or abroad. 
Include: destination name, highlights, best season, estimated budget, activities, nearby attractions, accommodation options,
and a match score (0–100) based on preferences.

USER PROFILE:
- Budget: {budget}
- Interests: {interests}
- Travel Duration: {duration}
- Travel Style: {style}
- Starting City: {city}

DESTINATION DATA:
{places}
""")

human_prompt = PromptTemplate.from_template("""
Create a warm and clear travel recommendation. For each suggested destination, include:
- Destination Name
- Why it matches the user
- Best Time to Visit
- Estimated Budget
- Top 3 Activities
- Accommodation Tip
- Match Score (0–100)

Finish with an inspiring note encouraging safe and fun travel.

DESTINATION DATA:
{filtered_places}
""")

# Chains
memory = ConversationBufferWindowMemory(k=5)
filter_chain = LLMChain(prompt=filter_prompt, llm=llm)
response_chain = LLMChain(prompt=human_prompt, llm=llm)

# ==== PDF Generation ====
def save_pdf_report(title: str, summary: str, filename="tour_plan.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"{title}\n\n{summary}")
    pdf.output(filename)
    return filename

# ==== Core Logic ====
def generate_tour_plan(user_profile: dict) -> tuple[str, Optional[str]]:
    try:
        query = f"Best destinations for budget {user_profile['budget']} with interests {user_profile['interests']}"
        retrieved_docs = retriever.get_relevant_documents(query)
        place_snippets = "\n".join([doc.page_content for doc in retrieved_docs])

        filter_input = {
            "budget": user_profile["budget"],
            "interests": ", ".join(user_profile["interests"]),
            "duration": user_profile["duration"],
            "style": user_profile["style"],
            "city": user_profile["city"],
            "places": place_snippets
        }

        filtered_places = filter_chain.run(filter_input)

        summarized = qa_nlp(
            filtered_places,
            max_length=800,
            min_length=300,
            do_sample=False
        )[0]['summary_text']

        final_summary = response_chain.run({"filtered_places": summarized})

        pdf_file = save_pdf_report("Your Tour Plan", final_summary)
        return final_summary, pdf_file
    except Exception as e:
        return f"⚠️ Error generating plan: {e}", None

# ==== FastAPI Endpoints ====
@app.post("/generate-tour")
def create_tour(user_profile: UserProfile):
    response, pdf_file = generate_tour_plan(user_profile.dict())
    if pdf_file:
        return {
            "summary": response,
            "pdf_url": f"/download/{os.path.basename(pdf_file)}"
        }
    raise HTTPException(status_code=500, detail=response)

@app.get("/download/{filename}")
def download_pdf(filename: str):
    file_path = os.path.join(".", filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=filename, media_type='application/pdf')
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/")
def root():
    return {"message": "YatraBot API is running"}

@app.post("/chat")
def chat_with_bot(req: ChatRequest):
    try:
        answer = qa_chain.run(req.message)
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
