from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import FileResponse
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from fpdf import FPDF
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import logging
import pyttsx3

# === Load environment ===
load_dotenv()

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yatrabot")

# === FastAPI app ===
app = FastAPI(title="🌍 YatraBot API", version="1.0")

# === Text-to-Speech ===
def init_tts():
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        if voices:
            engine.setProperty("voice", voices[0].id)
        return engine
    except Exception as e:
        logger.warning(f"TTS init failed: {e}")
        return None

engine = init_tts()

def speak_text(text: str):
    if engine:
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logger.warning(f"TTS error: {e}")

# === Models ===
class UserProfile(BaseModel):
    budget: str
    interests: List[str]
    duration: str
    style: str
    city: str

class ChatRequest(BaseModel):
    message: str

# === Load PDF and Vectorstore ===
def load_data():
    raw_docs = []

    try:
        if not os.path.exists("India Travel Guide.pdf"):
            raise FileNotFoundError("PDF not found.")

        loader = PyPDFLoader("India Travel Guide.pdf")
        raw_docs = loader.load()
        logger.info(f"✅ Loaded {len(raw_docs)} pages from PDF.")

    except Exception as e:
        logger.warning(f"❌ PDF load error: {e}")
        raw_docs = []

    if not raw_docs:
        logger.warning("⚠️ No documents loaded from PDF. Using fallback.")
        raw_docs = [Document(page_content="No data available", metadata={})]

    # Split documents
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    docs = splitter.split_documents(raw_docs)

    # Embeddings & Vector Store
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

# === Load AI components ===
qa_chain, qa_nlp, retriever, llm = load_data()

# === Prompt Templates ===
filter_prompt = PromptTemplate.from_template("""
Act as a professional tour planner. Based on the user's profile, plan the top 5 travel destinations in India or abroad.
STRICTLY ENSURE that the duration of each tour plan matches the user's specified duration. Do not suggest plans that exceed or fall short of it.

For each suggestion, include:
- Destination Name
- Highlights
- Best Season
- Estimated Budget
- Activities
- Nearby Attractions
- Accommodation Options
- Duration (must exactly match: {duration})
- Match Score (0–100) based on user preferences
Use bullet points for each destination.
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

# === Chains and Memory ===
from langchain_core.runnables import Runnable

filter_chain: Runnable = filter_prompt | llm
response_chain: Runnable = human_prompt | llm

# === PDF Generator ===
def save_pdf_report(title: str, summary: str, filename="tour_plan.pdf"):
    pdf = FPDF()
    pdf.add_page()

    # ✅ Add Unicode font (DejaVu Sans supports ₹)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if not os.path.exists(font_path):
        # Agar Codespaces/Docker pe ho aur font missing hai to install karna padega:
        # sudo apt-get update && sudo apt-get install -y fonts-dejavu
        raise FileNotFoundError("DejaVuSans.ttf not found. Install fonts-dejavu.")

    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", size=12)

    pdf.multi_cell(0, 10, f"{title}\n\n{summary}")
    pdf.output(filename)
    return filename


# === Core Logic ===
def generate_tour(user_profile: dict) -> tuple[str, Optional[str]]:
    try:
        query = f"Best destinations for budget {user_profile['budget']} with interests {user_profile['interests']}"
        retrieved_docs = retriever.invoke(query)  # ✅ Updated for LangChain >= 0.1.46
        place_snippets = "\n".join([doc.page_content for doc in retrieved_docs])

        filter_input = {
            "budget": user_profile["budget"],
            "interests": ", ".join(user_profile["interests"]),
            "duration": user_profile["duration"],
            "style": user_profile["style"],
            "city": user_profile["city"],
            "places": place_snippets
        }

        # Run through filter chain (LLM) and get response
        filtered_places = filter_chain.invoke(filter_input)

        # ✅ Ensure output is plain string
        if hasattr(filtered_places, "content"):
            filtered_places_text = filtered_places.content
        else:
            filtered_places_text = str(filtered_places)

        # ✅ Summarize using Hugging Face pipeline
        summarized = qa_nlp(
            filtered_places_text,
            max_length=800,
            min_length=300,
            do_sample=False
        )[0]['summary_text']

        # Final human-friendly output
        final_summary = response_chain.invoke({"filtered_places": summarized})
        final_summary_text = final_summary.content if hasattr(final_summary, "content") else str(final_summary)

        # Save as PDF
        pdf_file = save_pdf_report("Your Tour Plan", final_summary_text)

        return final_summary_text, pdf_file

    except Exception as e:
        logger.error(f"Error in generate_tour_plan: {e}")
        raise  # Let FastAPI return 500


# === FastAPI Routes ===

@app.post("/generate-tour")
def create_tour(user_profile: UserProfile):
    response, pdf_file = generate_tour(user_profile.dict())
    if pdf_file:
        return {
            "summary": response,
            "pdf_url": f"/download/{os.path.basename(pdf_file)}"
        }
    raise HTTPException(status_code=500, detail=response)


@app.post("/chat")
def chat_with_bot(req: ChatRequest):
    try:
        answer = qa_chain.run(req.message)
        # speak_text(answer)  # Optional: enable if TTS working
        return {"response": answer}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
