```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import FileResponse
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from fpdf import FPDF  # fpdf2
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import logging

# === Load environment ===
load_dotenv()

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yatrabot")

# === FastAPI app ===
app = FastAPI(title="🌍 YatraBot API", version="1.0")

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
    pdf_path = os.path.join(os.getcwd(), "India Travel Guide_compressed (1)_compressed_compressed-1-1000.pdf")

    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        raw_docs = loader.load()
        logger.info(f"✅ Loaded {len(raw_docs)} pages from PDF.")

    except Exception as e:
        logger.warning(f"❌ PDF load error: {e}")
        raw_docs = [Document(page_content="No travel guide data available", metadata={})]

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
STRICTLY ENSURE that the duration of each tour plan matches the user's specified duration.

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

# === Chains ===
from langchain_core.runnables import Runnable
filter_chain: Runnable = filter_prompt | llm
response_chain: Runnable = human_prompt | llm

# === PDF Generator (fpdf2) ===
def save_pdf_report(title: str, summary: str, filename="tour_plan.pdf"):
    pdf = FPDF()
    pdf.add_page()

    # ✅ fpdf2 supports UTF-8 by default if you use a TTF font
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if not os.path.exists(font_path):
        raise FileNotFoundError(
            "DejaVuSans.ttf not found. Install fonts-dejavu in Codespaces: "
            "sudo apt-get update && sudo apt-get install -y fonts-dejavu"
        )

    pdf.add_font("DejaVu", fname=font_path)
    pdf.set_font("DejaVu", size=12)

    pdf.multi_cell(0, 10, f"{title}\n\n{summary}")
    pdf.output(filename)
    return filename

# === Core Logic ===
def generate_tour(user_profile: dict) -> tuple[str, Optional[str]]:
    try:
        query = f"Best destinations for budget {user_profile['budget']} with interests {user_profile['interests']}"
        retrieved_docs = retriever.invoke(query)
        place_snippets = "\n".join([doc.page_content for doc in retrieved_docs])

        filter_input = {
            "budget": user_profile["budget"],
            "interests": ", ".join(user_profile["interests"]),
            "duration": user_profile["duration"],
            "style": user_profile["style"],
            "city": user_profile["city"],
            "places": place_snippets
        }

        filtered_places = filter_chain.invoke(filter_input)
        filtered_places_text = filtered_places.content if hasattr(filtered_places, "content") else str(filtered_places)

        summarized = qa_nlp(
            filtered_places_text,
            max_length=800,
            min_length=300,
            do_sample=False
        )[0]['summary_text']

        final_summary = response_chain.invoke({"filtered_places": summarized})
        final_summary_text = final_summary.content if hasattr(final_summary, "content") else str(final_summary)

        pdf_file = save_pdf_report("Your Tour Plan", final_summary_text)

        return final_summary_text, pdf_file

    except Exception as e:
        logger.error(f"Error in generate_tour: {e}")
        raise

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
        return {"response": answer}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
def download_pdf(filename: str):
    file_path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="application/pdf", filename=filename)

# === Run ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
```

