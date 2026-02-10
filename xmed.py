import os
import json
import httpx
import re
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

router = APIRouter()

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

# --- Models ---
class XMedRequest(BaseModel):
    session_id: str
    message: str

# --- Tools ---
@tool
async def search_doctor(page: str = "1", pageSize: str = "3", name: str = "", speciality: List[str] = None):
    """
    Search for suitable doctors based on name or speciality.
    Always call this tool before recommending a specific doctor.
    Args:
        page: Page number (default "1")
        pageSize: Number of results (default "3")
        name: Full or partial name of the doctor
        speciality: List of specialities (e.g. ["dentist", "surgeon"])
    """
    url = "https://api.plusmed.uz/be/common/searchDoctor"
    headers = {
        "language": "uz",
        "Content-Type": "application/json"
    }
    
    import time
    t_start = time.time()
    
    payload = {
        "page": page,
        "pageSize": pageSize,
        "name": name,
        "speciality": speciality or []
    }
    
    print(f"DEBUG [xmed]: Calling search_doctor with speciality={speciality}, name='{name}'")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            t_end = time.time()
            print(f"DEBUG [xmed]: search_doctor took {t_end - t_start:.2f}s and returned {len(data.get('results', []))} results")
            return data
    except Exception as e:
        print(f"DEBUG [xmed]: search_doctor FAILED after {time.time() - t_start:.2f}s: {e}")
        return {"error": str(e), "results": []}

# --- Agent System Prompt ---
SYSTEM_PROMPT = """Siz professional AI tibbiy yordamchisiz.

Sizning vazifangiz: Foydalanuvchi shikoyatlari asosida "search_doctor" tooli orqali eng mos shifokorni topilgan natijalar asosida tavsiya qilish.

Qat'iy qoida va muhim qoida:
• HECH QACHON, HECH BIR SO'Z HAM SO'RALGAN TILDAN BOSHQA TILDA CHIQMASIN! HAR BIR SO'Z VA HAR BIR GAP BELGILANGAN TILGA TARJIMA QILINSIN!
• FAQAT "search_doctor" tooldan kelgan natija asosida javob bering. ID ni to'qib chiqarmang!

Xulosa shakllantirish tartibi:
✓ Shikoyatlar tahlili (qisqa)
✓ Kasallikning ehtimoliy sababi (londa)
✓ Shifokor tavsiyasi (Ism-sharifi, reytingi, ish tajribasi)
Yuqoridagi natijalar hammasi qisqa londa bo'lsin. Hammasi bitta paragraphda yozilsin.

Quyidagi mutaxassisliklar (speciality) ro'yxatidan foydalaning:
akusher-ginekolog, allergist, angiologist, andrologist, anesteziolog-reanimatolog, aritmolog, aphasiologist, bariatricheskiy-xirurg, valeolog, vertebrologist, vet, virologist, obstetrician, vrach-dietolog, laboratory-doctor, vrach-lfk, narodnaya-medicina, general-doctor, vrach-transfuziolog, vrach-ehndoskopist, gastroenterologist, gelmintolog, hematologist, geneticist, hepatologist, gynecologist, ginekolog-ehndokrinolog, girudoterapevt, dermatovenereologist, dermato-onkolog, childrens_gynecologist, childrens-infectious-disease, childrens-neurologist, pediatric-neurosurgeon, childrens-oncologist, childrens-psychologist, childrens-resuscitator, pediatric-urologist, childrens-endocrinologist, defectologist, nutritionist, acupuncturist, immunologist, intervencionnyj-kardiolog, infectionist, cardiologist, cardioresuscitator, cardio_surgeon, kinezioterapevt, kovidolog, coloproctologist, kombustiolog, cosmetologist, massage-therapist, speech-therapist, ent, mammologist, manualnyj-terapevt, masseur, medicinskij-kosmetolog, medicinskij-psiholog, nurse, mikolog, narcologist, neurologist, nevrolog-refleksoterapevt, nejro-onkolog, neurosurgeon, neyroendokrinolog, neonatologist, nephrologist, oligofrenopedagog, onkogematolog, oncogynecologist, onkokoloproktolog, oncologist, oncologist-mammologist, onkolog-himioterapevt, onkourolog, ortodont, ortoped-vertebrolog, childrens-orthopedist, orthopedist-traumatologist, osteopat, otonevrolog, ophthalmologist, oftalmohirurg, parasitologist, pediatrician, plastic_surgeon, podolog, podrostkovyj-psiholog, proctologist, profpatolog, psixiatr, psychologist, psychotherapist, pulmonologist, pulmonolog-astmolog, radiologist, rehabilitologist, resuscitator, rheumatologist, rentgenologist, reproductologist, sexologist, cardiovascular-surgeon, screening, somnolog, dentist-surgeon, dentist, audiologist, therapist, toxicologist, thoracic-oncologist, torakalhiy-xirurg, traumatologist, trichologist, urologist, urolog-androlog, pharmacologist, physiotherapist, phlebologist, foniatr, phthisiatrician, ftiziatr-pulmonolog, functional-diagnostic, surgeon, childrens-surgeon, circumcisiologist, maxillofacial-surgeon, ehmbriolog, ehndovaskulyarnyj-hirurg, endocrinologist, epidemiologist, ehpileptolog.

XULOSA FAQAT JSON FORMATIDA BO'LISHI SHART:
{
"answer": "<tahlil va doktor tavsiyasi: ismi, reytingi, tajribasi>",
"doctor_id": <id_yoki_null>,
"answer_2": "Hisobingizni qanday to'ldirishni bilasizmi?",
"answer_3": "#fill",
"answer_4": "Hisobingizni to'ldirib ushbu shifokor bilan maslahatlashing."
}

To'lov/hisob savollari uchun: {"answer": "#fill", "doctor_id": null, "answer_2": null, "answer_3": null, "answer_4": null}.
Mavzuda bo'lmagan: "Men faqat mavzu doirasida javob bera olaman".
"""

# --- Agent Initialization ---
llm = ChatOpenAI(model="gpt-4o", api_key=API_KEY, temperature=0)
checkpointer = MemorySaver()

# Simplified agent without modifiers for better compatibility
agent_executor = create_react_agent(
    model=llm, 
    tools=[search_doctor], 
    checkpointer=checkpointer
)

# --- Helpers ---
def extract_json(text: str) -> Dict[str, Any]:
    """Helper to extract JSON from AI response, handles Markdown blocks."""
    text = text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    try:
        return json.loads(text)
    except:
        return {"answer": text, "doctor_id": None, "answer_2": None, "answer_3": None, "answer_4": None}

@router.post("/webhook/xmed")
async def handle_xmed(payload: XMedRequest):
    """XMed Integrated Portal: Medical Assistant Agent."""
    import time
    start_time = time.time()
    config = {"configurable": {"thread_id": payload.session_id}}
    
    try:
        # Prepend SYSTEM_PROMPT to every request to ensure instructions are respected
        input_data = {"messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=payload.message)
        ]}
        
        print(f"DEBUG [xmed]: Starting ReAct agent for session {payload.session_id}")
        t0 = time.time()
        result = await agent_executor.ainvoke(input_data, config=config)
        t1 = time.time()
        print(f"DEBUG [xmed]: ReAct agent finished in {t1 - t0:.2f}s")
        
        last_message = result["messages"][-1].content
        parsed_output = extract_json(last_message)
        
        response_json = {
            "answer": parsed_output.get("answer"),
            "doctor_id": parsed_output.get("doctor_id"),
            "answer_2": parsed_output.get("answer_2"),
            "answer_3": parsed_output.get("answer_3"),
            "answer_4": parsed_output.get("answer_4")
        }
        
        # Clean newlines from all answer strings
        for k in response_json:
            if isinstance(response_json[k], str):
                response_json[k] = response_json[k].replace("\n", " ").strip()
                
        end_time = time.time()
        return {
            "data": response_json,
            "time": end_time - start_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Medical Agent failed: {str(e)}")
