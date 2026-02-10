import os
import json
import io
from typing import Any, Dict, List
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import pandas as pd

router = APIRouter()

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

@router.post("/webhook/forma1")
async def handle_forma1(file: UploadFile = File(...)):
    """Migrated from forma1.json: Excel/CSV parsing via AI Agent."""
    
    import time
    start_time = time.time()
    # 1. Read file content based on type
    content = await file.read()
    filename = file.filename.lower()
    
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith((".xls", ".xlsx")):
            # Note: pandas uses openpyxl for xlsx and xlrd for xls
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV, XLS, or XLSX.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading Excel/CSV: {str(e)}")

    # 2. Convert to generic text representation
    lines = []
    for i, row in df.iterrows():
        # filter out empty values
        values = [str(v).strip() for v in row if pd.notna(v) and str(v).strip() not in ("", "-", "null", "None")]
        if values:
            lines.append(" | ".join(values))
    
    cleaned_data = "\n".join(lines)

    # 3. AI Agent Parsing (GPT-4o)
    llm = ChatOpenAI(model="gpt-4o", api_key=API_KEY, temperature=0)
    
    system_prompt = """Siz buxgalteriya hisobotlarini (Forma №1) tahlil qilish bo'yicha professional agentsiz.
Sizga Excel faylidan olingan matn beriladi. Matndagi har bir qator " | " belgisi bilan ajratilgan ustunlardan iborat.

VAZIFA:
Berilgan matndan quyidagi ma'lumotlarni structured JSON formatida chiqarib berishingiz kerak.

DIQQAT (MUHIM QOIDALAR):
1. "row_no" — bu jadval ichidagi satr kodi (masalan: 010, 011, 130, 400). Matndagi har bir qatorning boshida yoki ikkinchi ustunida bu kod bo'ladi. Satr kodini TO'G'RI aniqlang.
2. "sum_begin_period" va "sum_end_period" — bu tegishli satr kodidagi davr boshi va oxiri summalari. Agar qiymat bo'lmasa yoki nol bo'lsa, null (yoki "0") qaytaring.
3. Inflyatsiya yoki valyuta farqini EMAS, aynan hujjatda yozilgan raqamlarni oling.
4. "date" - hisobot sanasini hujjatdan toping (masalan: "31 dekabr 2024 y.").
5. CHIQUVCHI FORMAT FAQAT JSON:

{
    "date": "<Hujjat sanasi>",
    "company": "<Kompaniya nomi>",
    "network": "<Tarmoq nomi>",
    "inn": "<STIR/INN>",
    "address": "<Manzil>",
    "rows": [
        {
            "row_no": "<Satr kodi, masalan: 010>",
            "sum_begin_period": "<Qiymat>",
            "sum_end_period": "<Qiymat>"
        }
    ],
    "rows_out_of_balance": [
        {
            "row_no": "<Balansdan tashqari satr kodi>",
            "sum_begin_period": "<Qiymat>",
            "sum_end_period": "<Qiymat>"
        }
    ]
}

- "rows" ichiga asosiy balans (Aktiv va Passiv) jadvallarini kiriting.
- "rows_out_of_balance" ichiga "BALANSDAN TASHQARI SCHYOTLARDA..." deb nomlangan bo'limdagi ma'lumotlarni kiriting.
"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=cleaned_data)
        ])
        
        # Extract JSON from MD wrap if present
        text = response.content.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        
        end_time = time.time()
        return {
            "data": json.loads(text),
            "time": end_time - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Parsing failed: {str(e)}")
