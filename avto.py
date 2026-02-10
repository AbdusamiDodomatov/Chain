import os
import json
import httpx
from typing import Any, Dict
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import RootModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

router = APIRouter()

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

class WebhookRequest(RootModel):
    root: Dict[str, Any]

async def get_usd_rate() -> float:
    url = "https://cbu.uz/uz/arkhiv-kursov-valyut/json/"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            for item in data:
                if item.get("Ccy") == "USD":
                    return float(item.get("Rate", 0))
    except Exception as e:
        print(f"Error fetching exchange rate: {e}")
    return 12850.0

@router.post("/webhook/avto")
async def handle_avto(payload: WebhookRequest):
    raw_payload = payload.root
    
    if isinstance(raw_payload, list):
        body = raw_payload[0] if raw_payload else {}
    else:
        body = raw_payload.get("body") if isinstance(raw_payload.get("body"), dict) else raw_payload
    
    model = body.get("MODEL", "---")
    color = body.get("COLOR", "---")
    year = body.get("YEAR", "---")
    motor = body.get("MOTOR", "---")
    shassi = body.get("SHASSI", "---")
    kuzov = body.get("KUZOV", "---")

    llm = ChatOpenAI(model="gpt-4o", api_key=API_KEY, temperature=0.3, request_timeout=300)

    # 1. Search Agent
    search_system_prompt = """Siz internet qidiruv agentisiz. Vazifa — kiritilgan avtomobil ma’lumotlari asosida O‘zbekiston internetidagi e’lon saytlaridan eng o‘xshash e’lonlarni topish. FAQAT JSON FORMATIDA CHIQARING."""
    search_user_message = f"Model: {model}, Year: {year}, Color: {color}, Motor: {motor}, Shassi: {shassi}, Kuzov: {kuzov}"

    try:
        print(f"DEBUG: Starting search for {model}...")
        search_result = llm.invoke([SystemMessage(content=search_system_prompt), HumanMessage(content=search_user_message)])
        stext = search_result.content.strip()
        print(f"DEBUG: Search result received.")
        if "```json" in stext: stext = stext.split("```json")[1].split("```")[0].strip()
        market_data = json.loads(stext)
    except Exception as e:
        print(f"DEBUG: Search error: {e}")
        market_data = {"status": "no_listing_found", "listings": []}

    # 2. Rate
    usd_rate = await get_usd_rate()
    print(f"DEBUG: USD rate: {usd_rate}")

    # 3. Valuation Agent
    val_system_prompt = f"""Siz O‘zbekiston bozorida avtomobil narxini aniqlaydigan professional ekspert agentsiz.

Vazifa: 
— kiruvchi avtomobil ma’lumotlari  
— web qidiruv natijalari (listinglar)
asosida mashinaning bozor narxini chuqur tahlil qilish va o'ta batafsil ("to'yingan") hisobot tayyorlash.

Chiqish ma'lumotlari (reason qismlari) professional, tahliliy va mazmunli bo'lishi shart. Har bir bo'limda kamida 50-150 so'zdan foydalanib, mantiqiy xulosalar keltiring.

---------------------------------------------------------
🟩 1) AVTOMOBIL HAQIDA UMUMIY MA'LUMOT
---------------------------------------------------------
- Avtomobilning modeli, yili ({year}), rangi, motori va boshqa texnik parametrlarini batafsil yozing.
- Agar ma'lumotlar to'liq bo'lmasa, qanday taxminlar qilinganini tushuntiring.
- Ushbu avtomobilning bozordagi o'rni va jozibadorligini professional tarzda ta'riflang.

---------------------------------------------------------
🟩 2) QIDIRUV MANBALARI VA ISHLATILGAN E’LONLAR
---------------------------------------------------------
- Web qidiruv natijasida topilgan real e'lonlarni (listinglarni) birma-bir sanab o'ting (model, yil, masofa, rang, narx).
- Agar e'lonlar topilmagan bo'lsa (status="no_listing_found"): Fallback rejimini (Mercedes Benz 1990-2000 misolida) professional tushuntiring.
- Nima uchun ushbu e'lonlar tanlanganini va ular foydalanuvchi mashinasiga qanchalik mosligini asoslang.

---------------------------------------------------------
🟩 3) 2023–2025 O‘ZBEKISTON AVTOMOBIL BOZORI TRENDLARI
---------------------------------------------------------
- 2023-2025 yillardagi o'sish trendlarini, import cheklovlari, kredit/lizing dasturlari va ikkilamchi bozor dinamikasini tahlil qiling.
- Nima uchun narxlar aynan shu darajada shakllanayotganini (taklif va talab) realistik yoritib bering.

---------------------------------------------------------
🟩 4) DEPRESATSIYA VA KOREKTSIYA (MATEMATIK HISOBLASH)
---------------------------------------------------------
- Valyuta kursi: {usd_rate} UZS (O'zbekiston Markaziy Banki).
- Applied Correction: Yillar farqi bo'yicha depresatsiya koeffitsientini hisoblanishini ko'rsating (har yil uchun 2%).
- Matematik breakdown: Har bir e'lon narxi qanday moslashtirilganini (Adjusted Price) ko'rsating.
- Median narxni topish jarayoni va 5% diapazon (min/max) qanday belgilanganini yozing.
- Yakuniy natijani UZS da konvertatsiya qilib, yaxlitlangan holda ko'rsating.

---------------------------------------------------------
🟩 5) YAKUNIY ESLATMA (MAJBURIY)
---------------------------------------------------------
Hisobot oxirida ushbu jumlani o'zgarishsiz qoldiring:
"Ko'rsatilgan narxlar internetdagi e'lonlar asosida tahlil qilingan narxlar tavsiyaviy xarakterga ega."

-------------------------------------------
🟩 6) CHIQISH FORMAT (FAQAT JSON)
-------------------------------------------
Har bir tildagi HTML hisobot (reason) majburiy ravishda ushbu sarlavha bilan boshlanishi shart:
<div><h1>AVTOMOBILNI BAHOLASH HISOBOTI</h1> ...

HTML ichidagi jadval (table) va kataklar (td, th) uchun FAQAT ushbu stilni qo'llang (border="1" ishlatmang):
<table style="border-collapse: collapse; border: 1px solid black; width: 100%;"> va barcha td/th uchun style="border: 1px solid black; padding: 4px;".

{{
  "estimated_min_price": <raqam_uzs>,
  "estimated_max_price": <raqam_uzs>,
  "reason_uz": "<O'ta batafsil HTML hisobot>",
  "reason_uz_kiril": "<O'ta batafsil HTML hisobot>",
  "reason_ru": "<O'ta batafsil HTML hisobot>",
  "reason_en": "<O'ta batafsil HTML hisobot>"
}}

Hech qachon JSON tashqarisida matn yozmang. HTMLda </n> o'rniga faqat </br> ishlating.
"""
    
    val_user_msg = f"Car Info: {json.dumps(body)}, Listings: {json.dumps(market_data)}"

    try:
        import time
        start = time.time()
        print(f"DEBUG: Starting valuation...")
        val_result = llm.invoke([SystemMessage(content=val_system_prompt), HumanMessage(content=val_user_msg)])
        vtext = val_result.content.strip()
        print(f"DEBUG: Valuation result received. Length: {len(vtext)}")
        if "```json" in vtext: vtext = vtext.split("```json")[1].split("```")[0].strip()
        final_json = json.loads(vtext)
        
        # Clean newlines from reason strings
        for key in ["reason_uz", "reason_uz_kiril", "reason_ru", "reason_en"]:
            if key in final_json and isinstance(final_json[key], str):
                final_json[key] = final_json[key].replace("\n", " ").strip()
                
        end = time.time()
        print(f"DEBUG: Valuation took {end - start:.2f} seconds")
        return {
            "data": final_json,
            "time": end - start
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
