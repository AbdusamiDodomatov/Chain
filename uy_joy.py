import os
import json
import httpx
from typing import Any, Dict, List, Optional
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

# --- Models ---
class WebhookRequest(RootModel):
    root: Dict[str, Any]

# --- Helpers ---
async def get_usd_rate() -> float:
    """Fetches the current USD to UZS exchange rate from CBU API."""
    url = "https://cbu.uz/uz/arkhiv-kursov-valyut/json/"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            # Find USD
            for item in data:
                if item.get("Ccy") == "USD":
                    return float(item.get("Rate", 0))
    except Exception as e:
        print(f"Error fetching exchange rate: {e}")
    return 12850.0  # Fallback rate

# --- Core Logic ---
@router.post("/webhook/uy-joy")
async def handle_uy_joy(payload: WebhookRequest):
    raw_payload = payload.root
    
    if isinstance(raw_payload, list):
        body = raw_payload[0] if raw_payload else {}
    else:
        body = raw_payload.get("body") if isinstance(raw_payload.get("body"), dict) else raw_payload
    address = body.get("address", "---")
    area_info = body.get("area", {})
    actual_land_area = area_info.get("actualLandArea", 0)
    building_type = body.get("type")  # e.g., "FIRST_LINE"
    latitude = body.get("latitude")
    longitude = body.get("longitude")

    # 1. Search Agent - Simulate the market research
    llm_search = ChatOpenAI(model="gpt-5.1", api_key=API_KEY, temperature=0.3)
    
    search_system_prompt = """Siz internet qidiruv agentisiz. Vazifa — O‘zbekiston bozorida berilgan mulkka o‘xshash e’lonlarni topish.
Foydalaniladigan platformalar: olx.uz, domtut.uz, realt24.uz, uybor.uz, etc.

FAQAT JSON FORMATIDA CHIQARING.
{
 "status": "ok",
 "listings": [
   { "title": "...", "price_usd": ..., "area_m2": ..., "link": "..." }
 ]
}"""
    
    search_user_message = f"""Adress: {address}
Actual Land Area: {actual_land_area}
Type: {building_type}
Coords: {latitude}, {longitude}

Toshkent bo'yicha kamida 5 ta o'xshash elonlarni toping."""

    try:
        search_result = llm_search.invoke([
            SystemMessage(content=search_system_prompt),
            HumanMessage(content=search_user_message)
        ])
        # Try to parse JSON from search result
        search_text = search_result.content.strip()
        if "```json" in search_text:
            search_text = search_text.split("```json")[1].split("```")[0].strip()
        market_data = json.loads(search_text)
    except Exception as e:
        print(f"Search agent parsing error: {e}")
        market_data = {"status": "no_listing_found", "listings": []}

    # 2. Get Exchange Rate
    usd_rate = await get_usd_rate()

    # 3. Valuation Agent
    llm_val = ChatOpenAI(model="gpt-4o", api_key=API_KEY, temperature=0.3)
    
    valuation_system_prompt = f"""Siz O‘zbekiston bozorida ko‘chmas mulk narxini aniqlaydigan professional kredit baholash agentisiz.
Vazifa: kiruvchi ma'lumotlar va web qidiruv natijalari asosida obyektning bozor narxini chuqur tahlil qilish va o'ta batafsil ("to'yingan") hisobot tayyorlash.

Chiqish ma'lumotlari (reason qismi) professional, tahliliy va mazmunli bo'lishi shart. Har bir bo'limda kamida 50-150 so'zdan foydalanib, mantiqiy xulosalar keltiring.

-------------------------------------------
🟩 1) OBYEKT HAQIDA UMUMIY MA'LUMOT
-------------------------------------------
- Obyekt manzili, yer maydoni ({actual_land_area} m²) va foydali maydonlarini batafsil yozing.
- Joylashuv turini aniqlang (Agar type="FIRST_LINE" bo'lsa "Birinchi qatorda", aks holda "Birinchi qatorda joylashmagan").
- Joylashuv turi (ko'chadan ko'rinishi, mijoz oqimi, tijoriy jozibadorlik) obyekt narxiga va jozibadorligiga qanday ta'sir qilishini o'ta batafsil tushuntiring.

-------------------------------------------
🟩 2) FOYDALANILGAN MANBALAR VA WEB QIDIRUV
-------------------------------------------
- Agar e'lonlar topilgan bo'lsa (listings mavjud bo'lsa): Qaysi platformalardan (OLX, Uybor va h.k.) nechta e'lon topilganini va ularning m2 narxi qanday tahlil qilinganini yozing.
- Agar e'lonlar topilmagan bo'lsa (status="no_listing_found"): Bu holatni professional tushuntiring (noyob joylashuv yoki parametrlar). Fallback (zaxira) usuli qo'llanilgani, O'zbekiston bo'yicha median narxlar va bozor tahlillariga tayanganingizni batafsil bayon qiling.

-------------------------------------------
🟩 3) BOZOR NARXLARI DINAMIKASI (2023–2025)
-------------------------------------------
- 2023, 2024 va 2025 yillar uchun bozor trendlarini alohida tahlil qiling.
- Inflyatsiya, qurilish materiallari narxi va talab oshishi natijasida yuzaga kelgan o'zgarishlarni (foizlarda yoki USD diapazonida) realistik misollar bilan yoritib bering.
- Hududiy (masalan, Navoiy shahri sanoat markazi ekanligi) xususiyatlarni inobatga oling.

-------------------------------------------
🟩 4) KOREKTSIYA VA KURS (MATEMATIK HISOBLASH)
-------------------------------------------
- Valyuta kursi: {usd_rate} UZS (O'zbekiston Markaziy Banki).
- Applied Correction: Joylashuv turi uchun qo'llangan korektsiyani (-10% dan -20% gacha yoki +10%) tahliliy asoslang.
- Matematik breakdown: 1 m² uchun minimal va maksimal USD narxlarni aniqlang, so'ngra umumiy maydon ({actual_land_area} m²) ga ko'paytirib, yakuniy natijani UZS da ko'rsating.
- Formulani aniq matn ko'rinishida yozing (masalan: Area * Price * Correction = Total).

-------------------------------------------
🟩 5) YAKUNIY ESLATMA (MAJBURIY)
-------------------------------------------
Hisobot oxirida ushbu jumlani o'zgarishsiz qoldiring:
"Ko'rsatilgan narxlar internet tarmog'ida joylashtirilgan e'lonlar asosida tahlil qilingan narxlar tavsiyaviy xarakterga ega."

-------------------------------------------
🟩 6) CHIQISH FORMAT (FAQAT JSON)
-------------------------------------------
Har bir tildagi HTML hisobot (reason) majburiy ravishda ushbu sarlavha bilan boshlanishi shart:
<div><h1>KO'CHMAS MULKNI BAHOLASH BO'YICHA HISOBOT</h1> ...

HTML ichidagi jadval (table) va kataklar (td, th) uchun FAQAT ushbu stilni qo'llang (border="1" ishlatmang):
<table style="border-collapse: collapse; border: 1px solid black; width: 100%;"> va barcha td/th uchun style="border: 1px solid black; padding: 4px;".

{{
 "estimated_min_price": <raqam_uzs>,
 "estimated_max_price": <raqam_uzs>,
 "reason": {{
    "uz": "<O'ta batafsil HTML hisobot>",
    "uz_cyrl": "<O'ta batafsil HTML hisobot>",
    "ru": "<O'ta batafsil HTML hisobot>",
    "en": "<O'ta batafsil HTML hisobot>"
 }}
}}

Hech qachon JSON tashqarisida matn qaytarmang. Har doim professional va "to'yingan" tahlilni saqlab qoling. HTMLda </n> o'rniga </br> ishlating.
"""

    valuation_user_message = f"""
"user_natija": {{
    "address": "{address}",
    "actualLandArea": {actual_land_area},
    "type": "{building_type}",
    "totalArea": {area_info.get("totalArea", 0)},
    "usefulArea": {area_info.get("usefulArea", 0)}
}},
"web_natija": {json.dumps(market_data, ensure_ascii=False)}
"""

    try:
        import time
        start = time.time()
        val_result = llm_val.invoke([
            SystemMessage(content=valuation_system_prompt),
            HumanMessage(content=valuation_user_message)
        ])
        
        val_text = val_result.content.strip()
        if "```json" in val_text:
            val_text = val_text.split("```json")[1].split("```")[0].strip()
            
        final_json = json.loads(val_text)
        
        # Clean newlines from reason strings
        if "reason" in final_json:
            for lang in final_json["reason"]:
                if isinstance(final_json["reason"][lang], str):
                    final_json["reason"][lang] = final_json["reason"][lang].replace("\n", " ").strip()
                    
        end = time.time()
        print(f"Valuation took {end - start:.2f} seconds")
        return {
            "data": final_json,
            "time": end - start
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Valuation failed: {str(e)}")
