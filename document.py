import os
import json
import re
import math
from typing import Any, Dict, List
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import RootModel
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

router = APIRouter()

# ---------------- helpers (from code.py) ----------------

def is_nil(v: Any) -> bool:
    return v is None

def clean(v: Any) -> Any:
    return v.strip() if isinstance(v, str) else v

def val(v: Any) -> str:
    v = clean(v)
    if is_nil(v) or v == "" or v == "null" or v == "undefined":
        return "---"
    return str(v)

def num_format(x: Any) -> str:
    if is_nil(x):
        return "---"

    if isinstance(x, str):
        s = re.sub(r"\s+", "", x).replace(",", ".")
        if s in ("", "-", "null", "undefined"):
            return "---"
        try:
            n = float(s)
        except ValueError:
            return val(x)
        if not math.isfinite(n):
            return val(x)
        x = n

    if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return val(x)

    fixed = str(round(float(x) + 1e-12, 2))
    if "." in fixed:
        int_part, dec_part = fixed.split(".", 1)
    else:
        int_part, dec_part = fixed, ""

    grouped = re.sub(r"(?<=\d)(?=(\d{3})+(?!\d))", " ", int_part)
    dec_part = (dec_part + "00")[:2]
    return f"{grouped}.{dec_part}"

def safe_arr(a: Any) -> List[Any]:
    return a if isinstance(a, list) else []

def esc(s: Any) -> str:
    t = str(s)
    return (
        t.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&#39;")
    )

def join_address(region: Any, district: Any, street: Any) -> str:
    r = val(region)
    d = val(district)
    s = val(street)
    return ", ".join([r, d, s])

def bank_acc_format(acc: Any) -> str:
    raw = val(acc)
    if raw == "---":
        return "---"
    a = re.sub(r"\s+", "", raw)
    m = re.match(r"^(.{5})(.{3})(.{1})(.{8})(.*)$", a)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)} {m.group(5)}"
    return raw

def full_name(p: Any) -> str:
    if not isinstance(p, dict):
        return "---"
    parts = [p.get("lastName"), p.get("firstName"), p.get("middleName")]
    parts = [val(x) for x in parts if x]
    parts = [x for x in parts if x != "---"]
    return " ".join(parts) if parts else "---"

def find_row(rows: Any, row_no: str, field: str) -> str:
    rows_list = safe_arr(rows)
    r = next((x for x in rows_list if str((x or {}).get("row_no", "")) == str(row_no)), None)
    if not r:
        return "---"
    v = r.get(field)
    if is_nil(v):
        return "---"
    return num_format(v)

def year_from_any_date(v: Any) -> str:
    s = val(v)
    if s == "---":
        return "---"
    m = re.search(r"(19|20)\d{2}", str(s))
    return m.group(0) if m else "---"

def normalize_br(s: Any) -> str:
    return (
        str(s)
        .replace("</br>", "<br>")
        .replace("<br/>", "<br>")
        .replace("<br />", "<br>")
    )


def get_in(d: Any, path: List[Any], default=None):
    cur = d
    for key in path:
        if isinstance(key, int):
            if not isinstance(cur, list) or key < 0 or key >= len(cur):
                return default
            cur = cur[key]
        else:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
    return cur

# ---------------- dictionaries (from code.py) ----------------

DICT = {
    "uz": {
        "expertConclusionTitle": "Yakuniy kredit bo‘yicha ekspert xulosasi",
        "hProject": "Loyiha bo‘yicha ma’lumot",
        "hCompany": "Kompaniya haqida",
        "orgFull": "Tashkilotning to‘liq nomi",
        "inn": "INN",
        "oked": "Sohasi (OKED)",
        "cat": "Kompaniya kategoriyasi",
        "emp": "O'rta ishchilar soni",
        "addr": "Yuridik manzil",
        "fund": "Ustav fondi",
        "foundersTitle": "Ta'sischilar bo‘yicha ma’lumot",
        "fio": "F.I.O. / Nomi",
        "who": "Kim hisoblanadi",
        "share": "Ulushi (%)",
        "other": "Boshqa subyektlardagi ishtiroki",
        "director": "Direktor",
        "ind": "Jismoniy shaxs",
        "leg": "Yuridik shaxs",
        "bankReq": "Bank rekvizitlari:",
        "mainAcc": "Asosiy hisob:",
        "bankName": "Bank nomi",
        "acc": "Hisob raqami",
        "mfo": "MFO",
        "appShort": "Loyiha haqida qisqacha ma'lumot",
        "appDesc": "Loyiha tavsifi",
        "started": "-yilda faoliyatini boshlagan.",
        "mainAct": "Kompaniyaning asosiy faoliyati",
        "applied2": "bankka kompaniyaning",
        "applied3": "maqsadida asosiy vositalar va aylanma mablag'larni sotib olish uchun umumiy kredit shartnomasini tuzishni ko'rib chiqish taklifi bilan murojaat qildi, maksimal kredit limiti",
        "term": "oy muddatga mo'ljallangan.",
        "rates": "Kreditlar bo'yicha foiz stavkalari AQSh dollarida yillik 15%, yevroda yillik 15% va milliy valyutada yillik",
        "sourcePay": "Majburiyatlarni to'lash manbai kompaniyaning davom etayotgan faoliyatidan kelib chiqadi.",
        "bankFill": "Bank tomonidan to'ldiriladi",
        "mainObj": "Korxona nomidagi asosiy obyekt (ofis, bino, ombor, turar joy) quyidagi manzilda joylashgan:",
        "totalArea": "umumiy maydoni",
        "buildArea": "bino va inshootlar maydoni",
        "cad": "kadastr raqami",
        "belongs": "Ushbu yer",
        "belongs2": "ga tegishli.",
        "infra": "Hudud zarur kommunal xizmatlar va kommunikatsiyalar bilan jihozlangan.",
        "built": "Ushbu hududda kompaniya",
        "built2": "kv.m. maydonga ega yangi ma'muriy bino qurdi.",
        "collateralTitle": "Garov haqida ma’lumot",
        "reTitle": "Garovga qo'yilgan bino ma'lumotlari:",
        "carTitle": "Garovga qo'yilgan mashina ma'lumotlari:",
        "balanceTitle": "Balans bo'yicha hisobot:",
        "metric": "Ko'rsatkich nomi",
        "row": "Qator",
        "finTitle": "Moliyaviy ko'rsatkichlar ma'lumotlari",
        "shortAbout": "Korxona haqida qisqacha ma’lumot:",
        "objs": "Korxona nomidagi obyektlar:",
        "cars": "Korxona nomidagi mashinalar:",
        "hTin": "TIN",
        "hName": "Kompaniya nomi",
        "hType": "Turi",
        "hCad": "Kadastr raqami",
        "hObjName": "Bino nomi",
        "hAddr": "Manzili",
        "hShare": "Egalik qiluvchi ulushi (%)",
        "hInvCost": "Inventar narx",
        "hTotalArea": "Umumiy maydon",
        "hBuildArea": "Bino maydoni",
        "hExtraArea": "Qo'shimcha maydon",
        "hEstValue": "Garovning taxminiy qiymati",
        "hModel": "Modeli",
        "hColor": "Rangi",
        "hYear": "Yil",
        "hKuzov": "Kuzov raqami",
        "hMotor": "Motor",
        "hShassi": "Shassi",
        "hGosNumber": "Davlat raqami",
        "hRegDate": "Ro'yxatdan o'tgan sana",
        "hDivision": "Diviziya",
        "hOwner": "Egalik qiluvchi",
        "unitArea": "kv.m.",
    },
    "cyrl": {
        "expertConclusionTitle": "Якуний кредит бўйича эксперт хулосаси",
        "hProject": "Лойиҳа бўйича маълумот",
        "hCompany": "Компания ҳақида",
        "orgFull": "Ташкилотнинг тўлиқ номи",
        "inn": "ИНН",
        "oked": "Соҳаси (ОКЭД)",
        "cat": "Компания категорияси",
        "emp": "Ўрта ишчилар сони",
        "addr": "Юридик манзил",
        "fund": "Устав фонди",
        "foundersTitle": "Таъсисчилар бўйича маълумот",
        "fio": "Ф.И.О. / Номи",
        "who": "Ким ҳисобланади",
        "share": "Улуши (%)",
        "other": "Бошқа субъектлардаги иштироки",
        "director": "Директор",
        "ind": "Жисмоний шахс",
        "leg": "Юридик шахс",
        "bankReq": "Банк реквизитлари:",
        "mainAcc": "Асосий ҳисоб:",
        "bankName": "Банк номи",
        "acc": "Ҳисоб рақами",
        "mfo": "МФО",
        "appShort": "Лойиҳа ҳақида қисқача маълумот",
        "appDesc": "Лойиҳа тавсифи",
        "started": "-йилда фаолиятини бошлаган.",
        "mainAct": "Компаниянинг асосий фаолияти",
        "applied2": "банкка компаниянинг",
        "applied3": "мақсадида асосий воситалар ва айланма маблағларни сотиб олиш учун умумий кредит шартномасини тузишни кўриб чиқиш таклифи билан мурожаат қилди, максимал кредит лимити",
        "term": "ой муддатга мўлжалланган.",
        "rates": "Кредитлар бўйича фоиз ставкалари АҚШ долларида йиллик 15%, еврода йиллик 15% ва миллий валютада йиллик",
        "sourcePay": "Мажбуриятларни тўлаш манбаи компаниянинг давом этаётган фаолиятидан келиб чиқади.",
        "bankFill": "Банк томонидан тўлдирилади",
        "mainObj": "Корхона номидаги асосий объект (офис, бино, омбор, турар жой) қуйидаги манзилда жойлашган:",
        "totalArea": "умумий майдони",
        "buildArea": "бино ва иншоотлар майдони",
        "cad": "кадастр рақами",
        "belongs": "Ушбу ер",
        "belongs2": "га тегишли.",
        "infra": "Ҳудуд зарур коммунал хизматлар ва коммуникациялар билан жиҳозланган.",
        "built": "Ушбу ҳудудда компания",
        "built2": "кв.м. майдонга эга янги маъмурий бино қўрди.",
        "collateralTitle": "Гаров ҳақида маълумот",
        "reTitle": "Гаровга қўйилган бино маълумотлари:",
        "carTitle": "Гаровга қўйилган машина маълумотлари:",
        "balanceTitle": "Баланс бўйича ҳисобот:",
        "metric": "Кўрсаткич номи",
        "row": "Қатор",
        "finTitle": "Молиявий кўрсаткичлар маълумотлари",
        "shortAbout": "Корхона ҳақида қисқача маълумот:",
        "objs": "Корхона номидаги объектлар:",
        "cars": "Корхона номидаги машиналар:",
        "hTin": "TIN",
        "hName": "Компания номи",
        "hType": "Тури",
        "hCad": "Кадастр рақами",
        "hObjName": "Бино номи",
        "hAddr": "Манзили",
        "hShare": "Эгалик қилувчи улуши (%)",
        "hInvCost": "Инвентар нарх",
        "hTotalArea": "Умумий майдон",
        "hBuildArea": "Бино майдони",
        "hExtraArea": "Қўшимча майдон",
        "hEstValue": "Гаровнинг тахминий қиймати",
        "hModel": "Модели",
        "hColor": "Ранги",
        "hYear": "Йил",
        "hKuzov": "Кузов рақами",
        "hMotor": "Мотор",
        "hShassi": "Шасси",
        "hGosNumber": "Давлат рақами",
        "hRegDate": "Рўйхатдан ўтган сана",
        "hDivision": "Дивизия",
        "hOwner": "Эгалик қилувчи",
        "unitArea": "кв.м.",
    },
    "en": {
        "expertConclusionTitle": "Final credit expert conclusion",
        "hProject": "Project information",
        "hCompany": "Company details",
        "orgFull": "Full legal name",
        "inn": "TIN",
        "oked": "Business activity (OKED)",
        "cat": "Company category",
        "emp": "Average number of employees",
        "addr": "Legal address",
        "fund": "Charter capital",
        "foundersTitle": "Founders information",
        "fio": "Full name / Company name",
        "who": "Type",
        "share": "Share (%)",
        "other": "Participation in other entities",
        "director": "Director",
        "ind": "Individual",
        "leg": "Legal entity",
        "bankReq": "Bank details:",
        "mainAcc": "Main account:",
        "bankName": "Bank name",
        "acc": "Account number",
        "mfo": "MFO",
        "appShort": "Brief project information",
        "appDesc": "Project description",
        "started": "started operating in",
        "mainAct": "The company’s main activity is",
        "applied2": "submitted a request to the bank to consider signing a general loan agreement for the purpose of:",
        "applied3": "Maximum credit limit",
        "term": "term",
        "rates": "Interest rates: 15% p.a. in USD, 15% p.a. in EUR, and",
        "sourcePay": "The source of repayment is generated from the company’s ongoing business activities.",
        "bankFill": "To be completed by the bank",
        "mainObj": "The main asset of the enterprise (office, building, warehouse, residential property) is located at:",
        "totalArea": "total area",
        "buildArea": "building/structures area",
        "cad": "cadastral number",
        "belongs": "This land belongs to",
        "belongs2": ".",
        "infra": "The area is equipped with the necessary utilities and communications.",
        "built": "On this territory, the company constructed a new administrative building with an area of",
        "built2": "sq.m.",
        "collateralTitle": "Collateral information",
        "reTitle": "Collateralized real estate details:",
        "carTitle": "Collateralized vehicle details:",
        "balanceTitle": "Balance sheet report:",
        "metric": "Indicator name",
        "row": "Row",
        "finTitle": "Financial indicators",
        "shortAbout": "Company summary:",
        "objs": "Assets registered to the company:",
        "cars": "Vehicles registered to the company:",
        "hTin": "TIN",
        "hName": "Company name",
        "hType": "Type",
        "hCad": "Cadastral number",
        "hObjName": "Building name",
        "hAddr": "Address",
        "hShare": "Owner's share (%)",
        "hInvCost": "Inventory cost",
        "hTotalArea": "Total area",
        "hBuildArea": "Building area",
        "hExtraArea": "Additional area",
        "hEstValue": "Estimated collateral value",
        "hModel": "Model",
        "hColor": "Color",
        "hYear": "Year",
        "hKuzov": "Body number",
        "hMotor": "Motor",
        "hShassi": "Chassis",
        "hGosNumber": "License plate",
        "hRegDate": "Registration date",
        "hDivision": "Division",
        "hOwner": "Owner",
        "unitArea": "sq.m.",
    },
    "ru": {
        "expertConclusionTitle": "Итоговое экспертное заключение по кредиту",
        "hProject": "Информация о проекте",
        "hCompany": "О компании",
        "orgFull": "Полное наименование организации",
        "inn": "ИНН",
        "oked": "Сфера деятельности (ОКЭД)",
        "cat": "Категория компании",
        "emp": "Средняя численность сотрудников",
        "addr": "Юридический адрес",
        "fund": "Уставный фонд",
        "foundersTitle": "Информация об учредителях",
        "fio": "Ф.И.О. / Наименование",
        "who": "Кем является",
        "share": "Доля (%)",
        "other": "Участие в других субъектах",
        "director": "Директор",
        "ind": "Физическое лицо",
        "leg": "Юридическое лицо",
        "bankReq": "Банковские реквизиты:",
        "mainAcc": "Основной счёт:",
        "bankName": "Наименование банка",
        "acc": "Расчётный счёт",
        "mfo": "МФО",
        "appShort": "Краткая информация о проекте",
        "appDesc": "Описание проекта",
        "started": "начал деятельность в",
        "mainAct": "Основной вид деятельности компании",
        "applied2": "обратилось в банк с запросом по цели:",
        "applied3": "Максимальный кредитный лимит",
        "term": "Срок",
        "rates": "Процентные ставки по кредиту:",
        "sourcePay": "Источник погашения обязательств — доходы от текущей деятельности компании.",
        "bankFill": "Заполняется банком",
        "mainObj": "Основной объект предприятия (офис, здание, склад, жилое помещение) расположен по адресу:",
        "totalArea": "общая площадь",
        "buildArea": "площадь здания и сооружений",
        "cad": "кадастровый номер",
        "belongs": "Данный земельный участок принадлежит",
        "belongs2": ".",
        "infra": "Территория обеспечена необходимыми коммунальными услугами и коммуникациями.",
        "built": "На данной территории компания построила новое административное здание площадью",
        "built2": "кв.м.",
        "collateralTitle": "Информация о залоге",
        "reTitle": "Данные по заложенной недвижимости:",
        "carTitle": "Данные по заложенному транспортному средству:",
        "balanceTitle": "Отчёт по балансу:",
        "metric": "Наименование показателя",
        "row": "Строка",
        "finTitle": "Финансовые показатели",
        "shortAbout": "Краткая информация о компании:",
        "objs": "Объекты, зарегистрированные на компанию:",
        "cars": "Транспортные средства, зарегистрированные на компанию:",
        "hTin": "ИНН",
        "hName": "Наименование",
        "hType": "Тип",
        "hCad": "Кадастровый номер",
        "hObjName": "Наименование объекта",
        "hAddr": "Адрес",
        "hShare": "Доля владения (%)",
        "hInvCost": "Инвентарная стоимость",
        "hTotalArea": "Общая площадь",
        "hBuildArea": "Площадь здания",
        "hExtraArea": "Доп. площадь",
        "hEstValue": "Оценочная стоимость залога",
        "hModel": "Модель",
        "hColor": "Цвет",
        "hYear": "Год",
        "hKuzov": "Номер кузова",
        "hMotor": "Двигатель",
        "hShassi": "Шасси",
        "hGosNumber": "Гос. номер",
        "hRegDate": "Дата регистрации",
        "hDivision": "Подразделение",
        "hOwner": "Владелец",
        "unitArea": "кв.м.",
    },
}


# ---------------- main builder (adapted from code.py) ----------------

def build_html_for_lang(payload: Any, ai_conclusion: str, language: str) -> str:
    # If payload is a list, try using the first element or treat it as the body
    if isinstance(payload, list):
        body = payload[0] if payload else {}
    else:
        body = payload.get("body") if isinstance(payload.get("body"), dict) else payload
        
    language = str(language).lower()

    L = DICT.get(language) or DICT["uz"]

    company = get_in(body, ["company_info", "data"], {}) or {}
    bank_info = body.get("bankInfo") or {}
    app_data = get_in(body, ["applicationInfo", "applicationData"], {}) or {}
    tax_objects = body.get("taxObjects") or {}
    forma1 = get_in(body, ["forma_1", "data", 0, "rows"], []) or []
    forma2 = get_in(body, ["forma_2", "data", 0, "rows"], []) or []

    company_name = val(company.get("name"))
    tin = val(company.get("tin"))
    oked = val(get_in(company, ["okedDetail", "name"]))
    category = val(get_in(company, ["businessTypeDetail", "name"]))
    employee_count = num_format(body.get("employeeCount"))

    legal_address = join_address(
        get_in(company, ["companyBillingAddress", "region", "name"]),
        get_in(company, ["companyBillingAddress", "district", "name"]),
        get_in(company, ["companyBillingAddress", "streetName"]),
    )
    business_fund = num_format(company.get("businessFund"))
    director_name = full_name(company.get("director"))

    reg_date = val(bank_info.get("regDate"))

    def purpose_by_lang() -> str:
        a = get_in(body, ["applicationInfo", "applicationData"], {}) or {}
        if language == "cyrl":
            return val(a.get("purposeCyrl"))
        if language == "ru":
            return val(a.get("purposeRu"))
        if language == "en":
            return val(a.get("purposeEn"))
        return val(a.get("purposeUz"))

    purpose = purpose_by_lang()
    requested_amount = num_format(app_data.get("requestedAmount"))
    currency_raw = val(app_data.get("currency"))
    currency = "UZS" if currency_raw == "---" else currency_raw

    loan_term_months = val(app_data.get("loanTermMonths"))
    down_payment_percent = val(app_data.get("downPaymentPercent"))

    reg_date = val(bank_info.get("regDate"))
    foundation_year = year_from_any_date(reg_date)
    buildings = str(len(safe_arr(tax_objects.get("dataObject"))) or "---")


    def land_area_sum() -> str:
        s = 0.0
        for o in safe_arr(tax_objects.get("dataObject")):
            v = (o or {}).get("land_area")
            try:
                n = float(str(v or "").replace(" ", "").replace(",", "."))
            except ValueError:
                n = 0.0
            if math.isfinite(n):
                s += n
        return num_format(s) if s > 0 else "---"

    land_area = land_area_sum()

    first_tax_obj = (safe_arr(tax_objects.get("dataObject"))[0] if safe_arr(tax_objects.get("dataObject")) else {}) or {}
    main_obj_addr = val(first_tax_obj.get("address"))
    main_obj_total = num_format(first_tax_obj.get("total_area"))
    main_obj_land = num_format(first_tax_obj.get("land_area"))
    main_obj_cad = val(first_tax_obj.get("obj_code"))
    main_obj_extra = num_format(first_tax_obj.get("land_extra_area"))

    ua = "sq.m." if language == "en" else "кв.м."


    # founders
    founders_html = ""
    for f in safe_arr(company.get("founders")):
        is_individual = bool(get_in(f, ["founderIndividual"]))
        is_legal = bool(get_in(f, ["founderLegal"]))

        if is_individual:
            fi = get_in(f, ["founderIndividual"], {}) or {}
            name_parts = [fi.get("lastName"), fi.get("firstName"), fi.get("middleName")]
            name_parts = [val(x) for x in name_parts if x]
            name_parts = [x for x in name_parts if x != "---"]
            name = " ".join(name_parts) if name_parts else "---"
        else:
            name = val(get_in(f, ["founderLegal", "name"]))

        type_ = L.get("ind") if is_individual else (L.get("leg") if is_legal else "---")
        share = val(f.get("sharePercent"))

        founders_html += f"<tr><td style='border: 1px solid black; padding: 4px;'>{esc(name)}</td><td style='border: 1px solid black; padding: 4px;'>{esc(type_)}</td><td style='border: 1px solid black; padding: 4px;'>{esc(share)}</td><td style='border: 1px solid black; padding: 4px;'>---</td></tr>"

    # collateral
    collateral = safe_arr(get_in(body, ["applicationInfo", "collateralData"]))

    collateral_real_estate = ""
    for i in collateral:
        if (i or {}).get("yurTaxObjectData") is not None or (i or {}).get("collateralType") == "REAL_ESTATE":
            y = (i or {}).get("yurTaxObjectData") or {}
            collateral_real_estate += f"""
<tr>
<td style="border: 1px solid black; padding: 4px;">{esc(val(y.get("tin")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val(y.get("name")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val(y.get("type")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val(y.get("obj_code") or (i or {}).get("cadastreOrCarKuzov")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val(y.get("obj_name")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val(y.get("address") or (i or {}).get("address")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val(y.get("percentage")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(num_format(y.get("inv_cost")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(num_format(y.get("total_area")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(num_format(y.get("land_area")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(num_format(y.get("land_extra_area")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(num_format((i or {}).get("estimatedValue")))}</td>
</tr>
""".strip()

    collateral_cars = ""
    for i in collateral:
        if (i or {}).get("yurCarData") is not None or (i or {}).get("collateralType") == "VEHICLE":
            c = (i or {}).get("yurCarData") or {}
            collateral_cars += f"""
<tr>
<td style="border: 1px solid black; padding: 4px;">{esc(val(c.get("model")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val(c.get("color")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val((i or {}).get("carYear") or c.get("year")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val((i or {}).get("cadastreOrCarKuzov") or c.get("kuzov")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val(c.get("motor")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val(c.get("shassi")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val((i or {}).get("carLicensePlate") or c.get("gosNumber")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val(c.get("regDate")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val(c.get("division")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val(c.get("owner")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val(c.get("adres") or (i or {}).get("address")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(num_format((i or {}).get("estimatedValue")))}</td>
</tr>
""".strip()

    # company assets
    tax_obj_rows = ""
    for item in safe_arr(tax_objects.get("dataObject")):
        tax_obj_rows += f"""
<tr>
<td style="border: 1px solid black; padding: 4px;">{esc(val((item or {}).get("tin")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val((item or {}).get("name")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val((item or {}).get("type")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val((item or {}).get("obj_code")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val((item or {}).get("obj_name")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val((item or {}).get("address")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val((item or {}).get("percentage")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(num_format((item or {}).get("inv_cost")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(num_format((item or {}).get("total_area")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(num_format((item or {}).get("land_area")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(num_format((item or {}).get("land_extra_area")))}</td>
</tr>
""".strip()

    car_obj_rows = ""
    for car in safe_arr(tax_objects.get("carDataObject")):
        car_obj_rows += f"""
<tr>
<td style="border: 1px solid black; padding: 4px;">{esc(val((car or {}).get("model")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val((car or {}).get("color")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val((car or {}).get("year")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val((car or {}).get("kuzov")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val((car or {}).get("motor")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val((car or {}).get("shassi")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val((car or {}).get("gosNumber")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val((car or {}).get("regDate")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val((car or {}).get("division")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val((car or {}).get("owner")))}</td>
<td style="border: 1px solid black; padding: 4px;">{esc(val((car or {}).get("adres")))}</td>
</tr>
""".strip()

    # section 7 labels
    if language == "ru":
        S7 = {
            "a": "Остаточная стоимость основных средств (010-011)",
            "b": "Итого запасы (150+160+170+180), в том числе:",
            "c": "Итого дебиторская задолженность (220+240+250+260+270+280+290+300+310)",
            "d": "Общая сумма активов баланса (130 + 390)",
            "e": "В том числе: текущая кредиторская задолженность (610+630+650+670+680+690+700+710+720+760)",
            "f": "Долгосрочные банковские кредиты (7810)",
            "g": "Краткосрочные банковские кредиты (6810)",
            "h": "Итого обязательства (480+770)",
            "i": "Чистая выручка от реализации товаров (работ, услуг)",
            "j": "Чистая прибыль (убыток) за отчётный период (240-250-260)",
        }
    elif language == "en":
        S7 = {
            "a": "Residual value of fixed assets (010-011)",
            "b": "Total inventory (150+160+170+180), including:",
            "c": "Total receivables (220+240+250+260+270+280+290+300+310)",
            "d": "Total balance sheet assets (130 + 390)",
            "e": "Including: current accounts payable (610+630+650+670+680+690+700+710+720+760)",
            "f": "Long-term bank loans (7810)",
            "g": "Short-term bank loans (6810)",
            "h": "Total liabilities (480+770)",
            "i": "Net revenue from sales of goods (works, services)",
            "j": "Net profit (loss) for the reporting period (240-250-260)",
        }
    else:
        S7 = {
            "a": "Асосий воситаларнинг қолдиқ қиймати (010-011)" if language == "cyrl" else "Asosiy vositalarning qoldiq qiymati (010-011)",
            "b": "Жами инвентаризация (150+160+170+180-қаторлар), шу жумладан:" if language == "cyrl" else "Jami inventarizatsiya (150+160+170+180-qatorlar), shu jumladan:",
            "c": "Жами дебиторлар (220+240+250+260+270+280+290+300+310 қаторлар)" if language == "cyrl" else "Jami debitorlar (220+240+250+260+270+280+290+300+310 qatorlar)",
            "d": "Баланс активларининг умумий суммаси (130-қатор + 390-қатор)" if language == "cyrl" else "Balans aktivlarining umumiy summasi (130-qator + 390-qator)",
            "e": "Шу жумладан: жорий кредиторлик қарзлари (610+630+650+670+680+690+700+710+720+760-қаторлар)" if language == "cyrl" else "Shu jumladan: joriy kreditorlik qarzlari (610+630+650+670+680+690+700+710+720+760-qatorlar)",
            "f": "Узоқ muddatli bank kreditlari (7810)" if language == "cyrl" else "Uzoq muddatli bank kreditlari (7810)",
            "g": "Қисқа muddatli bank kreditlari (6810)" if language == "cyrl" else "Qisqa muddatli bank kreditlari (6810)",
            "h": "Жами мажбуриятлар (480+770-қаторлар)" if language == "cyrl" else "Jami majburiyatlar (480+770-qatorlar)",
            "i": "Маҳсулотлар (товарлар, ишлар ва хизматлар) сотишдан олинган соф даромад" if language == "cyrl" else "Mahsulotlar (tovarlar, ishlar va xizmatlar) sotishdan olingan sof daromad",
            "j": "Ҳисобот даври учун соф фойда (зарар) (240-250-260-қаторлар)" if language == "cyrl" else "Hisobot davri uchun sof foyda (zarar) (240-250-260-qatorlar)",
        }

    # build html
    html = f"""
<div style="font-family:'Times New Roman'; font-size:11pt;">

<h2>{L.get("hProject")}: {esc(company_name)}</h2>
<h3>{L.get("hCompany")}</h3>
<table border="1" cellspacing="0" cellpadding="4" width="100%" style="font-family:'Times New Roman'; font-size:11pt; border-collapse: collapse;">
<tr><td>{L.get("orgFull")}</td><td>{esc(company_name)}</td></tr>
<tr><td>{L.get("inn")}</td><td>{esc(tin)}</td></tr>
<tr><td>{L.get("oked")}</td><td>{esc(oked)}</td></tr>
<tr><td>{L.get("cat")}</td><td>{esc(category)}</td></tr>
<tr><td>{L.get("emp")}</td><td>{esc(employee_count)}</td></tr>
<tr><td>{L.get("addr")}</td><td>{esc(legal_address)}</td></tr>
<tr><td>{L.get("fund")}</td><td>{esc(business_fund)}</td></tr>
</table>

<h2>{L.get("foundersTitle")}</h2>
<table border="1" cellspacing="0" cellpadding="4" width="100%" style="font-family:'Times New Roman'; font-size:11pt; border-collapse: collapse;">
<tr><th>{L.get("fio")}</th><th>{L.get("who")}</th><th>{L.get("share")}</th><th>{L.get("other")}</th></tr>
{founders_html or "<tr><td>---</td><td>---</td><td>---</td><td>---</td></tr>"}
<tr><td><b>{L.get("director")}</b></td><td colspan="3">{esc(director_name)}</td></tr>
</table>

<p><em><u>{L.get("bankReq")}</u></em></p>
<p><strong>{L.get("mainAcc")}</strong></p>
<table border="1" cellspacing="0" cellpadding="4" width="100%" style="font-family:'Times New Roman'; font-size:11pt; border-collapse: collapse;">
<tr><td><b>{L.get("bankName")}</b></td><td>{esc(val(bank_info.get("ns2Name")))}</td></tr>
<tr><td><b>{L.get("acc")}</b></td><td>{esc(bank_acc_format(bank_info.get("account")))}</td></tr>
<tr><td><b>{L.get("mfo")}</b></td><td>{esc(val(bank_info.get("ns2Code")))}</td></tr>
</table>

<p><strong><u>{L.get("appShort")}</u></strong></br>
<strong><u>{L.get("appDesc")}</u></strong></br>
<strong><u>{esc(company_name)} {esc(reg_date)} {L.get("started")}</u></strong></br></br>
{L.get("mainAct")} {esc(oked)}.</br>
<strong><u>{esc(company_name)}</u></strong> {L.get("applied2")} {esc(purpose)}</br>
{ (f"{L.get('applied3')}: {esc(requested_amount)} {esc(currency)}; {L.get('term')}: {esc(loan_term_months)}.") if language == "ru"
  else (f"{L.get('applied3')}: {esc(requested_amount)} {esc(currency)}; {L.get('term')}: {esc(loan_term_months)} months.") if language == "en"
  else (f"{L.get('applied3')} {esc(requested_amount)} {esc(currency)}, {esc(loan_term_months)} {L.get('term')}")
}</br>
{L.get("rates")} {esc(num_format(down_payment_percent))}%.</br></br>
<strong><u>{L.get("sourcePay")}</u></strong></p>

<p><strong>{L.get("bankFill")}</strong></br></br>
<strong>{L.get("mainObj")}</strong> {esc(main_obj_addr)}, {L.get("totalArea")} {esc(main_obj_total)} {ua}, {L.get("buildArea")} {esc(main_obj_land)} {ua}, {L.get("cad")} {esc(main_obj_cad)}.</br>
{L.get("belongs")} {esc(company_name)}{L.get("belongs2")}</br>
{L.get("infra")}</br>
{L.get("built")} {esc(main_obj_extra)} {ua} {L.get("built2") if language in ("uz","cyrl") else ""}</p>

<h3>{L.get("collateralTitle")}</h3>
<p><em><u>{L.get("reTitle")}</u></em></p>
<table border="1" cellspacing="0" cellpadding="4" width="100%" style="font-family:'Times New Roman'; font-size:11pt; border-collapse: collapse;">
<tr>
<th>TIN</th>
<th>{"Company name" if language == "en" else ("Наименование" if language == "ru" else "Kompaniya nomi")}</th>
<th>{"Type" if language == "en" else ("Тип" if language == "ru" else "Turi")}</th>
<th>{"Cadastral number" if language == "en" else ("Кадастровый номер" if language == "ru" else "Kadastr raqami")}</th>
<th>{"Property name" if language == "en" else ("Наименование объекта" if language == "ru" else "Bino nomi")}</th>
<th>{"Address" if language == "en" else ("Адрес" if language == "ru" else "Manzili")}</th>
<th>{"Ownership share (%)" if language == "en" else ("Доля владения (%)" if language == "ru" else "Egalik qiluvchi ulushi (%)")}</th>
<th>{"Inventory value" if language == "en" else ("Инвентарная стоимость" if language == "ru" else "Inventar narx")}</th>
<th>{"Total area" if language == "en" else ("Общая площадь" if language == "ru" else "Umumiy maydon")}</th>
<th>{"Building area" if language == "en" else ("Площадь здания" if language == "ru" else "Bino maydoni")}</th>
<th>{"Additional area" if language == "en" else ("Дополнительная площадь" if language == "ru" else "Qo'shimcha maydon")}</th>
<th>{"Estimated collateral value" if language == "en" else ("Оценочная стоимость залога" if language == "ru" else "Garovning taxminiy qiymati")}</th>
</tr>
{collateral_real_estate or "<tr><td colspan='12'>---</td></tr>"}
</table>

</br>
<p><em><u>{L.get("carTitle")}</u></em></p>
<table border="1" cellspacing="0" cellpadding="4" width="100%" style="font-family:'Times New Roman'; font-size:11pt; border-collapse: collapse;">
<tr>
<th>{"Model" if language == "en" else ("Модель" if language == "ru" else "Modeli")}</th>
<th>{"Color" if language == "en" else ("Цвет" if language == "ru" else "Rangi")}</th>
<th>{"Year" if language == "en" else ("Год" if language == "ru" else "Yil")}</th>
<th>{"Body number" if language == "en" else ("Номер кузова" if language == "ru" else "Kuzov raqami")}</th>
<th>{"Engine" if language == "en" else ("Двигатель" if language == "ru" else "Motor")}</th>
<th>{"Chassis" if language == "en" else ("Шасси" if language == "ru" else "Shassi")}</th>
<th>{"License plate" if language == "en" else ("Гос. номер" if language == "ru" else "Davlat raqami")}</th>
<th>{"Registration date" if language == "en" else ("Дата регистрации" if language == "ru" else "Ro'yxatdan o'tgan sana")}</th>
<th>{"Division" if language == "en" else ("Подразделение" if language == "ru" else "Diviziya")}</th>
<th>{"Owner" if language == "en" else ("Владелец" if language == "ru" else "Egalik qiluvchi")}</th>
<th>{"Address" if language == "en" else ("Адрес" if language == "ru" else "Manzili")}</th>
<th>{"Estimated collateral value" if language == "en" else ("Оценочная стоимость залога" if language == "ru" else "Garovning taxminiy qiymati")}</th>
</tr>
{collateral_cars or "<tr><td colspan='12'>---</td></tr>"}
</table>

<p><i><u>{L.get("balanceTitle")}</u></i></p>
<table border="1" cellspacing="0" cellpadding="5" style="border-collapse: collapse; width: 100%;">
<tr><th>{L.get("metric")}</th><th>{L.get("row")}</th><th colspan="2">{esc(company_name)}</th></tr>
<tr><th></th><th></th><th></th><th></th></tr>
<tr><td>{S7["a"]}</td><td>010</td><td>{esc(find_row(forma1,"010","sum_begin_period"))}</td><td>{esc(find_row(forma1,"010","sum_end_period"))}</td></tr>
<tr><td>{S7["b"]}</td><td>140</td><td>{esc(find_row(forma1,"140","sum_begin_period"))}</td><td>{esc(find_row(forma1,"140","sum_end_period"))}</td></tr>
<tr><td>{S7["c"]}</td><td>210</td><td>{esc(find_row(forma1,"210","sum_begin_period"))}</td><td>{esc(find_row(forma1,"210","sum_end_period"))}</td></tr>
<tr><td>{S7["d"]}</td><td>400</td><td>{esc(find_row(forma1,"400","sum_begin_period"))}</td><td>{esc(find_row(forma1,"400","sum_end_period"))}</td></tr>
<tr><td>{S7["e"]}</td><td>601</td><td>{esc(find_row(forma1,"601","sum_begin_period"))}</td><td>{esc(find_row(forma1,"601","sum_end_period"))}</td></tr>
<tr><td>{S7["f"]}</td><td>570</td><td>{esc(find_row(forma1,"570","sum_begin_period"))}</td><td>{esc(find_row(forma1,"570","sum_end_period"))}</td></tr>
<tr><td>{S7["g"]}</td><td>730</td><td>{esc(find_row(forma1,"730","sum_begin_period"))}</td><td>{esc(find_row(forma1,"730","sum_end_period"))}</td></tr>
<tr><td>{S7["h"]}</td><td>780</td><td>{esc(find_row(forma1,"780","sum_begin_period"))}</td><td>{esc(find_row(forma1,"780","sum_end_period"))}</td></tr>
<tr><td>{L.get("finTitle")}</td><td></td><td></td><td></td></tr>
<tr><td>{S7["i"]}</td><td>010</td><td>{esc(find_row(forma2,"010","sum_period_doxod"))}</td><td>{esc(find_row(forma2,"010","sum_period_rasxod"))}</td></tr>
<tr><td>{S7["j"]}</td><td>270</td><td>{esc(find_row(forma2,"270","sum_period_doxod"))}</td><td>{esc(find_row(forma2,"270","sum_period_rasxod"))}</td></tr>
</table>

<p><strong>{L.get("shortAbout")}</strong></br>
{ (f"Предприятие {esc(company_name)} осуществляет деятельность с {esc(foundation_year)} года и в основном работает в сфере {esc(oked)}.</br>"
   f"Юридический адрес: {esc(legal_address)}.</br>"
   f"На балансе предприятия: {esc(buildings)} объект(ов) недвижимости и {esc(land_area)} {ua} земельных участков; территория обеспечена инфраструктурой и коммуникациями.</br>"
   f"Средняя численность сотрудников: {esc(employee_count)}.</br>"
   f"Финансовые показатели, деловая репутация и ликвидность активов формируют высокий уровень доверия к предприятию."
  ) if language == "ru" else (
   f"{esc(company_name)} has been operating since {esc(foundation_year)} and primarily provides services in the field of {esc(oked)}.</br>"
   f"Legal address: {esc(legal_address)}.</br>"
   f"The company’s balance includes {esc(buildings)} building(s) and {esc(land_area)} {ua} of land; the territory is supplied with the necessary infrastructure and communications.</br>"
   f"Average number of employees: {esc(employee_count)}.</br>"
   f"Financial performance, market reputation, and asset liquidity provide a strong basis for confidence in the company."
  ) if language == "en" else (
   f"{esc(company_name)} корхонаси {esc(foundation_year)} йилдан бери фаолият юритиб, асосан {esc(oked)} соҳасида хизмат кўрсатиб келади.</br>"
   f"Корхона юридик манзили: {esc(legal_address)}.</br>"
   f"Корхона балансида {esc(buildings)} дона бино ва {esc(land_area)} {ua} ер майдони мавжуд бўлиб, ҳудуд зарур инфратузилма ва коммуникациялар билан таъминланган.</br>"
   f"Ташкилотда {esc(employee_count)} нафар ходим ишлайди ва ишлаб чиқариш қуввати йил сайин ошмоқда.</br>"
   f"Молиявий кўрсаткичлар ва бозордаги обрў-эътибори, шунингдек, активларнинг ликвидлиги корхонага юқори ишонч беради."
  ) if language == "cyrl" else (
   f"{esc(company_name)} korxonasi {esc(foundation_year)} yildan beri faoliyat yuritib, asosan {esc(oked)} sohasida xizmat ko‘rsatib keladi.</br>"
   f"Korxona yuridik manzili: {esc(legal_address)}.</br>"
   f"Korxona balansida {esc(buildings)} dona bino va {esc(land_area)} {ua} yer maydoni mavjud bo‘lib, hudud zarur infratuzilma va kommunikatsiyalar bilan ta’minlangan.</br>"
   f"Tashkilotda {esc(employee_count)} nafar xodim ishlaydi va ishlab chiqarish quvvati yil sayin oshmoqda.</br>"
   f"Moliyaviy ko‘rsatkichlar va bozordagi obro‘-e’tibori, shuningdek, aktivlarning likvidligi korxonaga yuqori ishonch beradi."
  )
}
</p>

<p><em><u>{L.get("objs")}</u></em></p>
<table border="1" cellspacing="0" cellpadding="4" width="100%" style="font-family:'Times New Roman'; font-size:11pt; border-collapse: collapse;">
<tr>
<th style="border: 1px solid black; padding: 4px;">{L.get("hTin")}</th>
<th style="border: 1px solid black; padding: 4px;">{L.get("hName")}</th>
<th style="border: 1px solid black; padding: 4px;">{L.get("hType")}</th>
<th style="border: 1px solid black; padding: 4px;">{L.get("hCad")}</th>
<th style="border: 1px solid black; padding: 4px;">{L.get("hObjName")}</th>
<th style="border: 1px solid black; padding: 4px;">{L.get("hAddr")}</th>
<th style="border: 1px solid black; padding: 4px;">{L.get("hShare")}</th>
<th style="border: 1px solid black; padding: 4px;">{L.get("hInvCost")}</th>
<th style="border: 1px solid black; padding: 4px;">{L.get("hTotalArea")}</th>
<th style="border: 1px solid black; padding: 4px;">{L.get("hBuildArea")}</th>
<th style="border: 1px solid black; padding: 4px;">{L.get("hExtraArea")}</th>
</tr>
{tax_obj_rows or "<tr><td colspan='11'>---</td></tr>"}
</table>

</br>
<p><em><u>{L.get("cars")}</u></em></p>
<table border="1" cellspacing="0" cellpadding="4" width="100%" style="font-family:'Times New Roman'; font-size:11pt; border-collapse: collapse;">
<tr>
<th style="border: 1px solid black; padding: 4px;">{L.get("hModel")}</th><th style="border: 1px solid black; padding: 4px;">{L.get("hColor")}</th><th style="border: 1px solid black; padding: 4px;">{L.get("hYear")}</th><th style="border: 1px solid black; padding: 4px;">{L.get("hKuzov")}</th><th style="border: 1px solid black; padding: 4px;">{L.get("hMotor")}</th><th style="border: 1px solid black; padding: 4px;">{L.get("hShassi")}</th><th style="border: 1px solid black; padding: 4px;">{L.get("hGosNumber")}</th><th style="border: 1px solid black; padding: 4px;">{L.get("hRegDate")}</th><th style="border: 1px solid black; padding: 4px;">{L.get("hDivision")}</th><th style="border: 1px solid black; padding: 4px;">{L.get("hOwner")}</th><th style="border: 1px solid black; padding: 4px;">{L.get("hAddr")}</th>
</tr>
{car_obj_rows or "<tr><td colspan='11'>---</td></tr>"}
</table>

</div>
<br><h2>{esc(L.get("expertConclusionTitle"))}</h2><br>
{ai_conclusion}
""".strip()

    return normalize_br(html.replace("\n", " ").strip())


# ---------------- FastAPI App ----------------

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

llm = ChatOpenAI(
    model="gpt-4o",
    api_key=API_KEY,
    temperature=0.3,
)


class WebhookRequest(RootModel):
    root: dict | list

class WebhookResponse(RootModel):
    root: str

@router.post("/webhook/document")
async def webhook(payload: WebhookRequest):
    import time
    start_time = time.time()
    
    raw_payload = payload.root
    # Determine body
    if isinstance(raw_payload, list):
        body = raw_payload[0] if raw_payload else {}
    else:
        body = raw_payload.get("body") if isinstance(raw_payload.get("body"), dict) else raw_payload

    # 1. Prepare Logic Variables
    # LTV Logic
    app_info = body.get("applicationInfo", {}) or {}
    collateral_data = safe_arr(app_info.get("collateralData"))
    first_collateral = collateral_data[0] if collateral_data else {}
    estimated_value = float(first_collateral.get("estimatedValue") or 0)
    
    app_data = app_info.get("applicationData", {}) or {}
    requested_amount = float(app_data.get("requestedAmount") or 0)
    
    # "Garov (LTV - Kreditning qiymatga nisbati): Agar taqdim etilgan garov kredit miqdoridan sezilarli darajada ortiq bo'lsa, 
    # LTV (Kreditning qiymatga nisbati) 64% gacha bo'lgan nisbat (estimated * 0.64 > requested) ijobiy hisoblanadi."
    is_ltv_good = (estimated_value * 0.64) > requested_amount
    ltv_status_text = "Ijobiy" if is_ltv_good else "Salbiy"

    conclusion_field = str(body.get("conclusion", "NEUTRAL")).upper()


    # 2. Build Prompt
    # We pass the entire JSON as context
    try:
        input_json_str = json.dumps(raw_payload, ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as e:
        input_json_str = "{}" # Should not happen based on pydantic model

    system_prompt = f"""Siz professional AI yordamchisiz. 

Sizning vazifangiz: Yakuniy kredit bo‘yicha ekspert xulosasi yaratish,

Xulosa har doim 4 ta tilda bir vaqtning o‘zida yaratiladi:

1) O‘zbek lotin (UZ)
2) O‘zbek kirill (CYRL)
3) Rus tili (RU)
4) Ingliz tili (EN)

Har bir til alohida va mustaqil <div> da bo‘lishi shart.
Bir til ichida boshqa til aralashmasin.


<h3>Yakuniy kredit bo‘yicha ekspert xulosasi</h3>

Xulosa quyidagilar asosida avtomatik shakllantiriladi:

✓ Kompaniya faoliyat holati  
✓ Har doim forma bo‘yicha 
✓ Pul oqimi  
✓ Zalog likvidligi  
✓ Risklar  Kredit berish tavsiya qilinadi
✓ Kuchli tomonlar  
yuqoridagi natijalar hammasi qisqa londa bo'lsin. Hammasi bitta paragraphda yozilsin.

Kredit berish xulosasini tuzish bo'yicha logika:
- Moliyaviy yuk (Aylanma darajasi). Agar so'ralgan kredit summasi kompaniyaning yillik aylanmasidan oshiq bo'lsa, bunday holatni past kredit havfi va kredit bo'yicha qarzni to'lamaslik ehtimoli sifatida baholashingiz kerak.
- Garov (LTV - Kreditning qiymatga nisbati):
Agar taqdim etilgan garov kredit miqdoridan sezilarli darajada ortiq bo'lsa, LTV (Kreditning qiymatga nisbati) 64% gacha bo'lgan nisbat ({str(is_ltv_good).lower()}) ijobiy hisoblanadi.


XULOSA QUYIDAGILARDAN BIRI BO‘LISHI SHART, VA ULAR HAM BELGILANGAN TILGA TARJIMA QILINISHI SHART:

1) Agar kredit berish tavsiya etilsa: <strong> Kredit berish tavsiya qilinadi </strong> <- albatta {body.get('language', 'tegishli til')}ga tarjima qilinishi kerak
2) Agar kredit berish tavsiya etilmasa: <strong> Kommitet tomonidan ko'rib chiqiladi </strong> <- albatta {body.get('language', 'tegishli til')}ga tarjima qilinishi kerak

Yuqoridagi ikkita holat, hulosa ham albatta belgilangan tilgan tajrima qilish kerak, va o'sha tilda chiqarish kerak, boshqa tildagi qo'shimchalar bo'lmasin! Hech qachon belgilangan tildan boshqasiga tarjima qilmang!

Va albatta izohlar bilan:
– Nima sababdan berilishi kerak / berilmasligi kerak aynan jadvaldagi qaysi ma'lumot asosida olayapti.
– Qaysi ko‘rsatkichga asoslanildi aynan jadvaldagi qaysi ma'lumot asosida olayapti.
– Qaysi risklar mavjud aynan jadvaldagi qaysi ma'lumot asosida olayapti.
– Qaysi kuchli jihatlar kreditni qo‘llab-quvvatlaydi aynan jadvaldagi qaysi ma'lumot asosida olayapti.
- Hulosa rasmiy tilda yozilsin. jadval malumotlari korsatishda qavslar ichida emas, tabiiy shaklda yozilsin. Hulosa xar doim 9-bo'lim oxirida bulishi shart.
====================================================================================

Qat'iy qoidalar:
• Har doim belgilangan tilgan tarjima qilishinshi shart. 
• Ovoz ohangi — professional bank ekspertizasi
• Javob juda ham sifatli va juda ham tez bo'lishi lozim. Foydalanuvchi kutib qolmasligi uchun!
• Textlar toza natija bo'lishi kerkak.
• Chiquvchi text malumotlar ham tartibli kerakli joylar alohida korsatilgan holatda bolishi kerak.
• "language" fieldida so'ralgan tilda javob generatsiya qilish lozim!
• Kredit  tavsiya berish yoki bermaslik tavsiyasi <strong> tag ichida bulishi shart
• Agar {conclusion_field} = \"POSITIVE\" bo‘lsa, xulosada faqat ijobiy/kuchli jihatlar va kreditni qo‘llab-quvvatlovchi dalillar yoritilsin; salbiy tomonlar, kamchiliklar va risklar tilga olinmasin.
• Agar {conclusion_field} = \"NEGATIVE\" bo‘lsa, xulosada ko'proq salbiy tomonlar/risklar va kreditni rad etishga olib keladigan dalillar yoritilsin; ijobiy tomonlar va kuchli jihatlar kamaytirilsin olinmasin.
• Agar {conclusion_field} = NEUTRAL bo‘lsa, mavjud umumiy logika saqlanib qolsin, standart qoidalar asosida shakllantirilsin.
"""

    # 3. Call LLM
    # We want structured JSON output: { "uz": "...", "cyrl": "...", "ru": "...", "en": "..." }
    # langchain_openai ChatOpenAI supports binding response_format if using correct model params, 
    # but here we can just instruct it in the system prompt or use tool calling.
    # To keep it simple and robust with gpt-4o, we can use valid JSON mode if supported or just prompt instructions.
    
    # Let's enforce JSON output structure in the prompt or use with_structured_output if available (langchain 0.1+)
    # Given the import `from langchain.agents import create_agent`, we might be on an older version or using agents.
    # Let's just ask for JSON in the user message.
    
    user_prompt = f"""Process the following data and return the result as a raw JSON object with keys: "uz", "cyrl", "ru", "en".
Each key should contain the HTML string for the expert conclusion in that language.
Do not wrap the JSON in markdown code blocks.

Data:
{input_json_str}
"""

    # We can try to use standard invoke.
    try:
        # Check if we can use json_mode
        llm_json = llm.bind(response_format={"type": "json_object"})
        result = llm_json.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        ai_content = result.content
    except Exception as e:
        # Fallback if bind not supported
        result = llm.invoke([
            {"role": "system", "content": system_prompt + "\nRETURN JSON OBJECT ONLY."},
            {"role": "user", "content": user_prompt}
        ])
        ai_content = result.content

    # 4. Parse JSON
    try:
        # Clean potential markdown wrapping
        cleaned_content = ai_content.strip()
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content[7:]
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content[:-3]
        
        ai_conclusions = json.loads(cleaned_content)
    except Exception as e:
        print(f"Error parse JSON from AI: {e}")
        # Fallback: assign same content to all or error
        fallback_text = str(ai_content)
        ai_conclusions = {
            "uz": fallback_text,
            "cyrl": fallback_text,
            "ru": fallback_text,
            "en": fallback_text
        }

    # 5. Build HTML for each language
    html_results = {}
    for lang in ["uz", "cyrl", "ru", "en"]:
        conclusion_text = ai_conclusions.get(lang, "---")
        raw_html = build_html_for_lang(raw_payload, conclusion_text, lang)
        # Remove newlines for cleaner JSON string
        html_results[lang] = raw_html.replace("\n", " ").strip()

    end_time = time.time()
    
    # 6. Return
    return {
        "html": html_results,
        "time": end_time - start_time
    }
