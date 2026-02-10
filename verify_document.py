
import sys
import os
import json

# Add current directory to path
sys.path.append(os.getcwd())

from document import build_html_for_lang, DICT

# Dummy payload
payload = {
    "body": {
        "language": "uz",
        "company_info": {
            "data": {
                "name": "Test Company",
                "tin": "123456789",
                "founders": [
                    {"founderIndividual": {"firstName": "John", "lastName": "Doe"}, "sharePercent": 50},
                    {"founderLegal": {"name": "Other Corp"}, "sharePercent": 50}
                ]
            }
        },
        "applicationInfo": {
            "applicationData": {
                "requestedAmount": 100000,
                "loanTermMonths": 12,
                "currency": "USD"
            },
            "collateralData": [
                {
                    "estimatedValue": 200000,
                    "collateralType": "REAL_ESTATE",
                    "yurTaxObjectData": {
                        "name": "Office",
                        "totla_area": 100
                    }
                }
            ]
        },
        "conclusion": "POSITIVE"
    }
}

ai_conclusion = "<div>AI Conclusion HTML</div>"

print("Testing HTML generation for all languages...")
for lang in ["uz", "cyrl", "ru", "en"]:
    print(f"Testing {lang}...")
    try:
        html = build_html_for_lang(payload, ai_conclusion, lang)
        if "AI Conclusion HTML" not in html:
            print(f"Error: AI conclusion not found in {lang} output")
        if "Test Company" not in html:
            print(f"Error: Company name not found in {lang} output")
        
        # Check specific language strings
        l_dict = DICT.get(lang)
        
        if "\n" in html:
            print(f"Error: Newline found in {lang} output")
        else:
            print(f"Success {lang} (No newlines)")

    except Exception as e:
        print(f"Exception in {lang}: {e}")
        import traceback
        traceback.print_exc()

print("Verification complete.")
