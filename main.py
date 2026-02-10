import os
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn

import document
import uy_joy
import avto
import forma1
import forma2
import xmed

# Load environment 
load_dotenv()

app = FastAPI(title="Multi-Agent API")

# Register routes from each module
app.include_router(document.router)
app.include_router(uy_joy.router)
app.include_router(avto.router)
app.include_router(forma1.router)
app.include_router(forma2.router)
app.include_router(xmed.router)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Unified API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
