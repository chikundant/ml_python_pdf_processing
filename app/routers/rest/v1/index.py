from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

from fastapi.responses import FileResponse

@router.get("/", response_class=HTMLResponse)
async def web_ui():
    return FileResponse('static/index.html')