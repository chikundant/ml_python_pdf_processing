from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def web_ui():
    return FileResponse("static/index.html")
