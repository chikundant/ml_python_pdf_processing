from fastapi import APIRouter

router = APIRouter()


@router.get(path='/')
def get_all_tasks():
    return {"hi there"}
