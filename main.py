from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path


app = FastAPI()


@app.get("/get_image")
async def get_image():
    image_path = Path(
        "./Outputs/TOM/002599_1.jpg")
    if not image_path.is_file():
        return {"error": "Image not found on the server"}
    return FileResponse(image_path)


# @app.get("/get_image")
# async def get_image():
#     image_path = Path("gfglogo.jpg")
#     if not image_path.is_file():
#         return {"error": "Image not found on the server"}
#     return FileResponse(image_path)


# uvicorn main:app --reload
