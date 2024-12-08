from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
from gradio_client import Client
from apscheduler.schedulers.background import BackgroundScheduler
from io import BytesIO
import os
import sys

app = FastAPI()

def restart_server():
    print("Restarting server...")
    os.execv(sys.executable, ['uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8080', '--reload'])

scheduler = BackgroundScheduler()
scheduler.add_job(restart_server, 'interval', minutes=3)
scheduler.start()

@app.get("/")
def read_root():
    return {"status": "Server is running. Try /kivotos?text="}

def generate_image_with_kivotos(prompt: str) -> BytesIO:
    client = Client("Linaqruf/kivotos-xl-2.0")
    result = client.predict(
        prompt=prompt,
        negative_prompt="nsfw, (low quality, worst quality:1.2), 3d, watermark, signature, ugly, poorly drawn",
        seed=0,
        custom_width=1024,
        custom_height=1024,
        guidance_scale=7,
        num_inference_steps=28,
        sampler="Euler a",
        aspect_ratio_selector="896 x 1152",
        use_upscaler=False,
        upscaler_strength=0.55,
        upscale_by=1.5,
        add_quality_tags=True,
        api_name="/run"
    )

    image_path = result[0][0].get('image')

    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as file:
            image_bytes = file.read()
        return BytesIO(image_bytes)
    else:
        raise HTTPException(status_code=500, detail="Image not found or invalid path")

@app.get("/kivotos")
def kivotos_endpoint(text: str = Query(...)):
    try:
        image_data = generate_image_with_kivotos(text)
        return StreamingResponse(image_data, media_type="image/png", headers={"Content-Disposition": "inline; filename=output.png"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    finally:
        scheduler.shutdown()
