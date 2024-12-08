from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
from gradio_client import Client
from apscheduler.schedulers.background import BackgroundScheduler
from io import BytesIO
import os
import random
import signal
import sys

app = FastAPI()

def restart_server():
    print("Restarting server...")
    os.kill(os.getpid(), signal.SIGINT)

scheduler = BackgroundScheduler()
scheduler.add_job(restart_server, 'interval', minutes=5)
scheduler.start()

def get_random_proxy_from_file(file_path="all.txt"):
    try:
        with open(file_path, "r") as file:
            proxies = file.readlines()
            proxies = [proxy.strip() for proxy in proxies if proxy.strip()]
            if not proxies:
                raise HTTPException(status_code=500, detail="No proxies found in the file")
            selected_proxy = random.choice(proxies)
            print(f"Selected proxy: {selected_proxy}")
            return selected_proxy
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading proxy file: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "Server is running coba /kivotos?text="}

def generate_image_with_kivotos(prompt: str) -> BytesIO:
    retries = 5
    for _ in range(retries):
        random_proxy = get_random_proxy_from_file()
        proxies = {"http": f"http://{random_proxy}"}
        os.environ["http_proxy"] = proxies["http"]
        print(f"Using proxy: {proxies}")
        break

    print(f"Generating image with prompt: {prompt}")
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

    print(f"Result Kivotos: {result}")

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
    port = int(os.getenv("PORT", 2020))
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
    finally:
        scheduler.shutdown()
