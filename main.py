from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
from gradio_client import Client
from io import BytesIO
import os
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.connectionpool import HTTPConnectionPool
from urllib3.util.ssl_ import create_urllib3_context
from requests.packages.urllib3.poolmanager import PoolManager

app = FastAPI()

class HTTPConnectAdapter(HTTPAdapter):
    def __init__(self, proxy_url, *args, **kwargs):
        self.proxy_url = proxy_url
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        self.poolmanager = PoolManager(
            *args, proxy_url=self.proxy_url, ssl_context=context, **kwargs
        )

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

def setup_http_proxy(proxy_url):
    session = requests.Session()
    adapter = HTTPConnectAdapter(proxy_url)
    session.mount("https://", adapter)
    return session

def generate_image_with_kivotos(prompt: str) -> BytesIO:
    retries = 5
    for _ in range(retries):
        random_proxy = get_random_proxy_from_file()
        proxy_url = f"http://{random_proxy}"
        session = setup_http_proxy(proxy_url)
        
        print(f"Using HTTP CONNECT Proxy: {proxy_url}")
        
        try:
            response = session.get("https://huggingface.co/api/spaces/Linaqruf/kivotos-xl-2.0")
            print(f"Proxy Response: {response.status_code}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error using proxy: {str(e)}")
        
        break

    print(f"Generating image with prompt: {prompt}")
    client = Client("Linaqruf/kivotos-xl-2.0", session=session)
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
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
