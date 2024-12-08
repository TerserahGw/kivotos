from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
from gradio_client import Client
from io import BytesIO

app = FastAPI()

def generate_image_with_kivotos(prompt: str) -> BytesIO:
    x_ip_token = "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"  
    if not x_ip_token:
        raise HTTPException(status_code=400, detail="X-IP-Token tidak ditemukan. Pastikan token sudah disediakan.")

    client = Client("gradio/text-to-image", headers={"X-IP-Token": x_ip_token})
    
    print(f"Generating image for prompt: {prompt}")

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

    print("Result from ZeroGPU API:", result)

    image_path = result.get("image")
    if image_path:
        with open(image_path, "rb") as file:
            image_bytes = file.read()
        return BytesIO(image_bytes)
    else:
        raise HTTPException(status_code=500, detail="Image not found or invalid response")

@app.get("/kivotos")
def kivotos_endpoint(text: str = Query(...)):
    try:
        image_data = generate_image_with_kivotos(text)
        return StreamingResponse(image_data, media_type="image/png", headers={"Content-Disposition": "inline; filename=output.png"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = 2020
    print(f"Server is starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
