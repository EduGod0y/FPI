from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
import riffusion 
import torch
from PIL import Image
import threading
import typing as T
from spectro import wav_bytes_from_spectrogram_image
from diffusers import DiffusionPipeline



def get_pipeline():
    pipeline = DiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1")
    return pipeline


def run_txt2img(
    pipeline, 
    prompt: str,
    num_inference_steps: int=25,
    guidance: float=7,
    negative_prompt: str='',
    seed: int=42,
    width: int=512,
    height: int=512,
    device: str = "cpu",
) -> Image.Image:
   

    generator_device = "cpu" if device.lower().startswith("mps") else device
    generator = torch.Generator(device=generator_device).manual_seed(seed)

    output = pipeline(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance,
        negative_prompt=negative_prompt or None,
        generator=generator,
        width=width,
        height=height,
    )

    return output["images"][0]

def spec2audio(spec, text):
    print(spec)
    wav = wav_bytes_from_spectrogram_image(spec)
    with open(text+'.wav', "wb") as f:
        f.write(wav[0].getbuffer())
    return 


if __name__ == '__main__':

    text= "bb king guitar solo with a jazzy saxophone"
    pipeline = get_pipeline()
    img  = run_txt2img(pipeline,text)
    spec2audio(img,text)
    
    img.save("f {text}.jpeg")
    
