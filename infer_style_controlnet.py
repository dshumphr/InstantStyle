import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

import cv2
from PIL import Image

from ip_adapter import IPAdapterXL

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
#base_model_path = "stabilityai/sdxl-turbo"
image_encoder_path = "sdxl_models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl.bin"
device = "cuda"

controlnet_path = "diffusers/controlnet-canny-sdxl-1.0"
controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False, torch_dtype=torch.float16).to(device)

# load SDXL pipeline
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
pipe.enable_vae_tiling()

# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.0.attentions.1"] for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])

# style image
image = "./assets/iogirl.jpg"
image = Image.open(image)
image.resize((512, 512))

# control image
input_image = cv2.imread("./assets/iogirl.jpg")
detected_map = cv2.Canny(input_image, 85, 255)
canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))

# experimenting with adding two versions of same image
imagedark = "./assets/dark.png"
imagedark = Image.open(imagedark)
imagedark.resize((512, 512))
imagelight = "./assets/light.png"
imagelight = Image.open(imagelight)
imagelight.resize((512, 512))

# generate image
images = ip_model.generate(pil_image=[image,imagedark, imagelight],
                           pil_image_weights=[1, -0.5, 0.5],
                           prompt=["blonde woman smiling, realistic photo"],
                           negative_prompt=["text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"],
                           scale=1.0,
                           guidance_scale=5,
                           num_samples=1,
                           num_inference_steps=30, 
                           seed=42,
                           image=canny_map,
                           controlnet_conditioning_scale=0.8,
                          )

images[0].save("ioresult14basenot2.png")