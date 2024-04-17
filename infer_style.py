import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from ip_adapter import IPAdapterXL


# Does work w turbo
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
#base_model_path = "stabilityai/sdxl-turbo"
image_encoder_path = "sdxl_models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl.bin"
device = "cuda"

# load SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)

# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.0.attentions.1"] for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"])

image = "./assets/experiments/bear.webp"
image = Image.open(image)
image.resize((512, 512))
image2 = "./assets/experiments/caturai.webp"
image2 = Image.open(image2)
image2.resize((512, 512))
image3 = "./assets/experiments/bird.webp"
image3 = Image.open(image3)
image3.resize((512, 512))

# generate image
images = ip_model.generate(pil_image=[image, image2, image3],
                           pil_image_weights=[1,1,1],
                           prompt=["a handsome bunny, masterpiece, best quality, high quality"],
                           prompt_weights=None,
                           negative_prompt= ["text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"],
                           scale=1.0,
                           guidance_scale=5,
                           num_samples=1,
                           num_inference_steps=30, 
                           seed=42,
                           neg_content_prompt="flower, rose, plant",
                           neg_content_scale=0.35,
                          )

images[0].save("assets/experiments/result16.png")