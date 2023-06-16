import torch
from diffusers import StableDiffusionPipeline
torch.backends.cudnn.benchmark = True
# pipe = StableDiffusionPipeline.from_pretrained("IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1", torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained("IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1", torch_dtype=torch.float32)
pipe.to('cuda')

prompt = '飞流直下三千尺，油画'
image = pipe(prompt, guidance_scale=7.5).images[0]  
image.save("飞流.png")
