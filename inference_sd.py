import os

from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
import torch
from transformers import CLIPVisionModelWithProjection
from diffusers.models import UNet2DConditionModel

torch.manual_seed(42)

wPATH = "generated_images"
os.makedirs(wPATH,exist_ok=True)

def init_model(gpu_id: int = 0):
    device = torch.device(f'cuda:{gpu_id}')
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        'kandinsky-community/kandinsky-2-2-prior',
        subfolder='image_encoder'
    ).half().to(device)

    unet = UNet2DConditionModel.from_pretrained(
        'kandinsky-community/kandinsky-2-2-decoder',
        subfolder='unet'
    ).half().to(device)

    prior = KandinskyV22PriorPipeline.from_pretrained(
        'kandinsky-community/kandinsky-2-2-prior',
        image_encoder=image_encoder,
        torch_dtype=torch.float16
    ).to(device)

    decoder = KandinskyV22Pipeline.from_pretrained(
        'kandinsky-community/kandinsky-2-2-decoder',
        unet=unet,
        torch_dtype=torch.float16
    ).to(device)

    return prior, decoder


def main():
    prior, decoder = init_model()
    negative_prior_prompt = 'lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature'
    negative_emb = prior(
        prompt=negative_prior_prompt,
        num_inference_steps=25,
        num_images_per_prompt=1
    )

    cnt = 0
    while True:
        prompt = input("Input prompt:")
        img_emb = prior(
            prompt=prompt,
            num_inference_steps=25,
            num_images_per_prompt=1
        )

        images = decoder(
            image_embeds=img_emb.image_embeds,
            negative_image_embeds=negative_emb.image_embeds,
            num_inference_steps=75,
            height=512,
            width=512)

        cnt += 1
        images.images[0].save(f"{wPATH}/image_{cnt}.png")
        print("Save image ...")


if __name__ == '__main__':
    main()
