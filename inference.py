from kandinsky2 import get_kandinsky2
model = get_kandinsky2('cuda:0', task_type='text2img', model_version='2.2')
images = model.generate_text2img(
    "red cat, 4k photo",
    decoder_steps=50,
    batch_size=1,
    h=1024,
    w=768,
)

print(images)