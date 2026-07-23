from larpscaler import LarpScaler


upscaler = LarpScaler.from_pretrained("VladimirM388/larpscaler-v2-bf16")
image = upscaler.upscale(
    "input.png",
    scale=4,
    steps=1,
    noise_level=0.35,
    guidance_scale=1.0,
)
image.save("output-fast.png")
