from pathlib import Path

from larpscaler import LarpScaler


MODEL_ID = "VladimirM388/larpscaler-v2-bf16"
INPUT = Path("input.png")
OUTPUT = Path("output.png")


upscaler = LarpScaler.from_pretrained(MODEL_ID)
image = upscaler.upscale(
    INPUT,
    scale=4,
    steps=1,
    noise_level=0.85,
    guidance_scale=4.5,
    seed=1234,
)
image.save(OUTPUT)
print(f"Saved {image.size} to {OUTPUT}")
