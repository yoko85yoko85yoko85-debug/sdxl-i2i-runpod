import base64
import io
import math
import re
import threading
import warnings
from pathlib import PurePosixPath

import runpod
import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionXLImg2ImgPipeline
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
SIZES = {(640, 640), (768, 768), (1024, 1024), (1024, 768), (768, 1024)}
Image.MAX_IMAGE_PIXELS = 20_000_000
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
lock = threading.Lock()
loaded_lora = None
lora_failed = False


def parse_request(p):
    encoded = p.get("image_base64")
    if not isinstance(encoded, str) or not 1 <= len(encoded) <= 4_000_000:
        raise ValueError("元画像が必要です。送信画像は3MB以下にしてください。")
    raw = base64.b64decode(encoded, validate=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError("JPEG・PNG・WebP画像を選んでください。")
            image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = int(p.get("width", 1024)), int(p.get("height", 1024))
    if (width, height) not in SIZES:
        raise ValueError("未対応の出力サイズです。")
    prompt = str(p.get("prompt", "")).strip()
    negative = str(p.get("negative_prompt", ""))
    if not 1 <= len(prompt) <= 3000 or len(negative) > 3000:
        raise ValueError("プロンプトは1〜3000文字で入力してください。")
    steps, seed = int(p.get("steps", 25)), int(p.get("seed", 42))
    strength, guidance = float(p.get("strength", 0.45)), float(p.get("guidance", 6.0))
    if not 10 <= steps <= 50 or not 0 <= seed <= 2147483647:
        raise ValueError("ステップ数またはシード値を確認してください。")
    if not math.isfinite(strength) or not 0.1 <= strength <= 0.95 or not math.isfinite(guidance) or not 1 <= guidance <= 15:
        raise ValueError("変更の強さまたはCFGを確認してください。")
    lora = p.get("lora")
    if lora is not None:
        if not isinstance(lora, dict):
            raise ValueError("LoRA設定を確認してください。")
        repo, filename = lora.get("repo", ""), lora.get("filename", "")
        scale = float(lora.get("scale", 0.7))
        if not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("LoRAの公開Hugging FaceリポジトリIDを入力してください。")
        if not isinstance(filename, str) or len(filename) > 200 or not re.fullmatch(r"[A-Za-z0-9_./-]+\.safetensors", filename) or filename.startswith("/") or ".." in PurePosixPath(filename).parts:
            raise ValueError("LoRAはリポジトリ内の.safetensorsファイルを指定してください。")
        if not math.isfinite(scale) or not 0 < scale <= 1.5:
            raise ValueError("LoRA強度は0より大きく1.5以下にしてください。")
        lora = {"repo": repo, "filename": filename, "scale": scale}
    image = ImageOps.pad(image, (width, height), method=Image.Resampling.LANCZOS, color="black")
    return image, prompt, negative, width, height, steps, seed, strength, guidance, lora


def configure_lora(lora):
    global loaded_lora, lora_failed
    if lora_failed:
        raise RuntimeError("LoRA読み込みに失敗した状態です。RunPodワーカーを再起動してください。")
    if lora is None:
        if loaded_lora:
            pipe.disable_lora()
        return None
    identity = (lora["repo"], lora["filename"])
    mutating = False
    try:
        if loaded_lora is None or loaded_lora[:2] != identity:
            # Resolve a public repository to an immutable revision; do not accept URLs or pickle weights.
            info = HfApi().model_info(lora["repo"], files_metadata=True, token=False)
            file_info = next((item for item in info.siblings if item.rfilename == lora["filename"]), None)
            if file_info is None or file_info.size is None or file_info.size > 1024 * 1024 * 1024:
                raise ValueError("LoRAファイルが見つからないか、1GBを超えています。")
            path = hf_hub_download(lora["repo"], filename=lora["filename"], revision=info.sha, token=False)
            state = load_file(path, device="cpu")
            if not state or not any("lora" in key.lower() for key in state):
                raise ValueError("LoRA形式を確認できません。SDXL用の対応ファイルを選んでください。")
            mutating = True
            if loaded_lora:
                pipe.unload_lora_weights()
                loaded_lora = None
            pipe.load_lora_weights(state, adapter_name="image_style")
            loaded_lora = (*identity, info.sha)
        pipe.enable_lora()
        pipe.set_adapters(["image_style"], adapter_weights=[lora["scale"]])
        return {**lora, "revision": loaded_lora[2]}
    except Exception:
        if mutating:
            lora_failed = True
        raise


def handler(job):
    try:
        values = parse_request(job.get("input", {}))
    except Exception as error:
        return {"error": str(error)}
    image, prompt, negative, width, height, steps, seed, strength, guidance, lora = values
    with lock, torch.inference_mode():
        used_lora = configure_lora(lora)
        result = pipe(
            image=image, prompt=prompt, negative_prompt=negative,
            strength=strength, guidance_scale=guidance, num_inference_steps=steps,
            generator=torch.Generator("cpu").manual_seed(seed), num_images_per_prompt=1,
        ).images[0]
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    return {
        "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "mime_type": "image/png", "mode": "sdxl_i2i", "model": MODEL,
        "width": width, "height": height, "seed": seed, "lora": used_lora,
    }


runpod.serverless.start({"handler": handler})
