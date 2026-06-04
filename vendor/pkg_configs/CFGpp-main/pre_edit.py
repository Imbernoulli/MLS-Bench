"""Pre-edit operations for CFGpp-main package.

Adds CLIP score and Inception Score calculation to text_to_img.py.
"""

# Add all imports at the top
_ALL_IMPORTS = """\
import argparse
from pathlib import Path

from munch import munchify
from torchvision.utils import save_image
import torch
import clip

from latent_diffusion import get_solver
from latent_sdxl import get_solver as get_solver_sdxl
from utils.callback_util import ComposeCallback
from utils.log_util import create_workdir, set_seed
"""

# Replace the save and add all metric calculations
_ALL_METRICS = """\

    save_image(result, args.workdir.joinpath(f'result/generated.png'), normalize=True)

    # Calculate all metrics
    device = torch.device(args.device)
    img_path = args.workdir.joinpath(f'result/generated.png')

    # 1. CLIP score
    clip_model, preprocess = clip.load("ViT-B/32", device=device, download_root="/opt/model_weights")
    from PIL import Image
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    text = clip.tokenize([args.prompt]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        clip_score = (image_features @ text_features.T).item()

    # 2. Inception Score — compute batch IS only on the last prompt
    #    Load inception model once and process all generated images in one pass.
    is_last_prompt = str(args.workdir).rstrip('/').endswith('_99')
    inception_score = -1.0

    if is_last_prompt:
        try:
            from torchvision.models import inception_v3
            from torchvision import transforms
            import torch.nn.functional as F
            import numpy as np
            import glob

            inception_model = inception_v3(pretrained=False, transform_input=False)
            inception_model.load_state_dict(torch.load("/opt/model_weights/inception_v3_google.pth", map_location=device))
            inception_model = inception_model.to(device)
            inception_model.eval()

            inception_transform = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # Collect all generated images from sibling workdirs
            parent_dir = args.workdir.parent
            image_paths = sorted(glob.glob(str(parent_dir / '*/result/generated.png')))

            if image_paths:
                all_probs = []
                for p in image_paths:
                    image_pil = Image.open(p).convert('RGB')
                    inception_input = inception_transform(image_pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = inception_model(inception_input)
                        probs = F.softmax(logits, dim=1).cpu().numpy()
                    all_probs.append(probs)

                all_probs = np.concatenate(all_probs, axis=0)  # (N, 1000)
                py = np.mean(all_probs, axis=0)
                kl_div = all_probs * (np.log(all_probs + 1e-10) - np.log(py + 1e-10))
                inception_score = float(np.exp(np.mean(np.sum(kl_div, axis=1))))
        except Exception as e:
            print(f"Warning: Inception Score calculation failed: {e}", flush=True)

    if is_last_prompt and inception_score >= 0:
        print(f"GENERATION_METRICS method={args.method} cfg_guidance={args.cfg_guidance} NFE={args.NFE} seed={args.seed} clip_score={clip_score:.4f} inception_score={inception_score:.4f}", flush=True)
    else:
        print(f"GENERATION_METRICS method={args.method} cfg_guidance={args.cfg_guidance} NFE={args.NFE} seed={args.seed} clip_score={clip_score:.4f}", flush=True)
"""

_KDIFF_X_TO_DENOISED_PATCH = """    def kdiffusion_x_to_denoised(self, x, sigma, uc, c, cfg_guidance, t):
        xc = self.calculate_input(x, sigma)
        noise_uc, noise_c = self.predict_noise(xc, t, uc, c)
        noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
        denoised = self.calculate_denoised(x, noise_pred, sigma)
        # Preserve tuple-unpack compatibility while preventing sampler code from
        # accessing the pure unconditional denoised prediction.
        return denoised, denoised
"""

OPS = [
    # Bottom-to-top: higher line numbers first so earlier ops don't shift later ones
    {
        "op": "replace",
        "file": "CFGpp-main/examples/text_to_img.py",
        "start_line": 56,
        "end_line": 56,
        "content": _ALL_METRICS,
    },
    {
        "op": "replace",
        "file": "CFGpp-main/examples/text_to_img.py",
        "start_line": 1,
        "end_line": 10,
        "content": _ALL_IMPORTS,
    },
    {
        "op": "replace",
        "file": "CFGpp-main/latent_diffusion.py",
        "start_line": 235,
        "end_line": 241,
        "content": _KDIFF_X_TO_DENOISED_PATCH,
    },
]
