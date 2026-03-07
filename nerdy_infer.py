#!/usr/bin/env python
"""
nerdy_infer.py  –  Standalone CLI wrapper for nERdy+ ER segmentation.

Called by nERdyEnhance.m via MATLAB system():
    python nerdy_infer.py --input <in.tif> --output <out.tif>
                          [--model <weights.pth>]
                          [--nerdy_dir <path/to/nERdy+/>]
                          [--device auto|cuda|cpu]
                          [--no_postproc]
                          [--threshold <0-1>]

Outputs a binary mask (uint8: 0 = background, 255 = ER).

Exit codes: 0 = success, 1 = error (message on stderr).
"""

import argparse
import os
import sys


def resolve_nerdy_dir(nerdy_dir_arg):
    """Locate the nERdy+ directory containing model.py."""
    if nerdy_dir_arg:
        return nerdy_dir_arg
    # Default: nERdy/ subfolder relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, 'nERdy', 'nERdy+')
    if os.path.isfile(os.path.join(candidate, 'model.py')):
        return candidate
    raise FileNotFoundError(
        f"Cannot find nERdy+ directory. Expected: {candidate}\n"
        "Pass --nerdy_dir explicitly."
    )


def resolve_model_path(model_arg, nerdy_dir):
    """Return path to .pth weights file."""
    if model_arg:
        return model_arg
    default = os.path.join(nerdy_dir, 'NNet_groupy_p4m_v2_VecAdam.pth')
    if not os.path.isfile(default):
        raise FileNotFoundError(
            f"Pre-trained weights not found: {default}\n"
            "Pass --model explicitly."
        )
    return default


def load_image(path):
    """Load a TIFF or PNG image as float32 numpy array in [0, 1]."""
    from PIL import Image
    import numpy as np
    img = Image.open(path)
    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')
    arr = np.array(img, dtype=np.float32)
    arr_max = arr.max()
    if arr_max > 0:
        arr = arr / arr_max
    return arr


def run_nerdy(arr, nerdy_dir, model_path, device_str):
    """Run nERdy+ inference. Returns float32 probability map in [0, 1]."""
    import sys
    sys.path.insert(0, nerdy_dir)

    import torch
    from torchvision import transforms
    from model import D4nERdy

    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    # Load model
    model = D4nERdy(in_channels=1, out_channels=1)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()

    # Preprocess
    transform = transforms.Compose([transforms.ToTensor()])
    from PIL import Image
    import numpy as np
    pil_img = Image.fromarray((arr * 255).astype(np.uint8))
    tensor = transform(pil_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).cpu().squeeze().numpy()

    # Normalise to [0, 1]
    prob_min, prob_max = prob.min(), prob.max()
    if prob_max > prob_min:
        prob = (prob - prob_min) / (prob_max - prob_min)
    return prob


def postprocess(prob, threshold=None):
    """
    Convert probability map to binary mask (0 or 255).
    If threshold is None, use the nERdy+ default postprocessing
    (rolling-ball background subtraction + Otsu thresholding).
    """
    import sys as _sys
    import numpy as np

    if threshold is not None:
        mask = (prob >= threshold).astype(np.uint8) * 255
        return mask

    # nERdy+ default postprocessing
    # Ensure postprocessing module is importable (nerdy_dir already in sys.path)
    from postprocessing import postprocessing
    seg = postprocessing(prob)        # returns array with values 0 or 255
    return seg.astype(np.uint8)


def save_image(mask, path):
    """Save uint8 binary mask as TIFF or PNG."""
    from PIL import Image
    img = Image.fromarray(mask)
    img.save(path)


def main():
    parser = argparse.ArgumentParser(
        description='nERdy+ ER segmentation — standalone inference wrapper'
    )
    parser.add_argument('--input',     '-i', required=True,
                        help='Input image path (TIFF or PNG)')
    parser.add_argument('--output',    '-o', required=True,
                        help='Output mask path (TIFF or PNG)')
    parser.add_argument('--model',     '-m', default=None,
                        help='Path to .pth weight file (default: auto-detect)')
    parser.add_argument('--nerdy_dir', default=None,
                        help='Path to nERdy+ source directory (default: auto-detect)')
    parser.add_argument('--device',    default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Compute device (default: auto)')
    parser.add_argument('--no_postproc', action='store_true',
                        help='Output raw probability map [0,255] instead of binary mask')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Fixed threshold in [0,1] for binarisation (default: Otsu)')
    args = parser.parse_args()

    try:
        nerdy_dir  = resolve_nerdy_dir(args.nerdy_dir)
        model_path = resolve_model_path(args.model, nerdy_dir)

        if not os.path.isfile(args.input):
            raise FileNotFoundError(f"Input not found: {args.input}")

        arr  = load_image(args.input)
        prob = run_nerdy(arr, nerdy_dir, model_path, args.device)

        if args.no_postproc:
            import numpy as np
            out = (prob * 255).astype('uint8')
        else:
            out = postprocess(prob, threshold=args.threshold)

        save_image(out, args.output)

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()
