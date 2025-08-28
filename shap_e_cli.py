import argparse, os, torch
from pathlib import Path

from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.diffusion.sample import sample_latents
from shap_e.util.notebooks import decode_latent_mesh

def main():
    p = argparse.ArgumentParser(description="Shape-E text→mesh CLI")
    p.add_argument("--prompt", required=True, help="Text prompt")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--guidance-scale", type=float, default=15.0)
    p.add_argument("--karras-steps", type=int, default=64)
    p.add_argument("--sigma-min", type=float, default=1e-3)
    p.add_argument("--sigma-max", type=float, default=160.0)
    p.add_argument("--seed", type=int)
    p.add_argument("--out", default="/out", help="Output dir (mounted volume)")
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    # Load models once
    xm = load_model("transmitter", device=device)
    text_model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))

    # Sample latents
    latents = sample_latents(
        batch_size=args.batch_size,
        model=text_model,
        diffusion=diffusion,
        guidance_scale=args.guidance_scale,
        model_kwargs=dict(texts=[args.prompt] * args.batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=(device.type == "cuda"),
        use_karras=True,
        karras_steps=args.karras_steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        s_churn=0,
        device=device,
    )

    # Decode to meshes and save .obj
    for i, latent in enumerate(latents):
        mesh = decode_latent_mesh(xm, latent).tri_mesh()
        obj_path = out_dir / f"shape_e_{i}.obj"
        with obj_path.open("w") as f:
            f.write("# Shape-E OBJ\n")
            for v in mesh.verts:
                f.write(f"v {float(v[0])} {float(v[1])} {float(v[2])}\n")
            for face in mesh.faces:
                f.write(f"f {int(face[0])+1} {int(face[1])+1} {int(face[2])+1}\n")
        print("Wrote:", obj_path)

if __name__ == "__main__":
    main()
