from dotenv import load_dotenv
from huggingface_hub import snapshot_download
import os

load_dotenv()

vith_out_dir = "checkpoints/sam-3d-body-vith"
os.makedirs(vith_out_dir, exist_ok=True)

snapshot_download(
    repo_id="facebook/sam-3d-body-vith",
    local_dir=vith_out_dir,
    token=os.getenv("HF_TOKEN")
)

dinov3_out_dir = "checkpoints/sam-3d-body-dinov3"
os.makedirs(dinov3_out_dir, exist_ok=True)

snapshot_download(
    repo_id="facebook/sam-3d-body-dinov3",
    local_dir=dinov3_out_dir,
    token=os.getenv("HF_TOKEN")
)
