# SmolVLA inference server for closed-loop evaluation (run on the host in an environment with LeRobot installed)
#
#   python scripts/policy_server.py --ckpt outputs/train/<run>/checkpoints/last/pretrained_model
#
# POST /act  {"images": {"<camera>": absolute path, ...}, "state": [...], "task": "..." (optional)}
#   -> {"actions": [[speed command, steering command, lift command] x chunk], "latency": s}
# Images are passed by file path: the simulator saves JPEGs and sends their paths
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True, help="checkpoints/<step>/pretrained_model directory")
p.add_argument("--port", type=int, default=8001)
p.add_argument("--task", default="pick up the pallet, deliver it to the drop zone and come back")
p.add_argument("--seed", type=int, default=0, help="seed for the initial flow-matching noise")
args = p.parse_args()

from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

policy = SmolVLAPolicy.from_pretrained(args.ckpt)
policy.to("cuda").eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=args.ckpt)
torch.manual_seed(args.seed)
print(f"[server] loaded {args.ckpt}, chunk {policy.config.chunk_size}, inputs {list(policy.config.input_features)}", flush=True)


def to_tensor(path):
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(im).permute(2, 0, 1).contiguous()          # (3, H, W), 0-1 (same format as the training data)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/act":
            self.send_error(404)
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            obs = {f"observation.images.{k}": to_tensor(v) for k, v in req["images"].items()}
            obs["observation.state"] = torch.tensor(req["state"], dtype=torch.float32)
            obs["task"] = req.get("task", args.task)
            t = time.time()
            with torch.no_grad():
                batch = pre(obs)
                chunk = policy.predict_action_chunk(batch)          # (1, chunk, action dim), normalized values
                chunk = post(chunk)
            dt = time.time() - t
            acts = chunk[0, :, :3].float().cpu().numpy().tolist()
            body = json.dumps({"actions": acts, "latency": dt}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            msg = json.dumps({"error": repr(e)}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def log_message(self, *a):
        pass


print(f"[server] ready on port {args.port}", flush=True)
HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
