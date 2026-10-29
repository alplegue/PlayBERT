import argparse
import torch
from model import Decoder_Only_Transformer, generate

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="lorca.pt")
parser.add_argument("--prompt", default="\n")
parser.add_argument("--tokens", type=int, default=500)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top_k", type=int, default=20)
args = parser.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"

ckpt = torch.load(args.ckpt, map_location=device)
model = Decoder_Only_Transformer(**ckpt["config"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

chars = ckpt["chars"]
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

prompt = "".join(c for c in args.prompt if c in stoi) or "\n"   # gets rid of characters out of our voacb
idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)

print(prompt, end="", flush=True)
for tok in generate_stream(model, idx, args.tokens, temperature=args.temperature, top_k=args.top_k):
    print(itos[tok], end="", flush=True)
    time.sleep(0.02)                          # opcional: efecto "máquina de escribir"
print()