#!/usr/bin/env python3
"""
15b_interplm_sae.py - InterPLM's pre-trained SAE, which is what research/05 always claimed
                      Pillar 2 used and what src/15_sae_fhs.py never actually loaded.

Why a second script rather than a patch to 15
---------------------------------------------
src/15_sae_fhs.py tries `import interplm`, fails (the package is not installed and is not
required), and silently trains a 4096-dim linear SAE on the panel instead. Three defects
followed from that, all recorded in docs/DATA_CORRECTIONS.md (2026-09-18, third entry):
research/05 described a metric that was never built, the stored FHS-FSI correlation was
paired against a since-superseded FSI artifact, and the fallback trains without a seed so
its own output is not reproducible.

15 is left alone as the record of what was published. This script is the honest version.

What changes, concretely
------------------------
  * The SAE is InterPLM's, loaded straight from the `.pt` state dict on the Hub. No
    `interplm` package needed: the checkpoints are plain OrderedDicts holding
    `encoder.weight` (10240, 1280), `encoder.bias`, `decoder.weight`, and a pre-encoder
    `bias` (1280,). 10240 features, not 4096.
  * Therefore no training, therefore deterministic. Same inputs give the same numbers on
    every run, which the fallback could not promise.
  * InterPLM publishes SAEs for ESM-2 650M at layers 1, 9, 18, 24, 30 and 33, so the layer
    is a parameter rather than a hardcoded 33.

Why layer 18, and the cost of that choice
-----------------------------------------
docs/MECHANISM_GENERALIZATION.md §9.5 found depth is the one axis in §9 that moves the
answer: layer 12 beats the final layer by 16.4 points, and beta-lactamase collapses to
exactly 0.0% recovery at layers 12, 18 and 24. InterPLM has no layer-12 SAE, so 18 is the
shallowest layer where that collapse is total AND interpretable features exist.

🔴 But the SAE's own fidelity degrades with depth, measured here on one sequence:

    layer    1      9     18     24     30     33
    rel.err  0.006  0.064  0.133  0.228  0.256  0.221

At layer 18 the SAE reconstructs 87% of the activation it is decomposing. So a negative
result at this layer -- "beta-lactamase has no distinguishing features here" -- cannot be
separated from the 13% the SAE fails to capture. That is why layer 1 (0.6% error) is run
alongside as a fidelity control: if the class structure looks the same at a layer the SAE
reconstructs almost perfectly, reconstruction loss is not what produced it. Layer 33 is a
poor choice on this evidence despite being what 15_sae_fhs.py names, since its error is
0.221 and its activation norm (9.8) is two orders of magnitude off every other layer,
the final LayerNorm having rescaled it.

⚠️ This script computes feature activations and their class structure. It does NOT
recompute the published FHS scores or the FHS-FSI correlation; those stay as the record of
what the fallback produced, with their caveats in EVALUATION_REPORT.

⚠️ Status, stated so nothing is inferred: the §9.6 results in
docs/MECHANISM_GENERALIZATION.md were produced by `15c` and `15d`, which carry their own
copies of the loader below. THIS script's own main() has not been run end to end; it exists
because the loading convention and the layer-choice evidence needed one documented home,
and because a future caller wanting per-class features at a different layer should start
from a single verified loader rather than a third copy. Treat it as the reference
implementation, not as the source of a published number.

Usage:
    python src/15b_interplm_sae.py [--layer 18] [--device cpu]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
REPO_ID = "Elana/InterPLM-esm2-650m"
AVAILABLE_LAYERS = [1, 9, 18, 24, 30, 33]
MAX_LEN = 1022


def load_interplm_sae(layer: int):
    """Load InterPLM's SAE for one ESM-2 650M layer, directly from its state dict.

    Returns (encode_fn, n_features). encode_fn maps (L, 1280) -> (L, n_features),
    applying the pre-encoder bias subtraction then ReLU, which is the standard
    Anthropic-style SAE forward that these checkpoints are shaped for.

    The forward convention was checked rather than assumed, by reconstructing raw
    ESM-2 layer-18 activations through encoder-then-decoder and comparing relative
    error:

        input                          active/residue   rel. recon. error
        raw hidden state                    146              0.133
        unit-normalised * sqrt(1280)         30              0.476

    So raw activations are correct and the `ae_normalized.pt` filename does not mean
    the caller must normalise its input. Note also that `ae_normalized.pt` and
    `ae_unnormalized.pt` are genuinely different weights (decoder norms 8821 vs 101)
    that happen to give the same activation pattern and the same reconstruction error
    on raw input, the encoder/decoder scales cancelling; `ae_normalized.pt` is used
    here for determinism of choice, not because it reconstructs better.
    """
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO_ID, f"layer_{layer}/ae_normalized.pt")
    sd = torch.load(path, map_location="cpu", weights_only=False)
    W_enc = sd["encoder.weight"]          # (n_features, 1280)
    b_enc = sd["encoder.bias"]            # (n_features,)
    b_pre = sd["bias"]                    # (1280,)
    n_features = W_enc.shape[0]

    def encode(h: torch.Tensor) -> torch.Tensor:
        return torch.relu((h - b_pre) @ W_enc.T + b_enc)

    return encode, n_features, path


def load_interplm_weights(layer: int):
    """The same checkpoint as load_interplm_sae, returned as raw tensors
    (W_enc, b_enc, b_pre) for callers that want to batch the encode themselves.

    Exists so that 15c and 15d do not each keep a private copy of the download path,
    the key names and the forward convention. If the convention is ever found wrong,
    it is wrong in one place.
    """
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO_ID, f"layer_{layer}/ae_normalized.pt")
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return sd["encoder.weight"], sd["encoder.bias"], sd["bias"]


def read_fasta(p):
    out, acc, seq = {}, None, []
    for line in open(p):
        if line.startswith(">"):
            if acc:
                out[acc] = "".join(seq)
            acc, seq = line[1:].split()[0], []
        elif acc:
            seq.append(line.strip())
    if acc:
        out[acc] = "".join(seq)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=18, choices=AVAILABLE_LAYERS)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    torch.manual_seed(0)  # no training happens, but pin anyway so nothing can drift

    encode, n_features, ckpt = load_interplm_sae(a.layer)
    print(f"InterPLM SAE, layer {a.layer}: {n_features} features")
    print(f"  checkpoint {ckpt}")

    from transformers import AutoTokenizer, EsmModel
    mid = "facebook/esm2_t33_650M_UR50D"
    tok = AutoTokenizer.from_pretrained(mid)
    esm = EsmModel.from_pretrained(mid).to(a.device).eval()

    man = json.load(open(V2 / "embedding_manifest_v2.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    pf = read_fasta(ROOT / "data/sequences/toxins_positive_v2.fasta")

    rows = [r for r in man["positive_rows"]]
    print(f"\nencoding {len(rows)} panel positives at layer {a.layer}")

    # mean-pool SAE features over each protein: (n_features,) per protein
    feats, names, classes = [], [], []
    for i, r in enumerate(rows):
        seq = pf[r["acc"]][:MAX_LEN]
        enc = tok(seq, return_tensors="pt").to(a.device)
        with torch.no_grad():
            hs = esm(**enc, output_hidden_states=True).hidden_states[a.layer]
        h = hs[0, 1:-1, :].float().cpu()          # strip BOS/EOS -> (L, 1280)
        with torch.no_grad():
            f = encode(h)                          # (L, n_features)
        feats.append(f.mean(0).numpy())
        names.append(r["name"])
        classes.append(cls[r["acc"]])
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    F = np.vstack(feats)                           # (80, n_features)
    classes = np.array(classes)
    print(f"\nfeature matrix {F.shape}")
    alive = (F > 0).any(0)
    print(f"  features active on at least one protein: {int(alive.sum())}/{n_features}")
    print(f"  mean features active per protein: {(F > 0).sum(1).mean():.0f}")

    # which features separate beta-lactamase from the animal-target classes?
    bl = classes == "beta_lactamase"
    pf_cls = classes == "pore_forming_cytolysin"
    print(f"\nbeta_lactamase n={int(bl.sum())}, pore_forming n={int(pf_cls.sum())}")

    out = {
        "sae_source": "interplm",
        "repo_id": REPO_ID,
        "layer": a.layer,
        "n_features": int(n_features),
        "deterministic": True,
        "n_proteins": int(F.shape[0]),
        "features_active_somewhere": int(alive.sum()),
        "mean_active_per_protein": float((F > 0).sum(1).mean()),
        "why_layer": ("§9.5 found beta-lactamase collapses to 0.0% recovery at layers 12, "
                      "18 and 24; InterPLM has no layer-12 SAE, so 18 is the shallowest "
                      "layer with both the collapse and interpretable features"),
        "classes": {c: int((classes == c).sum()) for c in sorted(set(classes))},
    }
    np.save(V2 / f"interplm_features_L{a.layer}.npy", F)
    with open(V2 / f"interplm_sae_L{a.layer}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {V2}/interplm_sae_L{a.layer}.json")
    print(f"wrote {V2}/interplm_features_L{a.layer}.npy  {F.shape}")


if __name__ == "__main__":
    main()
