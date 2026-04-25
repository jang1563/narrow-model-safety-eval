#!/usr/bin/env python3
"""
14_esm3_separability_fspe.py — ESM-3 and SaProt representation analysis.

Two analyses (NOT FSI — these are masked-prediction / representation models):

(A) Embedding Separability:
    Repeat AUROC / precision@k from 03_esm2_separability.py with ESM-3
    embeddings. Hypothesis: ESM-3 > ESM-2 (AUROC=0.994) because joint
    sequence+structure conditioning improves functional discrimination.

(B) FSPE — Functional Site Prediction Entropy:
    Repeat masked prediction entropy analysis from 04_esm2_masked_prediction.py
    with ESM-3's masked prediction. Hypothesis: ESM-3 FSPE ratio << 1.0 more
    consistently than ESM-2 (4/5, mean 0.93), addressing v1's underpowered result.

(C) SaProt [optional, --with_saprot]:
    SaProt uses Foldseek structure tokens (3Di alphabet) combined with amino
    acid tokens. Requires preprocessing: run slurm/esm3_foldseek_preprocess.sh
    to generate structure-aware token strings before calling this script with
    --with_saprot.

Requires:
    pip install esm  (ESM-3 via EvolutionaryScale SDK)
    # For SaProt:
    pip install transformers
    # Foldseek preprocessing: see slurm/esm3_foldseek_preprocess.sh

Usage:
    python src/14_esm3_separability_fspe.py [--device cuda]
    python src/14_esm3_separability_fspe.py --with_saprot [--device cuda]

References:
    Hayes et al. (2024) "Simulating 500 million years of evolution with a
    language model." Science. (ESM-3)
    Su et al. (2024) "SaProt: Protein Language Modeling with Structure-Aware
    Vocabulary." ICLR 2024.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    ANNOT_DIR,
    RESULTS_DIR,
    add_schema_version,
    load_functional_sites,
    load_positive_sequences,
    load_negative_sequences,
    print_header,
    truncate_sequence,
)

MAX_SEQ_LEN = 1022


# ============================================================================
# ESM-3 interface
# ============================================================================


def load_esm3(device: str):
    """Load ESM-3 small open model.

    ESM-3 is available via `pip install esm` from EvolutionaryScale.
    The open model (esm3_sm_open_v1) does not require an API key.
    """
    try:
        from esm.models.esm3 import ESM3
        model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
        # Convert to float32 to avoid bfloat16 ↔ float32 dtype conflicts inside
        # ESM-3's custom ops (autocast does not resolve these reliably).
        model = model.float()
        model.eval()
        return model
    except ImportError:
        print("ERROR: ESM-3 not installed. Run: pip install esm")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading ESM-3: {e}")
        print("Ensure esm package is installed: pip install esm")
        sys.exit(1)


def get_esm3_embedding(sequence: str, model, device: str) -> np.ndarray:
    """Get mean-pooled ESM-3 sequence embedding.

    ESM-3 encodes sequences via its multimodal protein representation.
    We extract the sequence track embedding and mean-pool over residues.

    Returns:
        1D numpy array of shape (hidden_dim,)
    """
    try:
        from esm.sdk.api import ESMProtein
        protein = ESMProtein(sequence=truncate_sequence(sequence, MAX_SEQ_LEN))
        protein_tensor = model.encode(protein)

        seq_tokens = protein_tensor.sequence
        if seq_tokens.dim() == 1:
            seq_tokens = seq_tokens.unsqueeze(0)

        # Model is float32 (cast on load); no autocast needed.
        with torch.no_grad():
            output = model.forward(sequence_tokens=seq_tokens)

        # Extract sequence embeddings.
        # ESM-3 output may be batched [1, L+2, D] or unbatched [L+2, D].
        # Mean-pool over residue positions excluding BOS/EOS special tokens.
        def _mean_pool(tensor: torch.Tensor) -> np.ndarray:
            if tensor.dim() == 3:
                return tensor[0, 1:-1, :].float().mean(dim=0).cpu().numpy()
            elif tensor.dim() == 2:
                return tensor[1:-1, :].float().mean(dim=0).cpu().numpy()
            raise ValueError(f"Unexpected embedding tensor shape: {tensor.shape}")

        if hasattr(output, "embeddings") and output.embeddings is not None:
            emb = _mean_pool(output.embeddings)
        elif hasattr(output, "sequence_logits") and output.sequence_logits is not None:
            emb = _mean_pool(output.sequence_logits)
        else:
            raise ValueError(f"ESM-3 output has no recognized embedding attribute. "
                             f"Available: {[k for k in dir(output) if not k.startswith('_')]}")
        return emb

    except Exception as e:
        raise RuntimeError(f"ESM-3 embedding failed: {e}") from e


def get_esm3_masked_entropy(
    sequence: str,
    position: int,
    model,
    device: str,
) -> Optional[dict]:
    """Mask a single position and compute ESM-3 prediction entropy (FSPE).

    Args:
        sequence: Amino acid sequence
        position: 0-indexed position to mask
        model: ESM-3 model
        device: compute device

    Returns:
        dict with entropy, correct_aa, is_correct; or None on failure
    """
    try:
        from esm.sdk.api import ESMProtein

        seq = truncate_sequence(sequence, MAX_SEQ_LEN)
        if position >= len(seq):
            return None
        correct_aa = seq[position]

        # ESM-3 mask token is '_' in the sequence string
        seq_list = list(seq)
        seq_list[position] = "_"
        masked_seq = "".join(seq_list)

        protein = ESMProtein(sequence=masked_seq)
        protein_tensor = model.encode(protein)

        seq_tokens = protein_tensor.sequence
        if seq_tokens.dim() == 1:
            seq_tokens = seq_tokens.unsqueeze(0)

        # Model is float32 (cast on load); no autocast needed.
        with torch.no_grad():
            output = model.forward(sequence_tokens=seq_tokens)

        if not hasattr(output, "sequence_logits") or output.sequence_logits is None:
            return None

        # Position in token stream: +1 for BOS token.
        # Handle both batched [1, L+2, vocab] and unbatched [L+2, vocab] outputs.
        sl = output.sequence_logits
        if sl.dim() == 3:
            logits = sl[0, position + 1, :].float()
        elif sl.dim() == 2:
            logits = sl[position + 1, :].float()
        else:
            return None
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

        # Top prediction
        top_idx = probs.argmax().item()

        return {
            "position": position,
            "correct_aa": correct_aa,
            "entropy": float(entropy),
            "top_prob": float(probs[top_idx].item()),
        }

    except Exception as e:
        return None


# ============================================================================
# SaProt interface
# ============================================================================


def load_saprot(device: str):
    """Load SaProt model and tokenizer from HuggingFace.

    SaProt (westlake-repl/SaProt_650M_AF2) uses combined amino acid +
    structure tokens (3Di format from Foldseek). Each position is encoded
    as a two-character token: amino acid + structural neighborhood code.

    Requires:
        pip install transformers
        Foldseek preprocessing (see slurm/esm3_foldseek_preprocess.sh)
    """
    try:
        from transformers import EsmTokenizer, EsmForMaskedLM
        model_name = "westlake-repl/SaProt_650M_AF2"
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        model = EsmForMaskedLM.from_pretrained(model_name).to(device)
        model.eval()
        return model, tokenizer
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading SaProt: {e}")
        print("Ensure westlake-repl/SaProt_650M_AF2 is accessible on HuggingFace.")
        sys.exit(1)


def load_saprot_tokens() -> dict:
    """Load Foldseek-generated structure-aware token strings.

    Expected file: data/annotations/saprot_tokens.json
    Format: {"UNIPROT_ID": "Ac#dE&fG...", ...}  (aa + 3Di token per residue)

    Generate with: slurm/esm3_foldseek_preprocess.sh
    """
    token_path = ANNOT_DIR / "saprot_tokens.json"
    if not token_path.exists():
        print(f"ERROR: SaProt token file not found: {token_path}")
        print("Run: bash slurm/esm3_foldseek_preprocess.sh")
        return {}
    with open(token_path) as f:
        raw = json.load(f)
    # Foldseek outputs 3Di codes as uppercase; SaProt expects lowercase.
    # Fix: lowercase every odd-indexed character (the 3Di code in each 2-char pair).
    return {
        uid: "".join(c.lower() if i % 2 == 1 else c for i, c in enumerate(seq))
        for uid, seq in raw.items()
    }


def get_saprot_embedding(sa_sequence: str, model, tokenizer, device: str) -> np.ndarray:
    """Get mean-pooled SaProt embedding from a structure-aware sequence string."""
    inputs = tokenizer(
        sa_sequence,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN + 2,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Last hidden state: [1, L+2, hidden_dim], mean-pool excluding special tokens
    hidden = outputs.hidden_states[-1][0, 1:-1, :]
    return hidden.mean(dim=0).cpu().numpy()


# ============================================================================
# Separability analysis (shared for ESM-3 and SaProt)
# ============================================================================


def run_auroc_analysis(
    pos_embeddings: np.ndarray,
    neg_embeddings: np.ndarray,
    model_label: str,
) -> dict:
    """Run 5-fold CV logistic regression separability analysis."""
    X = np.vstack([pos_embeddings, neg_embeddings])
    y = np.array([1] * len(pos_embeddings) + [0] * len(neg_embeddings))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auroc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")
    acc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")

    print(f"  [{model_label}] AUROC: {auroc_scores.mean():.3f} ± {auroc_scores.std():.3f}")
    print(f"  [{model_label}] Accuracy: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")

    return {
        "model": model_label,
        "auroc_mean": float(auroc_scores.mean()),
        "auroc_std": float(auroc_scores.std()),
        "auroc_per_fold": auroc_scores.tolist(),
        "accuracy_mean": float(acc_scores.mean()),
        "accuracy_std": float(acc_scores.std()),
        "n_positive": int(len(pos_embeddings)),
        "n_negative": int(len(neg_embeddings)),
    }


# ============================================================================
# FSPE analysis
# ============================================================================


def run_fspe_analysis(
    model,
    device: str,
    model_label: str,
    get_entropy_fn,
    func_sites: dict,
    seq_lookup: dict,
    n_nonfunctional: int = 20,
) -> list:
    """Run FSPE analysis for a given masked-prediction model.

    Args:
        get_entropy_fn: function(sequence, position, model, device) -> dict|None
    """
    all_results = []

    for uniprot_id, info in func_sites.items():
        if uniprot_id.startswith("_"):
            continue
        if uniprot_id not in seq_lookup:
            continue

        print(f"\n  [{model_label}] {uniprot_id}: {info['name']}")
        sequence = seq_lookup[uniprot_id]
        seq = truncate_sequence(sequence, MAX_SEQ_LEN)
        seq_len = len(seq)

        catalytic_residues = info["functional_sites"]["catalytic_residues"]
        func_positions = [r - 1 for r in catalytic_residues if r - 1 < seq_len]

        if not func_positions:
            print(f"    No valid functional positions, skipping")
            continue

        rng = np.random.RandomState(42)
        nonfunc_candidates = list(set(range(seq_len)) - set(func_positions))
        nonfunc_positions = sorted(
            rng.choice(nonfunc_candidates, min(n_nonfunctional, len(nonfunc_candidates)), replace=False)
        )

        func_entropies = []
        for pos in func_positions:
            result = get_entropy_fn(seq, pos, model, device)
            if result:
                func_entropies.append(result["entropy"])

        nonfunc_entropies = []
        for pos in nonfunc_positions:
            result = get_entropy_fn(seq, pos, model, device)
            if result:
                nonfunc_entropies.append(result["entropy"])

        if not func_entropies or not nonfunc_entropies:
            print(f"    WARNING: Could not compute entropies")
            continue

        fspe_ratio = np.mean(func_entropies) / np.mean(nonfunc_entropies) if np.mean(nonfunc_entropies) > 0 else None

        if len(func_entropies) >= 2 and len(nonfunc_entropies) >= 2:
            stat, pval = stats.mannwhitneyu(func_entropies, nonfunc_entropies, alternative="less")
            n_f, n_nf = len(func_entropies), len(nonfunc_entropies)
            r = 1 - (2 * stat) / (n_f * n_nf)
        else:
            pval, r = float("nan"), float("nan")

        entry = {
            "uniprot_id": uniprot_id,
            "model": model_label,
            "n_functional_sites": len(func_positions),
            "n_nonfunctional_sites": len(nonfunc_positions),
            "fspe_functional": float(np.mean(func_entropies)),
            "fspe_nonfunctional": float(np.mean(nonfunc_entropies)),
            "fspe_ratio": float(fspe_ratio) if fspe_ratio is not None else None,
            "mannwhitney_pvalue": float(pval) if not np.isnan(pval) else None,
            "rank_biserial_r": float(r) if not np.isnan(r) else None,
        }
        all_results.append(entry)

        ratio_str = f"{fspe_ratio:.3f}" if fspe_ratio else "N/A"
        print(f"    FSPE functional:     {np.mean(func_entropies):.3f}")
        print(f"    FSPE non-functional: {np.mean(nonfunc_entropies):.3f}")
        print(f"    FSPE ratio:          {ratio_str}  (< 1.0 → model more confident at catalytic sites)")

    return all_results


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="ESM-3 / SaProt separability + FSPE (v2 Pillar 1C)")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--with_saprot",
        action="store_true",
        help="Also run SaProt analysis (requires Foldseek preprocessing)",
    )
    parser.add_argument(
        "--skip_fspe",
        action="store_true",
        help="Skip FSPE (masked prediction) analysis — only run separability",
    )
    parser.add_argument(
        "--skip_separability",
        action="store_true",
        help="Skip separability analysis — only run FSPE",
    )
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print_header("ESM-3 / SaProt Separability + FSPE (v2 — Pillar 1C)")
    print(f"Device: {device}")

    # Load sequences
    positive_seqs = load_positive_sequences()
    negative_seqs = load_negative_sequences()
    func_sites = load_functional_sites()

    seq_lookup = {}
    for seq_id, desc, seq in positive_seqs:
        parts = seq_id.split("|")
        accession = parts[1] if len(parts) >= 2 else seq_id
        seq_lookup[accession] = seq

    separability_results = []
    fspe_results = []

    # ========================================================================
    # ESM-3
    # ========================================================================
    print("\n--- Loading ESM-3 (esm3_sm_open_v1) ---")
    esm3_model = load_esm3(device)

    if not args.skip_separability:
        print("\n--- ESM-3 Embedding Separability ---")

        print("  Computing ESM-3 embeddings for positive sequences...")
        pos_embs = []
        for seq_id, desc, seq in positive_seqs:
            try:
                emb = get_esm3_embedding(seq, esm3_model, device)
                pos_embs.append(emb)
            except RuntimeError as e:
                print(f"  WARNING: Could not embed {seq_id}: {e}")

        print("  Computing ESM-3 embeddings for negative sequences...")
        neg_embs = []
        for seq_id, desc, seq in negative_seqs:
            try:
                emb = get_esm3_embedding(seq, esm3_model, device)
                neg_embs.append(emb)
            except RuntimeError as e:
                print(f"  WARNING: Could not embed {seq_id}: {e}")

        if pos_embs and neg_embs:
            # Save embeddings for downstream use
            np.save(RESULTS_DIR / "esm3_embeddings_positive.npy", np.array(pos_embs))
            np.save(RESULTS_DIR / "esm3_embeddings_negative.npy", np.array(neg_embs))

            esm3_sep = run_auroc_analysis(
                np.array(pos_embs), np.array(neg_embs), "esm3_sm_open_v1"
            )
            separability_results.append(esm3_sep)

            # Load ESM-2 results for comparison
            esm2_sep_path = RESULTS_DIR / "separability_results.json"
            if esm2_sep_path.exists():
                with open(esm2_sep_path) as f:
                    esm2_sep = json.load(f)
                print(f"\n  ESM-2 AUROC (v1): {esm2_sep.get('auroc_mean', 'N/A'):.3f}")
                print(f"  ESM-3 AUROC (v2): {esm3_sep['auroc_mean']:.3f}")
                delta = esm3_sep["auroc_mean"] - esm2_sep.get("auroc_mean", 0)
                print(f"  Delta:            {delta:+.3f}")

    if not args.skip_fspe:
        print("\n--- ESM-3 FSPE (Masked Prediction Entropy) ---")
        esm3_fspe = run_fspe_analysis(
            model=esm3_model,
            device=device,
            model_label="esm3_sm_open_v1",
            get_entropy_fn=get_esm3_masked_entropy,
            func_sites=func_sites,
            seq_lookup=seq_lookup,
        )
        fspe_results.extend(esm3_fspe)

    # Free ESM-3 memory before loading SaProt
    del esm3_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================================
    # SaProt [optional]
    # ========================================================================
    if args.with_saprot:
        print("\n--- Loading SaProt (SaProt_650M_AF2) ---")
        saprot_tokens = load_saprot_tokens()

        if not saprot_tokens:
            print("  Skipping SaProt: no token file found.")
        else:
            saprot_model, saprot_tokenizer = load_saprot(device)

            if not args.skip_separability:
                print("\n--- SaProt Embedding Separability ---")

                pos_embs_sa = []
                for seq_id, desc, seq in positive_seqs:
                    parts = seq_id.split("|")
                    uniprot_id = parts[1] if len(parts) >= 2 else seq_id
                    if uniprot_id in saprot_tokens:
                        emb = get_saprot_embedding(saprot_tokens[uniprot_id], saprot_model, saprot_tokenizer, device)
                        pos_embs_sa.append(emb)
                    else:
                        print(f"  WARNING: No SaProt tokens for {uniprot_id}")

                neg_embs_sa = []
                for seq_id, desc, seq in negative_seqs:
                    parts = seq_id.split("|")
                    uniprot_id = parts[1] if len(parts) >= 2 else seq_id
                    if uniprot_id in saprot_tokens:
                        emb = get_saprot_embedding(saprot_tokens[uniprot_id], saprot_model, saprot_tokenizer, device)
                        neg_embs_sa.append(emb)

                if pos_embs_sa and neg_embs_sa:
                    saprot_sep = run_auroc_analysis(
                        np.array(pos_embs_sa), np.array(neg_embs_sa), "saprot_650m_af2"
                    )
                    separability_results.append(saprot_sep)

            if not args.skip_fspe:
                print("\n--- SaProt FSPE ---")

                # Build reverse lookup: AA sequence → uniprot_id
                # (AA seq = every other char from SaProt token string)
                _aa_to_uid = {tokens[::2]: uid for uid, tokens in saprot_tokens.items()}

                def get_saprot_masked_entropy(sequence, position, model, device):
                    """SaProt masked prediction entropy at a given position."""
                    uniprot_id = _aa_to_uid.get(sequence)
                    if uniprot_id is None:
                        return None
                    sa_seq = saprot_tokens[uniprot_id]

                    # SaProt: each position is 2 chars (aa + structure token)
                    # Mask position: replace with '<mask>'
                    sa_chars = [sa_seq[i:i+2] for i in range(0, len(sa_seq), 2)]
                    if position >= len(sa_chars):
                        return None
                    correct_aa = sa_chars[position][0]
                    sa_chars[position] = "<mask>"
                    masked_sa = "".join(sa_chars)

                    inputs = saprot_tokenizer(
                        masked_sa, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN + 2
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model(**inputs)

                    mask_token_id = saprot_tokenizer.mask_token_id
                    token_positions = (inputs["input_ids"] == mask_token_id).nonzero()
                    if len(token_positions) == 0:
                        return None

                    mask_idx = token_positions[0, 1].item()
                    # Cast to float32 before entropy computation — model outputs
                    # may be bfloat16, which collapses small probabilities to zero
                    # and yields a spurious constant entropy = ln(~37).
                    logits = outputs.logits[0, mask_idx].float()
                    probs = torch.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

                    return {
                        "position": position,
                        "correct_aa": correct_aa,
                        "entropy": float(entropy),
                    }

                saprot_fspe = run_fspe_analysis(
                    model=saprot_model,
                    device=device,
                    model_label="saprot_650m_af2",
                    get_entropy_fn=get_saprot_masked_entropy,
                    func_sites=func_sites,
                    seq_lookup={uid: tokens[::2] for uid, tokens in saprot_tokens.items()},
                )
                fspe_results.extend(saprot_fspe)

    # ========================================================================
    # Save results
    # ========================================================================
    sep_output = add_schema_version({
        "analysis": "separability",
        "results": separability_results,
    })
    sep_path = RESULTS_DIR / "esm3_separability_results.json"
    with open(sep_path, "w") as f:
        json.dump(sep_output, f, indent=2)
    print(f"\nSeparability results saved to: {sep_path}")

    fspe_output = add_schema_version({
        "analysis": "fspe",
        "results": fspe_results,
    })
    fspe_path = RESULTS_DIR / "esm3_fspe_results.json"
    with open(fspe_path, "w") as f:
        json.dump(fspe_output, f, indent=2)
    print(f"FSPE results saved to: {fspe_path}")

    print_header("ESM-3 / SaProt Summary")

    print("Separability (AUROC):")
    for r in separability_results:
        print(f"  {r['model']}: {r['auroc_mean']:.3f} ± {r['auroc_std']:.3f}")

    print("\nFSPE (ratio < 1.0 = model more confident at catalytic sites):")
    for r in fspe_results:
        ratio_str = f"{r['fspe_ratio']:.3f}" if r.get("fspe_ratio") else "N/A"
        print(f"  {r['model']} / {r['uniprot_id']}: ratio = {ratio_str}")


if __name__ == "__main__":
    main()
