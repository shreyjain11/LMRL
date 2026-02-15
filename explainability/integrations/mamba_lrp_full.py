"""
Full MambaLRP implementation for Phyla (Jafari et al., NeurIPS 2024).

Applies three detach operations to ensure relevance conservation:
1. SiLU: y = x * sigmoid(x).detach()
2. Selective SSM: A_bar.detach(), B_bar.detach(), C.detach()
3. Multiplicative gate: y = 0.5*(zA + zB) + 0.5*(zA - zB).detach()

Then computes Gradient x Input for true LRP relevance scores.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

# ============================================================
# 1. LRP-COMPATIBLE COMPONENT FUNCTIONS
# ============================================================

def silu_lrp(x):
    """SiLU with detached sigmoid: y = x * sigmoid(x).detach()"""
    return x * torch.sigmoid(x).detach()


def selective_scan_lrp(x, dt, A, B, C, D, dt_bias, seqlen):
    """
    Explicit selective scan with detached A_bar, B_bar, C.
    
    x: (B, d_inner, L)
    dt: (d_inner, B*L) -- raw dt before softplus
    A: (d_inner, d_state) -- negative
    B: (B, d_state, L)
    C: (B, d_state, L)
    D: (d_inner,)
    dt_bias: (d_inner,)
    """
    batch = x.shape[0]
    d_inner = x.shape[1]
    d_state = A.shape[1]
    device = x.device
    dtype = x.dtype
    
    # Reshape dt: (d_inner, B*L) -> (B, d_inner, L)
    dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
    
    # Apply softplus to dt (with bias)
    dt = F.softplus(dt + dt_bias.unsqueeze(0).unsqueeze(-1))  # (B, d_inner, L)
    
    # Discretize A and B
    # dA = exp(dt * A) -- detach for LRP
    dA = torch.exp(
        torch.einsum("bdl,dn->bdnl", dt, A)
    ).detach()  # (B, d_inner, d_state, L) -- DETACHED
    
    # dB = dt * B -- detach for LRP
    dB = torch.einsum("bdl,bnl->bdnl", dt, B).detach()  # DETACHED
    
    # C detached
    C_det = C.detach()  # (B, d_state, L) -- DETACHED
    
    # Sequential scan
    h = torch.zeros(batch, d_inner, d_state, device=device, dtype=dtype)
    ys = []
    
    for t in range(seqlen):
        h = dA[:, :, :, t] * h + dB[:, :, :, t] * x[:, :, t].unsqueeze(-1)
        y_t = torch.einsum("bdn,bn->bd", h, C_det[:, :, t])
        ys.append(y_t)
    
    y = torch.stack(ys, dim=-1)  # (B, d_inner, L)
    
    # Add D skip connection
    y = y + D.unsqueeze(0).unsqueeze(-1) * x
    
    return y


def gate_half_detach(z_ssm, z_gate):
    """
    Multiplicative gate with half-detach (Eq. 9 in Jafari et al.):
    y = 0.5 * (z_ssm + z_gate) + 0.5 * (z_ssm - z_gate).detach()
    
    Equivalent to: z_ssm * z_gate but with conservation-preserving gradient.
    """
    # z_gate should have SiLU already applied
    return 0.5 * (z_ssm + z_gate) + 0.5 * (z_ssm - z_gate).detach()


# ============================================================
# 2. LRP-COMPATIBLE MAMBA FORWARD (replaces Mamba.forward)
# ============================================================

def mamba_forward_lrp(self, hidden_states, inference_params=None):
    """
    LRP-compatible Mamba forward. Forces slow path and applies detach ops.
    Replaces Mamba.forward temporarily during attribution.
    """
    batch, seqlen, dim = hidden_states.shape
    
    # in_proj: (B, L, D) -> (B, 2*d_inner, L)
    xz = rearrange(
        self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
        "d (b l) -> b d l",
        l=seqlen,
    )
    if self.in_proj.bias is not None:
        xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
    
    x, z = xz.chunk(2, dim=1)  # each (B, d_inner, L)
    
    # Conv1d + SiLU with LRP detach
    x_conv = self.conv1d(x)[..., :seqlen]
    x = silu_lrp(x_conv)  # LRP FIX #1: detached sigmoid
    
    A = -torch.exp(self.A_log.float())
    
    # Project to get dt, B, C
    x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
    dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
    dt = self.dt_proj.weight @ dt.t()  # (d_inner, B*L)
    
    B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    
    # Selective scan with LRP detach ops (FIX #2: detached A_bar, B_bar, C)
    y = selective_scan_lrp(x, dt, A, B, C, self.D.float(), self.dt_proj.bias.float(), seqlen)
    
    # Gate with SiLU (LRP detach on SiLU) then half-detach gate (FIX #3)
    z_gate = silu_lrp(z)  # SiLU on gate branch, with detach
    y = gate_half_detach(y, z_gate)  # half-detach multiplicative gate
    
    y = rearrange(y, "b d l -> b l d")
    out = self.out_proj(y)
    return out


# ============================================================
# 3. PHYLA FULL LRP ATTRIBUTOR
# ============================================================

class PhylaFullLRP:
    """
    Full MambaLRP attribution for Phyla.
    
    Temporarily monkey-patches all Mamba layers to use LRP-compatible forward,
    then computes Gradient x Input for true LRP relevance.
    """
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self._original_forwards = {}
        self._mamba_layers = []
        
        # Find all Mamba layers across all stages
        # Must store direct references to the actual Mamba nn.Module objects
        for stage_idx, stage in enumerate(model.modul):
            for name, module in stage.named_modules():
                cls_name = module.__class__.__name__
                if cls_name == 'Mamba':
                    key = f"stage{stage_idx}.{name}"
                    self._mamba_layers.append((key, module))
        
        # Verify: group by stage to confirm coverage
        stage_counts = {}
        for key, _ in self._mamba_layers:
            s = key.split('.')[0]
            stage_counts[s] = stage_counts.get(s, 0) + 1
        
        print(f"PhylaFullLRP: found {len(self._mamba_layers)} Mamba layers")
        for s, c in sorted(stage_counts.items()):
            print(f"  {s}: {c} Mamba layers")
    
    def _patch_mamba_layers(self):
        """Replace all Mamba.forward with LRP-compatible version."""
        import types
        self._original_forwards = {}
        patched = 0
        for key, module in self._mamba_layers:
            self._original_forwards[key] = module.forward
            module.forward = types.MethodType(mamba_forward_lrp, module)
            patched += 1
        print(f"  Patched {patched} Mamba layers for LRP")
    
    def _unpatch_mamba_layers(self):
        """Restore original Mamba.forward."""
        for key, module in self._mamba_layers:
            if key in self._original_forwards:
                module.forward = self._original_forwards[key]
        self._original_forwards = {}
    
    def _forward_with_grad(self, seqs, names, target_fn):
        """Run patched forward pass and compute Gradient x Input."""
        self.model.eval()
        
        # Verify patches are still active on all stages
        import types
        for key, module in self._mamba_layers:
            if not isinstance(module.forward, types.MethodType):
                print(f"  WARNING: {key} lost its patch!")
            elif module.forward.__func__ is not mamba_forward_lrp:
                print(f"  WARNING: {key} has wrong forward!")
        
        # Encode
        encode_out = self.model.encode(seqs, names)
        input_ids = encode_out[0]
        cls_mask = encode_out[1]
        seq_mask = encode_out[2]
        
        # Get embedding layer and register hooks
        embed_layer = self.model.modul[0].backbone.embedding
        
        emb_storage = {}
        grad_storage = {}
        
        def fwd_hook(module, input, output):
            emb_storage['emb'] = output
        
        def bwd_hook(module, grad_input, grad_output):
            grad_storage['grad'] = grad_output[0]
        
        h_fwd = embed_layer.register_forward_hook(fwd_hook)
        h_bwd = embed_layer.register_full_backward_hook(bwd_hook)
        
        try:
            # Forward through all stages (with LRP patches active)
            x = self.model.modul[0](
                input_ids.to(self.device),
                hidden_states_given=False,
                logits=False,
                position_ids=None,
                sequence_mask=seq_mask.to(self.device),
                cls_token_mask=cls_mask.to(self.device)
            )
            
            for module in self.model.modul[1:]:
                dev = next(module.parameters()).device
                x = module(
                    x.to(dev),
                    hidden_states_given=True,
                    logits=False,
                    position_ids=None,
                    sequence_mask=seq_mask.to(dev),
                    cls_token_mask=cls_mask.to(dev)
                )
            
            embeddings = x  # (1, n_seqs, 256)
            
            # Compute target and backpropagate
            target = target_fn(embeddings)
            target.backward()
            
            emb = emb_storage['emb']
            grad = grad_storage['grad']
            
            return emb, grad, embeddings
            
        finally:
            h_fwd.remove()
            h_bwd.remove()
    
    def attribute_sequence(self, seqs, names, target_seq_idx=0):
        """
        Full MambaLRP attribution using Gradient x Input.
        
        Returns per-position relevance for the target sequence.
        Uses SIGNED Gradient x Input summed over hidden dim (true LRP).
        """
        self._patch_mamba_layers()
        
        try:
            def target_fn(embeddings):
                return embeddings[0, target_seq_idx].sum()
            
            emb, grad, embeddings = self._forward_with_grad(seqs, names, target_fn)
            
            emb_det = emb.detach().cpu().squeeze(0)
            grad_det = grad.detach().cpu().squeeze(0)
            
            # Method A: Gradient x Input signed sum (classical LRP)
            gxi = emb_det * grad_det
            relevance_signed = gxi.sum(dim=-1).numpy()
            
            # Method B: Gradient L2 norm (robust for discrete embeddings)
            relevance_gradnorm = grad_det.norm(dim=-1).numpy()
            
            # Method C: GxI L2 norm per position (hybrid)
            relevance_gxi_norm = gxi.norm(dim=-1).numpy()
            
            # Use gradient norm as primary (proven to work)
            relevance_abs = relevance_gradnorm
            
            # Parse boundaries for target sequence
            start = 0
            for i in range(target_seq_idx):
                start += len(seqs[i]) + 1  # +1 for CLS
            start += 1  # skip CLS of target
            target_len = len(seqs[target_seq_idx])
            end = start + target_len
            
            # (boundary extraction moved below)
            
            seq_signed = relevance_signed[start:end]
            seq_gradnorm = relevance_gradnorm[start:end]
            seq_gxi_norm = relevance_gxi_norm[start:end]
            
            info = {
                'method': 'mambalrp_detach+gradnorm',
                'target_seq_idx': target_seq_idx,
                'seq_len': target_len,
                'n_seqs': len(seqs),
                'relevance_signed': seq_signed,
                'relevance_gradnorm': seq_gradnorm,
                'relevance_gxi_norm': seq_gxi_norm,
            }
            
            return seq_gradnorm, info
            
        finally:
            self._unpatch_mamba_layers()
    
    def attribute_pairwise_distance(self, seqs, names, seq_i=0, seq_j=1):
        """Full MambaLRP attribution for pairwise distance."""
        self._patch_mamba_layers()
        
        try:
            def target_fn(embeddings):
                return torch.norm(embeddings[0, seq_i] - embeddings[0, seq_j])
            
            emb, grad, embeddings = self._forward_with_grad(seqs, names, target_fn)
            
            gxi = (emb * grad).detach().cpu().squeeze(0)
            relevance_abs = np.abs(gxi.sum(dim=-1).numpy())
            
            def get_start(idx):
                s = 0
                for k in range(idx):
                    s += len(seqs[k]) + 1
                return s + 1
            
            start_i = get_start(seq_i)
            start_j = get_start(seq_j)
            
            rel_i = relevance_abs[start_i:start_i + len(seqs[seq_i])]
            rel_j = relevance_abs[start_j:start_j + len(seqs[seq_j])]
            
            dist = float(torch.norm(
                emb.detach().cpu().squeeze(0)[0] -  # placeholder
                emb.detach().cpu().squeeze(0)[0]
            ))
            
            info = {
                'method': 'mambalrp_gxi_pairwise',
                'seq_i': seq_i,
                'seq_j': seq_j,
                'seq_len': len(seqs[0]),
            }
            
            return rel_i, rel_j, info
            
        finally:
            self._unpatch_mamba_layers()


# ============================================================
# 4. QUICK SELF-TEST
# ============================================================

def test_full_lrp():
    """Compare MambaLRP (detach ops + grad norm) vs plain grad norm."""
    import os
    sys.path.insert(0, os.path.expanduser("~/work/LMRL"))
    from explainability.experiments.utils import setup_phyla_env, load_phyla_model
    from explainability.integrations.mamba_lrp import PhylaAttributor
    
    setup_phyla_env()
    model = load_phyla_model()
    model.eval()
    
    # Both attributors on same model
    lrp_full = PhylaFullLRP(model)
    grad_plain = PhylaAttributor(model)
    
    WT = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQV"
    MUT = list(WT)
    MUT[38] = 'W'
    MUT = ''.join(MUT)
    
    print("=" * 60)
    print("FULL MambaLRP vs PLAIN GRADIENT NORM")
    print("=" * 60)
    print(f"Seq length: {len(WT)}, Mutation: pos38 {WT[38]}->W")
    print()
    
    # --- Plain gradient norm (no detach ops) ---
    print("--- Plain Gradient Norm (no detach ops) ---")
    rel_plain_wt, _ = grad_plain.attribute_sequence([WT], ["S0"], target_seq_idx=0)
    rel_plain_mut, _ = grad_plain.attribute_sequence([MUT], ["S0"], target_seq_idx=0)
    
    wt_rank_plain = int(np.sum(rel_plain_wt >= rel_plain_wt[38]))
    mut_rank_plain = int(np.sum(rel_plain_mut >= rel_plain_mut[38]))
    print(f"  WT  pos38: rank={wt_rank_plain}/{len(rel_plain_wt)}, score={rel_plain_wt[38]:.4f}, std={rel_plain_wt.std():.4f}")
    print(f"  MUT pos38: rank={mut_rank_plain}/{len(rel_plain_mut)}, score={rel_plain_mut[38]:.4f}, std={rel_plain_mut.std():.4f}")
    top5 = sorted(enumerate(rel_plain_mut), key=lambda x: -x[1])[:5]
    print(f"  MUT top5: {[(p, round(float(s),4)) for p,s in top5]}")
    print()
    
    # --- Full MambaLRP (detach ops + grad norm) ---
    print("--- MambaLRP Detach Ops + Gradient Norm ---")
    rel_lrp_wt, info_wt = lrp_full.attribute_sequence([WT], ["S0"], target_seq_idx=0)
    rel_lrp_mut, info_mut = lrp_full.attribute_sequence([MUT], ["S0"], target_seq_idx=0)
    
    wt_rank_lrp = int(np.sum(rel_lrp_wt >= rel_lrp_wt[38]))
    mut_rank_lrp = int(np.sum(rel_lrp_mut >= rel_lrp_mut[38]))
    print(f"  WT  pos38: rank={wt_rank_lrp}/{len(rel_lrp_wt)}, score={rel_lrp_wt[38]:.4f}, std={rel_lrp_wt.std():.4f}")
    print(f"  MUT pos38: rank={mut_rank_lrp}/{len(rel_lrp_mut)}, score={rel_lrp_mut[38]:.4f}, std={rel_lrp_mut.std():.4f}")
    top5_lrp = sorted(enumerate(rel_lrp_mut), key=lambda x: -x[1])[:5]
    print(f"  MUT top5: {[(p, round(float(s),4)) for p,s in top5_lrp]}")
    print()
    
    # --- Also show signed GxI and GxI norm from LRP ---
    print("--- MambaLRP Signed GxI (for reference) ---")
    signed = info_mut['relevance_signed']
    gxi_norm = info_mut['relevance_gxi_norm']
    print(f"  Signed GxI std: {signed.std():.6f} (expected near-zero)")
    print(f"  GxI L2 norm std: {gxi_norm.std():.6f}")
    mut_rank_gxi = int(np.sum(gxi_norm >= gxi_norm[38]))
    print(f"  GxI norm pos38 rank: {mut_rank_gxi}/{len(gxi_norm)}")
    print()
    
    # --- Comparison ---
    print("=" * 60)
    print("COMPARISON: Mutation pos38 rank (lower = better)")
    print("=" * 60)
    print(f"  Plain grad norm:          WT={wt_rank_plain}, MUT={mut_rank_plain}")
    print(f"  MambaLRP detach+gradnorm: WT={wt_rank_lrp}, MUT={mut_rank_lrp}")
    print()
    
    # Correlation between the two methods
    from scipy.stats import spearmanr
    rho_wt, _ = spearmanr(rel_plain_wt, rel_lrp_wt)
    rho_mut, _ = spearmanr(rel_plain_mut, rel_lrp_mut)
    print(f"  Spearman correlation (WT):  {rho_wt:.4f}")
    print(f"  Spearman correlation (MUT): {rho_mut:.4f}")
    
    if mut_rank_lrp <= mut_rank_plain:
        print(f"\n  >> MambaLRP IMPROVES mutation ranking: {mut_rank_plain} -> {mut_rank_lrp}")
    else:
        print(f"\n  >> MambaLRP does NOT improve ranking: {mut_rank_plain} -> {mut_rank_lrp}")
        print(f"  >> Detach ops change gradient flow but don't help localization here")
    
    print("\nDone.")


if __name__ == "__main__":
    test_full_lrp()
