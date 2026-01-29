# mamba lrp for phyla
# gradient-based attribution for sequence embeddings

import torch
import numpy as np
from typing import Tuple, List, Optional


class PhylaAttributor:
    # computes input attribution for phyla embeddings using gradient norm
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def _forward_with_grad(self, seqs, names, target_fn):
        # forward pass with gradient capture on embeddings
        input_ids, cls_mask, seq_mask, _ = self.model.encode(seqs, names)
        input_ids = input_ids.to(self.device)
        cls_mask = cls_mask.to(self.device)
        seq_mask = seq_mask.to(self.device)
        
        embed_layer = self.model.modul[0].backbone.embedding
        self.model.zero_grad()
        
        grad_holder = {}
        emb_holder = {}
        
        def fwd_hook(module, inp, out):
            emb_holder['emb'] = out.clone()
        
        def bwd_hook(module, grad_in, grad_out):
            if grad_out[0] is not None:
                grad_holder['grad'] = grad_out[0].clone()
        
        h1 = embed_layer.register_forward_hook(fwd_hook)
        h2 = embed_layer.register_full_backward_hook(bwd_hook)
        
        try:
            x = self.model.modul[0](input_ids, logits=False, position_ids=None,
                                    sequence_mask=seq_mask, cls_token_mask=cls_mask)
            for module in self.model.modul[1:]:
                dev = next(module.parameters()).device
                x = module(x.to(dev), hidden_states_given=True, logits=False,
                           position_ids=None, sequence_mask=seq_mask.to(dev),
                           cls_token_mask=cls_mask.to(dev))
            
            embeddings = x[0]  # (n_seqs, hidden_dim)
            target = target_fn(embeddings)
            target.backward()
            
        finally:
            h1.remove()
            h2.remove()
        
        emb = emb_holder.get('emb')
        grad = grad_holder.get('grad')
        
        return emb, grad, embeddings.detach()
    
    def attribute_sequence(
        self, 
        seqs: List[str], 
        names: List[str], 
        target_seq_idx: int,
        method: str = 'grad_norm'
    ) -> Tuple[np.ndarray, dict]:
        # attribute a single sequence's embedding to input positions
        # this is the most effective method for finding important positions
        
        def target_fn(embeddings):
            return embeddings[target_seq_idx].sum()
        
        emb, grad, embeddings = self._forward_with_grad(seqs, names, target_fn)
        
        if grad is None:
            return None, {}
        
        grad = grad[0]  # (total_len, hidden)
        emb = emb[0]
        
        if method == 'grad_norm':
            relevance = grad.norm(dim=-1).cpu().numpy()
        elif method == 'grad_x_input':
            relevance = (grad * emb).abs().sum(dim=-1).cpu().numpy()
        else:
            relevance = grad.norm(dim=-1).cpu().numpy()
        
        # parse sequence boundaries
        # layout: [CLS s0 CLS s1 CLS s2 ...] 
        seq_len = len(seqs[0])
        n_seqs = len(seqs)
        
        # extract just the target sequence's positions
        start = target_seq_idx * (seq_len + 1) + 1  # skip CLS tokens
        end = start + seq_len
        seq_relevance = relevance[start:end] if end <= len(relevance) else relevance[start:]
        
        info = {
            'total_len': len(relevance),
            'seq_len': seq_len,
            'target_seq_idx': target_seq_idx,
            'start_pos': start,
            'end_pos': end,
            'embedding': embeddings[target_seq_idx].cpu().numpy()
        }
        
        return seq_relevance, info
    
    def attribute_pairwise_distance(
        self,
        seqs: List[str],
        names: List[str],
        seq_i: int,
        seq_j: int
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        # attribute pairwise distance to positions in both sequences
        # returns separate relevance for seq_i and seq_j
        
        def target_fn(embeddings):
            return torch.norm(embeddings[seq_i] - embeddings[seq_j])
        
        emb, grad, embeddings = self._forward_with_grad(seqs, names, target_fn)
        
        if grad is None:
            return None, None, {}
        
        grad = grad[0]
        relevance = grad.norm(dim=-1).cpu().numpy()
        
        seq_len = len(seqs[0])
        
        # extract per-sequence relevance
        start_i = seq_i * (seq_len + 1) + 1
        start_j = seq_j * (seq_len + 1) + 1
        
        rel_i = relevance[start_i:start_i + seq_len]
        rel_j = relevance[start_j:start_j + seq_len]
        
        dist = torch.norm(embeddings[seq_i] - embeddings[seq_j]).item()
        
        info = {
            'distance': dist,
            'seq_len': seq_len
        }
        
        return rel_i, rel_j, info
    
    def find_important_positions(
        self,
        seqs: List[str],
        names: List[str],
        target_seq_idx: int,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        # find top-k most important positions for a sequence
        
        relevance, info = self.attribute_sequence(seqs, names, target_seq_idx)
        
        if relevance is None:
            return []
        
        top_idx = np.argsort(relevance)[-top_k:][::-1]
        return [(int(idx), float(relevance[idx])) for idx in top_idx]


def test_attributor():
    import sys
    sys.path.insert(0, '/home/shrey/work/LMRL/explainability/experiments')
    from utils import setup_phyla_env, load_phyla_model
    setup_phyla_env()
    
    print("loading model...")
    model = load_phyla_model(device='cuda')
    
    base = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAA"
    seqs = [base, base[:40] + "W" + base[41:], base]  # mutation at pos 40
    names = ["wt", "mut", "wt2"]
    
    attr = PhylaAttributor(model)
    
    print("\n1. attribute mutated sequence...")
    rel, info = attr.attribute_sequence(seqs, names, target_seq_idx=1)
    print(f"   relevance shape: {rel.shape}")
    print(f"   mutation at pos 40, relevance: {rel[40]:.4f}")
    print(f"   rank of pos 40: {(rel > rel[40]).sum()}")
    
    top5 = attr.find_important_positions(seqs, names, target_seq_idx=1, top_k=5)
    print(f"   top 5 positions: {top5}")
    
    print("\n2. attribute wild-type sequence...")
    rel_wt, _ = attr.attribute_sequence(seqs, names, target_seq_idx=0)
    print(f"   top 5: {attr.find_important_positions(seqs, names, target_seq_idx=0, top_k=5)}")
    
    print("\n3. pairwise distance attribution...")
    rel_i, rel_j, info = attr.attribute_pairwise_distance(seqs, names, 0, 1)
    print(f"   distance: {info['distance']:.4f}")
    print(f"   mut pos 40 in seq_j: {rel_j[40]:.4f}, rank: {(rel_j > rel_j[40]).sum()}")
    
    print("\nattributor test complete")


if __name__ == "__main__":
    test_attributor()
