"""
conservation sensitivity test
tests if mutations at conserved positions cause larger embedding shifts
"""
from utils import setup_phyla_env, load_phyla_model, load_fasta, clean_sequences
from utils import STANDARD_AA, DATA_DIR
setup_phyla_env()

import torch
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr
import json
from datetime import datetime


def compute_conservation(sequences):
    """compute per-position conservation score (fraction of most common AA)"""
    n_seqs = len(sequences)
    seq_len = len(sequences[0])
    conservation = []
    
    for pos in range(seq_len):
        col = [s[pos] for s in sequences if pos < len(s)]
        col_aa = [c for c in col if c in STANDARD_AA]
        if not col_aa:
            conservation.append(0)
            continue
        from collections import Counter
        counts = Counter(col_aa)
        most_common = counts.most_common(1)[0][1]
        conservation.append(most_common / len(col_aa))
    
    return conservation


class ConservationEffectTest:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = load_phyla_model(device=device)
    
    def get_cls_embeddings(self, sequences, names):
        """extract cls token embeddings"""
        input_ids, cls_mask, seq_mask, _ = self.model.encode(sequences, names)
        input_ids = input_ids.to(self.device)
        cls_mask = cls_mask.to(self.device)
        seq_mask = seq_mask.to(self.device)
        
        with torch.no_grad():
            x = self.model.modul[0](input_ids, logits=False, position_ids=None,
                sequence_mask=seq_mask, cls_token_mask=cls_mask)
            for module in self.model.modul[1:]:
                dev = next(module.parameters()).device
                x = module(x.to(dev), hidden_states_given=True, logits=False,
                    position_ids=None, sequence_mask=seq_mask.to(dev),
                    cls_token_mask=cls_mask.to(dev))
        return x[0].cpu().numpy()
    
    def test_conservation_effect(self, sequences, names, n_samples=40):
        """test if conserved positions are more sensitive"""
        conservation = compute_conservation(sequences)
        orig_emb = self.get_cls_embeddings(sequences, names)
        seq = sequences[0]
        
        results = []
        positions = [i for i in range(len(seq)) if seq[i] in STANDARD_AA]
        
        for pos in np.random.choice(positions, min(len(positions), n_samples), replace=False):
            aa_orig = seq[pos]
            # pick a different AA
            other_aas = [aa for aa in STANDARD_AA if aa != aa_orig]
            aa_new = np.random.choice(list(other_aas))
            
            mut_seqs = list(sequences)
            mut_seqs[0] = seq[:pos] + aa_new + seq[pos+1:]
            mut_emb = self.get_cls_embeddings(mut_seqs, names)
            
            emb_dist = float(np.linalg.norm(mut_emb[0] - orig_emb[0]))
            results.append({
                'position': pos,
                'conservation': conservation[pos],
                'emb_distance': emb_dist
            })
        
        return results


def main():
    print("loading model...")
    tester = ConservationEffectTest()
    
    data_dir = Path(DATA_DIR)
    fastas = sorted(list(data_dir.glob("*_alignment.fasta")))
    
    all_results = []
    
    print("running conservation effect test...")
    
    processed = 0
    for fasta in fastas:
        if processed >= 30:
            break
        
        names, seqs = load_fasta(fasta)
        if not (6 <= len(seqs) <= 15 and len(seqs[0]) <= 300):
            continue
        
        clean = clean_sequences(seqs)
        
        try:
            results = tester.test_conservation_effect(clean, names)
            for r in results:
                r['alignment'] = fasta.stem
            all_results.extend(results)
            processed += 1
            
            if processed % 5 == 0:
                cons = [r['conservation'] for r in all_results]
                dists = [r['emb_distance'] for r in all_results]
                r, _ = spearmanr(cons, dists)
                print(f"[{processed:3d}] spearman r = {r:.3f} (n={len(all_results)})")
        
        except Exception as e:
            print(f"error {fasta.stem}: {e}")
            continue
    
    if not all_results:
        print("no results collected")
        return
    
    # analyze
    cons = [r['conservation'] for r in all_results]
    dists = [r['emb_distance'] for r in all_results]
    
    r_spearman, p_spearman = spearmanr(cons, dists)
    print(f"\nspearman r = {r_spearman:.3f}, p = {p_spearman:.2e}")
    
    # compare high vs low conservation
    median_cons = np.median(cons)
    high_cons = [r['emb_distance'] for r in all_results if r['conservation'] >= median_cons]
    low_cons = [r['emb_distance'] for r in all_results if r['conservation'] < median_cons]
    
    stat, pval = mannwhitneyu(high_cons, low_cons, alternative='greater')
    print(f"high vs low conservation: {np.mean(high_cons):.4f} vs {np.mean(low_cons):.4f}, p={pval:.2e}")
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f'conservation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
