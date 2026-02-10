"""
epistasis test
detects non-additive interactions between mutations
"""
from utils import NumpyEncoder,  setup_phyla_env, load_phyla_model, load_fasta, clean_sequences
from utils import NumpyEncoder,  STANDARD_AA, DATA_DIR
setup_phyla_env()

import torch
import numpy as np
from pathlib import Path
from scipy.stats import ttest_1samp
import json
from datetime import datetime


class EpistasisTest:
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
    
    def test_epistasis(self, sequences, names, n_pairs=15):
        """test for non-additive mutation effects"""
        seq = sequences[0]
        orig_emb = self.get_cls_embeddings(sequences, names)
        
        # find mutable positions
        positions = [i for i in range(len(seq)) if seq[i] in STANDARD_AA]
        if len(positions) < 10:
            return []
        
        results = []
        
        for _ in range(n_pairs):
            # pick two random positions
            pos1, pos2 = np.random.choice(positions, 2, replace=False)
            aa1_orig, aa2_orig = seq[pos1], seq[pos2]
            
            # pick new AAs
            other1 = [aa for aa in STANDARD_AA if aa != aa1_orig]
            other2 = [aa for aa in STANDARD_AA if aa != aa2_orig]
            aa1_new = np.random.choice(list(other1))
            aa2_new = np.random.choice(list(other2))
            
            # single mutations
            mut1_seqs = list(sequences)
            mut1_seqs[0] = seq[:pos1] + aa1_new + seq[pos1+1:]
            mut1_emb = self.get_cls_embeddings(mut1_seqs, names)
            delta1 = mut1_emb[0] - orig_emb[0]
            
            mut2_seqs = list(sequences)
            mut2_seqs[0] = seq[:pos2] + aa2_new + seq[pos2+1:]
            mut2_emb = self.get_cls_embeddings(mut2_seqs, names)
            delta2 = mut2_emb[0] - orig_emb[0]
            
            # double mutation
            double_seq = list(seq)
            double_seq[pos1] = aa1_new
            double_seq[pos2] = aa2_new
            double_seqs = list(sequences)
            double_seqs[0] = ''.join(double_seq)
            double_emb = self.get_cls_embeddings(double_seqs, names)
            delta_double = double_emb[0] - orig_emb[0]
            
            # epistasis = actual - expected (if additive)
            expected = delta1 + delta2
            epistasis = delta_double - expected
            epistasis_magnitude = float(np.linalg.norm(epistasis))
            
            # relative epistasis (normalized by expected magnitude)
            expected_mag = float(np.linalg.norm(expected))
            relative_epistasis = epistasis_magnitude / expected_mag if expected_mag > 0 else 0
            
            results.append({
                'pos1': pos1,
                'pos2': pos2,
                'distance': abs(pos2 - pos1),
                'epistasis_magnitude': epistasis_magnitude,
                'relative_epistasis': relative_epistasis,
                'expected_magnitude': expected_mag,
                'actual_magnitude': float(np.linalg.norm(delta_double))
            })
        
        return results


def main():
    print("loading model...")
    tester = EpistasisTest()
    
    data_dir = Path(DATA_DIR)
    fastas = sorted(list(data_dir.glob("*_alignment.fasta")))
    
    all_results = []
    
    print("running epistasis test...")
    
    processed = 0
    for fasta in fastas:
        if processed >= 30:
            break
        
        names, seqs = load_fasta(fasta)
        if not (6 <= len(seqs) <= 15 and len(seqs[0]) <= 300):
            continue
        
        clean = clean_sequences(seqs)
        
        try:
            results = tester.test_epistasis(clean, names)
            for r in results:
                r['alignment'] = fasta.stem
            all_results.extend(results)
            processed += 1
            
            if processed % 5 == 0:
                rel_ep = [r['relative_epistasis'] for r in all_results]
                print(f"[{processed:3d}] mean relative epistasis = {np.mean(rel_ep):.3f} (n={len(all_results)})")
        
        except Exception as e:
            print(f"error {fasta.stem}: {e}")
            continue
    
    if not all_results:
        print("no results collected")
        return
    
    # analyze
    rel_ep = [r['relative_epistasis'] for r in all_results]
    
    print(f"\nrelative epistasis: {np.mean(rel_ep):.3f} +/- {np.std(rel_ep):.3f}")
    
    # test if significantly different from 0
    t_stat, p_val = ttest_1samp(rel_ep, 0)
    print(f"t-test vs 0: t={t_stat:.2f}, p={p_val:.2e}")
    
    # analyze by distance
    close_pairs = [r['relative_epistasis'] for r in all_results if r['distance'] < 10]
    far_pairs = [r['relative_epistasis'] for r in all_results if r['distance'] >= 10]
    
    if close_pairs and far_pairs:
        print(f"close pairs (<10 aa): {np.mean(close_pairs):.3f}")
        print(f"far pairs (>=10 aa): {np.mean(far_pairs):.3f}")
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f'epistasis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, cls=NumpyEncoder)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
