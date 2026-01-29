"""
directional mutation test
verifies mutations toward a target sequence move embeddings toward target
"""
from utils import setup_phyla_env, load_phyla_model, load_fasta, clean_sequences
from utils import STANDARD_AA, DATA_DIR
setup_phyla_env()

import torch
import numpy as np
from pathlib import Path
from scipy.stats import binomtest
import json
from datetime import datetime


class DirectionalMutationTest:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = load_phyla_model(device=device)
    
    def get_cls_embeddings(self, sequences, names):
        """extract cls token embeddings for all sequences"""
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
    
    def test_directional_movement(self, sequences, names, n_mutations=5, n_trials=10):
        """test if mutating seq_a toward seq_b moves embedding toward seq_b"""
        results = []
        n_seqs = len(sequences)
        orig_emb = self.get_cls_embeddings(sequences, names)
        
        for _ in range(n_trials):
            idx_a, idx_b = np.random.choice(n_seqs, 2, replace=False)
            seq_a, seq_b = sequences[idx_a], sequences[idx_b]
            
            diff_positions = []
            for i in range(min(len(seq_a), len(seq_b))):
                if seq_a[i] != seq_b[i] and seq_a[i] in STANDARD_AA and seq_b[i] in STANDARD_AA:
                    diff_positions.append(i)
            
            if len(diff_positions) < n_mutations:
                continue
            
            orig_dist = np.linalg.norm(orig_emb[idx_a] - orig_emb[idx_b])
            mut_positions = np.random.choice(diff_positions, n_mutations, replace=False)
            mut_seq = list(seq_a)
            for pos in mut_positions:
                mut_seq[pos] = seq_b[pos]
            
            mut_sequences = list(sequences)
            mut_sequences[idx_a] = ''.join(mut_seq)
            mut_emb = self.get_cls_embeddings(mut_sequences, names)
            new_dist = np.linalg.norm(mut_emb[idx_a] - orig_emb[idx_b])
            
            moved_closer = new_dist < orig_dist
            results.append({
                'idx_a': idx_a, 'idx_b': idx_b, 'n_mutations': n_mutations,
                'orig_dist': float(orig_dist), 'new_dist': float(new_dist),
                'moved_closer': moved_closer
            })
        return results


def main():
    print("loading model...")
    tester = DirectionalMutationTest()
    
    data_dir = Path(DATA_DIR)
    fastas = sorted(list(data_dir.glob("*_alignment.fasta")))
    all_results = []
    
    print("running directional mutation test...")
    
    processed = 0
    for fasta in fastas:
        if processed >= 30:
            break
        names, seqs = load_fasta(fasta)
        if not (6 <= len(seqs) <= 15 and len(seqs[0]) <= 300):
            continue
        clean = clean_sequences(seqs)
        
        try:
            for n_mut in [1, 3, 5]:
                results = tester.test_directional_movement(clean, names, n_mutations=n_mut, n_trials=10)
                for r in results:
                    r['n_mutations'] = n_mut
                    r['alignment'] = fasta.stem
                all_results.extend(results)
            processed += 1
            if processed % 5 == 0:
                successes = sum(1 for r in all_results if r['moved_closer'])
                total = len(all_results)
                print(f"[{processed:3d}] {successes}/{total} moved closer ({100*successes/total:.1f}%)")
        except Exception as e:
            print(f"error {fasta.stem}: {e}")
            continue
    
    if not all_results:
        print("no results collected")
        return
    
    print("\nresults by mutation count:")
    for n_mut in [1, 3, 5]:
        subset = [r for r in all_results if r['n_mutations'] == n_mut]
        if subset:
            successes = sum(1 for r in subset if r['moved_closer'])
            total = len(subset)
            pval = binomtest(successes, total, 0.5, alternative='greater').pvalue
            print(f"  {n_mut} mutations: {successes}/{total} ({100*successes/total:.1f}%) p={pval:.2e}")
    
    successes = sum(1 for r in all_results if r['moved_closer'])
    total = len(all_results)
    pval = binomtest(successes, total, 0.5, alternative='greater').pvalue
    print(f"\noverall: {successes}/{total} ({100*successes/total:.1f}%) p={pval:.2e}")
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f'directional_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
