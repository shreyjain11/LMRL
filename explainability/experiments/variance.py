"""
variance decomposition test
measures how much embedding variance is explained by sequence identity
"""
from utils import setup_phyla_env, load_phyla_model, load_fasta, clean_sequences, seq_identity
from utils import DATA_DIR
setup_phyla_env()

import torch
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import json
from datetime import datetime


class VarianceDecomposition:
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
    
    def analyze_variance(self, sequences, names):
        """decompose embedding variance by sequence identity"""
        emb = self.get_cls_embeddings(sequences, names)
        n = len(sequences)
        
        # compute pairwise distances
        emb_distances = []
        seq_distances = []
        
        for i in range(n):
            for j in range(i+1, n):
                ed = float(np.linalg.norm(emb[i] - emb[j]))
                sd = 1 - seq_identity(sequences[i], sequences[j])
                emb_distances.append(ed)
                seq_distances.append(sd)
        
        if len(emb_distances) < 3:
            return None
        
        r, p = pearsonr(emb_distances, seq_distances)
        r_squared = r ** 2
        
        return {
            'r': r,
            'r_squared': r_squared,
            'p_value': p,
            'n_pairs': len(emb_distances),
            'mean_emb_dist': float(np.mean(emb_distances)),
            'mean_seq_dist': float(np.mean(seq_distances))
        }


def main():
    print("loading model...")
    analyzer = VarianceDecomposition()
    
    data_dir = Path(DATA_DIR)
    fastas = sorted(list(data_dir.glob("*_alignment.fasta")))
    
    all_results = []
    
    print("running variance decomposition...")
    
    processed = 0
    for fasta in fastas:
        if processed >= 30:
            break
        
        names, seqs = load_fasta(fasta)
        if not (6 <= len(seqs) <= 15 and len(seqs[0]) <= 300):
            continue
        
        clean = clean_sequences(seqs)
        
        try:
            result = analyzer.analyze_variance(clean, names)
            if result:
                result['alignment'] = fasta.stem
                all_results.append(result)
                processed += 1
                
                if processed % 5 == 0:
                    avg_r2 = np.mean([r['r_squared'] for r in all_results])
                    print(f"[{processed:3d}] avg R² = {avg_r2:.3f}")
        
        except Exception as e:
            print(f"error {fasta.stem}: {e}")
            continue
    
    if not all_results:
        print("no results collected")
        return
    
    # summary
    r_values = [r['r'] for r in all_results]
    r2_values = [r['r_squared'] for r in all_results]
    
    print(f"\npearson r: {np.mean(r_values):.3f} +/- {np.std(r_values):.3f}")
    print(f"R² (variance explained): {np.mean(r2_values):.3f} +/- {np.std(r2_values):.3f}")
    print(f"phylogenetic signal (unexplained): {1 - np.mean(r2_values):.3f}")
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f'riance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
