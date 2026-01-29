"""
blosum62 correlation test
measures if embedding distances correlate with blosum62 substitution scores
"""
from utils import setup_phyla_env, load_phyla_model, load_fasta, clean_sequences
from utils import STANDARD_AA, DATA_DIR
setup_phyla_env()

import torch
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import json
from datetime import datetime

# BLOSUM62 matrix
BLOSUM62 = {
    'A': {'A': 4, 'R':-1, 'N':-2, 'D':-2, 'C': 0, 'Q':-1, 'E':-1, 'G': 0, 'H':-2, 'I':-1, 'L':-1, 'K':-1, 'M':-1, 'F':-2, 'P':-1, 'S': 1, 'T': 0, 'W':-3, 'Y':-2, 'V': 0},
    'R': {'A':-1, 'R': 5, 'N': 0, 'D':-2, 'C':-3, 'Q': 1, 'E': 0, 'G':-2, 'H': 0, 'I':-3, 'L':-2, 'K': 2, 'M':-1, 'F':-3, 'P':-2, 'S':-1, 'T':-1, 'W':-3, 'Y':-2, 'V':-3},
    'N': {'A':-2, 'R': 0, 'N': 6, 'D': 1, 'C':-3, 'Q': 0, 'E': 0, 'G': 0, 'H': 1, 'I':-3, 'L':-3, 'K': 0, 'M':-2, 'F':-3, 'P':-2, 'S': 1, 'T': 0, 'W':-4, 'Y':-2, 'V':-3},
    'D': {'A':-2, 'R':-2, 'N': 1, 'D': 6, 'C':-3, 'Q': 0, 'E': 2, 'G':-1, 'H':-1, 'I':-3, 'L':-4, 'K':-1, 'M':-3, 'F':-3, 'P':-1, 'S': 0, 'T':-1, 'W':-4, 'Y':-3, 'V':-3},
    'C': {'A': 0, 'R':-3, 'N':-3, 'D':-3, 'C': 9, 'Q':-3, 'E':-4, 'G':-3, 'H':-3, 'I':-1, 'L':-1, 'K':-3, 'M':-1, 'F':-2, 'P':-3, 'S':-1, 'T':-1, 'W':-2, 'Y':-2, 'V':-1},
    'Q': {'A':-1, 'R': 1, 'N': 0, 'D': 0, 'C':-3, 'Q': 5, 'E': 2, 'G':-2, 'H': 0, 'I':-3, 'L':-2, 'K': 1, 'M': 0, 'F':-3, 'P':-1, 'S': 0, 'T':-1, 'W':-2, 'Y':-1, 'V':-2},
    'E': {'A':-1, 'R': 0, 'N': 0, 'D': 2, 'C':-4, 'Q': 2, 'E': 5, 'G':-2, 'H': 0, 'I':-3, 'L':-3, 'K': 1, 'M':-2, 'F':-3, 'P':-1, 'S': 0, 'T':-1, 'W':-3, 'Y':-2, 'V':-2},
    'G': {'A': 0, 'R':-2, 'N': 0, 'D':-1, 'C':-3, 'Q':-2, 'E':-2, 'G': 6, 'H':-2, 'I':-4, 'L':-4, 'K':-2, 'M':-3, 'F':-3, 'P':-2, 'S': 0, 'T':-2, 'W':-2, 'Y':-3, 'V':-3},
    'H': {'A':-2, 'R': 0, 'N': 1, 'D':-1, 'C':-3, 'Q': 0, 'E': 0, 'G':-2, 'H': 8, 'I':-3, 'L':-3, 'K':-1, 'M':-2, 'F':-1, 'P':-2, 'S':-1, 'T':-2, 'W':-2, 'Y': 2, 'V':-3},
    'I': {'A':-1, 'R':-3, 'N':-3, 'D':-3, 'C':-1, 'Q':-3, 'E':-3, 'G':-4, 'H':-3, 'I': 4, 'L': 2, 'K':-3, 'M': 1, 'F': 0, 'P':-3, 'S':-2, 'T':-1, 'W':-3, 'Y':-1, 'V': 3},
    'L': {'A':-1, 'R':-2, 'N':-3, 'D':-4, 'C':-1, 'Q':-2, 'E':-3, 'G':-4, 'H':-3, 'I': 2, 'L': 4, 'K':-2, 'M': 2, 'F': 0, 'P':-3, 'S':-2, 'T':-1, 'W':-2, 'Y':-1, 'V': 1},
    'K': {'A':-1, 'R': 2, 'N': 0, 'D':-1, 'C':-3, 'Q': 1, 'E': 1, 'G':-2, 'H':-1, 'I':-3, 'L':-2, 'K': 5, 'M':-1, 'F':-3, 'P':-1, 'S': 0, 'T':-1, 'W':-3, 'Y':-2, 'V':-2},
    'M': {'A':-1, 'R':-1, 'N':-2, 'D':-3, 'C':-1, 'Q': 0, 'E':-2, 'G':-3, 'H':-2, 'I': 1, 'L': 2, 'K':-1, 'M': 5, 'F': 0, 'P':-2, 'S':-1, 'T':-1, 'W':-1, 'Y':-1, 'V': 1},
    'F': {'A':-2, 'R':-3, 'N':-3, 'D':-3, 'C':-2, 'Q':-3, 'E':-3, 'G':-3, 'H':-1, 'I': 0, 'L': 0, 'K':-3, 'M': 0, 'F': 6, 'P':-4, 'S':-2, 'T':-2, 'W': 1, 'Y': 3, 'V':-1},
    'P': {'A':-1, 'R':-2, 'N':-2, 'D':-1, 'C':-3, 'Q':-1, 'E':-1, 'G':-2, 'H':-2, 'I':-3, 'L':-3, 'K':-1, 'M':-2, 'F':-4, 'P': 7, 'S':-1, 'T':-1, 'W':-4, 'Y':-3, 'V':-2},
    'S': {'A': 1, 'R':-1, 'N': 1, 'D': 0, 'C':-1, 'Q': 0, 'E': 0, 'G': 0, 'H':-1, 'I':-2, 'L':-2, 'K': 0, 'M':-1, 'F':-2, 'P':-1, 'S': 4, 'T': 1, 'W':-3, 'Y':-2, 'V':-2},
    'T': {'A': 0, 'R':-1, 'N': 0, 'D':-1, 'C':-1, 'Q':-1, 'E':-1, 'G':-2, 'H':-2, 'I':-1, 'L':-1, 'K':-1, 'M':-1, 'F':-2, 'P':-1, 'S': 1, 'T': 5, 'W':-2, 'Y':-2, 'V': 0},
    'W': {'A':-3, 'R':-3, 'N':-4, 'D':-4, 'C':-2, 'Q':-2, 'E':-3, 'G':-2, 'H':-2, 'I':-3, 'L':-2, 'K':-3, 'M':-1, 'F': 1, 'P':-4, 'S':-3, 'T':-2, 'W':11, 'Y': 2, 'V':-3},
    'Y': {'A':-2, 'R':-2, 'N':-2, 'D':-3, 'C':-2, 'Q':-1, 'E':-2, 'G':-3, 'H': 2, 'I':-1, 'L':-1, 'K':-2, 'M':-1, 'F': 3, 'P':-3, 'S':-2, 'T':-2, 'W': 2, 'Y': 7, 'V':-1},
    'V': {'A': 0, 'R':-3, 'N':-3, 'D':-3, 'C':-1, 'Q':-2, 'E':-2, 'G':-3, 'H':-3, 'I': 3, 'L': 1, 'K':-2, 'M': 1, 'F':-1, 'P':-2, 'S':-2, 'T': 0, 'W':-3, 'Y':-1, 'V': 4}
}


class BlosumCorrelationTest:
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
    
    def test_blosum_correlation(self, sequences, names, n_samples=50):
        """correlate embedding distances with blosum62 scores"""
        orig_emb = self.get_cls_embeddings(sequences, names)
        seq = sequences[0]
        
        blosum_scores = []
        emb_distances = []
        
        # sample random positions and substitutions
        positions = [i for i in range(len(seq)) if seq[i] in STANDARD_AA]
        if len(positions) < 10:
            return None
        
        sampled = 0
        for pos in np.random.choice(positions, min(len(positions), n_samples * 2), replace=False):
            if sampled >= n_samples:
                break
            aa_orig = seq[pos]
            for aa_new in STANDARD_AA:
                if aa_new == aa_orig:
                    continue
                # get blosum score
                blosum_score = BLOSUM62[aa_orig][aa_new]
                
                # make mutation
                mut_seqs = list(sequences)
                mut_seqs[0] = seq[:pos] + aa_new + seq[pos+1:]
                mut_emb = self.get_cls_embeddings(mut_seqs, names)
                
                emb_dist = float(np.linalg.norm(mut_emb[0] - orig_emb[0]))
                
                blosum_scores.append(blosum_score)
                emb_distances.append(emb_dist)
                sampled += 1
                break
        
        if len(blosum_scores) < 10:
            return None
        
        # higher blosum = more similar = expect smaller embedding distance
        # so we expect negative correlation
        r_pearson, p_pearson = pearsonr(blosum_scores, emb_distances)
        r_spearman, p_spearman = spearmanr(blosum_scores, emb_distances)
        
        return {
            'pearson_r': r_pearson,
            'pearson_p': p_pearson,
            'spearman_r': r_spearman,
            'spearman_p': p_spearman,
            'n_samples': len(blosum_scores)
        }


def main():
    print("loading model...")
    tester = BlosumCorrelationTest()
    
    data_dir = Path(DATA_DIR)
    fastas = sorted(list(data_dir.glob("*_alignment.fasta")))
    
    all_results = []
    
    print("running blosum correlation test...")
    
    processed = 0
    for fasta in fastas:
        if processed >= 30:
            break
        
        names, seqs = load_fasta(fasta)
        if not (6 <= len(seqs) <= 15 and len(seqs[0]) <= 300):
            continue
        
        clean = clean_sequences(seqs)
        
        try:
            result = tester.test_blosum_correlation(clean, names)
            if result:
                result['alignment'] = fasta.stem
                all_results.append(result)
                processed += 1
                
                if processed % 5 == 0:
                    avg_r = np.mean([r['spearman_r'] for r in all_results])
                    print(f"[{processed:3d}] avg spearman r = {avg_r:.3f}")
        
        except Exception as e:
            print(f"error {fasta.stem}: {e}")
            continue
    
    if not all_results:
        print("no results collected")
        return
    
    # summary
    pearson_rs = [r['pearson_r'] for r in all_results]
    spearman_rs = [r['spearman_r'] for r in all_results]
    
    print(f"\npearson r: {np.mean(pearson_rs):.3f} +/- {np.std(pearson_rs):.3f}")
    print(f"spearman r: {np.mean(spearman_rs):.3f} +/- {np.std(spearman_rs):.3f}")
    
    # count significant negative correlations
    sig_neg = sum(1 for r in all_results if r['spearman_r'] < 0 and r['spearman_p'] < 0.05)
    print(f"significant negative correlations: {sig_neg}/{len(all_results)}")
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f'blosum_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
