"""
substitution sensitivity test
measures if phyla distinguishes conservative from radical amino acid changes
"""
from utils import setup_phyla_env, load_phyla_model, load_fasta, clean_sequences
from utils import CONSERVATIVE_SUBS, RADICAL_SUBS, STANDARD_AA, DATA_DIR
setup_phyla_env()

import torch
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
import json
from datetime import datetime


class SubstitutionAnalyzer:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = load_phyla_model(device=device)
    
    def get_cls_embedding(self, sequences, names):
        """extract cls token embedding from final layer"""
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
        # output is (1, n_seqs, hidden_dim) -> squeeze batch dim
        return x[0].cpu().numpy()
    
    def measure_sub_effect(self, sequences, names, seq_idx, pos, old_aa, new_aa):
        """measure embedding shift from single substitution"""
        orig_emb = self.get_cls_embedding(sequences, names)
        mut_seqs = list(sequences)
        seq_list = list(mut_seqs[seq_idx])
        if seq_list[pos] != old_aa:
            return None
        seq_list[pos] = new_aa
        mut_seqs[seq_idx] = ''.join(seq_list)
        mut_emb = self.get_cls_embedding(mut_seqs, names)
        return float(np.linalg.norm(mut_emb[seq_idx] - orig_emb[seq_idx]))
    
    def analyze_alignment(self, sequences, names, n_subs=30):
        """collect conservative and radical substitution effects"""
        results = {'conservative': [], 'radical': []}
        for seq_idx in range(min(3, len(sequences))):
            seq = sequences[seq_idx]
            for pos in range(len(seq)):
                aa = seq[pos]
                if aa not in STANDARD_AA:
                    continue
                # try conservative substitutions
                for old, new in CONSERVATIVE_SUBS:
                    if aa == old and len(results['conservative']) < n_subs:
                        eff = self.measure_sub_effect(list(sequences), names, seq_idx, pos, old, new)
                        if eff:
                            results['conservative'].append(eff)
                        break
                # try radical substitutions
                for old, new in RADICAL_SUBS:
                    if aa == old and len(results['radical']) < n_subs:
                        eff = self.measure_sub_effect(list(sequences), names, seq_idx, pos, old, new)
                        if eff:
                            results['radical'].append(eff)
                        break
                if len(results['conservative']) >= n_subs and len(results['radical']) >= n_subs:
                    return results
        return results


def main():
    print("loading model...")
    analyzer = SubstitutionAnalyzer()
    
    data_dir = Path(DATA_DIR)
    fastas = sorted(list(data_dir.glob("*_alignment.fasta")))
    
    # sanity check on first valid alignment
    for fasta in fastas[:50]:
        names, seqs = load_fasta(fasta)
        if 6 <= len(seqs) <= 15 and len(seqs[0]) <= 300:
            print(f"test: {fasta.stem}, {len(seqs)} seqs, len={len(seqs[0])}")
            clean = clean_sequences(seqs)
            emb = analyzer.get_cls_embedding(clean, names)
            print(f"embedding shape: {emb.shape}")
            break
    
    all_c, all_r = [], []
    
    print("running substitution sensitivity analysis...")
    
    processed = 0
    for fasta in fastas:
        if processed >= 30:
            break
        names, seqs = load_fasta(fasta)
        if not (6 <= len(seqs) <= 15 and len(seqs[0]) <= 300):
            continue
        clean = clean_sequences(seqs)
        
        try:
            res = analyzer.analyze_alignment(clean, names)
            all_c.extend(res['conservative'])
            all_r.extend(res['radical'])
            processed += 1
            if processed % 5 == 0:
                cm = np.mean(all_c) if all_c else 0
                rm = np.mean(all_r) if all_r else 0
                print(f"[{processed:3d}] cons: {cm:.4f} | rad: {rm:.4f} | n={len(all_c)}/{len(all_r)}")
        except Exception as e:
            print(f"error {fasta.stem}: {e}")
            continue
    
    # final results
    if all_c and all_r:
        print(f"\nconservative (n={len(all_c)}): {np.mean(all_c):.4f} +/- {np.std(all_c):.4f}")
        print(f"radical (n={len(all_r)}): {np.mean(all_r):.4f} +/- {np.std(all_r):.4f}")
        
        stat, pval = mannwhitneyu(all_r, all_c, alternative='greater')
        print(f"mann-whitney p = {pval:.2e}")
        if pval < 0.05:
            effect = (np.mean(all_r) - np.mean(all_c)) / np.std(all_c)
            print(f"effect size = {effect:.2f}")
    else:
        print("no data collected")
    
    # save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f'substitution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump({'conservative': all_c, 'radical': all_r}, f)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
