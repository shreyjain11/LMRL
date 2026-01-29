"""
phyla vs esm-2 comparison
measures what msa context provides beyond single-sequence models
"""
from utils import setup_phyla_env, load_fasta, clean_sequences, seq_identity
from utils import CONSERVATIVE_SUBS, RADICAL_SUBS, STANDARD_AA, DATA_DIR
setup_phyla_env()

import torch
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, binomtest
import json
from datetime import datetime

# Import after setup
from phyla.model.model import Phyla, Config
import esm


class ModelComparison:
    def __init__(self, device='cuda'):
        self.device = device
        
        print("loading phyla...")
        config = Config()
        self.phyla = Phyla(config, device=device, name='phyla-beta')
        checkpoint_path = 'weights/11564369'
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)['state_dict']
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        self.phyla.load_state_dict(new_state_dict, strict=True)
        self.phyla.to(device)
        self.phyla.eval()
        
        print("loading esm-2 650m...")
        self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_model = self.esm_model.eval().to(device)
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
    
    def get_phyla_emb(self, sequences, names):
        """phyla cls embeddings (msa-aware)"""
        input_ids, cls_mask, seq_mask, _ = self.phyla.encode(sequences, names)
        input_ids = input_ids.to(self.device)
        cls_mask = cls_mask.to(self.device)
        seq_mask = seq_mask.to(self.device)
        with torch.no_grad():
            x = self.phyla.modul[0](input_ids, logits=False, position_ids=None,
                sequence_mask=seq_mask, cls_token_mask=cls_mask)
            for module in self.phyla.modul[1:]:
                dev = next(module.parameters()).device
                x = module(x.to(dev), hidden_states_given=True, logits=False,
                    position_ids=None, sequence_mask=seq_mask.to(dev),
                    cls_token_mask=cls_mask.to(dev))
        return x[0].cpu().numpy()
    
    def get_esm_emb(self, sequences, names):
        """esm-2 mean-pooled embeddings (single-sequence)"""
        clean_seqs = [s.replace("-", "") for s in sequences]
        embeddings = []
        for i, seq in enumerate(clean_seqs):
            data = [(names[i], seq)]
            _, _, batch_tokens = self.esm_batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)
            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
            emb = results["representations"][33][0, 1:-1, :].mean(dim=0).cpu().numpy()
            embeddings.append(emb)
        return np.array(embeddings)


def run_directional_test(get_emb, sequences, names, n_mutations=5):
    """test if mutations toward target move embedding toward target"""
    results = []
    n_seq = len(sequences)
    
    orig_emb = get_emb(sequences, names)
    
    for _ in range(15):
        idx_a, idx_b = np.random.choice(n_seq, 2, replace=False)
        seq_a, seq_b = sequences[idx_a], sequences[idx_b]
        
        diff_pos = [i for i in range(min(len(seq_a), len(seq_b)))
                   if seq_a[i] != seq_b[i] 
                   and seq_a[i] in STANDARD_AA 
                   and seq_b[i] in STANDARD_AA]
        
        if len(diff_pos) < n_mutations:
            continue
        
        orig_dist = np.linalg.norm(orig_emb[idx_a] - orig_emb[idx_b])
        
        positions = np.random.choice(diff_pos, n_mutations, replace=False)
        mut_seq = list(seq_a)
        for p in positions:
            mut_seq[p] = seq_b[p]
        mut_seqs = list(sequences)
        mut_seqs[idx_a] = "".join(mut_seq)
        mut_emb = get_emb(mut_seqs, names)
        new_dist = np.linalg.norm(mut_emb[idx_a] - orig_emb[idx_b])
        
        results.append(new_dist < orig_dist)
    
    return results


def run_variance_test(get_emb, sequences, names):
    """measure variance explained by sequence identity"""
    emb = get_emb(sequences, names)
    n = len(sequences)
    
    pairs_emb = []
    pairs_seq = []
    
    for i in range(n):
        for j in range(i+1, n):
            ed = float(np.linalg.norm(emb[i] - emb[j]))
            sd = 1 - seq_identity(sequences[i], sequences[j])
            pairs_emb.append(ed)
            pairs_seq.append(sd)
    
    if len(pairs_emb) > 2:
        r, _ = pearsonr(pairs_emb, pairs_seq)
        return r ** 2
    return None


def main():
    comp = ModelComparison()
    
    data_dir = Path(DATA_DIR)
    fastas = sorted(list(data_dir.glob("*_alignment.fasta")))
    
    results = {
        'phyla': {'directional': [], 'variance': []},
        'esm': {'directional': [], 'variance': []}
    }
    
    print("running phyla vs esm-2 comparison...")
    
    processed = 0
    for fasta in fastas:
        if processed >= 20:
            break
        
        names, seqs = load_fasta(fasta)
        if not (6 <= len(seqs) <= 12 and len(seqs[0]) <= 250):
            continue
        
        clean = clean_sequences(seqs)
        
        try:
            print(f"[{processed+1}] {fasta.stem}")
            
            for model_name, get_emb in [("phyla", comp.get_phyla_emb), ("esm", comp.get_esm_emb)]:
                dir_res = run_directional_test(get_emb, clean, names)
                results[model_name]['directional'].extend(dir_res)
                
                var = run_variance_test(get_emb, clean, names)
                if var:
                    results[model_name]['variance'].append(var)
            
            processed += 1
            
        except Exception as e:
            print(f"  error: {e}")
            continue
    
    print("\nresults summary")
    print(f"{'metric':<35} {'phyla':>15} {'esm-2':>15}")
    print("-" * 65)
    
    # directional accuracy
    phyla_dir = results['phyla']['directional']
    esm_dir = results['esm']['directional']
    phyla_dir_pct = 100 * sum(phyla_dir) / len(phyla_dir) if phyla_dir else 0
    esm_dir_pct = 100 * sum(esm_dir) / len(esm_dir) if esm_dir else 0
    print(f"{'directional accuracy (5-mut)':<35} {phyla_dir_pct:>14.1f}% {esm_dir_pct:>14.1f}%")
    
    # variance explained
    phyla_var = np.mean(results['phyla']['variance']) if results['phyla']['variance'] else 0
    esm_var = np.mean(results['esm']['variance']) if results['esm']['variance'] else 0
    print(f"{'seq identity variance explained':<35} {phyla_var*100:>14.1f}% {esm_var*100:>14.1f}%")
    
    # phylogenetic signal
    phyla_phylo = (1 - phyla_var) * 100
    esm_phylo = (1 - esm_var) * 100
    print(f"{'phylogenetic signal (unexplained)':<35} {phyla_phylo:>14.1f}% {esm_phylo:>14.1f}%")
    
    # statistical tests
    print("\nstatistical tests")
    
    if phyla_dir:
        p = binomtest(sum(phyla_dir), len(phyla_dir), 0.5, alternative="greater").pvalue
        print(f"phyla directional: {sum(phyla_dir)}/{len(phyla_dir)} toward, p = {p:.2e}")
    
    if esm_dir:
        p = binomtest(sum(esm_dir), len(esm_dir), 0.5, alternative="greater").pvalue
        print(f"esm-2 directional: {sum(esm_dir)}/{len(esm_dir)} toward, p = {p:.2e}")
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f'comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    save_results = {
        'phyla': {
            'directional_pct': phyla_dir_pct,
            'variance_explained': phyla_var,
            'n_directional': len(phyla_dir)
        },
        'esm': {
            'directional_pct': esm_dir_pct,
            'variance_explained': esm_var,
            'n_directional': len(esm_dir)
        }
    }
    
    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
