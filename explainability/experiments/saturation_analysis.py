"""
Saturation regime analysis: stratify families by sequence identity
and measure how Phyla structural/evolutionary signal changes.
"""
from utils import setup_phyla_env, load_phyla_model, load_fasta, clean_sequences
from utils import STANDARD_AA, DATA_DIR, NumpyEncoder, seq_identity
from utils import CONSERVATIVE_SUBS, RADICAL_SUBS
setup_phyla_env()

import torch
import numpy as np
import json
import sys
import os
import requests
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from datetime import datetime

sys.path.insert(0, '/home/shrey/work/LMRL/explainability')
from integrations.mamba_lrp import PhylaAttributor

AF_CACHE = Path(__file__).parent.parent / 'data' / 'alphafold_cache'
AF_CACHE.mkdir(parents=True, exist_ok=True)

def download_alphafold_pdb(acc):
    pdb_path = AF_CACHE / ('AF-' + acc + '-F1-model_v6.pdb')
    if pdb_path.exists():
        return pdb_path
    url = 'https://alphafold.ebi.ac.uk/files/AF-' + acc + '-F1-model_v6.pdb'
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            pdb_path.write_text(r.text)
            return pdb_path
    except:
        pass
    return None

def parse_pdb_ca_cb(pdb_path):
    ca_coords = {}
    cb_coords = {}
    with open(pdb_path) as fp:
        for line in fp:
            if not line.startswith('ATOM'):
                continue
            atom_name = line[12:16].strip()
            res_seq = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            if atom_name == 'CA':
                ca_coords[res_seq] = np.array([x, y, z])
            elif atom_name == 'CB':
                cb_coords[res_seq] = np.array([x, y, z])
    coords = {}
    for res in ca_coords:
        coords[res] = cb_coords.get(res, ca_coords[res])
    return coords

def compute_contact_map(coords, threshold=8.0):
    residues = sorted(coords.keys())
    n = len(residues)
    contacts = np.zeros((n, n), dtype=bool)
    coord_array = np.array([coords[r] for r in residues])
    for i in range(n):
        for j in range(i + 5, n):
            dist = np.linalg.norm(coord_array[i] - coord_array[j])
            if dist < threshold:
                contacts[i, j] = True
                contacts[j, i] = True
    return contacts, residues

def msa_to_ungapped_positions(aligned_seq):
    mapping = {}
    res_num = 0
    for col, char in enumerate(aligned_seq):
        if char != '-':
            res_num += 1
            mapping[col] = res_num
    return mapping

def compute_mean_pairwise_identity(seqs):
    ids = []
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            ids.append(seq_identity(seqs[i], seqs[j]))
    return np.mean(ids) if ids else 0

def compute_contact_enrichment(attributor, clean, names, coords):
    try:
        rel, info = attributor.attribute_sequence(clean, names, 0)
    except:
        return None
    pos_map = msa_to_ungapped_positions(clean[0])
    attr_by_res = {}
    for col_idx in range(len(rel)):
        if col_idx in pos_map:
            attr_by_res[pos_map[col_idx]] = float(rel[col_idx])
    contacts, res_list = compute_contact_map(coords)
    res_to_idx = {r: i for i, r in enumerate(res_list)}
    common_res = sorted(set(attr_by_res.keys()) & set(coords.keys()))
    if len(common_res) < 20:
        return None
    k = max(5, len(common_res) // 5)
    top_k_res = sorted(common_res, key=lambda r: attr_by_res.get(r, 0), reverse=True)[:k]
    top_contacts = 0
    top_pairs = 0
    for ii, r1 in enumerate(top_k_res):
        for r2 in top_k_res[ii + 1:]:
            if r1 in res_to_idx and r2 in res_to_idx:
                top_pairs += 1
                if contacts[res_to_idx[r1], res_to_idx[r2]]:
                    top_contacts += 1
    random_fracs = []
    for _ in range(100):
        rand_res = list(np.random.choice(common_res, k, replace=False))
        rc = 0
        rp = 0
        for ii, r1 in enumerate(rand_res):
            for r2 in rand_res[ii + 1:]:
                if r1 in res_to_idx and r2 in res_to_idx:
                    rp += 1
                    if contacts[res_to_idx[r1], res_to_idx[r2]]:
                        rc += 1
        random_fracs.append(rc / rp if rp > 0 else 0)
    top_frac = top_contacts / top_pairs if top_pairs > 0 else 0
    rand_mean = np.mean(random_fracs)
    enrichment = top_frac / rand_mean if rand_mean > 0 else 0
    return enrichment

def get_cls_embeddings(model, sequences, names, device='cuda'):
    input_ids, cls_mask, seq_mask, _ = model.encode(sequences, names)
    input_ids = input_ids.to(device)
    cls_mask = cls_mask.to(device)
    seq_mask = seq_mask.to(device)
    with torch.no_grad():
        x = model.modul[0](input_ids, logits=False, position_ids=None,
            sequence_mask=seq_mask, cls_token_mask=cls_mask)
        for module in model.modul[1:]:
            dev = next(module.parameters()).device
            x = module(x.to(dev), hidden_states_given=True, logits=False,
                position_ids=None, sequence_mask=seq_mask.to(dev),
                cls_token_mask=cls_mask.to(dev))
    return x[0].cpu().numpy()

def compute_substitution_d(model, sequences, names, device='cuda'):
    orig_emb = get_cls_embeddings(model, sequences, names, device)
    seq = sequences[0]
    positions = [i for i in range(len(seq)) if seq[i] in STANDARD_AA]
    if len(positions) < 10:
        return None
    conservative_shifts = []
    radical_shifts = []
    n_each = 15
    for aa_from, aa_to in CONSERVATIVE_SUBS[:5]:
        count = 0
        for pos in positions:
            if count >= n_each:
                break
            if seq[pos] == aa_from:
                mut = list(sequences)
                mut[0] = seq[:pos] + aa_to + seq[pos + 1:]
                mut_emb = get_cls_embeddings(model, mut, names, device)
                conservative_shifts.append(float(np.linalg.norm(mut_emb[0] - orig_emb[0])))
                count += 1
    for aa_from, aa_to in RADICAL_SUBS[:5]:
        count = 0
        for pos in positions:
            if count >= n_each:
                break
            if seq[pos] == aa_from:
                mut = list(sequences)
                mut[0] = seq[:pos] + aa_to + seq[pos + 1:]
                mut_emb = get_cls_embeddings(model, mut, names, device)
                radical_shifts.append(float(np.linalg.norm(mut_emb[0] - orig_emb[0])))
                count += 1
    if len(conservative_shifts) < 5 or len(radical_shifts) < 5:
        return None
    c = np.array(conservative_shifts)
    r = np.array(radical_shifts)
    pooled_std = np.sqrt((c.std()**2 + r.std()**2) / 2)
    if pooled_std == 0:
        return None
    d = (r.mean() - c.mean()) / pooled_std
    return d

def compute_directional_accuracy(model, sequences, names, n_muts=3, n_trials=20, device='cuda'):
    if len(sequences) < 2:
        return None
    orig_emb = get_cls_embeddings(model, sequences, names, device)
    seq_i = sequences[0]
    seq_j = sequences[1]
    diff_positions = [p for p in range(min(len(seq_i), len(seq_j)))
                      if seq_i[p] != seq_j[p] and seq_i[p] in STANDARD_AA and seq_j[p] in STANDARD_AA]
    if len(diff_positions) < n_muts + 1:
        return None
    correct = 0
    total = 0
    orig_dist = float(np.linalg.norm(orig_emb[0] - orig_emb[1]))
    for _ in range(n_trials):
        chosen = list(np.random.choice(diff_positions, n_muts, replace=False))
        mut_seq = list(seq_i)
        for p in chosen:
            mut_seq[p] = seq_j[p]
        mut_sequences = list(sequences)
        mut_sequences[0] = ''.join(mut_seq)
        mut_emb = get_cls_embeddings(model, mut_sequences, names, device)
        mut_dist = float(np.linalg.norm(mut_emb[0] - mut_emb[1]))
        if mut_dist < orig_dist:
            correct += 1
        total += 1
    return correct / total if total > 0 else None

def main():
    print("=" * 70)
    print("SATURATION REGIME ANALYSIS")
    print("Date: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70)
    np.random.seed(42)
    print("\nLoading model...")
    device = 'cuda'
    model = load_phyla_model(device=device)
    attributor = PhylaAttributor(model, device=device)
    data_dir = Path(DATA_DIR)
    fastas = sorted(list(data_dir.glob("*_alignment.fasta")))
    np.random.shuffle(fastas)
    bins = [
        ('< 30%', 0.0, 0.3),
        ('30-50%', 0.3, 0.5),
        ('50-70%', 0.5, 0.7),
        ('> 70%', 0.7, 1.01),
    ]
    bin_results = {b[0]: [] for b in bins}
    target_per_bin = 20
    total_processed = 0
    total_skipped_af = 0
    print("\nTarget: " + str(target_per_bin) + " families per bin (" +
          str(len(bins)) + " bins = " + str(target_per_bin * len(bins)) + " total)")
    print("Bins: " + ", ".join(b[0] for b in bins))
    print()
    for fasta in fastas:
        if all(len(bin_results[b[0]]) >= target_per_bin for b in bins):
            break
        names_raw, seqs = load_fasta(fasta)
        if not (4 <= len(seqs) <= 15 and len(seqs[0]) <= 300):
            continue
        clean = clean_sequences(seqs)
        mean_id = compute_mean_pairwise_identity(clean)
        target_bin = None
        for bname, blo, bhi in bins:
            if blo <= mean_id < bhi and len(bin_results[bname]) < target_per_bin:
                target_bin = bname
                break
        if target_bin is None:
            continue
        header = names_raw[0]
        parts = header.split('|')
        accession = parts[1] if len(parts) >= 2 else None
        contact_enrich = None
        if accession:
            pdb_path = download_alphafold_pdb(accession)
            if pdb_path is not None:
                try:
                    coords = parse_pdb_ca_cb(pdb_path)
                    if len(coords) >= 20:
                        contact_enrich = compute_contact_enrichment(
                            attributor, clean, names_raw, coords)
                except:
                    pass
            else:
                total_skipped_af += 1
        sub_d = compute_substitution_d(model, clean, names_raw, device)
        dir_acc = compute_directional_accuracy(model, clean, names_raw,
                                                n_muts=3, n_trials=20, device=device)
        result = {
            'family': fasta.stem,
            'accession': accession,
            'mean_identity': float(mean_id),
            'n_seqs': len(seqs),
            'seq_len': len(seqs[0]),
            'contact_enrichment': contact_enrich,
            'substitution_d': sub_d,
            'directional_accuracy': dir_acc,
        }
        bin_results[target_bin].append(result)
        total_processed += 1
        counts = {b[0]: len(bin_results[b[0]]) for b in bins}
        if total_processed % 10 == 0:
            status = ", ".join(b[0] + ":" + str(counts[b[0]]) for b in bins)
            print("[" + str(total_processed).rjust(3) + "] " + status)
    print("\n" + "=" * 70)
    print("RESULTS BY SEQUENCE IDENTITY BIN")
    print("=" * 70)
    summary = {}
    for bname, blo, bhi in bins:
        results = bin_results[bname]
        n = len(results)
        if n == 0:
            print("\n" + bname + ": NO DATA")
            continue
        ce_vals = [r['contact_enrichment'] for r in results if r['contact_enrichment'] is not None]
        sd_vals = [r['substitution_d'] for r in results if r['substitution_d'] is not None]
        da_vals = [r['directional_accuracy'] for r in results if r['directional_accuracy'] is not None]
        print("\n" + bname + " (n=" + str(n) + " families)")
        print("  Mean seq identity: {:.3f}".format(
            np.mean([r['mean_identity'] for r in results])))
        if ce_vals:
            print("  Contact enrichment: {:.2f}x +/- {:.2f} (n={})".format(
                np.mean(ce_vals), np.std(ce_vals), len(ce_vals)))
        else:
            print("  Contact enrichment: no data")
        if sd_vals:
            print("  Substitution d:     {:.2f} +/- {:.2f} (n={})".format(
                np.mean(sd_vals), np.std(sd_vals), len(sd_vals)))
        else:
            print("  Substitution d:     no data")
        if da_vals:
            print("  Directional acc:    {:.1f}% +/- {:.1f}% (n={})".format(
                np.mean(da_vals) * 100, np.std(da_vals) * 100, len(da_vals)))
        else:
            print("  Directional acc:    no data")
        summary[bname] = {
            'n_families': n,
            'mean_identity': float(np.mean([r['mean_identity'] for r in results])),
            'contact_enrichment_mean': float(np.mean(ce_vals)) if ce_vals else None,
            'contact_enrichment_std': float(np.std(ce_vals)) if ce_vals else None,
            'contact_enrichment_n': len(ce_vals),
            'substitution_d_mean': float(np.mean(sd_vals)) if sd_vals else None,
            'substitution_d_std': float(np.std(sd_vals)) if sd_vals else None,
            'substitution_d_n': len(sd_vals),
            'directional_acc_mean': float(np.mean(da_vals)) if da_vals else None,
            'directional_acc_std': float(np.std(da_vals)) if da_vals else None,
            'directional_acc_n': len(da_vals),
        }
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / ('saturation_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.json')
    all_out = {
        'summary': summary,
        'per_family': bin_results,
        'total_processed': total_processed,
        'skipped_no_alphafold': total_skipped_af,
    }
    with open(out_path, 'w') as f:
        json.dump(all_out, f, cls=NumpyEncoder)
    print("\nSaved: " + str(out_path))

if __name__ == "__main__":
    main()
