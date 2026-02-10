"""
Structural validation of Phyla attributions using AlphaFold v6 structures.
"""
from utils import setup_phyla_env, load_phyla_model, load_fasta, clean_sequences
from utils import STANDARD_AA, DATA_DIR, NumpyEncoder
setup_phyla_env()

import torch
import numpy as np
import json
import sys
import os
import requests
import time
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

def download_alphafold_plddt(acc):
    cache_path = AF_CACHE / ('AF-' + acc + '-F1-confidence_v6.json')
    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
        return data.get('confidenceScore', data) if isinstance(data, dict) else data
    url = 'https://alphafold.ebi.ac.uk/files/AF-' + acc + '-F1-confidence_v6.json'
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            cache_path.write_text(r.text)
            data = r.json()
            return data.get('confidenceScore', data) if isinstance(data, dict) else data
    except:
        pass
    return None

def parse_pdb_ca_cb(pdb_path):
    ca_coords = {}
    cb_coords = {}
    with open(pdb_path) as f:
        for line in f:
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

def parse_pdb_plddt(pdb_path):
    plddt = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            atom_name = line[12:16].strip()
            if atom_name != 'CA':
                continue
            res_seq = int(line[22:26].strip())
            bfactor = float(line[60:66])
            plddt[res_seq] = bfactor
    return plddt

def compute_contact_map(coords, threshold=8.0):
    residues = sorted(coords.keys())
    n = len(residues)
    contacts = np.zeros((n, n), dtype=bool)
    coord_array = np.array([coords[r] for r in residues])
    for i in range(n):
        for j in range(i+5, n):
            dist = np.linalg.norm(coord_array[i] - coord_array[j])
            if dist < threshold:
                contacts[i, j] = True
                contacts[j, i] = True
    return contacts, residues

def compute_contact_number(coords, threshold=8.0):
    residues = sorted(coords.keys())
    coord_array = np.array([coords[r] for r in residues])
    cn = {}
    for i, r in enumerate(residues):
        count = 0
        for j in range(len(residues)):
            if abs(i - j) < 5:
                continue
            if np.linalg.norm(coord_array[i] - coord_array[j]) < threshold:
                count += 1
        cn[r] = count
    return cn

def get_attribution_scores(attributor, sequences, names, target_idx=0):
    try:
        rel, info = attributor.attribute_sequence(sequences, names, target_idx)
        return rel
    except Exception as e:
        return None

def msa_to_ungapped_positions(aligned_seq):
    mapping = {}
    res_num = 0
    for col, char in enumerate(aligned_seq):
        if char != '-':
            res_num += 1
            mapping[col] = res_num
    return mapping

def main():
    print("=" * 70)
    print("STRUCTURAL VALIDATION OF PHYLA ATTRIBUTIONS")
    print("Date: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70)

    np.random.seed(42)

    print("\nPhase 0: Loading model...")
    model = load_phyla_model(device='cuda')
    attributor = PhylaAttributor(model, device='cuda')

    data_dir = Path(DATA_DIR)
    fastas = sorted(list(data_dir.glob("*_alignment.fasta")))
    np.random.shuffle(fastas)

    plddt_corrs = []
    contact_num_corrs = []
    contact_enrichments = []
    plddt_high_vs_low = []

    processed = 0
    skipped_no_af = 0
    skipped_other = 0
    target_families = 50

    print("\nPhase 1: Processing families (target: " + str(target_families) + ")...\n")

    for fasta in fastas:
        if processed >= target_families:
            break

        names, seqs = load_fasta(fasta)
        if not (4 <= len(seqs) <= 15 and len(seqs[0]) <= 300):
            continue

        clean = clean_sequences(seqs)

        header = names[0]
        parts = header.split('|')
        if len(parts) < 2:
            continue
        accession = parts[1]

        pdb_path = download_alphafold_pdb(accession)
        if pdb_path is None:
            skipped_no_af += 1
            continue

        try:
            coords = parse_pdb_ca_cb(pdb_path)
            plddt = parse_pdb_plddt(pdb_path)
        except Exception as e:
            skipped_other += 1
            continue

        if len(coords) < 20:
            skipped_other += 1
            continue

        attribution = get_attribution_scores(attributor, clean, names, target_idx=0)
        if attribution is None:
            skipped_other += 1
            continue

        pos_map = msa_to_ungapped_positions(clean[0])

        attr_by_res = {}
        for col_idx in range(len(attribution)):
            if col_idx in pos_map:
                res_num = pos_map[col_idx]
                attr_by_res[res_num] = float(attribution[col_idx])

        common_res = sorted(set(attr_by_res.keys()) & set(coords.keys()) & set(plddt.keys()))
        if len(common_res) < 20:
            skipped_other += 1
            continue

        attr_arr = np.array([attr_by_res[r] for r in common_res])
        plddt_arr = np.array([plddt[r] for r in common_res])

        # Experiment 1: Attribution vs pLDDT
        r_plddt, p_plddt = spearmanr(attr_arr, plddt_arr)
        plddt_corrs.append({'r': r_plddt, 'p': p_plddt, 'n': len(common_res),
                            'accession': accession, 'family': fasta.stem})

        high_plddt_mask = plddt_arr >= 70
        low_plddt_mask = plddt_arr < 50
        if high_plddt_mask.sum() >= 5 and low_plddt_mask.sum() >= 5:
            high_attr = attr_arr[high_plddt_mask].mean()
            low_attr = attr_arr[low_plddt_mask].mean()
            plddt_high_vs_low.append({
                'high_plddt_attr': float(high_attr),
                'low_plddt_attr': float(low_attr),
                'accession': accession
            })

        # Experiment 2: Attribution vs contact number
        cn = compute_contact_number(coords)
        cn_arr = np.array([cn.get(r, 0) for r in common_res])
        if cn_arr.std() > 0:
            r_cn, p_cn = spearmanr(attr_arr, cn_arr)
            contact_num_corrs.append({'r': r_cn, 'p': p_cn, 'n': len(common_res),
                                      'accession': accession, 'family': fasta.stem})

        # Experiment 3: Contact enrichment
        contacts, res_list = compute_contact_map(coords)
        res_to_idx = {r: i for i, r in enumerate(res_list)}

        k = max(5, len(common_res) // 5)
        top_k_res = sorted(common_res, key=lambda r: attr_by_res[r], reverse=True)[:k]

        top_contacts = 0
        top_pairs = 0
        for i_idx, r1 in enumerate(top_k_res):
            for r2 in top_k_res[i_idx+1:]:
                if r1 in res_to_idx and r2 in res_to_idx:
                    top_pairs += 1
                    if contacts[res_to_idx[r1], res_to_idx[r2]]:
                        top_contacts += 1

        random_contacts_list = []
        for _ in range(100):
            rand_res = list(np.random.choice(common_res, k, replace=False))
            rc = 0
            rp = 0
            for i_idx, r1 in enumerate(rand_res):
                for r2 in rand_res[i_idx+1:]:
                    if r1 in res_to_idx and r2 in res_to_idx:
                        rp += 1
                        if contacts[res_to_idx[r1], res_to_idx[r2]]:
                            rc += 1
            random_contacts_list.append(rc / rp if rp > 0 else 0)

        top_frac = top_contacts / top_pairs if top_pairs > 0 else 0
        rand_mean = np.mean(random_contacts_list)
        enrichment = top_frac / rand_mean if rand_mean > 0 else 0

        contact_enrichments.append({
            'top_k_contact_frac': float(top_frac),
            'random_contact_frac': float(rand_mean),
            'enrichment': float(enrichment),
            'k': k,
            'n_residues': len(common_res),
            'accession': accession,
            'family': fasta.stem
        })

        processed += 1
        if processed % 10 == 0:
            avg_plddt_r = np.mean([x['r'] for x in plddt_corrs])
            finite_e = [x['enrichment'] for x in contact_enrichments if x['enrichment'] > 0]
            avg_enrich = np.mean(finite_e) if finite_e else 0
            print("[" + str(processed).rjust(3) + "] pLDDT r=" + 
                  "{:.3f}".format(avg_plddt_r) + ", contact enrichment=" +
                  "{:.2f}".format(avg_enrich) + "x, skipped_af=" + str(skipped_no_af))

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS (n=" + str(processed) + " families, skipped: " + 
          str(skipped_no_af) + " no AF, " + str(skipped_other) + " other)")
    print("=" * 70)

    if plddt_corrs:
        rs = [x['r'] for x in plddt_corrs]
        print("\n1. ATTRIBUTION vs pLDDT (structured regions)")
        print("   Spearman r: {:.3f} +/- {:.3f}".format(np.mean(rs), np.std(rs)))
        print("   Significant positive (p<0.05): " + 
              str(sum(1 for x in plddt_corrs if x['r'] > 0 and x['p'] < 0.05)) +
              "/" + str(len(plddt_corrs)))
        if plddt_high_vs_low:
            high_attrs = [x['high_plddt_attr'] for x in plddt_high_vs_low]
            low_attrs = [x['low_plddt_attr'] for x in plddt_high_vs_low]
            print("   High pLDDT (>=70) mean attr: {:.4f}".format(np.mean(high_attrs)))
            print("   Low pLDDT (<50) mean attr:   {:.4f}".format(np.mean(low_attrs)))
            if len(high_attrs) >= 5:
                u, p = mannwhitneyu(high_attrs, low_attrs, alternative='greater')
                print("   Mann-Whitney p: {:.2e}".format(p))

    if contact_num_corrs:
        rs = [x['r'] for x in contact_num_corrs]
        print("\n2. ATTRIBUTION vs CONTACT NUMBER (burial proxy)")
        print("   Spearman r: {:.3f} +/- {:.3f}".format(np.mean(rs), np.std(rs)))
        print("   Significant positive (p<0.05): " +
              str(sum(1 for x in contact_num_corrs if x['r'] > 0 and x['p'] < 0.05)) +
              "/" + str(len(contact_num_corrs)))

    if contact_enrichments:
        finite = [x for x in contact_enrichments if x['enrichment'] > 0]
        if finite:
            es = [x['enrichment'] for x in finite]
            print("\n3. CONTACT ENRICHMENT (top-L/5 attributed residues)")
            print("   Enrichment ratio: {:.2f}x +/- {:.2f}".format(np.mean(es), np.std(es)))
            print("   Top-k contact frac:  {:.3f}".format(
                np.mean([x['top_k_contact_frac'] for x in finite])))
            print("   Random contact frac: {:.3f}".format(
                np.mean([x['random_contact_frac'] for x in finite])))
            print("   Enrichment > 1.0: " + str(sum(1 for e in es if e > 1.0)) +
                  "/" + str(len(es)))

    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / ('structural_validation_' + 
               datetime.now().strftime("%Y%m%d_%H%M%S") + '.json')

    all_results = {
        'n_families': processed,
        'skipped_no_alphafold': skipped_no_af,
        'skipped_other': skipped_other,
        'plddt_correlations': plddt_corrs,
        'contact_number_correlations': contact_num_corrs,
        'contact_enrichments': contact_enrichments,
        'plddt_high_vs_low': plddt_high_vs_low
    }

    with open(out_path, 'w') as f:
        json.dump(all_results, f, cls=NumpyEncoder)
    print("\nSaved: " + str(out_path))


if __name__ == "__main__":
    main()
