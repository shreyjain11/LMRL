#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from phyla.utils.utils import load_config
from phyla.eval.evo_reasoning_eval import Config, generate_tree, tier1_metric
from phyla import phyla
import pickle
from tqdm import tqdm


def create_random_model(config, device):
    model = phyla(config=config, name=config.model.model_name, custom_arch=False, device=device)
    model = model.to(device)
    model.eval()
    return model


def run_treebase_benchmark(model, dataset_names, dataset_type_label, device):
    seq_dir_path = "./treebase_benchmark/sequences/"
    tree_dir_path = "./treebase_benchmark/trees/"
    
    normrfs = []
    
    for dataset_name in tqdm(dataset_names, desc=dataset_type_label):
        try:
            sequence_path = "%s/%s.fa" % (seq_dir_path, dataset_name)
            tree_path = "%s/%s_tree.nh" % (tree_dir_path, dataset_name)
            
            if not os.path.exists(sequence_path):
                continue
                
            tree_dict = generate_tree(
                seq_file=sequence_path,
                tree_file=tree_path,
                model=model,
                alphabet_tokenizer=None,
                model_name="PHYLA",
                dataset_type="treebase",
                eval_mode=False,
                convert_to_aa=False,
                dictionary_data=None,
                random=42,
                device=device
            )
            
            tree_dicts = {"PHYLA": tree_dict}
            tier1_dict = tier1_metric(tree_dicts)
            normrfs.append(tier1_dict["PHYLA"]["norm_rf"])
            
        except Exception as e:
            print(f"Error on {dataset_name}: {e}")
            continue
    
    return normrfs


def run_treefam_benchmark(model, device):
    treefam_pickle = Path("phyla/eval/treefam.pickle")
    with open(treefam_pickle, 'rb') as f:
        dictionary_data = pickle.load(f)
    
    file_names = list(dictionary_data.keys())[:500]
    normrfs = []
    
    for dataset_name in tqdm(file_names, desc="TreeFam AA"):
        if dataset_name == "TF352211":
            continue
        try:
            tree_dict = generate_tree(
                seq_file=None,
                tree_file=None,
                model=model,
                alphabet_tokenizer=None,
                model_name="PHYLA",
                dataset_type="treefam",
                eval_mode=False,
                convert_to_aa=False,
                dictionary_data=dictionary_data[dataset_name],
                random=42,
                device=device
            )
            
            tree_dicts = {"PHYLA": tree_dict}
            tier1_dict = tier1_metric(tree_dicts)
            normrfs.append(tier1_dict["PHYLA"]["norm_rf"])
            
        except Exception as e:
            print(f"Error on {dataset_name}: {e}")
            continue
    
    return normrfs, len(file_names)


def main():
    cfg_path = "/home/shrey/work/Phyla/configs/sample_eval_config.yaml"
    device = "cuda:0"

    orig = sys.argv
    sys.argv = ["x", cfg_path]
    config = load_config(Config)
    sys.argv = orig

    # Random baseline config
    config.model.vocab_size = 29
    config.model.num_blocks = 10
    config.model.bidirectional = True
    config.model.model_name = "Phyla-beta"

    print("Creating random (untrained) model...")
    model = create_random_model(config, device)
    print(f"Model on device: {next(model.parameters()).device}")

    results = {}

    # TreeBase AA (protein)
    print("\n" + "="*60)
    print("BENCHMARK 1: TreeBase AA (protein)")
    print("="*60)
    
    protein_path = Path("phyla/eval/eval_preds/treebase/treebase_protein_datasets.txt")
    protein_datasets = [l.strip() for l in protein_path.read_text().splitlines() if l.strip()]
    print(f"TreeBase protein datasets: {len(protein_datasets)}")
    
    normrfs_aa = run_treebase_benchmark(model, protein_datasets, "TreeBase AA", device)
    avg_aa = sum(normrfs_aa)/len(normrfs_aa) if normrfs_aa else float('nan')
    results['treebase_aa'] = avg_aa
    print(f"TreeBase AA: {avg_aa:.4f} (success: {len(normrfs_aa)}/{len(protein_datasets)})")

    # TreeFam AA (protein)
    print("\n" + "="*60)
    print("BENCHMARK 2: TreeFam AA (protein)")
    print("="*60)
    
    normrfs_tf, total_tf = run_treefam_benchmark(model, device)
    avg_tf = sum(normrfs_tf)/len(normrfs_tf) if normrfs_tf else float('nan')
    results['treefam_aa'] = avg_tf
    print(f"TreeFam AA: {avg_tf:.4f} (success: {len(normrfs_tf)}/{total_tf})")

    # TreeBase NT (nucleotide)
    print("\n" + "="*60)
    print("BENCHMARK 3: TreeBase NT (nucleotide)")
    print("="*60)
    
    nuc_path = Path("phyla/eval/eval_preds/treebase/treebase_nucleotide_datasets.txt")
    nucleotide_datasets = [l.strip() for l in nuc_path.read_text().splitlines() if l.strip()]
    print(f"TreeBase nucleotide datasets: {len(nucleotide_datasets)}")
    
    normrfs_nt = run_treebase_benchmark(model, nucleotide_datasets, "TreeBase NT", device)
    avg_nt = sum(normrfs_nt)/len(normrfs_nt) if normrfs_nt else float('nan')
    results['treebase_nt'] = avg_nt
    print(f"TreeBase NT: {avg_nt:.4f} (success: {len(normrfs_nt)}/{len(nucleotide_datasets)})")

    # Final Summary
    print("\n" + "="*60)
    print("RANDOM BASELINE BENCHMARK RESULTS")
    print("="*60)
    print(f"TreeBase AA: {results['treebase_aa']:.4f}")
    print(f"TreeFam AA:  {results['treefam_aa']:.4f}")
    print(f"TreeBase NT: {results['treebase_nt']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
