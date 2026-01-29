#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from phyla.utils.utils import load_config
from phyla.eval.evo_reasoning_eval import Config, generate_tree, tier1_metric
from phyla import phyla
import torch
from tqdm import tqdm


def load_model(config, checkpoint_path, device):
    model = phyla(config=config, name=config.model.model_name, custom_arch=True, device=device)
    
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.modul.'):
            new_key = k.replace('model.modul.', 'model.')
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
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


def main():
    cfg_path = "/home/shrey/work/Phyla/configs/sample_eval_config.yaml"
    checkpoint_path = "<PATH_TO_PHYLA_NT_CHECKPOINT>"
    device = "cuda:0"

    orig = sys.argv
    sys.argv = ["x", cfg_path]
    config = load_config(Config)
    sys.argv = orig

    # Phyla NT config: nucleotide only
    config.model.vocab_size = 9
    config.model.num_blocks = 10
    config.model.bidirectional = True
    config.model.model_name = "Phyla-beta"

    print("Loading Phyla NT model...")
    model = load_model(config, checkpoint_path, device)
    print(f"Model loaded on device: {next(model.parameters()).device}")

    results = {}

    # TreeBase NT (nucleotide)
    print("\n" + "="*60)
    print("BENCHMARK: TreeBase NT (nucleotide)")
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
    print("PHYLA NT BENCHMARK RESULTS")
    print("="*60)
    print("TreeBase AA: NA (nucleotide-only model)")
    print("TreeFam AA:  NA (nucleotide-only model)")
    print(f"TreeBase NT: {results['treebase_nt']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
