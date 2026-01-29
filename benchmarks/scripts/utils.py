import os
import sys
import torch
import pickle
from pathlib import Path
from tqdm import tqdm

PHYLA_ROOT = os.environ.get("PHYLA_ROOT", "/home/shrey/work/Phyla")
sys.path.insert(0, PHYLA_ROOT)
sys.path.insert(0, os.path.join(PHYLA_ROOT, "phyla"))


def load_config(config_path: str):
    from phyla.utils.utils import load_config as _load_config
    from phyla.eval.evo_reasoning_eval import Config
    
    orig_argv = sys.argv
    sys.argv = ["x", config_path]
    config = _load_config(Config)
    sys.argv = orig_argv
    return config


def load_phyla_model(config, checkpoint_path: str, device: str = "cuda:0"):
    from phyla import phyla
    
    model = phyla(config=config, name=config.model.model_name, custom_arch=True, device=device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.modul.'):
            new_state_dict[key.replace('model.modul.', 'model.')] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def run_treebase_benchmark(model, dataset_file: str, device: str = "cuda:0", description: str = "TreeBase"):
    from phyla.eval.evo_reasoning_eval import generate_tree, tier1_metric
    
    seq_dir = "./treebase_benchmark/sequences/"
    tree_dir = "./treebase_benchmark/trees/"
    
    dataset_names = [l.strip() for l in Path(dataset_file).read_text().splitlines() if l.strip()]
    normrfs = []
    
    for dataset_name in tqdm(dataset_names, desc=description):
        try:
            sequence_path = f"{seq_dir}/{dataset_name}.fa"
            tree_path = f"{tree_dir}/{dataset_name}_tree.nh"
            
            if not os.path.exists(sequence_path):
                continue
            
            tree_dict = generate_tree(
                seq_file=sequence_path, tree_file=tree_path, model=model,
                alphabet_tokenizer=None, model_name="PHYLA", dataset_type="treebase",
                eval_mode=False, convert_to_aa=False, dictionary_data=None,
                random=42, device=device
            )
            
            tier1_dict = tier1_metric({"PHYLA": tree_dict})
            normrfs.append(tier1_dict["PHYLA"]["norm_rf"])
        except Exception as e:
            print(f"Error on {dataset_name}: {e}")
    
    return normrfs, len(normrfs), len(dataset_names)


def run_treefam_benchmark(model, device: str = "cuda:0", num_samples: int = 500):
    from phyla.eval.evo_reasoning_eval import generate_tree, tier1_metric
    
    with open("phyla/eval/treefam.pickle", 'rb') as f:
        dictionary_data = pickle.load(f)
    
    file_names = list(dictionary_data.keys())[:num_samples]
    normrfs = []
    
    for dataset_name in tqdm(file_names, desc="TreeFam AA"):
        if dataset_name == "TF352211":
            continue
        try:
            tree_dict = generate_tree(
                seq_file=None, tree_file=None, model=model,
                alphabet_tokenizer=None, model_name="PHYLA", dataset_type="treefam",
                eval_mode=False, convert_to_aa=False,
                dictionary_data=dictionary_data[dataset_name],
                random=42, device=device
            )
            
            tier1_dict = tier1_metric({"PHYLA": tree_dict})
            normrfs.append(tier1_dict["PHYLA"]["norm_rf"])
        except Exception as e:
            print(f"Error on {dataset_name}: {e}")
    
    return normrfs, len(normrfs), len(file_names)
