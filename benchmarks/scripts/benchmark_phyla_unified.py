import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, load_phyla_model, run_treebase_benchmark, run_treefam_benchmark, PHYLA_ROOT

CHECKPOINT = os.path.join(PHYLA_ROOT, "training_checkpoints/unified/epoch=00-step=001599-treefam_avg_normrf=0.5670.ckpt")
TREEBASE_AA_FILE = os.path.join(PHYLA_ROOT, "phyla/eval/eval_preds/treebase/treebase_protein_datasets.txt")
TREEBASE_NT_FILE = os.path.join(PHYLA_ROOT, "phyla/eval/eval_preds/treebase/treebase_nucleotide_datasets.txt")
DEVICE = "cuda:0"


def main():
    config = load_config(os.path.join(PHYLA_ROOT, "configs/sample_eval_config.yaml"))
    config.model.vocab_size = 56
    config.model.num_blocks = 10
    config.model.bidirectional = True
    config.model.model_name = "Phyla-unified"

    print("Loading Phyla Unified model...")
    model = load_phyla_model(config, CHECKPOINT, DEVICE)

    results = {}

    # TreeBase AA
    print("\n" + "="*50 + "\nTreeBase AA (Protein)\n" + "="*50)
    normrfs, success, total = run_treebase_benchmark(model, TREEBASE_AA_FILE, DEVICE, "TreeBase AA")
    results['TreeBase AA'] = sum(normrfs)/len(normrfs) if normrfs else None
    print(f"TreeBase AA: {results['TreeBase AA']:.4f} ({success}/{total})")

    # TreeBase NT
    print("\n" + "="*50 + "\nTreeBase NT (Nucleotide)\n" + "="*50)
    normrfs, success, total = run_treebase_benchmark(model, TREEBASE_NT_FILE, DEVICE, "TreeBase NT")
    results['TreeBase NT'] = sum(normrfs)/len(normrfs) if normrfs else None
    print(f"TreeBase NT: {results['TreeBase NT']:.4f} ({success}/{total})")

    # TreeFam AA
    print("\n" + "="*50 + "\nTreeFam AA (Protein)\n" + "="*50)
    normrfs, success, total = run_treefam_benchmark(model, DEVICE)
    results['TreeFam AA'] = sum(normrfs)/len(normrfs) if normrfs else None
    print(f"TreeFam AA: {results['TreeFam AA']:.4f} ({success}/{total})")

    print("\n" + "="*50 + "\nPHYLA UNIFIED RESULTS\n" + "="*50)
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if v else f"{k}: NA")


if __name__ == "__main__":
    main()
