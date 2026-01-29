"""
shared utilities for phyla explainability experiments
"""
import sys
import os

def setup_phyla_env():
    """configure paths and apply bimamba residual fix"""
    sys.path.insert(0, '/home/shrey/work/Phyla')
    sys.path.insert(0, '/home/shrey/work')
    os.chdir('/home/shrey/work/Phyla/phyla')
    
    from phyla.model.model import BiMambaWrapper
    
    def forward_with_residual(self, hidden_states, inference_params=None, cpu=None):
        out = self.mamba_fwd(hidden_states, inference_params=inference_params)
        if self.bidirectional:
            out_rev = self.mamba_rev(
                hidden_states.flip(dims=(1,)), 
                inference_params=inference_params
            ).flip(dims=(1,))
            if self.bidirectional_strategy == "add":
                out = out + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
        return hidden_states + out
    
    BiMambaWrapper.forward = forward_with_residual


def load_phyla_model(device='cuda'):
    """load phyla-beta model with correct state dict handling"""
    import torch
    from phyla.model.model import Phyla, Config
    
    config = Config()
    model = Phyla(config, device=device, name='phyla-beta')
    
    # Load checkpoint manually with correct key stripping
    checkpoint_path = 'weights/11564369'
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)['state_dict']
    # Strip 'model.' prefix (not 'model_name.')
    new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def load_fasta(path):
    """parse fasta file, return list of names and sequences"""
    names, seqs = [], []
    name, seq = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    names.append(name)
                    seqs.append("".join(seq))
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
    if name:
        names.append(name)
        seqs.append("".join(seq))
    return names, seqs


def clean_sequences(seqs):
    """keep only letters and gaps"""
    return ["".join(c for c in s if c.isalpha() or c == "-") for s in seqs]


def seq_identity(s1, s2):
    """compute pairwise sequence identity ignoring gaps"""
    matches = sum(1 for a, b in zip(s1, s2) if a == b and a != '-')
    total = sum(1 for a, b in zip(s1, s2) if a != '-' or b != '-')
    return matches / total if total > 0 else 0


CONSERVATIVE_SUBS = [
    ('V','I'), ('V','L'), ('I','L'), ('I','M'), ('D','E'),
    ('K','R'), ('S','T'), ('F','Y'), ('F','W')
]

RADICAL_SUBS = [
    ('V','D'), ('V','K'), ('V','E'), ('I','R'), ('L','D'),
    ('L','K'), ('K','D'), ('K','E'), ('R','D'), ('G','W'), ('A','W')
]

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

DATA_DIR = "/home/shrey/work/Cleaned_OpenProtein_Set/Cleaned_Open_Protein_Set"
