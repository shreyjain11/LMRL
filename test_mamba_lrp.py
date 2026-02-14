#!/usr/bin/env python3
import os, sys, torch, numpy as np, traceback
sys.path.insert(0, os.path.expanduser("~/phyla_final_workspace/Phyla"))
sys.path.insert(0, os.path.expanduser("~/work/LMRL"))

PASS_CT = 0
FAIL_CT = 0

def pr(msg=""):
    print(msg, flush=True)

def check(name, cond, detail=""):
    global PASS_CT, FAIL_CT
    if cond:
        PASS_CT += 1
        pr("  ok " + name)
    else:
        FAIL_CT += 1
        pr("  FAIL: " + name)
        if detail:
            pr("    -> " + detail)

WT = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
MUT1 = list(WT)
MUT1[38] = "W"
MUT1 = "".join(MUT1)

pr("=" * 60)
pr("1. IMPORTS")
pr("=" * 60)
from phyla.model.model import Phyla
check("Phyla import", True)
from explainability.integrations.mamba_lrp import PhylaAttributor
check("PhylaAttributor import", True)

pr()
pr("=" * 60)
pr("2. MODEL + ATTRIBUTOR")
pr("=" * 60)
model = Phyla(name="phyla-beta", device="cuda")
model.load()
model.eval()
check("Model loaded", True)
attributor = PhylaAttributor(model)
check("Attributor created", True)
eml = model.modul[0].backbone.embedding
check("Embedding dim=256: " + str(eml.weight.shape), eml.weight.shape[1] == 256)

pr()
pr("=" * 60)
pr("3. SINGLE SEQUENCE ATTRIBUTION")
pr("=" * 60)
rel, info = attributor.attribute_sequence([WT], ["S0"], target_seq_idx=0)
check("Returns array (not None)", rel is not None)
if rel is not None:
    check("Length match: " + str(len(rel)) + " == " + str(len(WT)), len(rel) == len(WT))
    check("Non-zero signal std=" + str(round(np.std(rel), 6)), np.std(rel) > 1e-6)
    check("No NaN", not np.any(np.isnan(rel)))
    check("No Inf", not np.any(np.isinf(rel)))
    check("All non-negative min=" + str(round(float(np.min(rel)), 6)), np.min(rel) >= 0)

pr()
pr("=" * 60)
pr("4. DETERMINISM")
pr("=" * 60)
rel2, _ = attributor.attribute_sequence([WT], ["S0"], target_seq_idx=0)
diff = float(np.max(np.abs(rel - rel2)))
check("Same output twice, max diff=" + str(round(diff, 10)), diff < 1e-5)

pr()
pr("=" * 60)
pr("5. MUTATION LOCALIZATION")
pr("=" * 60)
seqs_ctx = [WT, MUT1]
names_ctx = ["WT", "MUT"]
rel_wt, _ = attributor.attribute_sequence(seqs_ctx, names_ctx, target_seq_idx=0)
wt_rank = int(sum(1 for r in rel_wt if r > rel_wt[38])) + 1
pr("  WT:  pos38 score=" + str(round(float(rel_wt[38]), 4)) + " rank=" + str(wt_rank) + "/" + str(len(rel_wt)))

rel_mut, _ = attributor.attribute_sequence(seqs_ctx, names_ctx, target_seq_idx=1)
mut_rank = int(sum(1 for r in rel_mut if r > rel_mut[38])) + 1
pr("  MUT: pos38 score=" + str(round(float(rel_mut[38]), 4)) + " rank=" + str(mut_rank) + "/" + str(len(rel_mut)))

check("Mut pos ranked better in mutant " + str(mut_rank) + " vs WT " + str(wt_rank), mut_rank < wt_rank)

top5 = sorted(range(len(rel_mut)), key=lambda i: rel_mut[i], reverse=True)[:5]
pr("  MUT top-5: " + str([(p, round(float(rel_mut[p]), 4)) for p in top5]))
check("Pos 38 in top 5", 38 in top5)

if 38 not in top5:
    top10 = sorted(range(len(rel_mut)), key=lambda i: rel_mut[i], reverse=True)[:10]
    pr("  MUT top-10: " + str([(p, round(float(rel_mut[p]), 4)) for p in top10]))
    check("Pos 38 in top 10", 38 in top10)

pr()
pr("=" * 60)
pr("6. PAIRWISE DISTANCE ATTRIBUTION")
pr("=" * 60)
try:
    ri, rj, pinfo = attributor.attribute_pairwise_distance(seqs_ctx, names_ctx, seq_i=0, seq_j=1)
    check("Returns two arrays len=" + str(len(ri)) + "," + str(len(rj)), True)
    dist_val = pinfo.get("distance", 0)
    check("Distance > 0: " + str(round(dist_val, 6)), dist_val > 0)
    check("No NaN in rel_i", not np.any(np.isnan(ri)))
    check("No NaN in rel_j", not np.any(np.isnan(rj)))
    pw_rank = int(sum(1 for r in rj if r > rj[38])) + 1
    pr("  Pairwise: pos38 in seq_j rank=" + str(pw_rank) + "/" + str(len(rj)))
except Exception as e:
    check("Pairwise attribution", False, traceback.format_exc())

pr()
pr("=" * 60)
pr("7. SCALING (1,2,3,5 seqs)")
pr("=" * 60)
for n in [1, 2, 3, 5]:
    seqs = [WT] * n
    names = ["S" + str(i) for i in range(n)]
    try:
        r, _ = attributor.attribute_sequence(seqs, names, target_seq_idx=0)
        check("n=" + str(n) + ": len=" + str(len(r)) + " std=" + str(round(float(np.std(r)), 4)), len(r) == len(WT))
    except Exception as e:
        check("n=" + str(n), False, str(e))

pr()
pr("=" * 60)
pr("8. VARIABLE-LENGTH SEQS (boundary bug test)")
pr("=" * 60)
short_seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEA"
long_seq = WT + "EXTRARESIDUES"
seqs_var = [WT, short_seq, long_seq]
names_var = ["WT", "SHORT", "LONG"]
try:
    rv0, info0 = attributor.attribute_sequence(seqs_var, names_var, target_seq_idx=0)
    check("target_seq_idx=0 with variable lengths: len=" + str(len(rv0)), len(rv0) == len(WT))
except Exception as e:
    check("target_seq_idx=0 variable len", False, str(e))

try:
    rv1, info1 = attributor.attribute_sequence(seqs_var, names_var, target_seq_idx=1)
    expected_len = len(short_seq)
    got_len = len(rv1)
    check("target_seq_idx=1 variable lengths: got " + str(got_len) + " expected " + str(expected_len),
          got_len == expected_len)
    if got_len != expected_len:
        pr("    ** KNOWN BUG: boundary parsing assumes equal-length sequences **")
        pr("    ** Does NOT affect fitness results (always target_seq_idx=0) **")
except Exception as e:
    check("target_seq_idx=1 variable len", False, str(e))

pr()
pr("=" * 60)
pr("9. REAL HOMOLOGS")
pr("=" * 60)
from pathlib import Path
OPS = Path("/home/shrey/work/Cleaned_OpenProtein_Set/Cleaned_Open_Protein_Set")
fa = list(OPS.glob("*.fasta"))[0]
seqs_r = []
name_r = None
seq_r = []
with open(fa) as f:
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            if name_r:
                s = "".join(c for c in "".join(seq_r) if c.isalpha() and c != "-")
                if len(s) > 20:
                    seqs_r.append((name_r, s))
            name_r = line[1:].split()[0]
            seq_r = []
        else:
            seq_r.append(line)
    if name_r:
        s = "".join(c for c in "".join(seq_r) if c.isalpha() and c != "-")
        if len(s) > 20:
            seqs_r.append((name_r, s))
pr("  " + fa.name + ": " + str(len(seqs_r)) + " seqs")
use = seqs_r[:5]
try:
    rr, _ = attributor.attribute_sequence(
        [s for _, s in use],
        ["S" + str(i) for i in range(len(use))],
        target_seq_idx=0
    )
    check("Real homologs: len=" + str(len(rr)) + " std=" + str(round(float(np.std(rr)), 4)),
          len(rr) == len(use[0][1]))
except Exception as e:
    check("Real homologs", False, traceback.format_exc())

pr()
pr("=" * 60)
pr("10. HOOKS CLEANUP")
pr("=" * 60)
rb, _ = attributor.attribute_sequence([WT], ["S0"], target_seq_idx=0)
with torch.no_grad():
    ids, cm, sm, _ = model.encode([WT], ["S0"])
    d = next(model.modul[0].parameters()).device
    x = model.modul[0](
        ids.to(d), hidden_states_given=False, logits=False,
        position_ids=None, sequence_mask=sm.to(d), cls_token_mask=cm.to(d)
    )
check("Forward pass works after attribution", x is not None)
ra, _ = attributor.attribute_sequence([WT], ["S0"], target_seq_idx=0)
diff2 = float(np.max(np.abs(rb - ra)))
check("Still deterministic after interleaving diff=" + str(round(diff2, 10)), diff2 < 1e-5)

pr()
pr("=" * 60)
pr("RESULTS: " + str(PASS_CT) + " passed, " + str(FAIL_CT) + " failed")
pr("=" * 60)
if FAIL_CT == 0:
    pr("ALL TESTS PASSED")
else:
    pr(str(FAIL_CT) + " FAILURES -- review above")
