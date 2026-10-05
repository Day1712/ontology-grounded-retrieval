"""Macro concordance C_q between a run and the judged scores: the share of
gold-ordered document pairs the run orders the same way (ties count half),
with decisiveness and accuracy when deciding."""
import numpy as np

def concordance(run, judged):
    """Macro C_q over queries, plus decisiveness and accuracy when deciding."""
    cs = []
    conc_total = tied_total = ordered_total = 0
    for qid, row in judged.items():
        docs = list(row)
        g = np.array([row[d] for d in docs], dtype=np.float64)
        s = np.array([run[qid][d] for d in docs], dtype=np.float64)
        gs = np.sign(g[:, None] - g[None, :])
        ss = np.sign(s[:, None] - s[None, :])
        iu = np.triu_indices(len(docs), k=1)
        gs, ss = gs[iu], ss[iu]
        ordered = gs != 0
        n_ord = int(ordered.sum())
        if not n_ord:
            continue
        conc = int(((ss == gs) & ordered).sum())
        tied = int(((ss == 0) & ordered).sum())
        cs.append((conc + 0.5 * tied) / n_ord)
        conc_total += conc
        tied_total += tied
        ordered_total += n_ord
    disc_total = ordered_total - conc_total - tied_total
    return {
        "cq": float(np.mean(cs)),
        "decisiveness": 1 - tied_total / ordered_total,
        "accuracy_when_deciding": conc_total / (conc_total + disc_total),
    }
