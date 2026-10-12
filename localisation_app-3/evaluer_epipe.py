#!/usr/bin/env python3
"""Refait resume_global / resume_par_qualite_apr avec e_pipe = |estimation PnP - centre camera COLMAP|
(repere de la carte, sans GPS). Succes = image localisee ET e_pipe < seuil (taux sur toutes les images).

Usage : python evaluer_epipe.py MODEL_DIR reference_gps_colmap.txt details_par_image.csv [sortie_dir]
Sorties : resume_global_epipe.csv, resume_par_qualite_epipe.csv, details_epipe.csv + tableaux Markdown a l'ecran.

La ligne 'PG-PnP + repli (SIMULE)' est un scenario what-if (repli sur le PnP classique quand le filtrage echoue) :
ce n'est PAS une mesure de l'application tant que le repli n'est pas implemente et re-mesure.
"""
import sys, os, math
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_pnp_colmap import read_gps
from decomposer_erreur import centers, fit_sim2d

RT = 6371000.0
CL, RF, PG = "PnP classique", "RANSAC + filtrage geo", "PG-PnP complet"
SIM = "PG-PnP + repli PnP classique (SIMULE)"

def short(v):
    return CL if v.startswith("PnP cl") else RF if v.startswith("RANSAC") else PG

def n_ransac(e, p=0.99, nmin=4):
    e = min(max(e, 0.0), 0.99); w = (1 - e) ** nmin
    return 1 if w >= 1 else max(1, math.ceil(math.log(1 - p) / math.log(1 - w)))

def stats(g):
    n = len(g); ok = g[g.success]; e = ok.e_pipe
    s = lambda t: 100 * (e < t).sum() / n
    eo = np.median(ok.outlier_rate_e) if len(ok) else float("nan")
    return dict(n_images=n, n_localisees=len(ok), n_echecs=n - len(ok), taux_echec_pct=round(100 * (n - len(ok)) / n, 1),
                taux_succes_2m_pct=round(s(2), 1), taux_succes_0p5m_pct=round(s(0.5), 1), taux_succes_0p25m_pct=round(s(0.25), 1),
                succes_2m_sur_localisees_pct=round(100 * (e < 2).mean(), 1) if len(ok) else float("nan"),
                e_pipe_mediane_m=round(e.median(), 3), e_pipe_p75_m=round(e.quantile(.75), 3), e_pipe_p90_m=round(e.quantile(.9), 3),
                temps_median_ms=round(ok.elapsed_ms.median(), 1), taux_outliers_e=round(eo, 3), n_ransac=n_ransac(eo))

def table(d, by=None):
    rows = []
    groups = [(None, d)] if by is None else list(d.groupby(by, sort=True))
    for key, sub in groups:
        base = None
        for v in (CL, RF, PG, SIM):
            g = sub[sub.v == v]
            if g.empty: continue
            r = stats(g)
            if v == CL: base = r["n_ransac"]; r["reduction"] = "--"
            else: r["reduction"] = f"x{base / r['n_ransac']:.1f}" if base else "--"
            r = {**({by: key} if by else {}), "variante": v, **r}; rows.append(r)
    return pd.DataFrame(rows)

def main(model, gps, csv, out="."):
    C3 = centers(model); ref = read_gps(gps)
    names = [n for n in C3 if n in ref]
    ll = np.array([ref[n] for n in names]); lat0, lon0 = ll[:, 0].mean(), ll[:, 1].mean()
    enu = lambda lat, lon: np.c_[(np.asarray(lon) - lon0) * np.pi/180 * RT * np.cos(np.radians(lat0)),
                                 (np.asarray(lat) - lat0) * np.pi/180 * RT]
    C_all = np.array([C3[n] for n in names]); mu = C_all.mean(0)
    _, _, Vt = np.linalg.svd(C_all - mu, full_matrices=False); H = lambda C: (C - mu) @ Vt[:2].T
    f, _ = fit_sim2d(H(C_all), enu(ll[:, 0], ll[:, 1]))

    d = pd.read_csv(csv); d["name"] = d.image.map(os.path.basename); d["v"] = d.variante.map(short)
    d = d[d.name.isin(C3)].copy()
    Cm = {n: f(H(C3[n][None]))[0] for n in d.name.unique()}
    d["e_pipe"] = np.nan
    ok = d.success
    E = enu(d.loc[ok, "lat"], d.loc[ok, "lon"]); Cc = np.array([Cm[n] for n in d.loc[ok, "name"]])
    d.loc[ok, "e_pipe"] = np.linalg.norm(E - Cc, axis=1)

    # scenario what-if : repli sur le PnP classique quand PG-PnP echoue
    pg = d[d.v == PG].set_index("name"); cl = d[d.v == CL].set_index("name"); sim = pg.copy()
    for n in sim.index[~sim.success]:
        if n in cl.index and cl.loc[n, "success"]:
            for c in ("success", "lat", "lon", "e_pipe", "outlier_rate_e"): sim.loc[n, c] = cl.loc[n, c]
            sim.loc[n, "elapsed_ms"] = pg.loc[n, "elapsed_ms"] + cl.loc[n, "elapsed_ms"]
    sim["v"] = SIM; d = pd.concat([d, sim.reset_index()], ignore_index=True)
    d["success"] = d.success.astype(bool)

    os.makedirs(out, exist_ok=True)
    g = table(d); q = table(d, "apr_quality")
    g.to_csv(os.path.join(out, "resume_global_epipe.csv"), index=False)
    q.to_csv(os.path.join(out, "resume_par_qualite_epipe.csv"), index=False)
    d.drop(columns=["name"]).to_csv(os.path.join(out, "details_epipe.csv"), index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    print("=== GLOBAL (e_pipe) ===\n", g.to_string(index=False))
    print("\n=== PAR QUALITE DU PRIOR APR (e_pipe) ===\n", q.to_string(index=False))

    a = d[d.v == CL].set_index("name"); b = d[d.v == PG].set_index("name")
    com = a.index[a.success & b.reindex(a.index).success.fillna(False).astype(bool)]
    diff = (b.loc[com, "e_pipe"] - a.loc[com, "e_pipe"]).values
    try: p = wilcoxon(diff).pvalue
    except ValueError: p = float("nan")
    print(f"\nPaire PG-PnP - PnP classique sur {len(com)} images communes : delta e_pipe moyen {diff.mean():+.3f} m, "
          f"mediane {np.median(diff):+.3f} m, Wilcoxon p={p:.3g}")

if __name__ == "__main__":
    main(*sys.argv[1:4], *(sys.argv[4:5] or ["."]))
