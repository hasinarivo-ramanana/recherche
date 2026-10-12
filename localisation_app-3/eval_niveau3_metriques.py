#!/usr/bin/env python3
"""Evaluation du Niveau 3 (PnP classique / RANSAC+filtrage / PG-PnP) avec des metriques defendables.

Usage : python eval_niveau3_metriques.py details_par_image.csv [sortie_dir]

Principe : la reference GPS smartphone n'a pas une precision absolue < 2 m. On rapporte donc
  (A) l'erreur absolue vs GPS            -> plafond de reference, a documenter honnetement
  (B) l'erreur relative (RPE) a L metres -> coherence locale, independante de la derive basse frequence
  (C) l'ATE apres retrait d'offset local -> idem, sur fenetre glissante
  (D) tests apparies (Wilcoxon) + IC bootstrap entre variantes, sur les images communes
Aucune de (B)(C) n'est une precision absolue : a presenter comme telle dans la these.
"""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

R_T = 6371000.0
VARS = {"PnP classique": "PnP cl", "RANSAC+filtrage": "RANSAC +", "PG-PnP": "PG-PnP c"}


def enu(lat, lon, lat0, lon0):
    return np.c_[(lon - lon0) * np.pi / 180 * R_T * np.cos(np.radians(lat0)),
                 (lat - lat0) * np.pi / 180 * R_T]


def boot_ci(x, f=np.median, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = [f(rng.choice(x, len(x))) for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def per_variant(d, key, lat0, lon0):
    g = d[d.variante.str.startswith(key)].copy()
    g["idx"] = g.image.str.extract(r"(\d+)")[0].astype(int)
    g = g.sort_values("idx").reset_index(drop=True)
    g["fail"] = ~g.success
    ok = g[g.success].reset_index(drop=True)
    G = enu(ok.gt_lat.values, ok.gt_lon.values, lat0, lon0)
    E = enu(ok.lat.values, ok.lon.values, lat0, lon0)
    return g, ok, G, E


def rpe(G, E, idx, L):
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(G, axis=0).T))]
    out = []
    for i in range(len(G)):
        j = np.searchsorted(s, s[i] + L)
        if j < len(G) and abs(idx[j] - idx[i]) <= 400:
            out.append(np.hypot(*((E[j] - E[i]) - (G[j] - G[i]))))
    return np.array(out)


def local_ate(err, k):
    n = len(err); out = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - k // 2), min(n, i + k // 2 + 1)
        out[i] = np.hypot(*(err[i] - err[lo:hi].mean(0)))
    return out


def row(name, x, extra=None):
    ci = boot_ci(x)
    r = {"metrique": name, "n": len(x), "mediane_m": np.median(x),
         "IC95_bas": ci[0], "IC95_haut": ci[1], "p90_m": np.percentile(x, 90),
         "pct_<2m": 100 * (x < 2).mean()}
    if extra: r.update(extra)
    return r


def main(path, outdir="."):
    d = pd.read_csv(path)
    lat0, lon0 = d.gt_lat.mean(), d.gt_lon.mean()
    rows = []
    cache = {}
    for name, key in VARS.items():
        g, ok, G, E = per_variant(d, key, lat0, lon0)
        cache[name] = (g, ok, G, E)
        err = E - G
        base = {"variante": name, "taux_echec_pct": 100 * g.fail.mean()}
        rows.append(row("A. absolue vs GPS", np.hypot(*err.T), base))
        for L in (5, 10, 20):
            rows.append(row(f"B. RPE L={L} m", rpe(G, E, ok.idx.values, L), base))
        for k in (10, 20):
            rows.append(row(f"C. ATE local (fenetre {k} img)", local_ate(err, k), base))
    res = pd.DataFrame(rows)
    os.makedirs(outdir, exist_ok=True)
    res.to_csv(os.path.join(outdir, "resultats_metriques.csv"), index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print(res.round(2).to_string(index=False))

    # D. tests apparies sur images communes aux 3 variantes
    common = set.intersection(*[set(cache[n][1].image) for n in VARS])
    print(f"\nD. Tests apparies sur {len(common)} images reussies par les 3 variantes")
    def errs(n):
        ok, G, E = cache[n][1], cache[n][2], cache[n][3]
        m = ok.image.isin(common).values
        o = ok[m].reset_index(drop=True); e = np.hypot(*(E[m] - G[m]).T)
        return pd.Series(e, index=o.image).sort_index()
    ref = errs("PnP classique")
    for n in ("RANSAC+filtrage", "PG-PnP"):
        e = errs(n); diff = (e - ref).values
        try: p = wilcoxon(diff).pvalue
        except ValueError: p = float("nan")
        ci = boot_ci(diff, np.mean)
        print(f"  {n} - PnP classique : delta moyen {diff.mean():+.3f} m  IC95 [{ci[0]:+.3f}, {ci[1]:+.3f}]  Wilcoxon p={p:.3g}")
    print("\nTaux d'echec (a rapporter separement, jamais melange aux erreurs) :")
    for n in VARS:
        g = cache[n][0]; print(f"  {n}: {g.fail.sum()}/{len(g)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ".")
