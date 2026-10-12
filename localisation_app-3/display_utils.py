# -*- coding: utf-8 -*-
"""
display_utils.py
=================
Petites fonctions d'affichage console (sans dependance externe) pour
produire des rapports lisibles et detailles a chaque niveau du pipeline.
"""

WIDTH = 78


def hr(char="-"):
    print(char * WIDTH)


def title(text, char="="):
    hr(char)
    pad = (WIDTH - len(text) - 2) // 2
    print(" " * max(pad, 0) + text)
    hr(char)


def section(text):
    print()
    print(f"── {text} " + "─" * max(0, WIDTH - len(text) - 4))


def kv(key, value, unit="", indent=2):
    key_str = f"{' ' * indent}{key}"
    print(f"{key_str:<40}: {value}{(' ' + unit) if unit else ''}")


def bullet(text, indent=2):
    print(f"{' ' * indent}- {text}")


def table(headers, rows, col_widths=None):
    """Affiche un tableau simple aligne."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            w = len(str(h))
            for r in rows:
                w = max(w, len(str(r[i])))
            col_widths.append(w + 2)

    def fmt_row(vals):
        return "".join(str(v).ljust(w) for v, w in zip(vals, col_widths))

    print(fmt_row(headers))
    print("-" * sum(col_widths))
    for r in rows:
        print(fmt_row(r))


def warn(text):
    print(f"[ATTENTION] {text}")


def ok(text):
    print(f"[OK] {text}")


def fail(text):
    print(f"[ECHEC] {text}")


def step(n, text):
    print(f"\n[{n}] {text}")
