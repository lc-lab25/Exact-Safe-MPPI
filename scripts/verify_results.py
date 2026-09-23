"""Compare fresh results against results/expected/expected_results.json.

Prints a PASS / FAIL / SKIP table. Never writes to the expected file. Exits non-zero if
anything compared actually FAILs; entries with no fresh result are SKIP, not failure, so
the quick subset can be verified without pretending it covers the full grid.
"""
import argparse, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXPECTED = ROOT / "results" / "expected" / "expected_results.json"


def load(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


def cmp_num(got, spec):
    val = spec["value"]
    if isinstance(val, bool):
        return bool(got) == val, f"{got}"
    tol = spec.get("tol", spec.get("tol_abs", 0.0))
    if spec.get("cmp") == "<=":
        return got <= val + tol, f"{got:.3e} <= {val:.3e}"
    if spec.get("cmp") == ">=0":
        return got >= 0.0, f"{got:.4g} >= 0"
    return abs(got - val) <= tol, f"{got:.6g} vs {val:.6g} (tol {tol:g})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--quick", action="store_true",
                    help="label run-count mismatches as SKIP, since quick uses a reduced grid")
    a = ap.parse_args()
    R = pathlib.Path(a.results)
    exp = load(EXPECTED)
    rows, n_fail = [], 0

    # Checks the quick grid cannot test. Quick uses one width per geometry, three seeds and a
    # shortened horizon, so any run COUNT and any outcome RATE is a property of the reduced
    # grid rather than a disagreement with the paper. These are skipped only under --quick;
    # at full scale they are compared and can fail.
    QUICK_COUNT = ("n_states", "n_softmin_runs", "n_exact_runs", "n_below_threshold",
                   "runs_per_controller")
    QUICK_RATE = ("deadlocked", "moving", "threaded", "runs", "smallest_successful_kappa")

    def add(name, spec, got):
        nonlocal n_fail
        if got is None:
            rows.append(("SKIP", name, "no fresh result")); return
        if a.quick and isinstance(got, float) and got != got:            # NaN
            rows.append(("SKIP", name, "nan  [not produced by the reduced grid]")); return
        ok, detail = cmp_num(got, spec)
        if not ok and a.quick:
            leaf = name.rsplit(".", 1)[-1]
            if leaf in QUICK_COUNT:
                rows.append(("SKIP", name, detail + "  [reduced grid]")); return
            if name.startswith("table1.") and leaf in QUICK_RATE:
                rows.append(("SKIP", name, detail + "  [short horizon]")); return
        rows.append(("PASS" if ok else "FAIL", name, detail))
        n_fail += (not ok)

    cf = load(R / "closed_form_checks" / "closed_form_checks.json")
    for k, spec in exp["closed_form_checks"].items():
        if k == "sup_hrho_rho2_positive":
            add("closed_form.sup_hrho_rho2_positive", spec,
                (cf["sup_hrho_rho2"] > 0) if cf else None)
        else:
            add(f"closed_form.{k}", spec, cf.get(k) if cf else None)

    lv = load(R / "lcp_vs_scalar" / "lcp_vs_scalar.json")
    for k, spec in exp["lcp_vs_scalar"].items():
        add(f"lcp_vs_scalar.{k}", spec, lv.get(k) if lv else None)

    gt = load(R / "gap_threshold" / "summary.json")
    for k, spec in exp["gap_threshold"].items():
        if k == "table1":
            continue
        add(f"gap_threshold.{k}", spec, gt.get(k) if gt else None)
    if gt:
        band_key = {"<ln2": "lt_ln2", "[ln2,1)": "ln2_to_1", ">=1": "ge_1"}
        got = {band_key[b["band"]]: b for b in gt.get("table1", [])}
        for bk, spec in exp["gap_threshold"]["table1"].items():
            g = got.get(bk)
            for field in ("runs", "threaded", "deadlocked", "moving"):
                if field not in spec:
                    continue
                add(f"table1.{bk}.{field}",
                    {"value": spec[field], "tol": spec.get("tol", 0)},
                    g[field] if g else None)

    nc = load(R / "nine_constraint" / "summary.json")
    for k, spec in exp["nine_constraint"]["violations"].items():
        add(f"nine_constraint.violations.{k}", spec,
            nc[k]["violations"] if nc and k in nc else None)

    w = max(len(r[1]) for r in rows) + 2
    print(f"{'status':7s} {'check':{w}s} detail")
    print("-" * (10 + w + 40))
    for st, name, detail in rows:
        print(f"{st:7s} {name:{w}s} {detail}")
    npass = sum(r[0] == "PASS" for r in rows)
    nskip = sum(r[0] == "SKIP" for r in rows)
    print("-" * (10 + w + 40))
    print(f"{npass} passed, {n_fail} failed, {nskip} skipped")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
