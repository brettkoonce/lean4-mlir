import pickle, re, sys, os
from collections import defaultdict, Counter
from pathlib import Path
G = pickle.load(open(Path(os.environ.get("CENSUS_DIR", "/tmp/audit_census")) / "graph.pkl", "rb"))
consts, env_users, tok_users, pins = G["consts"], G["env_users"], G["tok_users"], G["pins"]
pinset = {p for p, _ in pins}
def users(n):
    return set(env_users.get(n, ())) | set(tok_users.get(n, ()))
audit_only = [p for p, _ in pins if not users(p)]
if __name__ == "__main__":
    print("pins", len(pinset), "audit-only", len(audit_only))
    c = Counter(consts[p]["file"] for p in audit_only)
    for f, k in c.most_common(): print(f"{k:4d} {f}")
    (Path(os.environ.get("CENSUS_DIR", "/tmp/audit_census")) / "audit_only.txt").write_text(
        "\n".join(sorted(audit_only)) + "\n")
