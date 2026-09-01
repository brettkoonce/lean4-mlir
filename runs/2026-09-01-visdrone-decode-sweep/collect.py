import re, sys, pathlib
rows = []
for p in sorted(pathlib.Path(sys.argv[1]).glob("*.log")):
    m = re.match(r"t(\d+)_k(\d+)_f([\d.]+)\.log", p.name)
    if not m: continue
    topk, k, floor = int(m.group(1)), int(m.group(2)), float(m.group(3))
    txt = p.read_text()
    mp = re.search(r"mAP@0\.50 = ([\d.]+)", txt)
    ca = re.search(r"class-agnostic localization AP@0\.50 = ([\d.]+)\s+\(GT boxes=(\d+), dets=(\d+), TP=(\d+), recall=([\d.]+)\)", txt)
    if not (mp and ca): continue
    per = dict(re.findall(r"^\s+([\w-]+): AP@0\.50 = ([\d.]+)", txt, re.M))
    rare = sum(float(per.get(c, 0)) for c in ("bicycle", "awning-tri", "tricycle", "truck")) / 4
    rows.append(dict(topk=topk, k=k, floor=floor, mAP=float(mp.group(1)),
                     caAP=float(ca.group(1)), dets=int(ca.group(3)),
                     recall=float(ca.group(5)), rare4=rare))
rows.sort(key=lambda r: -r["mAP"])
print(f"{'topk':>5} {'ml-k':>4} {'floor':>5} {'mAP':>7} {'ca-AP':>6} {'recall':>6} {'dets':>8} {'rare4':>6}")
for r in rows:
    print(f"{r['topk']:>5} {r['k']:>4} {r['floor']:>5} {r['mAP']:>7.4f} {r['caAP']:>6.4f} "
          f"{r['recall']:>6.4f} {r['dets']:>8} {r['rare4']:>6.4f}")
print(f"\n{len(rows)} points")
