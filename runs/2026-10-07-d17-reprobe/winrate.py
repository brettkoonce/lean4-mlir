import sys,re,statistics as st
# per-100-step window ms/step from wall-stamped trainer lines
rows=[]
for l in open(sys.argv[1]):
    ts,_,msg=l.partition('\t')
    m=re.search(r'step (\d+)/',msg)
    if m: rows.append((float(ts),int(m.group(1))))
w=[(t2-t1)/(s2-s1)*1000 for (t1,s1),(t2,s2) in zip(rows,rows[1:]) if s2>s1]
if not w: print("no windows"); sys.exit()
print(f"windows={len(w)} first_step={rows[0][1]} last_step={rows[-1][1]}")
print("ms/step per window:", " ".join(f"{x:.0f}" for x in w))
ss=w[2:] if len(w)>4 else w
print(f"steady (skip 2): median {st.median(ss):.0f}  mean {st.mean(ss):.0f}  max {max(ss):.0f}  min {min(ss):.0f}")
