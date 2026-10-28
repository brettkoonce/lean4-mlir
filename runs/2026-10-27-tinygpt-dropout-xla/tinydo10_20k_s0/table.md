# the ladder under TinyGPT — runs/2026-10-27-tinygpt-dropout-xla/tinydo10_20k_s0

| row | P1 chunked (T=128) | P2 streaming |
|---|---|---|
| uniform | 6.0224 | 6.0224 |
| KN n=1 | 4.8291 | 4.8284 |
| KN n=2 | 3.5830 | 3.5825 |
| KN n=3 | 2.9451 | 2.9405 |
| KN n=4 | 2.4917 | 2.4805 |
| KN n=5 | 2.2685 | 2.2529 |
| KN n=6 | 2.2142 | 2.1975 |
| KN n=7 | 2.2056 | 2.1887 |
| KN n=8 | 2.2092 | 2.1926 |
| KN n=9 | 2.2135 | 2.1966 |
| KN n=10 | 2.2142 | 2.1973 |
| TinyGPT tiny-do10 | 2.2106 | 2.1799 |

plateau (P2): order 6 at 2.1975 bits/char (best order 7 at 2.1887)
gap, TinyGPT tiny-do10 − KN n=7 (P2): -0.0088 bits/char, 95% block-bootstrap CI [-0.0253, +0.0091] over 111,412 positions
gap, TinyGPT tiny-do10 − KN n=7 (P1): +0.0050 bits/char; P1 − P2 for the net: +0.0307
