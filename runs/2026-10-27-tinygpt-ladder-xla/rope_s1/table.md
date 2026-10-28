# the ladder under TinyGPT — runs/2026-10-27-tinygpt-ladder-xla/rope_s1

| row | P1 chunked (T=64) | P2 streaming |
|---|---|---|
| uniform | 6.0224 | 6.0224 |
| KN n=1 | 4.8291 | 4.8287 |
| KN n=2 | 3.5830 | 3.5826 |
| KN n=3 | 2.9503 | 2.9404 |
| KN n=4 | 2.5045 | 2.4803 |
| KN n=5 | 2.2861 | 2.2526 |
| KN n=6 | 2.2334 | 2.1971 |
| KN n=7 | 2.2249 | 2.1883 |
| KN n=8 | 2.2283 | 2.1922 |
| KN n=9 | 2.2322 | 2.1963 |
| KN n=10 | 2.2330 | 2.1969 |
| TinyGPT nano-rope-nopos | 2.2800 | 2.2245 |

plateau (P2): order 6 at 2.1971 bits/char (best order 7 at 2.1883)
gap, TinyGPT nano-rope-nopos − KN n=7 (P2): +0.0362 bits/char, 95% block-bootstrap CI [+0.0268, +0.0457] over 111,476 positions
gap, TinyGPT nano-rope-nopos − KN n=7 (P1): +0.0551 bits/char; P1 − P2 for the net: +0.0554
