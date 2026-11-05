"""Row 53: the scale cost probe with the cascade (ADR-010 part C and the amendment's section 3; C.4's two conditions), read
cold from the files `cortex_c2b.scale_probe` retained under `metrics/mqar/` and from `PROVENANCE-scale-probe.json`, by
`cortex_c2b.scale_reading`. Every number of the row recomputed from the files; these tests fail if a digit moves. The
tables below hold every file of the record as it arrives (the three encoders' five ten thousand draws under both gates,
the hundred thousand draw on seed 2024's encoder), complete on 2026-11-05: 79 files, every one in the tables."""
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
M = ROOT / "metrics" / "mqar"

from cortex_c2b import scale_reading as SR                       # noqa: E402
from cortex_c2b import scale_probe as SP                         # noqa: E402
from cortex_c2b import sealed_families as S                      # noqa: E402


def close(a, b, rel=1e-9):
    """Equality of two readings up to the last bits of their floats: the exact intervals are solved by bisection on binomial tails
    computed with `math.lgamma`, whose last bits differ between the founder's C runtime (Windows) and the instance's (Linux); every
    count and label must be equal, every float within a relative 1e-9 (the founder's run of 2026-11-05 found the difference)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], rel) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool) and abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)
    return a == b

S2024, S7, S1337 = "4cfd6f6e1406", "2e1dc71913b1", "8ceba5db8d0b"
THRESHOLD = {S1337: 0.5360028743743896, S2024: 0.6122402548789978, S7: 0.5907554626464844}
FREEZE_SHA256 = "33b57d451ca1887dffb86664fbc01944f6e63aa39485d86498155382a27c1439"
FROZEN_TREE = "676060c734b6d3d795ba401d6610d7a0f3fe8ee5a0f713ee26ff68273ca157d0"
LIFECYCLE_HASH = "4901a04848d1d29a"
PART_C_COMMITS = ("dfe7145", "a3c312e", "2d59e5c", "1d824db", "f90ec79")   # the code states of part C's sessions: the tree after zip 55 (seed 2024's twenty five sessions), after
                                                                             # the seed 2024 files' commit (seed 1337's twenty five), after the seed 1337 files' commit (seed 7's first
                                                                             # seven) and after the working notes of 2026-11-03 (seed 7's other eighteen), after the seed 7 files' commit
                                                                             # (the hundred thousand draw's five); the modules equal on every session (SCALE_PROBE_SHA, the freeze)
SCALE_PROBE_SHA = "965dbdacf840170e"
CONSOLIDATE_BATCH, PERIOD = 8, 16

# ---- the files as they date them (seed 2024, 2026-10-30; the encoders of 1337 and 7 and the hundred thousand draw added when retained) ----
ENCODING = {
    (50000, S2024): dict(diff=1.7881393432617188e-07, rounding=1, flipped=0, repeated=[], plan="43eab4af76340c21", threads=8, start="13:10:59", end="13:12:19", day="2026-10-30", commit="dfe7145"),
    (50001, S2024): dict(diff=1.7881393432617188e-07, rounding=1, flipped=0, repeated=[], plan="7f1c2bf7502b06f5", threads=8, start="13:41:42", end="13:43:04", day="2026-10-30", commit="dfe7145"),
    (50002, S2024): dict(diff=1.7881393432617188e-07, rounding=1, flipped=0, repeated=[], plan="d0ec67561fcf38c3", threads=8, start="14:12:14", end="14:13:39", day="2026-10-30", commit="dfe7145"),
    (50003, S2024): dict(diff=1.7881393432617188e-07, rounding=1, flipped=0, repeated=[], plan="6df561eb134b1dd5", threads=8, start="14:42:50", end="14:44:05", day="2026-10-30", commit="dfe7145"),
    (50004, S2024): dict(diff=1.7881393432617188e-07, rounding=1, flipped=0, repeated=["Dazu-73c7"], plan="a7084075ae3d5ea1", threads=8, start="15:12:34", end="15:13:48", day="2026-10-30", commit="dfe7145"),
    (50000, S1337): dict(diff=1.7881393432617188e-07, rounding=2, flipped=0, repeated=[], plan="43eab4af76340c21", threads=8, day="2026-10-30", start="16:13:51", end="16:16:37", commit="a3c312e"),
    (50001, S1337): dict(diff=1.7881393432617188e-07, rounding=2, flipped=0, repeated=[], plan="7f1c2bf7502b06f5", threads=8, day="2026-10-30", start="20:42:41", end="20:43:58", commit="a3c312e"),
    (50002, S1337): dict(diff=1.7881393432617188e-07, rounding=2, flipped=0, repeated=[], plan="d0ec67561fcf38c3", threads=8, day="2026-10-30", start="21:12:05", end="21:13:15", commit="a3c312e"),
    (50003, S1337): dict(diff=1.7881393432617188e-07, rounding=2, flipped=0, repeated=[], plan="6df561eb134b1dd5", threads=8, day="2026-10-30", start="21:41:20", end="21:42:31", commit="a3c312e"),
    (50004, S1337): dict(diff=1.7881393432617188e-07, rounding=2, flipped=0, repeated=['Dazu-73c7'], plan="a7084075ae3d5ea1", threads=8, day="2026-10-30", start="22:10:30", end="22:11:40", commit="a3c312e"),
    (50000, S7): dict(diff=2.384185791015625e-07, rounding=2, flipped=0, repeated=[], plan="43eab4af76340c21", threads=8, day="2026-11-03", start="21:18:45", end="21:20:01", commit="2d59e5c"),
    (50001, S7): dict(diff=2.384185791015625e-07, rounding=2, flipped=0, repeated=[], plan="7f1c2bf7502b06f5", threads=8, day="2026-11-03", start="21:48:07", end="21:49:29", commit="2d59e5c"),
    (50002, S7): dict(diff=2.384185791015625e-07, rounding=2, flipped=0, repeated=[], plan="d0ec67561fcf38c3", threads=8, day="2026-11-03", start="22:17:50", end="22:19:08", commit="1d824db"),
    (50003, S7): dict(diff=2.384185791015625e-07, rounding=2, flipped=0, repeated=[], plan="6df561eb134b1dd5", threads=8, day="2026-11-03", start="22:47:18", end="22:48:34", commit="1d824db"),
    (50004, S7): dict(diff=2.384185791015625e-07, rounding=2, flipped=0, repeated=['Dazu-73c7'], plan="a7084075ae3d5ea1", threads=8, day="2026-11-03", start="23:16:56", end="23:18:10", commit="1d824db"),
    (60000, S2024): dict(diff=1.7881393432617188e-07, rounding=1, flipped=0, repeated=['Divu-1e8e', 'Giku-d078', 'Muzo-000b', 'Nuvi-2551', 'Roro-0aab', 'Safu-a238', 'Tise-d5d4', 'Vaza-8c17', 'Voru-8939', 'Zovi-7c9a'], plan="559d30bfba3b18a2", threads=8, day="2026-11-03", start="23:58:24", end="00:11:25", commit="f90ec79"),
}
# C.2: the planting (admitted, refused, per kind admitted and under the floor 0.2), the sample's own line in the window per index and lever, I0's
# candidates (median, p95, max), the ties at the boundary, the agreement (own in both, same set, same first), the never planted control (at or above,
# n; by distance), the pairs (admitted, own in window, duplicate in window, contested, the mark ranks the duplicate first, R4 not following),
# R1's citation of the duplicate with the model's citation taken as the duplicate, the key's contract (reads, both lines, R4 abstains), the time
# per query medians in ns (I0, I1, I2) interleaved and blocked, the traced bytes, I0's buckets, the session's start and end (UTC)
C2 = {
    (50000, S2024, "identity"): dict(admitted=11000, refused=0, by_kind={"fact": (10000, 88), "E1": (500, 499), "E2": (500, 465)}, own={"I0": 2000, "I1": 2000, "I2": 2000, "16x12": 1999, "32x10": 2000, "32x12": 2000}, cand=(275.0, 372.0, 497.0), ties=0, agree=(2000, 596, 2000), never=(906, 1000), never_dist={"1": (3, 3), "2": (102, 107), "3": (744, 824), "4": (57, 66)}, e1=(500, 500, 500, 500, 243, 0), e2=(500, 500, 500, 500, 66, 0), e1_sel=252, e2_sel=68, kc=(0, 0, 0), t_inter=(1447300.0, 26107300.0, 3752550.0), t_block=(1286350.0, 25880900.0, 3140050.0), bytes=(8137646, 678468, 6596204), buckets=15927, start="13:14:38", end="13:18:55", day="2026-10-30", commit="dfe7145"),
    (50000, S2024, "surprise"): dict(admitted=10047, refused=953, by_kind={"fact": (9958, 44), "E1": (11, 10), "E2": (78, 43)}, own={"I0": 1996, "I1": 1996, "I2": 1996, "16x12": 1995, "32x10": 1996, "32x12": 1996}, cand=(251.0, 338.0, 446.0), ties=0, agree=(1996, 495, 2000), never=(903, 1000), never_dist={"1": (3, 3), "2": (101, 107), "3": (742, 824), "4": (57, 66)}, e1=(11, 11, 11, 11, 7, 0), e2=(78, 78, 78, 78, 4, 0), e1_sel=7, e2_sel=4, kc=(0, 0, 0), t_inter=(1348250.0, 24398200.0, 3500950.0), t_block=(1200550.0, 24447400.0, 3097100.0), bytes=(7441264, 448060, 5886810), buckets=15890, start="13:20:51", end="13:24:55", day="2026-10-30", commit="dfe7145"),
    (50001, S2024, "identity"): dict(admitted=11000, refused=0, by_kind={"fact": (10000, 82), "E1": (500, 499), "E2": (500, 459)}, own={"I0": 2000, "I1": 2000, "I2": 2000, "16x12": 1999, "32x10": 2000, "32x12": 2000}, cand=(274.0, 370.04999999999995, 498.0), ties=0, agree=(2000, 579, 2000), never=(898, 1000), never_dist={"1": (3, 3), "2": (92, 101), "3": (758, 847), "4": (45, 49)}, e1=(500, 500, 500, 500, 254, 0), e2=(500, 500, 500, 500, 62, 0), e1_sel=257, e2_sel=66, kc=(0, 0, 0), t_inter=(1460450.0, 26339250.0, 3715550.0), t_block=(1384000.0, 26276500.0, 3175400.0), bytes=(8136718, 678468, 6587818), buckets=15931, start="13:45:23", end="13:49:42", day="2026-10-30", commit="dfe7145"),
    (50001, S2024, "surprise"): dict(admitted=10070, refused=930, by_kind={"fact": (9970, 51), "E1": (7, 6), "E2": (93, 52)}, own={"I0": 1995, "I1": 1995, "I2": 1995, "16x12": 1994, "32x10": 1995, "32x12": 1995}, cand=(251.0, 340.04999999999995, 456.0), ties=0, agree=(1995, 466, 2000), never=(895, 1000), never_dist={"1": (3, 3), "2": (91, 101), "3": (756, 847), "4": (45, 49)}, e1=(7, 7, 7, 7, 3, 0), e2=(93, 91, 93, 91, 5, 0), e1_sel=3, e2_sel=5, kc=(0, 0, 0), t_inter=(1333850.0, 23906800.0, 3373750.0), t_block=(1183300.0, 24070700.0, 2843050.0), bytes=(7454240, 448612, 5885307), buckets=15894, start="13:51:35", end="13:55:31", day="2026-10-30", commit="dfe7145"),
    (50002, S2024, "identity"): dict(admitted=11000, refused=0, by_kind={"fact": (10000, 77), "E1": (500, 499), "E2": (500, 472)}, own={"I0": 2000, "I1": 2000, "I2": 2000, "16x12": 1999, "32x10": 2000, "32x12": 2000}, cand=(274.0, 373.0, 478.0), ties=0, agree=(2000, 565, 2000), never=(901, 1000), never_dist={"1": (4, 4), "2": (98, 105), "3": (747, 829), "4": (52, 62)}, e1=(500, 500, 500, 500, 256, 0), e2=(500, 500, 500, 500, 45, 0), e1_sel=266, e2_sel=49, kc=(0, 0, 0), t_inter=(1424500.0, 25889350.0, 3414750.0), t_block=(1261350.0, 25898100.0, 3031000.0), bytes=(8133150, 678468, 6608324), buckets=15929, start="14:15:57", end="14:20:16", day="2026-10-30", commit="dfe7145"),
    (50002, S2024, "surprise"): dict(admitted=10041, refused=959, by_kind={"fact": (9964, 40), "E1": (9, 8), "E2": (68, 40)}, own={"I0": 1996, "I1": 1996, "I2": 1996, "16x12": 1995, "32x10": 1996, "32x12": 1996}, cand=(251.0, 341.0, 451.0), ties=0, agree=(1996, 473, 2000), never=(901, 1000), never_dist={"1": (4, 4), "2": (98, 105), "3": (747, 829), "4": (52, 62)}, e1=(9, 9, 9, 9, 3, 0), e2=(68, 68, 68, 68, 3, 0), e1_sel=3, e2_sel=3, kc=(0, 0, 0), t_inter=(1329450.0, 23686800.0, 3152500.0), t_block=(1189600.0, 24024550.0, 2668700.0), bytes=(7435616, 447916, 5895659), buckets=15896, start="14:22:10", end="14:26:09", day="2026-10-30", commit="dfe7145"),
    (50003, S2024, "identity"): dict(admitted=11000, refused=0, by_kind={"fact": (10000, 72), "E1": (500, 500), "E2": (500, 464)}, own={"I0": 2000, "I1": 2000, "I2": 2000, "16x12": 1999, "32x10": 2000, "32x12": 2000}, cand=(271.0, 365.0, 488.0), ties=0, agree=(2000, 555, 2000), never=(899, 1000), never_dist={"1": (2, 2), "2": (93, 103), "3": (757, 840), "4": (47, 55)}, e1=(500, 500, 500, 500, 266, 0), e2=(500, 500, 500, 500, 52, 0), e1_sel=273, e2_sel=55, kc=(0, 0, 0), t_inter=(1416900.0, 26066550.0, 3723250.0), t_block=(1245700.0, 25366000.0, 3009550.0), bytes=(8136166, 678468, 6601350), buckets=15936, start="14:46:28", end="14:50:49", day="2026-10-30", commit="dfe7145"),
    (50003, S2024, "surprise"): dict(admitted=10060, refused=940, by_kind={"fact": (9968, 39), "E1": (9, 9), "E2": (83, 47)}, own={"I0": 1993, "I1": 1993, "I2": 1993, "16x12": 1992, "32x10": 1993, "32x12": 1993}, cand=(248.0, 334.0, 450.0), ties=0, agree=(1993, 475, 2000), never=(894, 1000), never_dist={"1": (2, 2), "2": (92, 103), "3": (753, 840), "4": (47, 55)}, e1=(9, 7, 9, 7, 3, 0), e2=(83, 82, 83, 82, 2, 0), e1_sel=3, e2_sel=2, kc=(0, 0, 0), t_inter=(1287600.0, 23147100.0, 3182950.0), t_block=(1125200.0, 23119400.0, 2768600.0), bytes=(7450768, 448372, 5894990), buckets=15916, start="14:52:35", end="14:56:21", day="2026-10-30", commit="dfe7145"),
    (50004, S2024, "identity"): dict(admitted=11000, refused=0, by_kind={"fact": (10000, 99), "E1": (500, 499), "E2": (500, 466)}, own={"I0": 2000, "I1": 2000, "I2": 2000, "16x12": 1999, "32x10": 2000, "32x12": 2000}, cand=(277.0, 372.0, 504.0), ties=0, agree=(2000, 561, 2000), never=(897, 1000), never_dist={"1": (2, 3), "2": (119, 132), "3": (716, 801), "4": (60, 64)}, e1=(500, 500, 500, 500, 231, 0), e2=(500, 500, 500, 500, 49, 0), e1_sel=242, e2_sel=52, kc=(1, 1, 1), t_inter=(1422450.0, 25115200.0, 3532950.0), t_block=(1340900.0, 25159200.0, 3110950.0), bytes=(8133686, 678468, 6608093), buckets=15934, start="15:16:01", end="15:20:09", day="2026-10-30", commit="dfe7145"),
    (50004, S2024, "surprise"): dict(admitted=10064, refused=936, by_kind={"fact": (9959, 52), "E1": (10, 9), "E2": (95, 61)}, own={"I0": 1994, "I1": 1994, "I2": 1994, "16x12": 1993, "32x10": 1994, "32x12": 1994}, cand=(252.0, 338.0, 448.0), ties=0, agree=(1994, 475, 2000), never=(892, 1000), never_dist={"1": (2, 3), "2": (119, 132), "3": (712, 801), "4": (59, 64)}, e1=(10, 10, 10, 10, 6, 0), e2=(95, 93, 95, 93, 2, 0), e1_sel=6, e2_sel=2, kc=(1, 1, 1), t_inter=(1340050.0, 23657900.0, 3437550.0), t_block=(1180100.0, 23597900.0, 3097700.0), bytes=(7448128, 448468, 5903911), buckets=15902, start="15:21:56", end="15:25:50", day="2026-10-30", commit="dfe7145"),
    (50000, S1337, "identity"): dict(admitted=11000, refused=0, by_kind={'fact': (10000, 90), 'E1': (500, 500), 'E2': (500, 442)}, own={'I0': 2000, 'I1': 2000, 'I2': 2000, '16x12': 2000, '32x10': 2000, '32x12': 2000}, cand=(271.0, 369.0, 522.0), ties=0, agree=(2000, 581, 2000), never=(999, 1000), never_dist={'1': (3, 3), '2': (107, 107), '3': (823, 824), '4': (66, 66)}, e1=(500, 500, 500, 500, 242, 0), e2=(500, 500, 498, 498, 49, 0), e1_sel=252, e2_sel=51, kc=(0, 0, 0), t_inter=(1461700.0, 27495850.0, 3770200.0), t_block=(1391050.0, 27461150.0, 3390650.0), bytes=(8143534, 678468, 6596204), buckets=15987, day="2026-10-30", start="16:19:00", end="16:23:49", commit="a3c312e"),
    (50000, S1337, "surprise"): dict(admitted=10080, refused=920, by_kind={'fact': (9967, 55), 'E1': (8, 8), 'E2': (105, 47)}, own={'I0': 1989, 'I1': 1989, 'I2': 1989, '16x12': 1989, '32x10': 1989, '32x12': 1989}, cand=(248.0, 337.0, 476.0), ties=0, agree=(1989, 500, 2000), never=(999, 1000), never_dist={'1': (3, 3), '2': (107, 107), '3': (823, 824), '4': (66, 66)}, e1=(8, 8, 8, 8, 3, 0), e2=(105, 103, 103, 101, 2, 0), e1_sel=3, e2_sel=2, kc=(0, 0, 0), t_inter=(1369100.0, 25079750.0, 3746300.0), t_block=(1210750.0, 24837200.0, 3274150.0), bytes=(7464568, 448852, 5895404), buckets=15963, day="2026-10-30", start="20:21:34", end="20:25:50", commit="a3c312e"),
    (50001, S1337, "identity"): dict(admitted=11000, refused=0, by_kind={'fact': (10000, 101), 'E1': (500, 500), 'E2': (500, 439)}, own={'I0': 2000, 'I1': 2000, 'I2': 2000, '16x12': 2000, '32x10': 2000, '32x12': 2000}, cand=(269.0, 371.04999999999995, 504.0), ties=0, agree=(2000, 600, 2000), never=(994, 1000), never_dist={'1': (3, 3), '2': (100, 101), '3': (842, 847), '4': (49, 49)}, e1=(500, 500, 500, 500, 259, 0), e2=(500, 500, 498, 498, 37, 0), e1_sel=264, e2_sel=43, kc=(0, 0, 0), t_inter=(1409900.0, 25288450.0, 3605100.0), t_block=(1235500.0, 25169300.0, 3136650.0), bytes=(8139614, 678468, 6587818), buckets=16009, day="2026-10-30", start="20:46:17", end="20:50:23", commit="a3c312e"),
    (50001, S1337, "surprise"): dict(admitted=10081, refused=919, by_kind={'fact': (9960, 59), 'E1': (12, 12), 'E2': (109, 48)}, own={'I0': 1987, 'I1': 1987, 'I2': 1987, '16x12': 1987, '32x10': 1987, '32x12': 1987}, cand=(247.0, 341.0999999999999, 471.0), ties=0, agree=(1987, 513, 2000), never=(994, 1000), never_dist={'1': (3, 3), '2': (100, 101), '3': (842, 847), '4': (49, 49)}, e1=(12, 11, 12, 11, 5, 0), e2=(109, 109, 107, 107, 1, 0), e1_sel=5, e2_sel=1, kc=(0, 0, 0), t_inter=(1302950.0, 22851300.0, 3203400.0), t_block=(1155700.0, 22850850.0, 2835600.0), bytes=(7462536, 448876, 5885148), buckets=15983, day="2026-10-30", start="20:52:12", end="20:55:57", commit="a3c312e"),
    (50002, S1337, "identity"): dict(admitted=11000, refused=0, by_kind={'fact': (10000, 95), 'E1': (500, 500), 'E2': (500, 440)}, own={'I0': 2000, 'I1': 2000, 'I2': 2000, '16x12': 1999, '32x10': 2000, '32x12': 2000}, cand=(270.0, 371.0, 534.0), ties=0, agree=(2000, 553, 2000), never=(993, 1000), never_dist={'1': (4, 4), '2': (105, 105), '3': (822, 829), '4': (62, 62)}, e1=(500, 500, 500, 500, 226, 0), e2=(500, 500, 500, 500, 60, 0), e1_sel=236, e2_sel=64, kc=(0, 0, 0), t_inter=(1352400.0, 25366700.0, 3506950.0), t_block=(1216800.0, 25316250.0, 2971300.0), bytes=(8134526, 678468, 6608324), buckets=15969, day="2026-10-30", start="21:15:28", end="21:19:34", commit="a3c312e"),
    (50002, S1337, "surprise"): dict(admitted=10076, refused=924, by_kind={'fact': (9966, 57), 'E1': (10, 10), 'E2': (100, 40)}, own={'I0': 1995, 'I1': 1995, 'I2': 1995, '16x12': 1994, '32x10': 1995, '32x12': 1995}, cand=(248.0, 342.0, 489.0), ties=0, agree=(1995, 470, 2000), never=(993, 1000), never_dist={'1': (4, 4), '2': (105, 105), '3': (822, 829), '4': (62, 62)}, e1=(10, 10, 10, 10, 4, 0), e2=(100, 100, 100, 100, 4, 0), e1_sel=5, e2_sel=4, kc=(0, 0, 0), t_inter=(1257350.0, 22973750.0, 3112550.0), t_block=(1104450.0, 22878050.0, 2751650.0), bytes=(7457648, 448756, 5905949), buckets=15944, day="2026-10-30", start="21:21:23", end="21:25:07", commit="a3c312e"),
    (50003, S1337, "identity"): dict(admitted=11000, refused=0, by_kind={'fact': (10000, 105), 'E1': (500, 500), 'E2': (500, 427)}, own={'I0': 2000, 'I1': 2000, 'I2': 2000, '16x12': 1998, '32x10': 2000, '32x12': 2000}, cand=(273.0, 373.04999999999995, 473.0), ties=0, agree=(2000, 564, 2000), never=(993, 1000), never_dist={'1': (2, 2), '2': (102, 103), '3': (835, 840), '4': (54, 55)}, e1=(500, 500, 500, 500, 219, 0), e2=(500, 500, 499, 499, 43, 0), e1_sel=229, e2_sel=48, kc=(0, 0, 0), t_inter=(1380900.0, 25287450.0, 3609950.0), t_block=(1222450.0, 25162550.0, 3175300.0), bytes=(8143030, 678468, 6601350), buckets=15982, day="2026-10-30", start="21:44:43", end="21:48:48", commit="a3c312e"),
    (50003, S1337, "surprise"): dict(admitted=10095, refused=905, by_kind={'fact': (9969, 73), 'E1': (9, 9), 'E2': (117, 44)}, own={'I0': 1992, 'I1': 1992, 'I2': 1992, '16x12': 1990, '32x10': 1992, '32x12': 1992}, cand=(250.0, 343.04999999999995, 433.0), ties=0, agree=(1992, 475, 2000), never=(991, 1000), never_dist={'1': (2, 2), '2': (102, 103), '3': (833, 840), '4': (54, 55)}, e1=(9, 8, 9, 8, 4, 0), e2=(117, 117, 116, 116, 1, 0), e1_sel=4, e2_sel=1, kc=(0, 0, 0), t_inter=(1281150.0, 23026800.0, 3322100.0), t_block=(1133850.0, 22929850.0, 2803000.0), bytes=(7470744, 449212, 5906266), buckets=15957, day="2026-10-30", start="21:50:37", end="21:54:22", commit="a3c312e"),
    (50004, S1337, "identity"): dict(admitted=11000, refused=0, by_kind={'fact': (10000, 94), 'E1': (500, 500), 'E2': (500, 427)}, own={'I0': 2000, 'I1': 2000, 'I2': 2000, '16x12': 2000, '32x10': 2000, '32x12': 2000}, cand=(273.0, 365.0, 511.0), ties=0, agree=(2000, 615, 2000), never=(989, 1000), never_dist={'1': (3, 3), '2': (132, 132), '3': (790, 801), '4': (64, 64)}, e1=(500, 500, 500, 500, 239, 0), e2=(500, 500, 500, 500, 47, 0), e1_sel=248, e2_sel=49, kc=(1, 1, 1), t_inter=(1395350.0, 25193550.0, 3634150.0), t_block=(1229750.0, 25130950.0, 3356200.0), bytes=(8137966, 678468, 6608093), buckets=15971, day="2026-10-30", start="22:13:53", end="22:17:59", commit="a3c312e"),
    (50004, S1337, "surprise"): dict(admitted=10095, refused=905, by_kind={'fact': (9967, 59), 'E1': (6, 6), 'E2': (122, 49)}, own={'I0': 1994, 'I1': 1994, 'I2': 1994, '16x12': 1994, '32x10': 1994, '32x12': 1994}, cand=(250.0, 336.0, 465.0), ties=0, agree=(1994, 527, 2000), never=(989, 1000), never_dist={'1': (3, 3), '2': (132, 132), '3': (790, 801), '4': (64, 64)}, e1=(6, 6, 6, 6, 1, 0), e2=(122, 122, 122, 122, 2, 0), e1_sel=1, e2_sel=2, kc=(1, 1, 1), t_inter=(1280300.0, 22826900.0, 3251200.0), t_block=(1133400.0, 22911700.0, 2976800.0), bytes=(7471432, 449212, 5911214), buckets=15947, day="2026-10-30", start="22:19:47", end="22:23:33", commit="a3c312e"),
    (50000, S7, "identity"): dict(admitted=11000, refused=0, by_kind={'fact': (10000, 76), 'E1': (500, 498), 'E2': (500, 490)}, own={'I0': 2000, 'I1': 2000, 'I2': 2000, '16x12': 1999, '32x10': 2000, '32x12': 2000}, cand=(267.0, 366.0, 484.0), ties=0, agree=(2000, 546, 2000), never=(944, 1000), never_dist={'1': (3, 3), '2': (106, 107), '3': (772, 824), '4': (63, 66)}, e1=(500, 500, 500, 500, 245, 0), e2=(500, 500, 500, 500, 96, 0), e1_sel=255, e2_sel=105, kc=(0, 0, 0), t_inter=(1371100.0, 25194500.0, 3498700.0), t_block=(1225350.0, 25065050.0, 3049100.0), bytes=(8146494, 678468, 6596204), buckets=16001, day="2026-11-03", start="21:22:13", end="21:26:17", commit="2d59e5c"),
    (50000, S7, "surprise"): dict(admitted=9998, refused=1002, by_kind={'fact': (9962, 38), 'E1': (6, 4), 'E2': (30, 20)}, own={'I0': 1996, 'I1': 1996, 'I2': 1996, '16x12': 1995, '32x10': 1996, '32x12': 1996}, cand=(242.5, 333.0, 432.0), ties=0, agree=(1996, 452, 2000), never=(940, 1000), never_dist={'1': (3, 3), '2': (106, 107), '3': (768, 824), '4': (63, 66)}, e1=(6, 5, 6, 5, 2, 0), e2=(30, 30, 30, 30, 2, 0), e1_sel=2, e2_sel=2, kc=(0, 0, 0), t_inter=(1277700.0, 22859800.0, 3198250.0), t_block=(1151900.0, 22834750.0, 2836550.0), bytes=(7423608, 446884, 5873581), buckets=15971, day="2026-11-03", start="21:28:13", end="21:32:02", commit="2d59e5c"),
    (50001, S7, "identity"): dict(admitted=11000, refused=0, by_kind={'fact': (10000, 73), 'E1': (500, 499), 'E2': (500, 483)}, own={'I0': 2000, 'I1': 2000, 'I2': 2000, '16x12': 2000, '32x10': 2000, '32x12': 2000}, cand=(270.0, 370.0, 520.0), ties=0, agree=(2000, 527, 2000), never=(942, 1000), never_dist={'1': (3, 3), '2': (94, 101), '3': (803, 847), '4': (42, 49)}, e1=(500, 500, 500, 500, 228, 0), e2=(500, 500, 500, 500, 81, 0), e1_sel=235, e2_sel=87, kc=(0, 0, 0), t_inter=(1383450.0, 25908900.0, 3616400.0), t_block=(1229600.0, 25727500.0, 3081600.0), bytes=(8141926, 678468, 6587818), buckets=16000, day="2026-11-03", start="21:51:49", end="21:56:04", commit="1d824db", session_commit="2d59e5c"),
    (50001, S7, "surprise"): dict(admitted=10025, refused=975, by_kind={'fact': (9968, 41), 'E1': (10, 9), 'E2': (47, 30)}, own={'I0': 1994, 'I1': 1994, 'I2': 1994, '16x12': 1994, '32x10': 1994, '32x12': 1994}, cand=(245.0, 337.0, 472.0), ties=0, agree=(1994, 425, 2000), never=(938, 1000), never_dist={'1': (3, 3), '2': (94, 101), '3': (799, 847), '4': (42, 49)}, e1=(10, 10, 10, 10, 4, 0), e2=(47, 46, 47, 46, 2, 0), e1_sel=4, e2_sel=2, kc=(0, 0, 0), t_inter=(1326000.0, 24236200.0, 3630300.0), t_block=(1156100.0, 23417750.0, 2671050.0), bytes=(7437432, 447532, 5871365), buckets=15981, day="2026-11-03", start="21:57:55", end="22:01:45", commit="1d824db"),
    (50002, S7, "identity"): dict(admitted=11000, refused=0, by_kind={'fact': (10000, 86), 'E1': (500, 499), 'E2': (500, 491)}, own={'I0': 2000, 'I1': 2000, 'I2': 2000, '16x12': 2000, '32x10': 2000, '32x12': 2000}, cand=(267.0, 361.04999999999995, 462.0), ties=0, agree=(2000, 555, 2000), never=(955, 1000), never_dist={'1': (4, 4), '2': (103, 105), '3': (791, 829), '4': (57, 62)}, e1=(500, 500, 500, 500, 230, 0), e2=(500, 500, 500, 500, 91, 0), e1_sel=239, e2_sel=98, kc=(0, 0, 0), t_inter=(1336100.0, 25580400.0, 3517900.0), t_block=(1194700.0, 25846300.0, 3129900.0), bytes=(8144022, 678468, 6608324), buckets=15970, day="2026-11-03", start="22:21:23", end="22:25:36", commit="1d824db"),
    (50002, S7, "surprise"): dict(admitted=9996, refused=1004, by_kind={'fact': (9956, 40), 'E1': (10, 9), 'E2': (30, 21)}, own={'I0': 1991, 'I1': 1991, 'I2': 1991, '16x12': 1991, '32x10': 1991, '32x12': 1991}, cand=(242.0, 328.0, 422.0), ties=0, agree=(1991, 463, 2000), never=(953, 1000), never_dist={'1': (4, 4), '2': (103, 105), '3': (789, 829), '4': (57, 62)}, e1=(10, 10, 10, 10, 4, 0), e2=(30, 30, 30, 30, 2, 0), e1_sel=4, e2_sel=2, kc=(0, 0, 0), t_inter=(1236850.0, 23412000.0, 3148050.0), t_block=(1074250.0, 23287100.0, 2879900.0), bytes=(7421968, 446836, 5884412), buckets=15940, day="2026-11-03", start="22:27:23", end="22:31:08", commit="1d824db"),
    (50003, S7, "identity"): dict(admitted=11000, refused=0, by_kind={'fact': (10000, 72), 'E1': (500, 499), 'E2': (500, 487)}, own={'I0': 1999, 'I1': 2000, 'I2': 2000, '16x12': 1999, '32x10': 2000, '32x12': 2000}, cand=(267.0, 359.0, 476.0), ties=0, agree=(1999, 533, 1999), never=(948, 1000), never_dist={'1': (2, 2), '2': (98, 103), '3': (798, 840), '4': (50, 55)}, e1=(500, 500, 500, 500, 247, 0), e2=(500, 500, 500, 500, 80, 0), e1_sel=256, e2_sel=87, kc=(0, 0, 0), t_inter=(1385050.0, 25783650.0, 3658200.0), t_block=(1203950.0, 25965950.0, 3097000.0), bytes=(8144310, 678468, 6601350), buckets=15982, day="2026-11-03", start="22:50:50", end="22:55:02", commit="1d824db"),
    (50003, S7, "surprise"): dict(admitted=10013, refused=987, by_kind={'fact': (9962, 34), 'E1': (10, 9), 'E2': (41, 28)}, own={'I0': 1993, 'I1': 1994, 'I2': 1994, '16x12': 1993, '32x10': 1994, '32x12': 1994}, cand=(243.0, 327.0, 441.0), ties=0, agree=(1993, 444, 1999), never=(947, 1000), never_dist={'1': (2, 2), '2': (98, 103), '3': (797, 840), '4': (50, 55)}, e1=(10, 10, 10, 10, 7, 0), e2=(41, 41, 41, 41, 2, 0), e1_sel=7, e2_sel=2, kc=(0, 0, 0), t_inter=(1255400.0, 23274450.0, 3441250.0), t_block=(1122350.0, 23204450.0, 3011600.0), bytes=(7435688, 447244, 5884037), buckets=15955, day="2026-11-03", start="22:56:52", end="23:00:41", commit="1d824db"),
    (50004, S7, "identity"): dict(admitted=11000, refused=0, by_kind={'fact': (10000, 80), 'E1': (500, 500), 'E2': (500, 488)}, own={'I0': 1999, 'I1': 2000, 'I2': 2000, '16x12': 2000, '32x10': 2000, '32x12': 2000}, cand=(265.0, 366.04999999999995, 479.0), ties=0, agree=(1999, 514, 1999), never=(941, 1000), never_dist={'1': (3, 3), '2': (130, 132), '3': (748, 801), '4': (60, 64)}, e1=(500, 500, 500, 500, 258, 0), e2=(500, 500, 500, 500, 93, 0), e1_sel=263, e2_sel=99, kc=(1, 1, 1), t_inter=(1381650.0, 25907500.0, 3841550.0), t_block=(1294650.0, 25453400.0, 3226600.0), bytes=(8140982, 678468, 6608093), buckets=15990, day="2026-11-03", start="23:20:25", end="23:24:39", commit="1d824db"),
    (50004, S7, "surprise"): dict(admitted=10002, refused=998, by_kind={'fact': (9958, 37), 'E1': (6, 6), 'E2': (38, 26)}, own={'I0': 1991, 'I1': 1992, 'I2': 1992, '16x12': 1992, '32x10': 1992, '32x12': 1992}, cand=(241.0, 332.04999999999995, 433.0), ties=0, agree=(1991, 425, 1999), never=(937, 1000), never_dist={'1': (3, 3), '2': (130, 132), '3': (744, 801), '4': (60, 64)}, e1=(6, 6, 6, 6, 4, 0), e2=(38, 37, 38, 37, 1, 0), e1_sel=4, e2_sel=1, kc=(1, 1, 1), t_inter=(1265900.0, 23354550.0, 3422950.0), t_block=(1133550.0, 22893800.0, 2721950.0), bytes=(7425000, 446980, 5888088), buckets=15961, day="2026-11-03", start="23:26:30", end="23:30:18", commit="1d824db"),
    (60000, S2024, "identity"): dict(r1_ci={"pairs_E1": (0.505, 0.533), "pairs_E2": (0.111, 0.129)}, repeated_apart_read={"pairs_E1": (1, 1, 0), "pairs_E2": (2, 2, 1)}, admitted=110000, refused=0, by_kind={'fact': (100000, 7365), 'E1': (5000, 4998), 'E2': (5000, 4698)}, own={'I0': 2000, 'I1': 2000, 'I2': 2000, '16x12': 1999, '32x10': 2000, '32x12': 2000}, cand=(2769.0, 3675.2, 5076.0), ties=0, agree=(2000, 1332, 2000), never=(1000, 1000), never_dist={'1': (31, 31), '2': (683, 683), '3': (286, 286)}, e1=(4999, 4999, 4999, 4999, 2595, 0), e2=(4998, 4997, 4995, 4994, 597, 0), e1_sel=2680, e2_sel=647, kc=(1, 1, 1), t_inter=(13096600.0, 303044050.0, 65242900.0), t_block=(12858100.0, 306383500.0, 64970600.0), bytes=(61496582, 6484180, 54655543), buckets=16382, day="2026-11-04", start="02:40:25", end="04:31:53", commit="f90ec79"),
    (60000, S2024, "surprise"): dict(repeated_apart_read={"pairs_E1": (0, 0, 0), "pairs_E2": (0, 0, 0)}, admitted=97565, refused=12435, by_kind={'fact': (96662, 3847), 'E1': (114, 105), 'E2': (789, 478)}, own={'I0': 1934, 'I1': 1934, 'I2': 1934, '16x12': 1933, '32x10': 1934, '32x12': 1934}, cand=(2453.0, 3260.1, 4519.0), ties=0, agree=(1934, 1239, 2000), never=(1000, 1000), never_dist={'1': (31, 31), '2': (683, 683), '3': (286, 286)}, e1=(114, 75, 114, 75, 36, 0), e2=(789, 750, 786, 747, 27, 0), e1_sel=37, e2_sel=26, kc=(1, 1, 1), t_inter=(11549050.0, 263272650.0, 54882300.0), t_block=(11301500.0, 266230750.0, 54726750.0), bytes=(55236632, 6185740, 42582271), buckets=16381, day="2026-11-04", start="06:47:16", end="13:05:58", commit="f90ec79", timer_overhead=200.0),
}
# C.3: the phases, the entries (the sentinels included), the admitted, the states at the end (live, consolidated, demoted, evicted), the applied
# totals (consolidated, evicted, withdrawn, demoted groups, replay failed), the near duplicates (admitted, evicted, live, read before eviction as
# the final file counts it, over the sample reads alone), the reads after the last write (n, sample, pairs, own in window, holes, duplicates in
# window), the reconstruction reading at the last phase (reconstructions, patterns, evicted addresses, still reconstructed under the bound), the
# defect bounds, the phase times (sum, first, last, max), the linear fit's slope and r2, the read lines the final file counts, the session (UTC)
CASCADE = {
    (50000, S2024, "identity"): dict(probe_s=9.1, phases=687, entries=11020, admitted=11000, states=(4464, 95, 0, 6453), applied=(5496, 6453, 5401, 0, 0), e1=(500, 499, 1, 0), e2=(500, 465, 35, 8), last=(1925, 925, 1000, 1382, 4038, 36), recon=(5496, 95, 5401, 0), defect=(0.1698, 0.5610), time=(486.7, 0.062, 1.36, 1.593), slope=0.00181, r2=0.983, lines=2398, start="13:25:14", end="13:33:37", day="2026-10-30", commit="dfe7145"),
    (50000, S2024, "surprise"): dict(probe_s=9.64, phases=627, entries=10067, admitted=10047, states=(4939, 103, 0, 5010), applied=(5016, 5010, 4913, 0, 0), e1=(11, 10, 1, 0), e2=(78, 43, 35, 11), last=(2021, 1021, 1000, 1527, 3188, 36), recon=(5016, 103, 4913, 0), defect=(0.1620, 0.5717), time=(455.1, 0.063, 1.36, 1.578), slope=0.00202, r2=0.984, lines=2763, start="13:33:40", end="13:41:32", day="2026-10-30", commit="dfe7145"),
    (50001, S2024, "identity"): dict(probe_s=9.02, phases=687, entries=11020, admitted=11000, states=(4476, 98, 0, 6438), applied=(5496, 6438, 5398, 0, 0), e1=(500, 499, 1, 0), e2=(500, 459, 41, 11), last=(1906, 906, 1000, 1331, 4050, 42), recon=(5496, 98, 5398, 0), defect=(0.1356, 0.5851), time=(481.1, 0.063, 1.297, 1.438), slope=0.00175, r2=0.986, lines=2401, start="13:55:52", end="14:04:10", day="2026-10-30", commit="dfe7145"),
    (50001, S2024, "surprise"): dict(probe_s=9.82, phases=629, entries=10090, admitted=10070, states=(4943, 100, 0, 5041), applied=(5032, 5041, 4932, 0, 0), e1=(7, 6, 1, 0), e2=(93, 52, 41, 14), last=(1996, 996, 1000, 1465, 3292, 42), recon=(5032, 100, 4932, 0), defect=(0.1391, 0.5977), time=(453.9, 0.063, 1.406, 1.562), slope=0.00201, r2=0.980, lines=2754, start="14:04:13", end="14:12:04", day="2026-10-30", commit="dfe7145"),
    (50002, S2024, "identity"): dict(probe_s=8.9, phases=687, entries=11020, admitted=11000, states=(4468, 98, 0, 6446), applied=(5496, 6446, 5398, 0, 0), e1=(500, 499, 1, 0), e2=(500, 472, 28, 8), last=(1873, 873, 1000, 1346, 3934, 29), recon=(5496, 98, 5398, 0), defect=(0.1576, 0.6000), time=(485.7, 0.047, 1.406, 1.844), slope=0.00180, r2=0.974, lines=2302, start="14:26:30", end="14:34:52", day="2026-10-30", commit="dfe7145"),
    (50002, S2024, "surprise"): dict(probe_s=9.56, phases=627, entries=10061, admitted=10041, states=(4948, 101, 0, 5003), applied=(5016, 5003, 4915, 0, 0), e1=(9, 8, 1, 0), e2=(68, 40, 28, 8), last=(1967, 967, 1000, 1477, 3071, 29), recon=(5016, 101, 4915, 0), defect=(0.1527, 0.6505), time=(450.3, 0.063, 1.531, 1.531), slope=0.00198, r2=0.979, lines=2707, start="14:34:54", end="14:42:41", day="2026-10-30", commit="dfe7145"),
    (50003, S2024, "identity"): dict(probe_s=8.38, phases=687, entries=11020, admitted=11000, states=(4480, 97, 0, 6435), applied=(5496, 6435, 5399, 0, 0), e1=(500, 500, 0, 0), e2=(500, 464, 36, 8), last=(1919, 919, 1000, 1358, 4003, 36), recon=(5496, 97, 5399, 0), defect=(0.1714, 0.6080), time=(468.2, 0.062, 1.297, 1.5), slope=0.00173, r2=0.988, lines=2429, start="14:56:41", end="15:04:45", day="2026-10-30", commit="dfe7145"),
    (50003, S2024, "surprise"): dict(probe_s=9.79, phases=628, entries=10080, admitted=10060, states=(4949, 101, 0, 5018), applied=(5024, 5018, 4923, 0, 0), e1=(9, 9, 0, 0), e2=(83, 47, 36, 10), last=(2019, 1019, 1000, 1508, 3147, 36), recon=(5024, 101, 4923, 0), defect=(0.1645, 0.5708), time=(440.4, 0.047, 1.282, 1.469), slope=0.00196, r2=0.981, lines=2809, start="15:04:47", end="15:12:25", day="2026-10-30", commit="dfe7145"),
    (50004, S2024, "identity"): dict(probe_s=9.16, phases=687, entries=11020, admitted=11000, states=(4452, 99, 0, 6461), applied=(5496, 6461, 5397, 0, 0), e1=(500, 499, 1, 0), e2=(500, 466, 34, 7), last=(1872, 872, 1000, 1334, 4019, 35), recon=(5496, 99, 5397, 0), defect=(0.1605, 0.5895), time=(483.8, 0.078, 1.281, 1.531), slope=0.00178, r2=0.984, lines=2335, start="15:26:10", end="15:34:31", day="2026-10-30", commit="dfe7145"),
    (50004, S2024, "surprise"): dict(probe_s=9.99, phases=629, entries=10084, admitted=10064, states=(4930, 101, 0, 5053), applied=(5032, 5053, 4931, 0, 0), e1=(10, 9, 1, 0), e2=(95, 61, 34, 8), last=(1981, 981, 1000, 1483, 3190, 35), recon=(5032, 101, 4931, 0), defect=(0.1630, 0.5758), time=(441.3, 0.063, 1.25, 1.515), slope=0.00189, r2=0.985, lines=2692, start="15:34:33", end="15:42:11", day="2026-10-30", commit="dfe7145"),
    (50000, S1337, "identity"): dict(probe_s=8.92, phases=687, entries=11020, admitted=11000, states=(4484, 95, 0, 6433), applied=(5496, 6433, 5401, 0, 0), e1=(500, 500, 0, 0), e2=(500, 442, 58, 17), last=(1920, 920, 1000, 1343, 3971, 56), recon=(5496, 95, 5401, 0), defect=(0.1634, 0.6105), time=(489.3, 0.063, 1.313, 1.985), slope=0.00179, r2=0.946, lines=2420, day="2026-10-30", start="20:26:12", end="20:34:38", commit="a3c312e"),
    (50000, S1337, "surprise"): dict(probe_s=9.95, phases=630, entries=10100, admitted=10080, states=(4950, 102, 0, 5048), applied=(5040, 5048, 4938, 0, 0), e1=(8, 8, 0, 0), e2=(105, 47, 58, 24), last=(2034, 1034, 1000, 1493, 3227, 56), recon=(5040, 102, 4938, 0), defect=(0.1551, 0.5982), time=(453.9, 0.062, 1.313, 1.375), slope=0.00195, r2=0.983, lines=2791, day="2026-10-30", start="20:34:41", end="20:42:32", commit="a3c312e"),
    (50001, S1337, "identity"): dict(probe_s=8.46, e2_late=2, under_apart=1, phases=687, entries=11020, admitted=11000, states=(4477, 97, 0, 6438), applied=(5496, 6438, 5399, 0, 0), e1=(500, 500, 0, 0), e2=(500, 441, 59, 14), last=(1921, 921, 1000, 1365, 3994, 57), recon=(5496, 97, 5399, 0), defect=(0.1170, 0.5624), time=(470.4, 0.062, 1.25, 1.422), slope=0.00173, r2=0.991, lines=2375, day="2026-10-30", start="20:56:15", end="21:04:21", commit="a3c312e"),
    (50001, S1337, "surprise"): dict(probe_s=8.85, phases=630, entries=10101, admitted=10081, states=(4941, 101, 0, 5058), applied=(5040, 5058, 4939, 0, 0), e1=(12, 12, 0, 0), e2=(109, 50, 59, 16), last=(2008, 1008, 1000, 1488, 3213, 57), recon=(5040, 101, 4939, 0), defect=(0.1391, 0.5697), time=(437.5, 0.063, 1.265, 1.5), slope=0.00192, r2=0.992, lines=2735, day="2026-10-30", start="21:04:23", end="21:11:57", commit="a3c312e"),
    (50002, S1337, "identity"): dict(probe_s=8.68, e2_late=1, phases=687, entries=11020, admitted=11000, states=(4481, 96, 0, 6435), applied=(5496, 6435, 5400, 0, 0), e1=(500, 500, 0, 0), e2=(500, 441, 59, 10), last=(1873, 873, 1000, 1334, 3944, 59), recon=(5496, 96, 5400, 0), defect=(0.1163, 0.6189), time=(474.0, 0.078, 1.375, 1.5), slope=0.00170, r2=0.989, lines=2330, day="2026-10-30", start="21:25:26", end="21:33:36", commit="a3c312e"),
    (50002, S1337, "surprise"): dict(probe_s=8.62, phases=629, entries=10096, admitted=10076, states=(4945, 101, 0, 5038), applied=(5032, 5038, 4931, 0, 0), e1=(10, 10, 0, 0), e2=(100, 41, 59, 13), last=(1974, 974, 1000, 1479, 3137, 59), recon=(5032, 101, 4931, 0), defect=(0.1656, 0.5711), time=(436.8, 0.062, 1.281, 1.391), slope=0.00191, r2=0.992, lines=2714, day="2026-10-30", start="21:33:39", end="21:41:11", commit="a3c312e"),
    (50003, S1337, "identity"): dict(probe_s=8.04, phases=687, entries=11020, admitted=11000, states=(4484, 98, 0, 6430), applied=(5496, 6430, 5398, 0, 0), e1=(500, 500, 0, 0), e2=(500, 427, 73, 15), last=(1872, 872, 1000, 1323, 3965, 72), recon=(5496, 98, 5398, 0), defect=(0.1565, 0.5908), time=(471.2, 0.062, 1.265, 1.406), slope=0.00175, r2=0.992, lines=2326, day="2026-10-30", start="21:54:42", end="22:02:48", commit="a3c312e"),
    (50003, S1337, "surprise"): dict(probe_s=8.96, phases=630, entries=10115, admitted=10095, states=(4934, 101, 0, 5065), applied=(5040, 5065, 4939, 0, 0), e1=(9, 9, 0, 0), e2=(117, 44, 73, 23), last=(1978, 978, 1000, 1473, 3206, 72), recon=(5040, 101, 4939, 0), defect=(0.1628, 0.5534), time=(434.5, 0.062, 1.281, 1.563), slope=0.00191, r2=0.991, lines=2692, day="2026-10-30", start="22:02:50", end="22:10:21", commit="a3c312e"),
    (50004, S1337, "identity"): dict(probe_s=8.48, phases=687, entries=11020, admitted=11000, states=(4495, 96, 0, 6421), applied=(5496, 6421, 5400, 0, 0), e1=(500, 500, 0, 0), e2=(500, 427, 73, 26), last=(1905, 905, 1000, 1361, 4023, 73), recon=(5496, 96, 5400, 0), defect=(0.2018, 0.6204), time=(469.7, 0.062, 1.266, 1.437), slope=0.00173, r2=0.992, lines=2401, day="2026-10-30", start="22:23:52", end="22:31:57", commit="a3c312e"),
    (50004, S1337, "surprise"): dict(probe_s=8.87, phases=630, entries=10115, admitted=10095, states=(4946, 101, 0, 5053), applied=(5040, 5053, 4939, 0, 0), e1=(6, 6, 0, 0), e2=(122, 49, 73, 28), last=(2004, 1004, 1000, 1509, 3221, 73), recon=(5040, 101, 4939, 0), defect=(0.1415, 0.5965), time=(436.0, 0.047, 1.265, 1.438), slope=0.00191, r2=0.991, lines=2750, day="2026-10-30", start="22:32:00", end="22:39:31", commit="a3c312e"),
    (50000, S7, "identity"): dict(probe_s=9.26, phases=687, entries=11020, admitted=11000, states=(4452, 98, 0, 6462), applied=(5496, 6462, 5398, 0, 0), e1=(500, 498, 2, 0), e2=(500, 490, 10, 1), last=(1942, 942, 1000, 1378, 4126, 12), recon=(5496, 98, 5398, 0), defect=(0.174, 0.5793), time=(467.0, 0.078, 1.281, 1.36), slope=0.00171, r2=0.997, lines=2469, day="2026-11-03", start="21:32:21", end="21:40:25", commit="2d59e5c"),
    (50000, S7, "surprise"): dict(probe_s=9.74, phases=624, entries=10018, admitted=9998, states=(4950, 100, 0, 4954), applied=(4992, 4954, 4892, 0, 0), e1=(6, 4, 2, 0), e2=(30, 20, 10, 2), last=(2061, 1061, 1000, 1549, 3256, 12), recon=(4992, 100, 4892, 0), defect=(0.1776, 0.5959), time=(432.7, 0.062, 1.344, 1.359), slope=0.00192, r2=0.993, lines=2868, day="2026-11-03", start="21:40:28", end="21:47:58", commit="2d59e5c"),
    (50001, S7, "identity"): dict(probe_s=8.89, phases=687, entries=11020, admitted=11000, states=(4461, 98, 0, 6453), applied=(5496, 6453, 5398, 0, 0), e1=(500, 499, 1, 0), e2=(500, 483, 17, 2), last=(1861, 861, 1000, 1279, 4100, 18), recon=(5496, 98, 5398, 0), defect=(0.1757, 0.5859), time=(465.3, 0.062, 1.218, 1.343), slope=0.00172, r2=0.994, lines=2318, day="2026-11-03", start="22:02:05", end="22:10:06", commit="1d824db"),
    (50001, S7, "surprise"): dict(probe_s=9.18, phases=626, entries=10045, admitted=10025, states=(4948, 102, 0, 4986), applied=(5008, 4986, 4906, 0, 0), e1=(10, 9, 1, 0), e2=(47, 30, 17, 5), last=(1973, 973, 1000, 1439, 3273, 18), recon=(5008, 102, 4906, 0), defect=(0.1324, 0.5876), time=(434.3, 0.063, 1.328, 1.563), slope=0.00196, r2=0.986, lines=2707, day="2026-11-03", start="22:10:09", end="22:17:40", commit="1d824db"),
    (50002, S7, "identity"): dict(probe_s=8.89, phases=687, entries=11020, admitted=11000, states=(4440, 98, 0, 6474), applied=(5496, 6474, 5398, 0, 0), e1=(500, 499, 1, 0), e2=(500, 491, 9, 2), last=(1876, 876, 1000, 1308, 4059, 10), recon=(5496, 98, 5398, 0), defect=(0.1565, 0.5899), time=(468.9, 0.063, 1.281, 1.375), slope=0.00172, r2=0.988, lines=2300, day="2026-11-03", start="22:31:28", end="22:39:33", commit="1d824db"),
    (50002, S7, "surprise"): dict(probe_s=8.98, phases=624, entries=10016, admitted=9996, states=(4942, 101, 0, 4961), applied=(4992, 4961, 4891, 0, 0), e1=(10, 9, 1, 0), e2=(30, 21, 9, 3), last=(1979, 979, 1000, 1469, 3162, 10), recon=(4992, 101, 4891, 0), defect=(0.1572, 0.5961), time=(436.0, 0.062, 1.282, 1.422), slope=0.00197, r2=0.985, lines=2699, day="2026-11-03", start="22:39:35", end="22:47:08", commit="1d824db"),
    (50003, S7, "identity"): dict(probe_s=8.89, under_apart=1, phases=687, entries=11020, admitted=11000, states=(4459, 100, 0, 6453), applied=(5496, 6453, 5396, 0, 0), e1=(500, 499, 1, 0), e2=(500, 487, 13, 6), last=(1878, 878, 1000, 1328, 4002, 14), recon=(5496, 100, 5396, 0), defect=(0.1179, 0.6111), time=(475.7, 0.078, 1.453, 1.468), slope=0.0017, r2=0.98, lines=2333, day="2026-11-03", start="23:01:01", end="23:09:13", commit="1d824db"),
    (50003, S7, "surprise"): dict(probe_s=9.13, under_apart=2, phases=625, entries=10033, admitted=10013, states=(4951, 100, 0, 4969), applied=(5000, 4969, 4900, 0, 0), e1=(10, 9, 1, 0), e2=(41, 27, 14, 7), last=(1973, 973, 1000, 1475, 3128, 15), recon=(5000, 100, 4900, 0), defect=(0.1435, 0.6066), time=(435.5, 0.062, 1.25, 1.406), slope=0.00191, r2=0.98, lines=2707, day="2026-11-03", start="23:09:15", end="23:16:47", commit="1d824db"),
    (50004, S7, "identity"): dict(probe_s=8.49, phases=687, entries=11020, admitted=11000, states=(4448, 96, 0, 6468), applied=(5496, 6468, 5400, 0, 0), e1=(500, 500, 0, 0), e2=(500, 488, 12, 3), last=(1912, 912, 1000, 1380, 4082, 12), recon=(5496, 96, 5400, 0), defect=(0.1573, 0.6075), time=(466.1, 0.063, 1.25, 1.484), slope=0.00175, r2=0.988, lines=2451, day="2026-11-03", start="23:30:38", end="23:38:40", commit="1d824db"),
    (50004, S7, "surprise"): dict(probe_s=9.79, phases=625, entries=10022, admitted=10002, states=(4951, 100, 0, 4969), applied=(5000, 4969, 4900, 0, 0), e1=(6, 6, 0, 0), e2=(38, 26, 12, 6), last=(2015, 1015, 1000, 1527, 3157, 12), recon=(5000, 100, 4900, 0), defect=(0.1532, 0.5884), time=(434.3, 0.047, 1.234, 1.61), slope=0.00194, r2=0.975, lines=2829, day="2026-11-03", start="23:38:42", end="23:46:13", commit="1d824db"),
    (60000, S2024, "identity"): dict(probe_s=312.06, e2_late=13, phases=6875, entries=110020, admitted=110000, states=(37959, 76, 0, 71985), applied=(55000, 71985, 54924, 0, 0), e1=(5000, 4998, 2, 0), e2=(5000, 4711, 289, 7), last=(10758, 758, 10000, 4505, 29357, 288), recon=(55000, 76, 54924, 0), defect=(0.0803, 0.7462), time=(52713.3, 0.094, 11.14, 10814.0), slope=0.00158, r2=0.001, lines=1804, day="2026-11-04", start="13:07:39", end="07:59:53", commit="f90ec79", sessions=2, budget=50400.0),
}
# the hundred thousand draw (C.1 (a), C.3 (b)): the plan's counts and the read phases differ from the ten thousand draws'; an entry gives them where they differ
TEN_THOUSAND = dict(n_statements=11000, n_questions=11000, n_pairs=(500, 500), n_reads=4000, read_phases=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, "last"], sessions=1, budget=None,
                    summaries=0, demoted=0, still_under_bound=0, repeated_apart=None, timer_overhead=100.0)
HUNDRED_THOUSAND = dict(n_statements=110000, n_questions=101000, n_pairs=(4999, 4998), n_reads=13000, read_phases=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, "last"],
                        repeated_apart=(1, 2))        # the pairs whose name is one of the draw's ten repeated names, read apart (E1 one, E2 two): 4 999 and 4 998 pairs in the pools


def expect(e: dict, draw: int) -> dict:
    """An entry's expectations: the ten thousand defaults, the hundred thousand draw's counts, the entry's own fields over both."""
    return {**TEN_THOUSAND, **(HUNDRED_THOUSAND if draw >= 60000 else {}), **e}


def sha_lf(p: Path, n: int = 16) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:n]


@pytest.fixture(scope="module")
def reading():
    return SR.read_rows(M)


@pytest.fixture(scope="module")
def raw():
    out = {}
    for e in SR.expected_files():
        p = M / e["name"]
        if p.exists():
            out[(e["kind"], e["draw"], e["run_id"], e["gate"])] = json.loads(p.read_text(encoding="utf-8"))
    return out


def present(raw, kind, draw, run_id, gate=None):
    d = raw.get((kind, draw, run_id, gate))
    if d is None:
        pytest.skip(f"{kind} {draw} {run_id} {gate}: not retained yet (part C running)")
    return d


# ---- the files, the logs, the identity -------------------------------------------------------------------------------------
def test_the_tables_name_files_that_exist_and_the_reading_reads_every_file_present(reading, raw):
    for (draw, run_id) in ENCODING:
        assert ("encoding", draw, run_id, None) in raw
    for (draw, run_id, gate) in C2:
        assert ("c2", draw, run_id, gate) in raw
    for (draw, run_id, gate) in CASCADE:
        assert ("cascade", draw, run_id, gate) in raw
    assert reading["files"]["present"] == len(raw) and reading["files"]["outside_the_expected_set"] == []


def test_the_logs_printed_hashes_equal_the_committed_files(raw):
    for (kind, draw, run_id, gate), d in raw.items():
        if kind == "encoding":
            continue
        log = M / "logs" / f"scale-{kind}-{draw}-{run_id}.log"
        assert log.exists(), f"{log.name}: a final file without its log"            # the log travels with the file
        raw_bytes = log.read_bytes()
        assert raw_bytes[:2] == b"\xff\xfe"                                           # UTF-16 as Tee-Object writes it
        text = raw_bytes.decode("utf-16")
        name = f"scale-{kind}-{draw}-{run_id}-{gate}.json"
        m = re.search(r"retained: metrics\\mqar\\" + re.escape(name) + r"\s+sha256 ([0-9a-f]{16})", text)
        assert m, name
        assert m.group(1) == sha_lf(M / name) == hashlib.sha256((M / name).read_bytes()).hexdigest()[:16]


def test_every_file_is_bound_to_the_freeze_the_seal_the_checkpoint_and_the_launcher(raw):
    for (kind, draw, run_id, gate), d in raw.items():
        c = d["identity"]["compared"]
        assert c["freeze_sha256"] == FREEZE_SHA256 and c["frozen_tree_sha256"] == FROZEN_TREE and c["draw"] == draw and c["run_id"] == run_id
        assert c["environment"]["PYTHONHASHSEED"] == "0"
        if kind == "encoding":
            assert c["plan_content_sha256"] == d["plan_content_sha256"]
        seal = json.loads(S.SEAL_PATH.read_text(encoding="utf-8"))["scale_draws"][str(draw)]["content_sha256"]
        assert c["plan_content_sha256"] == seal
        for s in d["sessions"]:
            assert s["code_state"]["commit"] in PART_C_COMMITS and s["code_state"]["tree_clean"] is True
            assert s["code_state"]["new_modules_sha256_lf"]["cortex_c2b/scale_probe.py"] == SCALE_PROBE_SHA
            assert "[tools/adr010_launcher.py]" in s["command"] and s["traceback"] is None
        if kind != "encoding":
            assert c["gate"] == gate and c["admission"] == gate
        if kind == "cascade":
            assert c["lifecycle_config_hash"] == LIFECYCLE_HASH == d["lifecycle_config"]["hash"]


# ---- the encodings (C.1 (c)) ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("key", sorted(ENCODING))
def test_the_encoding(raw, key):
    draw, run_id = key
    d = present(raw, "encoding", draw, run_id)
    e = expect(ENCODING[key], draw)
    b = d["batched_against_single"]
    assert (d["n_statements"], d["n_questions"], d["batch_size"]) == (e["n_statements"], e["n_questions"], 64)
    assert b["largest_abs_cue_difference"] == e["diff"] and b["components_rounding_changed"] == e["rounding"] and b["components"] == 12800
    assert b["signatures_flipped"] == e["flipped"] and b["signatures"] == 3200
    assert sorted(d["repeated_names"]) == e["repeated"] and d["plan_content_sha256"].startswith(e["plan"])
    assert len(d["sessions"]) == 1 and d["sessions"][0]["reason"] == "complete" and d["sessions"][0]["start"] == f"{e['day']}T{e['start']}+00:00" and d["sessions"][0]["end"].endswith(f"T{e['end']}+00:00")
    assert d["sessions"][0]["code_state"]["commit"] == e["commit"]
    assert d["device"]["threads"] == e["threads"]                 # the encoding is not pinned: the model's forward passes


# ---- C.2 -----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("key", sorted(C2))
def test_c2_the_planting_the_sample_the_never_planted_control_and_the_pairs(raw, key):
    draw, run_id, gate = key
    d = present(raw, "c2", draw, run_id, gate)
    e = expect(C2[key], draw)
    p, a, s = d["planting"], d["aggregates"], d["aggregates"]["sample"]
    assert (p["admitted"], p["refused"], p["entries"], p["n_decisions"]) == (e["admitted"], e["refused"], e["admitted"], e["n_statements"]) and p["phases"] == 0 and p["sentinels"] == 0
    assert {k: (v["admitted"], v["salience"]["under_floor_0.2"]) for k, v in p["by_kind"].items()} == e["by_kind"]
    assert p["shadow"] == "not asked for" and p["admission_record"] == gate and p["run_in_sessions"] == 1 and p["resumption_fault"] == []
    assert d["threshold"]["value"] == THRESHOLD[run_id] == d["threshold"]["row47"] and d["threshold"]["within_1e-6"] is True
    assert d["k"] == 4 and d["bm25"] == {"k1": 1.5, "b": 0.75} and d["entries"] == e["admitted"]
    assert s["n"] == 2000 and {"I0": s["i0"]["own_in_window"], "I1": s["i1"]["own_in_window"], "I2": s["i2"]["own_in_window"], **{k: v["own_in_window"] for k, v in s["levers"].items()}} == e["own"]
    assert (s["i0"]["candidates"]["median"], s["i0"]["candidates"]["p95"], s["i0"]["candidates"]["max"]) == e["cand"] and s["i0"]["ties_at_boundary"] == e["ties"]
    assert s["i0"]["own_in_candidates"] == s["i0"]["own_in_window"]        # the own line among the candidates is in the window at k = 4 on every sample read
    assert (s["agree"]["own_in_both"], s["agree"]["same_set"], s["agree"]["same_first"]) == e["agree"]
    assert s["i1"]["compared"]["median"] == e["admitted"]                   # I1 compares every entry
    nn = a["never_planted"]
    assert (nn["max_mark_at_or_above_threshold"], nn["n"]) == e["never"] and {k: (v["at_or_above"], v["n"]) for k, v in nn["by_distance"].items()} == e["never_dist"]
    assert sum(v["n"] for v in nn["by_distance"].values()) == 1000 and nn["ties_at_boundary"] == 0 and nn["label"] == SP.NEVER_PLANTED_LABEL
    lo, hi = nn["clopper_pearson_95"]
    assert lo < e["never"][0] / 1000 <= hi and hi - lo < 0.04              # at 1 000 of 1 000 the share is the interval's upper bound
    for kind, exp, sel in (("pairs_E1", e["e1"], e["e1_sel"]), ("pairs_E2", e["e2"], e["e2_sel"])):
        x = a[kind]
        assert (x["duplicate_admitted"], x["own_in_window"], x["duplicate_in_window"], x["contested"], x["mark_ranks_duplicate_first"], x["r4_not_following"]) == exp
        assert x["n"] == e["n_pairs"][0 if kind == "pairs_E1" else 1] and x["r4_outcome_by_construction"] == x["contested"]
        if "r1_ci" in e:                                                    # the hundred thousand draw's intervals, as the row prints them
            assert tuple(round(v, 3) for v in x["r1_error_clopper_pearson_95"]) == e["r1_ci"][kind]
        ra = x["repeated_name_pairs_apart"]
        if e["repeated_apart"] is None:
            assert ra is None
        else:                                                               # the hundred thousand draw: the pairs on a repeated name, read apart, never pooled
            assert ra["n"] == e["repeated_apart"][0 if kind == "pairs_E1" else 1] and ra["r4_not_following"] == 0 and x["n"] + ra["n"] == 5000
            assert (ra["duplicate_admitted"], ra["contested"], ra["mark_ranks_duplicate_first"]) == e["repeated_apart_read"][kind]
        assert x["selection_counts"]["duplicate"]["R1"]["cites_duplicate"] == sel and x["selection_counts"]["duplicate"]["R1"]["abstain"] == 0
        if kind == "pairs_E1":                                              # two values of one entity: R4 abstains on every contested read
            assert x["selection_counts"]["duplicate"]["R4"] == {"abstain": x["contested"], "cites_own": 0, "cites_duplicate": 0}
        else:                                                               # two names: the own line is R4's single candidate
            assert x["selection_counts"]["duplicate"]["R4"] == {"abstain": 0, "cites_own": x["contested"], "cites_duplicate": 0}
        assert sum(v["n"] for v in x["margins_by_raw_cosine"].values()) == x["contested"]
    kc = a["key_contract"]
    assert (kc["n_sample_reads_of_cross_schema_repeated_names"], kc["both_lines_in_window"], kc["r4_abstains_with_both_lines"]) == e["kc"] and kc["claimed_for"] == "nothing"
    assert kc["r4_keeps_own_with_both_lines"] == 0 and kc["n_sample_reads_of_cross_schema_repeated_names"] == (1 if draw in (50004, 60000) else 0)   # eight files: six on draw 50 004, two on 60 000
    assert {k: v["n"] for k, v in d["planting"]["by_kind"].items()} == ({"fact": 100000, "E1": 5000, "E2": 5000} if draw == 60000 else {"fact": 10000, "E1": 500, "E2": 500})
    t = d["time_per_query"]
    assert tuple(t["interleaved"][ix]["median"] for ix in ("I0", "I1", "I2")) == e["t_inter"] and tuple(t["blocked"][ix]["median"] for ix in ("I0", "I1", "I2")) == e["t_block"]
    assert t["n_queries"] == 200 and t["repetitions"] == 6 and t["flagged_interleaved"] == [] and t["flagged_blocked"] == [] and t["timer_overhead_ns"] == e.get("timer_overhead", 100.0)
    assert all(t[mode][ix]["n"] == 1200 for mode in ("interleaved", "blocked") for ix in ("I0", "I1", "I2"))
    assert tuple(d["index_sizes"]["traced_bytes"][ix] for ix in ("I0", "I1", "I2")) == e["bytes"] and d["index_sizes"]["I0"]["buckets"] == e["buckets"]
    assert d["index_sizes"]["I0"]["vectors"] == d["index_sizes"]["I1"]["vectors"] == d["index_sizes"]["I2"]["documents"] == e["admitted"]
    assert d["n_reads"] == e["n_reads"] and d["environment"]["torch_threads"] == 1 and d["environment"]["thread_variables"] == {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    assert len(d["sessions"]) == 1 and d["sessions"][0]["reason"] == "complete" and d["sessions"][0]["start"] == f"{e['day']}T{e['start']}+00:00" and d["sessions"][0]["end"].endswith(f"T{e['end']}+00:00")
    assert d["code_commit"] == e["commit"] and d["sessions"][0]["code_state"]["commit"] == e.get("session_commit", e["commit"])   # one file: the commit of 2026-11-03 21:55 UTC fell between its session's start and its writing


# ---- C.3 -----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("key", sorted(CASCADE))
def test_c3_the_cascade(raw, key):
    draw, run_id, gate = key
    d = present(raw, "cascade", draw, run_id, gate)
    e = expect(CASCADE[key], draw)
    assert d["phases"] == e["phases"] and d["entries"] == e["entries"] and d["admitted"] == e["admitted"] and d["sentinels"] == 20
    if gate == "identity":
        assert d["phases"] == d["expected_phases_identity"] == SR.EXPECTED_PHASES[d["size"]]
    assert tuple(d["states"][s] for s in ("live", "consolidated", "demoted", "evicted")) == e["states"]
    ap = d["applied_totals"]
    assert tuple(ap[k] for k in ("consolidated", "evicted", "withdrawn", "demoted_groups", "replay_failed")) == e["applied"]
    assert ap["consolidated"] == CONSOLIDATE_BATCH * d["phases"] and ap["evicted"] == d["states"]["evicted"]
    assert d["faults"] == {"phases": [], "n": 0, "condition": "the withdrawal under cascade holds on every phase"}
    assert d["phases_refused"] == 0 and d["replay_failed_total"] == 0 and d["summaries"]["written"] == e["summaries"]
    nd = d["near_duplicates"]["by_kind"]
    for kind, exp in (("E1", e["e1"]), ("E2", e["e2"])):
        assert (nd[kind]["admitted"], nd[kind]["evicted"], nd[kind]["live"], nd[kind]["read_before_eviction"]) == exp
        assert nd[kind]["demoted"] == e["demoted"] and nd[kind]["grouped_with_partner"] == e["demoted"]
        assert nd[kind]["admitted"] == nd[kind]["evicted"] + nd[kind]["live"]
    assert len(d["near_duplicates"]["rows"]) == nd["E1"]["admitted"] + nd["E2"]["admitted"]
    if gate == "identity":                                                  # an E1 partner goes at the phase of its write (j // 16 + 1); an E2 partner too but for the few admitted above the floor
        late = {k: [r for r in d["near_duplicates"]["rows"] if r["kind"] == k and r.get("evicted_at_phase") is not None and r["evicted_at_phase"] != r["j"] // PERIOD + 1] for k in ("E1", "E2")}
        assert late["E1"] == [] and len(late["E2"]) == e.get("e2_late", 0)
        assert all(1 <= x["evicted_at_phase"] - x["j"] // PERIOD - 1 <= 3700 for x in late["E2"])
    assert abs(math.fsum(d["probe_times_seconds"]) - e["probe_s"]) <= 0.005 + 1e-9   # the probe's own time after the phases (the checks, the counts, the reads, the reconstruction), at two decimals;
    # 2026-11-05: `round(sum(...), 2)` read 8.895 as 8.9 under Python 3.11 (the CI's) and as 8.89 under 3.12 and later, whose sum() compensates the rounding of floats (the CI's red runs from 8263e61)
    last = d["reads_by_phase"]["last"]
    assert tuple(last[k] for k in ("n", "sample_read", "pairs_read", "own_in_window", "holes_total", "duplicates_in_window")) == e["last"]
    assert d["read_phases"] == e["read_phases"] and set(d["reads_by_phase"]) == {str(x) for x in e["read_phases"]}
    rc = d["reconstruction_by_phase"]["last"]
    assert (rc["reconstructions"], rc["patterns"], rc["evicted_address_reading"]["entries"], rc["evicted_address_reading"]["still_reconstructed_under_bound"]) == e["recon"]
    assert rc["bound"] == 0.05 and rc["radius"] == 0.9 and rc["evicted_address_reading"]["share"] == 0.0
    assert rc["reconstructions"] == ap["consolidated"] and rc["evicted_address_reading"]["entries"] == ap["withdrawn"]
    assert round(rc["evicted_address_reading"]["min_defect_evicted"], 4) == e["defect"][0] and round(rc["evicted_address_reading"]["max_defect_evicted"], 4) == e["defect"][1]
    for ph, v in d["reconstruction_by_phase"].items():
        assert v["evicted_address_reading"]["still_reconstructed_under_bound"] == e["still_under_bound"] and v["bound"] == 0.05
    pt = d["phase_times_seconds"]
    assert len(pt) == d["phases"] == d["n_units"] and (round(sum(pt), 1), pt[0], pt[-1], max(pt)) == e["time"]
    fit = SR._linear_fit(pt)
    assert round(fit["slope_seconds_per_phase"], 5) == e["slope"] and round(fit["r2"], 3) == e["r2"]
    assert d["phase_times_label"] == SP.TIME_LABEL and d["n_read_lines"] == e["lines"]
    assert d["n_read_lines"] == sum(v["sample_read"] for v in d["reads_by_phase"].values())   # the final file counts the sample reads' lines alone (ADR-010 section 11)
    assert len(d["sessions"]) == d["run_in_sessions"] == e["sessions"] and d["sessions"][-1]["reason"] == "complete" and d["resumption_fault"] == [] and d["read_lines_recorded_twice_compared"] == 0
    assert d["sessions"][0]["start"] == f"{e['day']}T{e['start']}+00:00" and d["sessions"][-1]["end"].endswith(f"T{e['end']}+00:00") and all(s["budget_seconds"] == e["budget"] for s in d["sessions"])
    if e["sessions"] > 1:                                                   # C.3 (b): every session but the last stopped by the budget at a phase boundary, resumed by the same command
        assert all(s["reason"] == "budget" for s in d["sessions"][:-1]) and all(s["resumed_from"] is not None for s in d["sessions"][1:])
    assert d["lifecycle_config"]["fields"]["period_writes"] == PERIOD and d["lifecycle_config"]["fields"]["consolidate_batch"] == CONSOLIDATE_BATCH
    assert d["lifecycle_config"]["fields"]["withdraw_at_eviction"] is True and d["lifecycle_config"]["fields"]["index_tables"] == 16 and d["lifecycle_config"]["fields"]["sentinel_seed"] == 2
    assert d["k"] == 4 and d["code_commit"] == e["commit"] == d["sessions"][0]["code_state"]["commit"]


@pytest.mark.parametrize("key", sorted(CASCADE))
def test_c3_the_evictions_without_a_pattern_are_c2s_under_floor_writes(raw, key):
    """evicted minus withdrawn counts the entries that went at their first phase under the floor without a consolidation (the
    near duplicates admitted under the identity rule, ADR-007 section 4 (c), and the facts whose surprise is under 0.2), as
    C.2's planting of the same cues counts them, less those written after the last phase (at most the period's remainder,
    8 of 11 000 under the identity rule), which no phase evicts; the exact tail count is in the progress file."""
    draw, run_id, gate = key
    cas = present(raw, "cascade", draw, run_id, gate); c2 = present(raw, "c2", draw, run_id, gate)
    under = sum(v["salience"]["under_floor_0.2"] for v in c2["planting"]["by_kind"].values())
    without_pattern = cas["applied_totals"]["evicted"] - cas["applied_totals"]["withdrawn"]
    e = expect(CASCADE[key], draw)
    assert under - without_pattern == e.get("under_apart", 0) <= cas["admitted"] - 16 * cas["phases"]   # equal on 28 of 31; one apart on two cascades, two on one
    assert cas["admitted"] == c2["planting"]["admitted"]                   # the two plantings of the same cues admit the same entries


# ---- C.4 and the pooled numbers -------------------------------------------------------------------------------------------------
def test_c4_the_two_conditions_on_the_files_present(reading):
    w = reading["c4"]["withdrawal_under_cascade"]
    assert w["holds_on_every_journal_and_phase"] and w["faults"] == [] and w["read_first"] and w["runs_read"] == len(reading["cascades"])
    never = reading["c4"]["never_planted_above_threshold"]
    for size, rows in never.items():
        for r in rows:
            assert r["n"] == 1000 and 0 < r["at_or_above"] <= 1000 and r["clopper_pearson_95"][0] < r["at_or_above"] / 1000 <= r["clopper_pearson_95"][1]
    assert reading["c4"]["label"] == SP.NEVER_PLANTED_LABEL and reading["c4"]["decision_13_successor"] == SP.DECISION_13_SUCCESSOR


# the pooled numbers per encoder and gate at ten thousand (five files each): the never planted control (at or above, by distance), the pairs (contested, the mark ranks the duplicate first)
POOLED_PER_ENCODER = {
    ("1337", "identity"): dict(never=(4968, 5000), never_bounds=(989, 999), by_distance={"1": (15, 15), "2": (546, 548), "3": (4112, 4141), "4": (295, 296)}, e1=(1185, 2500), e2=(236, 2495)),
    ("1337", "surprise"): dict(never=(4966, 5000), never_bounds=(989, 999), by_distance={"1": (15, 15), "2": (546, 548), "3": (4110, 4141), "4": (295, 296)}, e1=(17, 43), e2=(10, 546)),
    ("2024", "identity"): dict(never=(4501, 5000), never_bounds=(897, 906), by_distance={"1": (14, 15), "2": (504, 548), "3": (3722, 4141), "4": (261, 296)}, e1=(1250, 2500), e2=(274, 2500)),
    ("2024", "surprise"): dict(never=(4485, 5000), never_bounds=(892, 903), by_distance={"1": (14, 15), "2": (501, 548), "3": (3710, 4141), "4": (260, 296)}, e1=(22, 44), e2=(16, 412)),
    ("7", "identity"): dict(never=(4730, 5000), never_bounds=(941, 955), by_distance={"1": (15, 15), "2": (531, 548), "3": (3912, 4141), "4": (272, 296)}, e1=(1208, 2500), e2=(441, 2500)),
    ("7", "surprise"): dict(never=(4715, 5000), never_bounds=(937, 953), by_distance={"1": (15, 15), "2": (531, 548), "3": (3897, 4141), "4": (272, 296)}, e1=(21, 41), e2=(9, 184)),
}
# the hundred thousand draw (one file per gate, the encoder of seed 2024): the never planted control (at or above, n; by distance), the pairs
POOLED_HUNDRED_THOUSAND = {
    ("2024", "identity"): dict(never=(1000, 1000), by_distance={"1": (31, 31), "2": (683, 683), "3": (286, 286)}, e1=(2595, 4999), e2=(597, 4994)),
    ("2024", "surprise"): dict(never=(1000, 1000), by_distance={"1": (31, 31), "2": (683, 683), "3": (286, 286)}, e1=(36, 75), e2=(27, 747)),
}


def test_the_pooled_numbers_per_encoder_and_gate_at_ten_thousand(reading):
    for (enc, gate), e in POOLED_PER_ENCODER.items():
        p = reading["pooled"].get(f"ten_thousand/{gate}")
        if p is None or enc not in p["never_planted"]["per_encoder"]:
            pytest.skip(f"the encoder of seed {enc} under the {gate} rule: not retained yet")
        nn = p["never_planted"]["per_encoder"][enc]
        assert (nn["at_or_above"], nn["n"]) == e["never"] and (nn["per_file_bounds"]["min"], nn["per_file_bounds"]["max"], nn["per_file_bounds"]["n"]) == (*e["never_bounds"], 5)
        assert {k: (v["at_or_above"], v["n"]) for k, v in nn["by_distance"].items()} == e["by_distance"] and nn["ties_at_boundary"] == 0
        lo, hi = nn["clopper_pearson_95_pooled"]
        assert lo < e["never"][0] / e["never"][1] < hi and hi - lo < 0.02
        for kind, (m, n) in (("pairs_E1", e["e1"]), ("pairs_E2", e["e2"])):
            x = p[kind]["per_encoder"][enc]
            assert (x["mark_ranks_duplicate_first"], x["contested"], x["r4_not_following"]) == (m, n, 0)
            f = reading["decision_13_successor"]["filled"][f"ten_thousand/{enc}/{gate}/{kind[-2:]}"]
            assert (f["n"], f["N"], f["files"]) == (m, n, 5)
    # the pool over the encoders present is the sum of the per encoder pools
    for key, p in reading["pooled"].items():
        if "never_planted" in p:
            assert p["never_planted"]["at_or_above"] == sum(v["at_or_above"] for v in p["never_planted"]["per_encoder"].values())
            assert p["encoders"] == [int(k) for k in p["never_planted"]["per_encoder"]] == ([1337, 2024, 7] if key.startswith("ten_thousand") else [2024])   # the record's order
            if "cascade" in p:                                                                 # the hundred thousand draw has no cascade under the surprise rule (C.1 (a))
                assert p["cascade"]["faults_total"] == 0 and p["cascade"]["applied_totals"]["demoted_groups"]["max"] == 0 and p["cascade"]["still_reconstructed_under_bound_total"] == 0
    assert reading["decision_13_successor"]["label"] == SP.DECISION_13_SUCCESSOR


def test_the_pooled_numbers_per_encoder_and_gate_at_hundred_thousand(reading):
    for (enc, gate), e in POOLED_HUNDRED_THOUSAND.items():
        p = reading["pooled"][f"hundred_thousand/{gate}"]
        nn = p["never_planted"]["per_encoder"][enc]
        assert (nn["at_or_above"], nn["n"]) == e["never"] and nn["per_file_bounds"] == {"min": 1000, "max": 1000, "n": 1}
        assert {k: (v["at_or_above"], v["n"]) for k, v in nn["by_distance"].items()} == e["by_distance"] and nn["ties_at_boundary"] == 0
        lo, hi = nn["clopper_pearson_95_pooled"]
        assert round(lo, 4) == 0.9963 and hi == 1.0                              # 1 000 of 1 000: the exact interval's lower bound
        for kind, (m, n) in (("pairs_E1", e["e1"]), ("pairs_E2", e["e2"])):
            x = p[kind]["per_encoder"][enc]
            assert (x["mark_ranks_duplicate_first"], x["contested"], x["r4_not_following"]) == (m, n, 0)
            f = reading["decision_13_successor"]["filled"][f"hundred_thousand/{enc}/{gate}/{kind[-2:]}"]
            assert (f["n"], f["N"], f["files"]) == (m, n, 1)
        assert p["encoders"] == [2024]
    assert reading["pooled"]["hundred_thousand/identity"]["cascade"]["faults_total"] == 0 and "cascade" not in reading["pooled"]["hundred_thousand/surprise"]


def test_c3_the_hundred_thousand_cascade_ran_in_two_sessions_and_the_machine_slept_in_two_phases(raw):
    """C.3 (b): the first session stopped by the budget at the first phase boundary past fourteen hours, the second resumed it by the
    same command and completed it; the phase times carry the machine's sleep of 2026-11-04 in two phases (the founder's statement;
    the probe measures wall clock), 10 814.0 s and 699.9 s where the neighbours take 5.6 to 6.6 s; the line of the phase times
    against the phase number is fitted with and without those two phases, both reported by the row."""
    d = present(raw, "cascade", 60000, S2024, "identity")
    s1, s2 = d["sessions"]
    assert (s1["reason"], s1["resumed_from"], s1["first_unit"], s1["last_unit"], s1["budget_seconds"]) == ("budget", None, 1, 6641, 50400.0)
    assert (s2["reason"], s2["resumed_from"], s2["first_unit"], s2["last_unit"], s2["budget_seconds"]) == ("complete", 106256, 6642, 6875, 50400.0)
    assert s1["start"] == "2026-11-04T13:07:39+00:00" and s1["end"] == "2026-11-05T03:07:48+00:00" and s2["start"] == "2026-11-05T07:13:18+00:00" and s2["end"] == "2026-11-05T07:59:53+00:00"
    assert d["restored_index_at_last_reopening"]["vectors"] == 106276 and d["restored_index_at_last_reopening"]["evicted_entries_in_index"] == 69318
    assert s2["reopening_seconds"] == 247.062 and s2["reopening"]["action"] == "plain reopening" and s2["cuts"] == [] and s2["restored"] == [] and s2["moved"] == [] and s2["phases_at_start"] == 6641
    assert s2["reopening"]["log_length_before"] == s2["reopening"]["log_length_recorded"] == 152014760 and s2["reopening"]["lines_moved"] == 0 and s2["reopening"]["archive_tail_bytes"] == 0
    defects = {ph: v["evicted_address_reading"]["min_defect_evicted"] for ph, v in d["reconstruction_by_phase"].items()}
    assert {ph: (round(x, 4) if x is not None else None) for ph, x in defects.items()} == {"1": None, "2": None, "4": None, "8": None, "16": None, "32": 0.2338, "64": 0.1881, "128": 0.2137, "256": 0.2024, "512": 0.1875, "1024": 0.1398, "2048": 0.0895, "4096": 0.0792, "last": 0.0803}
    assert (min(pt_ := d["phase_times_seconds"]), pt_.index(min(pt_)) + 1) == (0.062, 2) and max(x for i, x in enumerate(pt_) if i + 1 not in (3170, 3171)) == 11.984
    pt = d["phase_times_seconds"]
    top = sorted(range(len(pt)), key=lambda i: -pt[i])[:3]
    assert [(i + 1, pt[i]) for i in top] == [(3170, 10814.0), (3171, 699.891), (6581, 11.984)]
    assert [round(pt[i - 1], 3) for i in (3167, 3168, 3169, 3172, 3173, 3174)] == [6.36, 6.234, 5.954, 6.641, 5.766, 5.594]
    fit_all = SR._linear_fit(pt); fit_ex = SR._linear_fit([x for i, x in enumerate(pt) if i + 1 not in (3170, 3171)])
    assert (round(fit_all["slope_seconds_per_phase"], 5), round(fit_all["r2"], 3)) == (0.00158, 0.001)
    assert (round(fit_ex["slope_seconds_per_phase"], 5), round(fit_ex["r2"], 3), fit_ex["n"]) == (0.00169, 0.996, 6873)
    assert round(sum(pt), 1) == 52713.3 and round(sum(pt) - pt[3169] - pt[3170], 1) == 41199.4 and (pt[0], pt[-1]) == (0.094, 11.14)
    assert d["read_lines_recorded_twice_compared"] == 0 and d["resumption_fault"] == []  # the resumption fell between read phases: no line recorded twice


CONTESTED_CI = {  # the contested reads pooled per size, encoder, gate and kind, with the exact interval as the row prints it (over the pairs whose original is not a repeated name)
    ("ten_thousand", "2024", "identity"): {"E1": (2500, 2500, 0.9985, 1.0), "E2": (2500, 2500, 0.9985, 1.0)}, ("ten_thousand", "7", "identity"): {"E1": (2500, 2500, 0.9985, 1.0), "E2": (2500, 2500, 0.9985, 1.0)},
    ("ten_thousand", "1337", "identity"): {"E1": (2500, 2500, 0.9985, 1.0), "E2": (2495, 2500, 0.9953, 0.9994)},
    ("ten_thousand", "2024", "surprise"): {"E1": (44, 2500, 0.0128, 0.0236), "E2": (412, 2500, 0.1505, 0.1799)}, ("ten_thousand", "7", "surprise"): {"E1": (41, 2500, 0.0118, 0.0222), "E2": (184, 2500, 0.0637, 0.0845)},
    ("ten_thousand", "1337", "surprise"): {"E1": (43, 2500, 0.0125, 0.0231), "E2": (546, 2500, 0.2023, 0.2351)},
    ("hundred_thousand", "2024", "identity"): {"E1": (4999, 4999, 0.9993, 1.0), "E2": (4994, 4998, 0.9980, 0.9998)}, ("hundred_thousand", "2024", "surprise"): {"E1": (75, 4999, 0.0118, 0.0188), "E2": (747, 4998, 0.1397, 0.1596)},
}


def test_the_contested_reads_and_their_exact_interval_as_the_row_prints_them(reading):
    from cortex_c2b.sealed_reading import clopper_pearson
    for (size, enc, gate), kinds in CONTESTED_CI.items():
        p = reading["pooled"].get(f"{size}/{gate}")
        if p is None or enc not in p["pairs_E1"]["per_encoder"]:
            pytest.skip(f"the encoder of seed {enc} under the {gate} rule at {size}: not retained yet")
        for kind, (m, n, lo, hi) in kinds.items():
            x = p[f"pairs_{kind}"]["per_encoder"][enc]
            assert (x["contested"], x["pairs"]) == (m, n)
            assert tuple(round(v, 4) for v in x["contested_clopper_pearson_95"]) == tuple(round(v, 4) for v in clopper_pearson(m, n)) == (lo, hi)


def test_the_keys_contract_reads_are_eight_on_the_thirty_two_files(reading):
    kc = {n: c["key_contract"]["n_sample_reads_of_cross_schema_repeated_names"] for n, c in reading["c2"].items()}
    assert sum(kc.values()) == 8 and {n for n, v in kc.items() if v} == {n for n in kc if "-50004-" in n or "-60000-" in n} and len([n for n in kc if "-50004-" in n]) == 6


def test_c3_the_evictions_without_a_pattern_match_c2_across_every_file(reading):
    for n, c in reading["cascades"].items():
        c2 = reading["c2"].get(n.replace("scale-cascade-", "scale-c2-"))
        assert c2 is not None
        under = sum(v["under_floor_0.2"] for v in c2["planting"]["by_kind"].values())
        assert 0 <= under - (c["applied_totals"]["evicted"] - c["applied_totals"]["withdrawn"]) <= c["admitted"] - 16 * c["phases"]


# the founder's progress files (never committed; read through a copy, verified by the final file's hash) when at hand: the environment variable
# QC_SCALE_PROGRESS_DIR names a folder of flat copies cascade-<draw>-<run id>-<gate>.progress.jsonl; the test skips without it
PROGRESS = {  # read lines (sample, pair); the sample reads over the read phases (n, own in window); the pair reads (n, originals not evicted, own in window);
              # read before eviction over every read line of the numbered read phases (E1, E2) and as the final file counts it, then with the reads of the last phase (`rbe_last`),
              # the partners read then evicted (`rbe_evicted`, 0 but at the larger size) and the live partners the last phase does not read (`live_unread`); the states at phase 1 and
              # at the last phase; the memory's patterns at most
    (50000, S2024, "identity"): dict(rbe_last=(1, 35), lines=(2398, 2493), sample=(2398, 2398), pairs=(2493, 1211, 1211), rbe=(0, 26), rbe_final=(0, 8), states_1=(26, 8, 0, 2), states_last=(4464, 95, 0, 6453), patterns_max=134),
    (50000, S2024, "surprise"): dict(rbe_last=(1, 35), lines=(2763, 2631), sample=(2763, 2763), pairs=(2631, 1400, 1400), rbe=(1, 29), rbe_final=(0, 11), states_1=(28, 8, 0, 0), states_last=(4939, 103, 0, 5010), patterns_max=134),
    (50001, S2024, "identity"): dict(rbe_last=(1, 41), lines=(2401, 2493), sample=(2401, 2401), pairs=(2493, 1143, 1143), rbe=(1, 28), rbe_final=(0, 11), states_1=(26, 8, 0, 2), states_last=(4476, 98, 0, 6438), patterns_max=136),
    (50001, S2024, "surprise"): dict(rbe_last=(1, 41), lines=(2754, 2629), sample=(2754, 2754), pairs=(2629, 1314, 1314), rbe=(1, 35), rbe_final=(0, 14), states_1=(28, 8, 0, 0), states_last=(4943, 100, 0, 5041), patterns_max=136),
    (50002, S2024, "identity"): dict(rbe_last=(1, 28), lines=(2302, 2493), sample=(2302, 2302), pairs=(2493, 1260, 1260), rbe=(1, 21), rbe_final=(0, 8), states_1=(27, 8, 0, 1), states_last=(4468, 98, 0, 6446), patterns_max=135),
    (50002, S2024, "surprise"): dict(rbe_last=(1, 28), lines=(2707, 2632), sample=(2707, 2707), pairs=(2632, 1438, 1438), rbe=(1, 22), rbe_final=(0, 8), states_1=(28, 8, 0, 0), states_last=(4948, 101, 0, 5003), patterns_max=135),
    (50003, S2024, "identity"): dict(rbe_last=(0, 36), lines=(2429, 2493), sample=(2429, 2429), pairs=(2493, 1164, 1164), rbe=(0, 28), rbe_final=(0, 8), states_1=(26, 8, 0, 2), states_last=(4480, 97, 0, 6435), patterns_max=133),
    (50003, S2024, "surprise"): dict(rbe_last=(0, 36), lines=(2809, 2629), sample=(2809, 2809), pairs=(2629, 1354, 1354), rbe=(0, 30), rbe_final=(0, 10), states_1=(28, 8, 0, 0), states_last=(4949, 101, 0, 5018), patterns_max=133),
    (50004, S2024, "identity"): dict(rbe_last=(1, 34), lines=(2335, 2493), sample=(2335, 2335), pairs=(2493, 1222, 1222), rbe=(1, 21), rbe_final=(0, 7), states_1=(26, 8, 0, 2), states_last=(4452, 99, 0, 6461), patterns_max=134),
    (50004, S2024, "surprise"): dict(rbe_last=(1, 34), lines=(2692, 2630), sample=(2692, 2692), pairs=(2630, 1389, 1389), rbe=(1, 23), rbe_final=(0, 8), states_1=(28, 8, 0, 0), states_last=(4930, 101, 0, 5053), patterns_max=134),
    (60000, S2024, "identity"): dict(rbe_last=(2, 292), rbe_evicted=(0, 5), live_unread=2, lines=(1804, 21920), sample=(1804, 1804), pairs=(21920, 8874, 8874), rbe=(1, 163), rbe_final=(0, 7), states_1=(26, 8, 0, 2), states_last=(37959, 76, 0, 71985), patterns_max=135),
}


@pytest.mark.parametrize("key", sorted(PROGRESS))
def test_c3_the_progress_file_read_apart_the_own_line_present_whenever_not_evicted(raw, key):
    pdir = os.environ.get("QC_SCALE_PROGRESS_DIR")
    if not pdir:
        pytest.skip("QC_SCALE_PROGRESS_DIR not set: the founder's progress files are not at hand")
    draw, run_id, gate = key
    final = present(raw, "cascade", draw, run_id, gate)
    path = Path(pdir) / f"cascade-{draw}-{run_id}-{gate}.progress.jsonl"
    if not path.exists():
        pytest.skip(f"{path.name} not at hand")
    e = PROGRESS[key]
    r = SR.read_progress(path, final)
    assert r["equal_to_the_final_files"] and r["faults"] == [] and r["read_faults"] == [] and r["torn_tail"] is None and r["last_session_closed"]
    assert (r["n_sample_lines"], r["n_pair_lines"]) == e["lines"] and r["n_sample_lines"] == final["n_read_lines"]     # the final file counts the sample lines alone
    S = sum(b["sample"]["n"] for b in r["reads_by_phase"].values()); SO = sum(b["sample"]["own_in_window"] for b in r["reads_by_phase"].values())
    P = sum(b["pairs"][k]["n"] for b in r["reads_by_phase"].values() for k in ("E1", "E2"))
    PN = sum(b["pairs"][k]["own_not_evicted"] for b in r["reads_by_phase"].values() for k in ("E1", "E2"))
    PO = sum(b["pairs"][k]["own_in_window"] for b in r["reads_by_phase"].values() for k in ("E1", "E2"))
    assert (S, SO) == e["sample"] and S == SO                                                   # every sample read (an entry not yet evicted) returns its own line
    assert (P, PN, PO) == e["pairs"] and PN == PO                                               # a pair's original is in the window exactly when it is not evicted
    for ph, b in r["reads_by_phase"].items():                                                   # the lines sum to the final file's summaries at every read phase
        f = final["reads_by_phase"][ph]
        assert (b["sample"]["n"] + b["pairs"]["E1"]["n"] + b["pairs"]["E2"]["n"], b["sample"]["own_in_window"] + b["pairs"]["E1"]["own_in_window"] + b["pairs"]["E2"]["own_in_window"]) == (f["n"], f["own_in_window"])
        assert b["sample"]["holes"] + b["pairs"]["E1"]["holes"] + b["pairs"]["E2"]["holes"] == f["holes_total"]
        assert all(k in ("live", "consolidated", "demoted") for k in b["sample"]["own_state"])
    rb = r["near_duplicates_read_before_eviction"]
    assert (rb["E1"]["read_before_eviction_sample_and_pair_reads"], rb["E2"]["read_before_eviction_sample_and_pair_reads"]) == e["rbe"]
    assert (rb["E1"]["read_before_eviction_in_the_final_file_sample_reads_only"], rb["E2"]["read_before_eviction_in_the_final_file_sample_reads_only"]) == e["rbe_final"]
    assert rb["E1"]["read_before_eviction_in_the_final_file_sample_reads_only"] == final["near_duplicates"]["by_kind"]["E1"]["read_before_eviction"]
    assert (rb["E1"]["read_before_eviction_with_the_last_phase"], rb["E2"]["read_before_eviction_with_the_last_phase"]) == e["rbe_last"]
    for k in ("E1", "E2"):                                                                      # the partners read: those still live at the end, and those evicted later
        assert rb[k]["read_live_at_the_end"] + rb[k]["read_then_evicted"] == rb[k]["read_before_eviction_sample_and_pair_reads"]
        assert rb[k]["read_live_at_the_end_with_the_last_phase"] + rb[k]["read_then_evicted_with_the_last_phase"] == rb[k]["read_before_eviction_with_the_last_phase"]
        assert rb[k]["read_then_evicted"] == rb[k]["read_then_evicted_with_the_last_phase"] == e.get("rbe_evicted", (0, 0))[k == "E2"]
    assert rb["E2"]["read_live_at_the_end_with_the_last_phase"] == final["near_duplicates"]["by_kind"]["E2"]["live"] - e.get("live_unread", 0)   # the last phase reads every pair: every live partner but two at the larger size
    pp = r["per_phase"]
    assert tuple(pp[1]["states"][s] for s in ("live", "consolidated", "demoted", "evicted")) == e["states_1"]
    assert tuple(pp[max(pp)]["states"][s] for s in ("live", "consolidated", "demoted", "evicted")) == e["states_last"] == tuple(final["states"][s] for s in ("live", "consolidated", "demoted", "evicted"))
    assert max(v["memory_patterns"] for v in pp.values()) == e["patterns_max"] and all(not v["fault"] for v in pp.values())
    assert len(pp) == final["phases"] and abs(sum(v["lifecycle_seconds"] for v in pp.values()) - sum(final["phase_times_seconds"])) < 1e-6
    # the entries evicted without a pattern are C.2's under floor writes less those written after the last phase (the tail's decisions)
    c2 = present(raw, "c2", draw, run_id, gate)
    under = sum(v["salience"]["under_floor_0.2"] for v in c2["planting"]["by_kind"].values())
    assert under - r["tail_under_floor_writes"] == final["applied_totals"]["evicted"] - final["applied_totals"]["withdrawn"]


def test_the_expected_set_is_complete():
    f = SR.load_files(M)
    assert f["missing"] == [] and len(f["present"]) == 79 and f["outside_the_expected_set"] == []   # part C complete on 2026-11-05 (the founder's commit e4932eb)


# ---- the provenance file and the founder's reading file -----------------------------------------------------------------------
def test_the_provenance_file_pins_every_file_log_and_progress_hash_and_the_reading(reading, raw):
    prov = json.loads((M / "PROVENANCE-scale-probe.json").read_text(encoding="utf-8"))
    assert prov["rows"] == "row 53" and prov["set"] == {"expected": 79, "present": 79, "missing": []} and len(prov["files"]) == 79 and len(prov["logs"]) == 48
    for name, f in prov["files"].items():
        assert f["sha256_lf"] == sha_lf(M / name)
        if f["console_sha256"] is not None:
            assert f["console_sha256"] == f["sha256_lf"]
        assert f["freeze_bound"] and f["PYTHONHASHSEED"] == "0" and f["scale_probe_py"] == SCALE_PROBE_SHA and f["code_commit"] in PART_C_COMMITS
        assert f["torch_threads"] == (8 if f["kind"] == "encoding" else 1)
    assert sum(1 for f in prov["files"].values() if f["code_commit_at_writing"] not in (None, f["code_commit"])) == 1          # scale-c2-50001-2e1dc71913b1-identity.json
    assert all((f["code_commit_at_writing"] is None) == (f["kind"] == "encoding") for f in prov["files"].values())   # the encoding files carry no code_commit field
    assert prov["files"]["scale-c2-50001-2e1dc71913b1-identity.json"]["commit_of_every_session"] == ["2d59e5c"]
    assert prov["files"]["scale-cascade-60000-4cfd6f6e1406-identity.json"]["reasons"] == ["budget", "complete"]
    # the hundred thousand draw's C.2 progress files (the plantings' and the reads'), hashed against the final files, and the gaps between their recorded points:
    # the machine's sleep of 2026-11-04 lies inside the surprise reads, in two gaps (the committee of 2026-11-05)
    for gate, big in (("identity", []), ("surprise", [(10803.0, 7123, 7124, "2026-11-04T07:34:32+00:00", "2026-11-04T10:34:35+00:00"), (5887.0, 7135, 7136, "2026-11-04T10:34:50+00:00", "2026-11-04T12:12:57+00:00")])):
        pf = prov["progress_files"][f"scale-c2-60000-4cfd6f6e1406-{gate}.json"]
        assert pf["hash_equal_here"] is True and pf["planting_hash_equal_here"] is True
        g = pf["wall_clock_gaps"]
        assert (g["planting"]["recorded_points"], g["planting"]["gaps_over_60_s"], g["reads"]["recorded_points"]) == (1719, [], 13000) and g["planting"]["largest_gap_seconds"] <= 11.0
        assert [(x["seconds"], x["from_point"], x["to_point"], x["from"], x["to"]) for x in g["reads"]["gaps_over_60_s"]] == big
        assert g["reads"]["largest_gap_seconds"] == (1.0 if gate == "identity" else 10803.0)
    assert len(prov["notes"]) == 4
    read_here = {n: v["read_here"] for n, v in prov["progress_files"].items() if "read_here" in v}            # the eleven cascade progress files read through copies at the graving: seed 2024's ten, the hundred thousand draw's
    assert len(read_here) == 31 and {n for n, v in read_here.items() if v is True} == {n for n in read_here if "-4cfd6f6e1406-" in n} and all(v in (True, None) for v in read_here.values())
    for name, lg in prov["logs"].items():
        assert lg["sha256_lf"] == sha_lf(M / "logs" / name) and lg["retained"]
    assert prov["code"]["scale_probe_py_sha256_lf"] == [SCALE_PROBE_SHA] and sorted(prov["code"]["commit_of_every_session"]) == sorted(PART_C_COMMITS)
    assert prov["code"]["scale_reading_py_sha256_lf"] == sha_lf(ROOT / "cortex_c2b" / "scale_reading.py")
    assert prov["freeze"]["freeze_sha256"] == FREEZE_SHA256 and prov["freeze"]["frozen_tree_sha256"] == FROZEN_TREE
    sealed = json.loads((M / "PROVENANCE-sealed-set.json").read_text(encoding="utf-8"))
    for run_id, c in prov["checkpoints"].items():
        assert c["as_in_PROVENANCE_sealed_set"] == sealed["checkpoints"][run_id] and c["sha256"] == raw[("encoding", 50000, run_id, None)]["identity"]["compared"]["checkpoint_sha256"]
    for name, p in prov["progress_files"].items():
        assert p["progress_sha256"] == raw[(prov["files"][name]["kind"], prov["files"][name]["draw"], prov["files"][name]["run_id"], prov["files"][name]["gate"])]["progress_sha256"]
    assert all(p["read_here"] for n, p in prov["progress_files"].items() if prov["files"][n]["kind"] == "cascade" and (prov["files"][n]["run_id"] == S2024))   # the ten of seed 2024 and the hundred thousand draw's
    lines = lambda r: [l for l in r if not l.startswith("checks:")]          # the checks line counts the progress files at hand, read or not
    assert lines(prov["reading"]["recomputed_here"]["summary_lines"]) == lines(reading["summary_lines"])
    assert close(prov["row53"]["pooled"], reading["pooled"]) and close(prov["row53"]["c4"], reading["c4"]) and prov["row53"]["checks"]["progress_files_read"] == 11
    assert len(prov["notes"]) == 4


def test_the_founders_reading_file_equals_the_recomputation_when_retained(reading):
    p = M / SR.READING_NAME
    if not p.exists():
        pytest.skip(f"{SR.READING_NAME} not retained yet (the founder writes it from the module at the graving's commit)")
    fr = json.loads(p.read_text(encoding="utf-8"))
    lines = lambda r: [l for l in r if not l.startswith("checks:")]
    assert lines(fr["summary_lines"]) == lines(reading["summary_lines"]) and close(fr["pooled"], reading["pooled"]) and close(fr["c4"], reading["c4"]) and fr["checks"]["freeze_bound_on_every_file"]
    assert fr["files"]["present"] == 79 and fr["files"]["missing"] == []
