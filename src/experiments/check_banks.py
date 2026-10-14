import numpy as np, os
d = r"D:\programme file\ResearchArticle\results\perception"
for f in ["memory_bank_80.npy", r"banks\金属螺母(metal_nut).npy", r"banks\钢带(NEU-DET).npy"]:
    p = os.path.join(d, f)
    if os.path.exists(p):
        a = np.load(p)
        print(f, a.shape)
    else:
        print(f, "NOT FOUND")
