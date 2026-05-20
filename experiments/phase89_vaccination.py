# -*- coding: utf-8 -*-
"""
Phase 89: Immune Memory (Vaccination)
First exposure: slow (recompile). Second exposure: instant (memory recall).
Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def compile_prog(model, tok, train, layer, device, seed=42, epochs=100):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device)*0.01; vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for ep in range(epochs):
        for p, t in train:
            tid = tok.encode(t)[-1]; inp = tok(p, return_tensors='pt').to(device)
            def inj(m,i,o,v=vec): return replace_last_token(o,v)
            h = model.model.layers[layer].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()

def evaluate_vec(model, tok, vec, data, layer, device):
    c = 0
    for p, e in data:
        def inj(m,i,o,v=vec): return replace_last_token(o,v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        h.remove()
        if tok.decode(out.logits[0,-1,:].argmax().item()).strip() == e: c += 1
    return c / len(data)

class ImmuneMemory:
    def __init__(self, threshold=0.85):
        self.antibodies = []; self.threshold = threshold
    def recognize(self, virus_vec):
        if not self.antibodies: return None, -1
        v = virus_vec.cpu().numpy().flatten()
        best_sim, best_idx = -1, -1
        for i, (sig, _, _) in enumerate(self.antibodies):
            sim = np.dot(v, sig)/(np.linalg.norm(v)*np.linalg.norm(sig)+1e-8)
            if sim > best_sim: best_sim, best_idx = sim, i
        if best_sim >= self.threshold: return self.antibodies[best_idx], best_sim
        return None, best_sim
    def store(self, virus_vec, clean_vec, ts):
        self.antibodies.append((virus_vec.cpu().numpy().flatten(), clean_vec.cpu().numpy().flatten(), ts))
    def neutralize(self, virus_vec, svd_basis):
        match, sim = self.recognize(virus_vec)
        if match is not None:
            return torch.tensor(match[1], device=virus_vec.device, dtype=torch.float32), 'memory', sim
        v = virus_vec.cpu().numpy().flatten()
        f = (v @ svd_basis.T) @ svd_basis
        return torch.tensor(f, device=virus_vec.device, dtype=torch.float32), 'innate', sim

def main():
    print("[P89] Immune Memory (Vaccination)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hs = model.config.hidden_size; tl = 8
    for p in model.parameters(): p.requires_grad = False
    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),("4, 6) =","4"),("9, 3) =","3")]
    test_data = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),("5, 4) =","4"),("3, 8) =","3")]
    all_data = min_data + test_data
    # SVD basis
    variants = []
    for s in range(10):
        v = compile_prog(model, tok, min_data, tl, DEVICE, seed=s*100, epochs=80)
        variants.append(v.cpu().numpy().flatten())
    Vk = np.linalg.svd(np.array(variants), full_matrices=False)[2][:10,:]
    clean = compile_prog(model, tok, min_data, tl, DEVICE, seed=42)
    clean_acc = evaluate_vec(model, tok, clean, all_data, tl, DEVICE)
    print(f"  Clean: {clean_acc:.0%}")
    memory = ImmuneMemory(threshold=0.85)
    # 1st exposure
    print("  1st exposure...")
    strains = []; first_res = []
    for sid in range(5):
        np.random.seed(sid*1000)
        n = np.random.randn(hs).astype(np.float32); n = n/np.linalg.norm(n)*5
        vv = torch.tensor(clean.cpu().numpy().flatten()+n, device=DEVICE, dtype=torch.float32)
        strains.append(vv.clone())
        t0 = time.time(); nv, method, sim = memory.neutralize(vv, Vk); t1 = time.time()
        acc = evaluate_vec(model, tok, nv, all_data, tl, DEVICE)
        t2 = time.time()
        rec = compile_prog(model, tok, min_data, tl, DEVICE, seed=42+sid)
        t3 = time.time()
        memory.store(vv, rec, time.time())
        first_res.append({'strain':sid,'method':method,'acc':round(acc,4),'resp_time':round(t1-t0,4),'recomp_time':round(t3-t2,2)})
        print(f"    S{sid}: {method}, acc={acc:.0%}, recomp={t3-t2:.1f}s")
    # 2nd exposure
    print("  2nd exposure (same strains)...")
    second_res = []
    for sid in range(5):
        t0 = time.time(); nv, method, sim = memory.neutralize(strains[sid], Vk); t1 = time.time()
        acc = evaluate_vec(model, tok, nv, all_data, tl, DEVICE)
        second_res.append({'strain':sid,'method':method,'acc':round(acc,4),'resp_time':round(t1-t0,4),'sim':round(float(sim),4)})
        print(f"    S{sid}: {method}, acc={acc:.0%}, time={t1-t0:.4f}s")
    # Mutants
    print("  Mutant strains...")
    mut_res = []
    for sid in range(5):
        np.random.seed(sid*1000+1)
        n = np.random.randn(hs).astype(np.float32)*5; m = np.random.randn(hs).astype(np.float32)*0.5
        mn = 0.8*n+0.2*m; mn = mn/np.linalg.norm(mn)*5
        vv = torch.tensor(clean.cpu().numpy().flatten()+mn, device=DEVICE, dtype=torch.float32)
        t0 = time.time(); nv, method, sim = memory.neutralize(vv, Vk); t1 = time.time()
        acc = evaluate_vec(model, tok, nv, all_data, tl, DEVICE)
        mut_res.append({'strain':sid,'method':method,'acc':round(acc,4),'sim':round(float(sim),4)})
        print(f"    M{sid}: {method}, acc={acc:.0%}, sim={sim:.3f}")
    # Summary
    t_first = [r['recomp_time'] for r in first_res]
    t_second = [r['resp_time'] for r in second_res]
    speedup = np.mean(t_first)/(np.mean(t_second)+1e-8)
    output = {'phase':89,'name':'immune_memory_vaccination','clean_accuracy':round(clean_acc,4),
              'n_antibodies':len(memory.antibodies),'first_exposure':first_res,
              'second_exposure':second_res,'mutant_exposure':mut_res,
              'speedup':round(float(speedup),1),'elapsed':round(time.time()-start,1)}
    with open(os.path.join(RESULTS_DIR,'phase89_vaccination.json'),'w') as f: json.dump(output,f,indent=2)
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    x = np.arange(5); w = 0.25
    axes[0].bar(x-w,[r['acc'] for r in first_res],w,label='1st (innate)',color='tab:orange',edgecolor='black')
    axes[0].bar(x,[r['acc'] for r in second_res],w,label='2nd (memory)',color='tab:green',edgecolor='black')
    axes[0].bar(x+w,[r['acc'] for r in mut_res],w,label='Mutant',color='tab:purple',edgecolor='black')
    axes[0].set_xlabel('Strain'); axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Immune Response Accuracy',fontweight='bold'); axes[0].legend(fontsize=8); axes[0].set_ylim(0,1.1)
    axes[1].bar(['1st\n(recompile)'],[np.mean(t_first)],color='tab:orange',edgecolor='black')
    axes[1].bar(['2nd\n(memory)'],[np.mean(t_second)],color='tab:green',edgecolor='black')
    axes[1].set_ylabel('Time (s)'); axes[1].set_title(f'Speedup: {speedup:.0f}x',fontweight='bold')
    mem_ct = sum(1 for r in second_res if r['method']=='memory')
    inn_ct = 5 - mem_ct
    axes[2].pie([inn_ct+5, mem_ct], labels=['Innate','Memory'], colors=['tab:orange','tab:green'],
                autopct='%1.0f%%', textprops={'fontweight':'bold'})
    axes[2].set_title('Defense Methods', fontweight='bold')
    plt.suptitle('Phase 89: Immune Memory (Vaccination)\n"The body remembers every battle"',fontsize=13,fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR,'phase89_vaccination.png'),dpi=150,bbox_inches='tight'); plt.close()
    print(f"\n  Speedup: {speedup:.0f}x")
    print(f"  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
