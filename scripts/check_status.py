import json
with open("session_state.json") as f:
    s = json.load(f)
for mid in s.get("queue", []):
    m = s["models"].get(mid, {})
    st = m.get("status", "?")
    ep = m.get("current_epoch", 0)
    ncc = m.get("best_ncc", 0)
    print(f"  {mid:30s}  {st:8s}  ep={ep:3d}  ncc={ncc:.4f}")
