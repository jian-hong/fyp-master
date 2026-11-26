from pathlib import Path

f = Path("evaluation/benchmark_runner.py")
txt = f.read_text(encoding="utf-8", errors="replace")

old = "model.load_state_dict(_load_checkpoint_state(ckpt))"
new = "model.load_state_dict(_load_checkpoint_state(ckpt), strict=False)"

if old in txt:
    f.write_text(txt.replace(old, new), encoding="utf-8")
    print("Patched OK")
else:
    for i, l in enumerate(txt.splitlines()):
        if "load_state_dict" in l:
            print(f"  {i+1}: {l}")
