from pathlib import Path
f = Path("config/training_config.yaml")
txt = f.read_text(encoding="utf-8")
txt = txt.replace("num_workers: 4", "num_workers: 0")
f.write_text(txt, encoding="utf-8")
print("Fixed: num_workers set to 0")
