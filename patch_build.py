from pathlib import Path

f = Path("data/build_test_sets.py")
txt = f.read_text()

old = '        Path(r"D:\\New folder (2)\\Data\\OneDrive_2025-12-13\\Datset from Q.lin"),\n'
if old in txt:
    f.write_text(txt.replace(old, ""))
    print("Removed dead OneDrive fallback path")
else:
    print("Line not found - already clean or different formatting")
