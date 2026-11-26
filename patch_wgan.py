from pathlib import Path

f = Path("data/wgan/wgan_noise.py")
txt = f.read_text()

old = r'Path(r"D:\New folder (2)\Data\OneDrive_2025-12-13\Datset from Q.lin")'
new = r'Path(r"D:\New folder (2)\Data\FYP_MASTER\Datset from Q.lin")'

if old in txt:
    f.write_text(txt.replace(old, new))
    print("Patched OK")
else:
    print("Not found - printing lines 70-85:")
    for i, l in enumerate(txt.splitlines()[69:85], 70):
        print(f"{i}: {l}")
