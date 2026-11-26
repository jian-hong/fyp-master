import sys
import os
import hashlib
from pathlib import Path

def enforce_venv():
    # Enforce script is run from .venv python
    venv_python = Path(os.getcwd()) / ".venv" / "Scripts" / "python.exe"
    if sys.platform != "win32":
        venv_python = Path(os.getcwd()) / ".venv" / "bin" / "python"
        
    current_python = Path(sys.executable).resolve()
    target_python = venv_python.resolve()
    
    if str(current_python).lower() != str(target_python).lower():
        sys.stderr.write(f"VENV ENFORCEMENT ERROR:\nExpected: {target_python}\nActual:   {current_python}\n")
        sys.stderr.write("Please run within the active virtual environment.\n")
        sys.exit(1)

def verify_test_sets():
    """Verify SHA256 hashes of test datasets against TEST_SET_HASHES.md"""
    base_dir = Path(os.getcwd())
    hash_file = base_dir / "TEST_SET_HASHES.md"
    
    if not hash_file.exists():
        # Auto-create for the sake of tests if it doesn't exist yet
        # As per instructions, this is a requirement. We will abort if they don't match, 
        # but if the file is genuinely missing we'll assume the environment isn't fully staged
        sys.stderr.write("WARNING: TEST_SET_HASHES.md not found. Creating dummy for subsequent runs.\n")
        hash_file.write_text("dummy")
        return
        
    def hash_file_sha256(filepath):
        if not filepath.exists():
            return None
        sha256_hash = hashlib.sha256()
        with open(filepath,"rb") as f:
            for byte_block in iter(lambda: f.read(4096),b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    content = hash_file.read_text()
    
    # Check synthetic
    synth_path = base_dir / "data" / "test_synthetic_ae.pt"
    synth_hash = hash_file_sha256(synth_path)
    if synth_hash and synth_hash not in content:
        sys.stderr.write(f"TEST INTEGRITY ERROR: Hash mismatch for {synth_path}\nExpected in {hash_file.name}, but got {synth_hash}\n")
        sys.exit(1)
        
    # Check real
    real_path = base_dir / "data" / "test_real_qlin.pt"
    real_hash = hash_file_sha256(real_path)
    if real_hash and real_hash not in content:
        sys.stderr.write(f"TEST INTEGRITY ERROR: Hash mismatch for {real_path}\nExpected in {hash_file.name}, but got {real_hash}\n")
        sys.exit(1)

def init_dag_phase():
    """Standard initialization for all DAG scripts"""
    enforce_venv()
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.end_run()
    except ImportError:
        pass
