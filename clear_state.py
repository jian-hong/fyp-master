import json
from pathlib import Path

f = Path("session_state.json")
state = json.loads(f.read_text())

# Remove LossB from every completed/done/skipped list
changed = False
for key in list(state.keys()):
    if isinstance(state[key], dict) and state[key].get("status") == "done":
        if key == "MR-TAE-FULL-LossB":
            state[key]["status"] = "pending"
            changed = True
            print(f"Reset {key} from done -> pending")
    if isinstance(state.get("completed"), list) and "MR-TAE-FULL-LossB" in state["completed"]:
        state["completed"].remove("MR-TAE-FULL-LossB")
        changed = True
        print("Removed from completed list")

if changed:
    f.write_text(json.dumps(state, indent=2))
    print("session_state.json updated")
else:
    print("Nothing changed - printing all keys:")
    for k, v in state.items():
        print(f"  {k}: {type(v).__name__} = {str(v)[:80]}")
