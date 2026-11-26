# Metric Reconciliation: NCC Discrepancy

**Thesis Reported NCC:** `0.5145`
**Dashboard Reported NCC:** `~0.9480`

**Explanation:**
The discrepancy arises from two major differences in evaluation protocol: dataset difficulty and sliding window boundary logic. The initial thesis reported **global NCC** calculated over a significantly harder, composite evaluation set (which included mixed out-of-band noise not present in the isolated training bins) and correlated across the entire $N=2048$ window, artificially collapsing the metric due to flat-noise correlation. The dashboard currently reports **pulse-region NCC** restricted solely to synthetic tests over the mask regions where pulses actually exist, which is a mathematically denser correlation. Moving forward, both `ncc_pulse` and `ncc_global` will be explicitly reported universally across synthetic and real-world test sets to prevent ambiguity during examination.
