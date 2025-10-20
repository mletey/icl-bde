#!/usr/bin/env python3
import os
import yaml
import subprocess
import itertools

# ---------- helpers ----------
def sanitize(val):
    """Make values filename/job-name safe: 0.5 -> 0p5, True/False -> true/false."""
    if isinstance(val, float):
        return str(val).replace(".", "p")
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val)

def build_varcontext_flag(v):
    """True -> --variablecontext ; False -> --no-variablecontext"""
    return "--variablecontext" if v else "--no-variablecontext"

# ---------- load config ----------
with open("jobs.yaml", "r") as f:
    cfg = yaml.safe_load(f)

d = cfg["defaults"]
grid = cfg["grid"]
extra_flags = cfg.get("extra_flags", [])

taus = grid["tau"]
alphas = grid["alpha"]
vcs = grid["variablecontext"]

# ---------- directories ----------
log_dir = "log_files"
sbatch_dir = "sbatch_scripts"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(sbatch_dir, exist_ok=True)

# ---------- generate + submit ----------
for tau, alpha, vc in itertools.product(taus, alphas, vcs):
    rn = f"tau{sanitize(tau)}_alpha{sanitize(alpha)}_vc{sanitize(vc)}"
    job_name = f"{d['job_name']}_{rn}"

    # Resolve AUTO placeholder in extra flags into run_name
    resolved_extras = [flag.replace("AUTO", rn) for flag in extra_flags]

    flags = [
        f"--tau {tau}",
        f"--alpha {alpha}",
        build_varcontext_flag(vc),
    ]

    full_cmd = " ".join([d["base_cmd"], *resolved_extras, *flags])

    sbatch_script = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --account={d['account']}
#SBATCH -p {d['partition']}
#SBATCH --gres={d['gres']}
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t {d['time']}
#SBATCH --cpus-per-gpu={d['cpus_per_gpu']}
#SBATCH --mem-per-gpu={d['mem_per_gpu']}
#SBATCH -o {os.path.join(log_dir, job_name)}.out
#SBATCH -e {os.path.join(log_dir, job_name)}.err
#SBATCH --mail-type=END
#SBATCH --mail-user={d['email']}

module load {d.get('cuda_module', 'cuda/12.2')}
source ~/.bashrc
conda activate {d['conda_env']}

# ---- Weights & Biases: force a descriptive run name (and stable id) ----
export WANDB_NAME="{rn}"
export WANDB_RUN_ID="{rn}"   # optional but useful for stable resuming
# export WANDB_PROJECT="linear-attn-demo"  # optional if you prefer env var over --project

{full_cmd}
"""

    sbatch_path = os.path.join(sbatch_dir, f"{job_name}.sh")
    with open(sbatch_path, "w") as f:
        f.write(sbatch_script)

    subprocess.run(["sbatch", sbatch_path], check=False)