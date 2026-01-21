import subprocess

def get_best_gpu_cc():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        text=True
    )
    gpus = []
    for line in out.strip().split("\n"):
        name, cc = line.split(",")
        gpus.append({"name": name.strip(), "cc": float(cc.strip())})
    # выбираем GPU с максимальной compute capability
    best = max(gpus, key=lambda x: x["cc"])
    return best

def pick_torch_wheel(cc: float):
    if cc >= 8.0:
        return "cu128"
    elif cc >= 7.0:
        return "cu126"
    else:
        return "cu126"  # старые карты только FP32

if __name__ == "__main__":
    best_gpu = get_best_gpu_cc()
    wheel = pick_torch_wheel(best_gpu["cc"])
    print("Best GPU:", best_gpu["name"], "CC:", best_gpu["cc"])
    print("Suggested torch wheel:", wheel)
