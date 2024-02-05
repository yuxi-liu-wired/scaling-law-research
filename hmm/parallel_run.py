import subprocess
from concurrent.futures import ThreadPoolExecutor

def run_script(index):
    print(f"Starting run {index}")
    subprocess.run(["python", "run.py"])
    print(f"Finished run {index}")

n_runs = 5

with ThreadPoolExecutor(max_workers=n_runs) as executor:
    # Submit tasks to the executor
    futures = [executor.submit(run_script, i) for i in range(n_runs)]
    # Wait for all futures to complete
    for future in futures:
        future.result()

print("All runs completed.")
