import subprocess
import time
import csv
import sys

# --- Data Configuration ---
# SM frequency: from 2100 to 300 in steps of 150, plus minimum 210
SM_CLOCKS = list(range(2100, 299, -150)) + [210]
# Memory frequency: key levels
MEM_CLOCKS = [810, 5001, 9251, 9501]
OUTPUT_FILE = "3080Ti_power_simple.csv"

def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return res.stdout.strip()
    except:
        return ""

def get_gpu_status():
    # Only capture: actual core frequency, actual memory frequency, real-time power
    raw = run_cmd("nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw --format=csv,noheader,nounits")
    if raw and "not a valid" not in raw:
        return [x.strip() for x in raw.split(',')]
    return ["0", "0", "0"]

def main():
    print(f"--- 3080Ti Power Test (Simple Mode) ---")
    run_cmd("sudo nvidia-smi -pm 1")

    total = len(MEM_CLOCKS) * len(SM_CLOCKS)
    count = 0

    # buffering=1 enables line buffering
    with open(OUTPUT_FILE, 'w', newline='', buffering=1) as f:
        writer = csv.writer(f)
        # Only record actual physical values
        writer.writerow(["Actual_SM", "Actual_Mem", "Power_W"])

        try:
            for m_clk in MEM_CLOCKS:
                run_cmd(f"sudo nvidia-smi -lmc {m_clk}")
                time.sleep(2)

                for s_clk in SM_CLOCKS:
                    count += 1
                    run_cmd(f"sudo nvidia-smi -lgc {s_clk}")

                    sys.stdout.write(f"\rProgress: {count}/{total} | Target SM: {s_clk}MHz | Memory: {m_clk}MHz")
                    sys.stdout.flush()

                    # Wait for physical frequency change to stabilize
                    time.sleep(1.5)

                    # Sample 3 times and average for better accuracy
                    samples = []
                    for _ in range(3):
                        status = get_gpu_status()
                        samples.append([float(x) for x in status])
                        time.sleep(0.3)

                    avg_data = [round(sum(col)/len(col), 2) for col in zip(*samples)]

                    # Write and flush to disk immediately
                    writer.writerow(avg_data)
                    f.flush()

        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        finally:
            run_cmd("sudo nvidia-smi -rgc")
            run_cmd("sudo nvidia-smi -rmc")
            print(f"\nTest completed! File saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()