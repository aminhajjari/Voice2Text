import os, json, subprocess, time

# Check running Python processes
p = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/NH'], capture_output=True, text=True)
py_count = sum(1 for l in p.stdout.split('\n') if 'python' in l.lower())
print(f"Running Python processes: {py_count}")

# Check diagnostic output files
files = sorted([f for f in os.listdir('output') if 'diagnostic' in f])
print(f"Diagnostic output files: {files}")

# Check sizes
for f in files:
    size = os.path.getsize(f'output/{f}')
    print(f"  {f}: {size} bytes")

# Check log file contents
log_path = 'output/diagnostic_test2_run.log'
if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
    with open(log_path, 'r', encoding='utf-8') as fh:
        lines = fh.readlines()
    print(f"\nLog file last {min(20, len(lines))} lines:")
    for line in lines[-20:]:
        print(line.rstrip())