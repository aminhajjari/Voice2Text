import os
import time
import subprocess

max_wait_minutes = 30
check_interval = 60  # seconds

start = time.time()
for iteration in range(int(max_wait_minutes * 60 / check_interval)):
    elapsed = time.time() - start
    elapsed_min = int(elapsed // 60)
    elapsed_sec = int(elapsed % 60)
    
    # Check for output JSON files
    json_files = [f for f in os.listdir('output') if f.startswith('diagnostic_test2_') and f.endswith('.json') and 'comparison' not in f]
    comparison_file = [f for f in os.listdir('output') if f == 'diagnostic_test2_comparison.json']
    
    # Check running Python processes
    p = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/NH'], capture_output=True, text=True)
    py_count = sum(1 for l in p.stdout.split('\n') if 'python' in l.lower())
    
    status = {
        'elapsed': f'{elapsed_min:02d}:{elapsed_sec:02d}',
        'python_processes': py_count,
        'json_files': json_files,
        'has_comparison': len(comparison_file) > 0,
    }
    
    # Write status to file
    with open('output/_wait_status.txt', 'w') as f:
        f.write(f"Elapsed: {elapsed_min:02d}:{elapsed_sec:02d}\n")
        f.write(f"Python processes: {py_count}\n")
        f.write(f"JSON result files: {json_files}\n")
        f.write(f"Has comparison: {len(comparison_file) > 0}\n")
        
        if json_files:
            for jf in json_files:
                size = os.path.getsize(f'output/{jf}')
                f.write(f"  {jf}: {size} bytes\n")
        
        # Check log file
        if os.path.exists('output/diagnostic_test2_run.log'):
            log_size = os.path.getsize('output/diagnostic_test2_run.log')
            f.write(f"Log file: {log_size} bytes\n")
            if log_size > 0:
                with open('output/diagnostic_test2_run.log', 'r', encoding='utf-8') as lf:
                    lines = lf.readlines()
                f.write(f"Log lines: {len(lines)}\n")
                for line in lines[-5:]:
                    f.write(f"  {line.rstrip()}\n")
    
    print(f"[{elapsed_min:02d}:{elapsed_sec:02d}] Python procs: {py_count}, JSON files: {json_files}")
    
    # If we have both result files and comparison, we're done
    if len(json_files) >= 2 and comparison_file:
        print("All result files found!")
        break
    
    # If python process count dropped significantly (diagnostic script finished)
    if py_count < 3 and json_files:
        print("Process finished, results found")
        break
    
    time.sleep(check_interval)

print("Monitoring complete")