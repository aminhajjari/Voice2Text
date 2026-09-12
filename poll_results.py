import os
import sys
import time
import json

def check():
    out_dir = 'output'
    results = {
        'elapsed': time.strftime('%H:%M:%S'),
        'files': {},
        'any_new': False
    }
    
    for f in ['ab_test_A_True.json', 'ab_test_B_False.json', 'ab_test_comparison.json']:
        path = os.path.join(out_dir, f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            mtime = time.ctime(os.path.getmtime(path))
            results['files'][f] = {'exists': True, 'size': size, 'mtime': mtime}
            results['any_new'] = True
        else:
            results['files'][f] = {'exists': False}
    
    # Also check log
    log_path = os.path.join(out_dir, 'ab_test_log.txt')
    if os.path.exists(log_path):
        size = os.path.getsize(log_path)
        results['files']['ab_test_log.txt'] = {'exists': True, 'size': size, 'mtime': time.ctime(os.path.getmtime(log_path))}
        results['any_new'] = True
        if size > 0:
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            results['log_lines'] = len(lines)
            results['log_tail'] = lines[-10:] if len(lines) >= 10 else lines
    
    return results

print("Waiting for results...")
for attempt in range(120):
    r = check()
    out_path = os.path.join('output', '_poll_result.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    
    if r.get('files', {}).get('ab_test_comparison.json', {}).get('exists'):
        print(f"COMPARISON FILE FOUND at attempt {attempt}")
        break
    
    if r.get('files', {}).get('ab_test_log.txt', {}).get('exists') and r['files']['ab_test_log.txt']['size'] > 0:
        log = r.get('log_tail', [])
        if any('Model loaded' in l for l in log):
            print(f"  Model loaded at attempt {attempt}, waiting for transcription...")
    
    if attempt % 5 == 0:
        print(f"  Attempt {attempt}: {', '.join(k for k,v in r['files'].items() if v.get('exists'))}")
    
    time.sleep(10)

print("Done polling")
