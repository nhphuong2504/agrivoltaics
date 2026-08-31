import json, subprocess, sys, tempfile, time, os

conn = {
    "shell_port": 55231, "iopub_port": 55232, "stdin_port": 55233,
    "control_port": 55234, "hb_port": 55235,
    "ip": "127.0.0.1", "key": "testkey", "transport": "tcp",
    "signature_scheme": "hmac-sha256", "kernel_name": "solar-venv",
}
fd, path = tempfile.mkstemp(suffix='.json')
os.write(fd, json.dumps(conn).encode()); os.close(fd)

py = r'C:\Users\nhphuong\Desktop\Solar\all_data\venv\Scripts\python.exe'
p = subprocess.Popen([py, '-Xfrozen_modules=off', '-m', 'ipykernel_launcher', '-f', path],
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(15)
rc = p.poll()
print('after 15s, returncode =', rc, '(None = still alive)')
if rc is not None:
    out, err = p.communicate()
    print('STDOUT:', out.decode(errors='replace')[:3000])
    print('STDERR:', err.decode(errors='replace')[:3000])
else:
    p.kill()
    print('kernel process is alive and waiting for connections')
os.remove(path)
