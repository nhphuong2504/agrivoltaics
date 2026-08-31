import asyncio, sys, time, threading
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from jupyter_client.manager import KernelManager

km = KernelManager(kernel_name='solar-venv')
t0 = time.time()
km.start_kernel()
print('kernel starting...', flush=True)
proc = km.provisioner.process if hasattr(km, 'provisioner') else None

def watch():
    time.sleep(20)
    print(f'[t+20s] has_kernel={km.has_kernel}', flush=True)

threading.Thread(target=watch, daemon=True).start()

try:
    kc = km.client()
    kc.start_channels()
    kc.wait_for_ready(timeout=120)
    print('READY in', round(time.time() - t0, 1), 's', flush=True)
    msg_id = kc.execute("print('hello from kernel'); import pandas; print(pandas.__version__)")
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            msg = kc.get_iopub_msg(timeout=5)
        except Exception:
            continue
        if msg['parent_header'].get('msg_id') != msg_id:
            continue
        if msg['msg_type'] == 'stream':
            print('STREAM:', msg['content']['text'].strip(), flush=True)
        if msg['msg_type'] == 'status' and msg['content']['execution_state'] == 'idle':
            break
    kc.stop_channels()
except Exception as e:
    print('FAILED after', round(time.time() - t0, 1), 's:', type(e).__name__, e, flush=True)
finally:
    try:
        km.shutdown_kernel(now=True)
    except Exception:
        pass
