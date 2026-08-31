import asyncio, sys, os, time
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
print('proxy envs:', {k: v for k, v in os.environ.items() if 'proxy' in k.lower()}, flush=True)
import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_code_cell("import sys; print('ok', sys.version_info[:3])")]
nb.metadata = {'kernelspec': {'display_name': 'x', 'language': 'python', 'name': 'solar-venv'},
               'language_info': {'name': 'python'}}
t0 = time.time()
try:
    client = NotebookClient(nb, timeout=120, startup_timeout=90, kernel_name='solar-venv')
    client.execute()
    print('EXECUTED in', round(time.time() - t0, 1), 's')
    print(nb.cells[0].outputs)
except Exception as e:
    print('FAILED after', round(time.time() - t0, 1), 's:', type(e).__name__, e)
