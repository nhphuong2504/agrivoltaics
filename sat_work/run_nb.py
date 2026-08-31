"""Execute saturation_detection.ipynb in-place with the project venv."""
import asyncio, sys
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import nbformat
from nbclient import NotebookClient

p = r'c:\Users\nhphuong\Desktop\Solar\all_data\saturation_detection.ipynb'
nb = nbformat.read(p, as_version=4)
client = NotebookClient(nb, timeout=600, startup_timeout=180, kernel_name='solar-venv',
                        resources={'metadata': {'path': r'c:\Users\nhphuong\Desktop\Solar\all_data'}})
client.execute()
nbformat.write(nb, p)
print('executed OK')
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        for out in cell.get('outputs', []):
            if out.get('output_type') == 'error':
                print(f'ERROR in cell {i}:', out.get('ename'), out.get('evalue'))
