import nbformat, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

p = r'c:\Users\nhphuong\Desktop\Solar\all_data\saturation_detection.ipynb'
nb = nbformat.read(p, as_version=4)
print('cells:', len(nb.cells))
n_img = 0
for i, c in enumerate(nb.cells):
    if c.cell_type != 'code':
        continue
    for o in c.get('outputs', []):
        if o.get('output_type') == 'stream':
            txt = o.get('text', '')
            print(f'--- cell {i} stdout ---')
            print(txt[:1200])
        if o.get('output_type') in ('display_data', 'execute_result'):
            data = o.get('data', {})
            if 'image/png' in data:
                n_img += 1
                print(f'--- cell {i}: [PNG figure] ---')
            elif 'text/plain' in data:
                t = data['text/plain']
                print(f'--- cell {i} result ---')
                print(t[:900])
print('total figures:', n_img)
for f in ['saturation_flags.csv', 'data_quality_events.csv']:
    fp = rf'c:\Users\nhphuong\Desktop\Solar\all_data\{f}'
    print(f, os.path.exists(fp), os.path.getsize(fp) if os.path.exists(fp) else '')
