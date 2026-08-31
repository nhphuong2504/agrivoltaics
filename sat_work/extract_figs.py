import nbformat, base64, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

p = r'c:\Users\nhphuong\Desktop\Solar\all_data\saturation_detection.ipynb'
out = r'c:\Users\nhphuong\Desktop\Solar\all_data\sat_work'
nb = nbformat.read(p, as_version=4)
n = 0
for i, c in enumerate(nb.cells):
    if c.cell_type != 'code':
        continue
    for o in c.get('outputs', []):
        data = o.get('data', {})
        if 'image/png' in data:
            n += 1
            fp = os.path.join(out, f'nb_fig{n}_cell{i}.png')
            with open(fp, 'wb') as f:
                f.write(base64.b64decode(data['image/png']))
            print('saved', fp)
print('done,', n, 'figures')
