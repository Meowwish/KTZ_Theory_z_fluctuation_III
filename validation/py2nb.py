import nbformat, re, sys
src = open(sys.argv[1]).read()
cells = []
for chunk in re.split(r'(?m)^(?=# %%)', src):
    if not chunk.strip():
        continue
    if chunk.startswith('# %% [markdown]'):
        body = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in chunk.splitlines()[1:])
        cells.append(nbformat.v4.new_markdown_cell(body.strip()))
    else:
        cells.append(nbformat.v4.new_code_cell(chunk.rstrip()))
nb = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
nbformat.write(nb, sys.argv[2])
print(len(cells), 'cells')
