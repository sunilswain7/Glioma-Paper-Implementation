"""Reorder thebibliography in refs.tex by order of first citation (IEEE style)."""
import re
files = [l.split('{')[1].rstrip('}\n') for l in open('report.tex') if l.startswith(r'\input')]
order = []
for f in files:
    if f == 'refs':
        continue
    for m in re.finditer(r'\\cite\{([^}]*)\}', open(f + '.tex').read()):
        for k in m.group(1).split(','):
            k = k.strip()
            if k not in order:
                order.append(k)
s = open('refs.tex').read()
head, rest = s.split(r'\bibitem', 1)
body, tail = ('\\bibitem' + rest).rsplit(r'\end{thebibliography}', 1)
items = {re.match(r'\\bibitem\{([^}]*)\}', b).group(1): b.strip() + '\n\n'
         for b in re.split(r'(?=\\bibitem\{)', body) if b.strip()}
missing = [k for k in order if k not in items]
unused = [k for k in items if k not in order]
assert not missing, f"cited but no bibitem: {missing}"
new = head + ''.join(items[k] for k in order + unused) + r'\end{thebibliography}' + tail
open('refs.tex', 'w').write(new)
print("order:", order)
print("uncited:", unused)
