"""Build the audited reproducibility archive from the manuscript and its data."""
from pathlib import Path
import hashlib
import re
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
FIGURES = re.findall(r'\\includegraphics\[[^\]]+\]\{([^}]+)\}',
                     (ROOT/'main.tex').read_text(encoding='utf-8'))
MANIFEST = [
    'main.tex', 'main.pdf', 'verification.md', 'references.tex',
    'references_verified.json', 'literature_search.json', 'requirements.txt',
    'formal/UQ.lean', 'formal/lakefile.toml', 'formal/lake-manifest.json', 'formal/lean-toolchain',
]
files = [ROOT/name for name in MANIFEST]
files += sorted((ROOT/'scripts').glob('*.py'))
files += [p for p in sorted((ROOT/'results').iterdir())
          if p.is_file() and p.name != 'illustrative_gain_fields.npz']
for name in FIGURES:
    figure = ROOT/name
    files.extend((figure, figure.with_suffix('.png')))

final_pdf = ROOT/'output/pdf/Energy_and_Waveform_Uncertainty.pdf'
final_pdf.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(ROOT/'main.pdf', final_pdf)
with ZipFile(ROOT/'Soliton_UQ_Reproducibility.zip', 'w', ZIP_DEFLATED, compresslevel=6) as z:
    for path in files:
        z.write(path, path.relative_to(ROOT).as_posix())
with ZipFile(ROOT/'Soliton_UQ_Reproducibility.zip') as z:
    assert z.testzip() is None
    assert z.read('main.pdf') == final_pdf.read_bytes()
    assert len(z.namelist()) == len(set(z.namelist())) == len(files)
    print(f'ZIP verified: {len(files)} files, {len(FIGURES)} figure pairs')
    print('PDF SHA-256:', hashlib.sha256(z.read('main.pdf')).hexdigest())
