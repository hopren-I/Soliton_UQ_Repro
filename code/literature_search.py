"""Targeted OpenAlex discovery; metadata alone do not establish novelty."""
import concurrent.futures
import json
from pathlib import Path
import requests

queries = [
    'uncertainty quantification nonlinear Schrodinger equation soliton random coefficients',
    'nonlinear Schrodinger sensitivity analysis random dispersion soliton',
    'nonlinear Schrodinger polynomial chaos uncertainty quantification 2023',
    'soliton uncertainty quantification waveform fidelity stochastic gain',
]

def search(q):
    try:
        r = requests.get('https://api.openalex.org/works', params={
            'search': q, 'filter': 'from_publication_date:2022-01-01', 'per-page': 8,
            'select':'title,doi,publication_year'}, timeout=14)
        r.raise_for_status()
        data=r.json()
        return {'query': q, 'total': data['meta']['count'], 'items':data['results']}
    except requests.RequestException as e:
        return {'query':q,'error':str(e),'items':[]}

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        out=list(pool.map(search, queries))
    Path(__file__).resolve().parents[1].joinpath('literature_search.json').write_text(
        json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    for q in out:
        print(q['query'])
        for x in q['items']: print(x.get('doi'),x['title'])
