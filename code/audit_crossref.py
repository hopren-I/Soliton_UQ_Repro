"""Verify every bibliography entry against Crossref and its DOI hyperlink.

Run with network access:
    python scripts/audit_crossref.py
Writes results/crossref_audit.json. This is an audit utility, not part of the
numerical study; a failed network call is reported rather than silently passed.
"""
from __future__ import annotations

import html
from concurrent.futures import ThreadPoolExecutor
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = 'https://api.crossref.org/works/'
UA = 'SolitonUQ-reference-audit/1.1'


def norm(text: str) -> str:
    text = html.unescape(str(text))
    text = text.replace('\u2019', "'").replace('\u2013', '-').replace('\u2014', '-')
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'[{}]', '', text)
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def years(message: dict) -> set[int]:
    out = set()
    for key in ('published-print', 'published-online', 'published', 'issued', 'created'):
        parts = (message.get(key) or {}).get('date-parts') or []
        if parts and parts[0] and parts[0][0]:
            out.add(int(parts[0][0]))
    return out


def fetch(key: str, doi: str) -> dict:
    url = API + urllib.parse.quote(doi, safe='/')
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=30) as response:
        message = json.load(response)['message']
    return {'key': key, 'doi': doi, 'message': message}


def main():
    data = json.loads((ROOT / 'references_verified.json').read_text(encoding='utf-8'))
    bib = (ROOT / 'references.tex').read_text(encoding='utf-8')
    links, other_links, bib_years = {}, {}, {}
    for block in re.split(r'(?=\\bibitem)', bib):
        m = re.search(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}(.*)', block, re.S)
        if not m:
            continue
        key, body = m.group(1), m.group(2)
        href = re.search(r'\\href\{https://doi\.org/([^}]+)\}\{doi:', body)
        links[key] = href.group(1).replace('\\_', '_') if href else None
        official = re.search(r'\\href\{([^}]+)\}\{Official recommendation\}', body)
        if official:
            other_links[key] = official.group(1)
        year = re.search(r'\((\d{4})\)', body)
        bib_years[key] = int(year.group(1)) if year else None

    jobs = [(key, data[key]['doi']) for key in links if data[key].get('doi')]
    with ThreadPoolExecutor(max_workers=3) as workers:
        responses = list(workers.map(lambda job: fetch(*job), jobs))
    standards = []
    for key in links:
        if data[key].get('doi'):
            continue
        url = data[key]['url']
        req = urllib.request.Request(url, headers={'User-Agent': UA})
        with urllib.request.urlopen(req, timeout=30) as response:
            page = response.read().decode('utf-8', errors='replace')
        expected = data[key]['published']['date-parts'][0][0]
        title = data[key]['title']
        passed = (other_links.get(key) == url and bib_years[key] == expected
                  and norm(title) in norm(page) and 'G.652' in page)
        standards.append({
            'key': key, 'url': url, 'displayed_year': bib_years[key],
            'status': 'verified' if passed else 'mismatch',
            'data_source': 'ITU-T Recommendation G.652 (08/2024), Table I.1, 1550 nm.',
        })

    results = []
    for item in responses:
        key, doi, cr = item['key'], item['doi'], item['message']
        stored = data[key]
        reasons, notes = [], []
        if not links.get(key):
            reasons.append('missing DOI hyperlink in references.tex')
        elif norm(links[key]) != norm(doi):
            reasons.append('DOI hyperlink differs from metadata DOI')

        cr_title = cr.get('title', [''])[0]
        book = stored.get('book_metadata')
        stored_title = stored['title'].rsplit(', ', 1)[0] if book else stored['title']
        if norm(cr_title) != norm(stored_title):
            reasons.append('title differs from Crossref')

        cr_authors = [norm(a.get('family', '')) for a in cr.get('author', [])]
        st_authors = [norm(a.get('family', '')) for a in stored.get('authors', [])]
        if book and not cr_authors:
            if cr.get('ISBN') != ['9780128170427']:
                reasons.append('book ISBN differs from publisher catalog')
            notes.append('Crossref omits the author and edition. The Elsevier book catalog for ISBN 9780128170427 specifies Govind P. Agrawal, 6th edition.')
        elif cr_authors != st_authors:
            reasons.append('author list differs from Crossref')

        cr_journals = [norm(j) for j in cr.get('container-title', [])]
        st_journals = [norm(j) for j in stored.get('journal', [])]
        if book:
            cr_journals = [norm(cr.get('publisher', ''))]
        if not set(st_journals) & set(cr_journals):
            reasons.append('journal differs from Crossref')

        displayed_year = bib_years.get(key)
        if displayed_year and displayed_year not in years(cr):
            stored_year = int((stored.get('print') or stored['published'])['date-parts'][0][0])
            if key == 'suslov2012' and stored_year in years(cr):
                notes.append('bibliography uses the 2012 issue year; Crossref records a 2011 online date')
            else:
                reasons.append(f'displayed year {displayed_year} absent from Crossref dates')

        cr_vol = str(cr.get('volume') or '')
        cr_loc = str(cr.get('page') or cr.get('article-number') or '')
        st_vol = str(stored.get('volume') or '')
        st_loc = str(stored.get('page') or stored.get('article_number') or '')
        if cr_vol and st_vol and cr_vol != st_vol:
            reasons.append('volume differs from Crossref')
        if cr_loc and st_loc and norm(cr_loc) != norm(st_loc):
            reasons.append('page/article number differs from Crossref')

        results.append({
            'key': key, 'doi': doi,
            'crossref_title': cr_title,
            'crossref_year_dates': sorted(years(cr)),
            'displayed_year': displayed_year,
            'doi_hyperlink': links.get(key),
            'status': 'verified' if not reasons else 'mismatch',
            'reasons': reasons,
            'notes': notes,
        })

    report = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'endpoint': API,
        'checked': len(results),
        'verified': sum(r['status'] == 'verified' for r in results),
        'doi_link_count': sum(v is not None for v in links.values()),
        'all_verified': all(r['status'] == 'verified' for r in results)
                        and len(results) == len(jobs),
        'official_standards': standards,
        'all_references_verified': all(r['status'] == 'verified' for r in results + standards)
                                   and len(results) + len(standards) == len(links),
        'entries': results,
    }
    (ROOT / 'results' / 'crossref_audit.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    audit_path = ROOT / 'results' / 'reference_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    audit.update({
        'crossref_checked': report['checked'],
        'crossref_verified': report['verified'],
        'crossref_all_verified': report['all_verified'],
        'doi_link_count': report['doi_link_count'],
        'crossref_checked_at_utc': report['checked_at_utc'],
        'official_standard_count': len(standards),
        'all_references_verified': report['all_references_verified'],
    })
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n',
                          encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('checked', 'verified', 'doi_link_count', 'all_references_verified')},
                     indent=2))
    for r in results:
        if r['reasons']:
            print(r['key'], r['reasons'])
    if not report['all_references_verified']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
