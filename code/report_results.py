"""Create manuscript numbers and tables directly from saved propagation results."""
from pathlib import Path
import html
import json
import re
import numpy as np
from itertools import product
from numpy.polynomial.legendre import legval

ROOT=Path(__file__).resolve().parents[1]
res=json.loads((ROOT/'results/study.json').read_text(encoding='utf-8'))
def collect(a,e,q='aligned_loss'):
    return np.array([x['mean'][q] for x in res['records'] if x['control']==a and x['epsilon']==e])
def sem(x): return np.std(x,ddof=1)/np.sqrt(len(x))
def texnum(x,digits=2):
    m,p=f'{x:.{digits}e}'.split('e')
    return f'${m}\\times10^{{{int(p)}}}$'
def percent(x): return f'{100*x:.2f}\\%'

def total_degree_indices(d, p):
    return [m for m in product(range(p + 1), repeat=d) if sum(m) <= p]

def orthonormal_legendre_matrix(x, indices):
    out = np.ones((x.shape[0], len(indices)))
    for j, multi in enumerate(indices):
        for k, degree in enumerate(multi):
            coeff = np.zeros(degree + 1)
            coeff[degree] = 1.0
            out[:, j] *= np.sqrt(2 * degree + 1) * legval(x[:, k], coeff)
    return out

def pce_comparison():
    """Fit a degree-two PCE on two scramblings and test it on two held out scramblings."""
    epsilon = 0.1
    train_x, train_y, test_x, test_y = [], [], [], []
    for rep in (0, 1, 2, 3):
        path = ROOT / 'results' / f'samples_r{rep}_a+0.000_e{epsilon:.3f}.npz'
        with np.load(path) as sample:
            if rep < 2:
                train_x.append(sample['xi'])
                train_y.append(sample['aligned_loss'])
            else:
                test_x.append(sample['xi'])
                test_y.append(sample['aligned_loss'])
    train_x, train_y = np.concatenate(train_x), np.concatenate(train_y)
    test_x, test_y = np.concatenate(test_x), np.concatenate(test_y)
    indices = total_degree_indices(train_x.shape[1], 2)
    design = orthonormal_legendre_matrix(train_x, indices)
    test_design = orthonormal_legendre_matrix(test_x, indices)
    coefficients, *_ = np.linalg.lstsq(design, train_y, rcond=None)
    prediction = test_design @ coefficients
    constant = indices.index((0, 0, 0, 0))
    pce_mean = float(coefficients[constant])
    pce_variance = float(np.sum(coefficients**2) - coefficients[constant]**2)
    direct_mean = float(np.mean(test_y))
    direct_variance = float(np.var(test_y, ddof=1))
    quadratic = epsilon**2 * np.einsum(
        'ni,ij,nj->n', test_x, np.asarray(res['responses']['0.0']['aligned_gram']), test_x
    )
    return {
        'epsilon': epsilon,
        'basis': 'orthonormal Legendre, total degree 2',
        'terms': len(indices),
        'training_replicates': [0, 1],
        'test_replicates': [2, 3],
        'test_rmse': float(np.sqrt(np.mean((prediction - test_y)**2))),
        'quadratic_test_rmse': float(np.sqrt(np.mean((quadratic - test_y)**2))),
        'test_mean': direct_mean,
        'pce_mean': pce_mean,
        'quadratic_mean': float(np.mean(quadratic)),
        'test_variance': direct_variance,
        'pce_variance': pce_variance,
        'quadratic_variance': float(np.var(quadratic, ddof=1)),
        'pce_mean_relative_error': float(abs(pce_mean / direct_mean - 1)),
        'quadratic_mean_relative_error': float(abs(np.mean(quadratic) / direct_mean - 1)),
        'pce_variance_relative_error': float(abs(pce_variance / direct_variance - 1)),
        'quadratic_variance_relative_error': float(abs(np.var(quadratic, ddof=1) / direct_variance - 1)),
    }

pce = pce_comparison()
(ROOT / 'results' / 'pce_comparison.json').write_text(json.dumps(pce, indent=2) + '\n', encoding='utf-8')
best=res['design']['best']['control']
base=collect(0.,.1); selected=collect(best,.1); dif=base-selected
paired_sample_differences=[]
for rep in range(res['parameters']['replicates']):
    base_path=ROOT/'results'/f'samples_r{rep}_a+0.000_e0.100.npz'
    selected_path=ROOT/'results'/f'samples_r{rep}_a{best:+.3f}_e0.100.npz'
    with np.load(base_path) as base_sample, np.load(selected_path) as selected_sample:
        paired_sample_differences.append(base_sample['aligned_loss']-selected_sample['aligned_loss'])
paired_sample_differences=np.concatenate(paired_sample_differences)
v=res['validation']; responses=res['responses']
macro={
 'BestControl':f'{best:g}',
 'DesignReduction':percent(dif.mean()/base.mean()),
 'RKError':texnum(max(v['rk4_tangent_relative_errors'])),
 'FDError':texnum(max(v['finite_difference_tangent_errors'])),
 'SpatialChange':texnum(v['spatial_trace_change']),
 'EnergyLawError':texnum(max(x['energy_law_max_error'] for x in res['records'])),
 'CentreLawError':texnum(max(x['centre_law_max_error'] for x in res['records'])),
 'CornerChange':texnum(max(max(x['relative_field_change']) for x in res['perturbed_refinement'])),
 'SilentMax':texnum(max(res['energy_silent']['aligned_loss'])),
 'BaseRisk':f"{responses['0.0']['aligned_trace']/3:.6f}",
 'BestRisk':f"{responses[str(best)]['aligned_trace']/3:.6f}",
 'PairedDifference':texnum(dif.mean()),'PairedSE':texnum(sem(dif))}
macro.update({
 'PairedQlow': texnum(np.quantile(paired_sample_differences, 0.05), 2),
 'PairedQmid': texnum(np.quantile(paired_sample_differences, 0.50), 2),
 'PairedQhigh': texnum(np.quantile(paired_sample_differences, 0.95), 2),
 'PairedWorse': percent(np.mean(paired_sample_differences < 0)),
 'VarianceReplicateSD': texnum(np.std([
     x['variance']['aligned_loss'] for x in res['records']
     if x['control'] == 0.0 and x['epsilon'] == 0.1
 ], ddof=1), 2),
 'PCEterms': str(pce['terms']),
 'PCEtestRMSE': texnum(pce['test_rmse'], 2),
 'PCEquadraticRMSE': texnum(pce['quadratic_test_rmse'], 2),
 'PCEmeanError': percent(pce['pce_mean_relative_error']),
 'PCEquadraticMeanError': percent(pce['quadratic_mean_relative_error']),
 'PCEvarianceError': percent(pce['pce_variance_relative_error']),
 'PCEquadraticVarianceError': percent(pce['quadratic_variance_relative_error']),
})
for eps,label in [(.025,'SmallError'),(.1,'MediumError'),(.2,'LargeError')]:
    actual=collect(0.,eps).mean(); pred=eps**2*responses['0.0']['aligned_trace']/3
    macro[label]=percent(abs(pred/actual-1))
fiber = json.loads((ROOT/'results/fiber.json').read_text(encoding='utf-8'))
extensions = json.loads((ROOT/'results/extensions.json').read_text(encoding='utf-8'))
net_gain = extensions['net_gain_only']
redistribution = extensions['redistribution_only']
macro.update({
    'NetGainBase': texnum(net_gain[0]['mean'], 3),
    'NetGainSelected': texnum(net_gain[1]['mean'], 3),
    'NetGainChange': percent(extensions['net_gain_change']),
    'RedistributionBase': texnum(redistribution[0]['mean'], 3),
    'RedistributionSelected': texnum(redistribution[1]['mean'], 3),
    'RedistributionChange': percent(extensions['redistribution_change']),
    'RedistributionQuadChange': texnum(max(row['relative_quadrature_change'] for row in redistribution), 2),
})
for label, key in {
    'FiberSweep': 'gain_only_max_loss',
    'FiberGauge': 'maximum_gauge_field_difference',
    'FiberDirect': 'direct_mean',
    'FiberSE': 'direct_SE',
    'FiberQuad': 'quad_mean',
    'FiberEnergyError': 'maximum_energy_relative_error',
    'FiberRKError': 'maximum_independent_RK4_field_difference',
    'FiberStepError': 'maximum_split_step_halving_difference',
    'FiberSpatialError': 'maximum_spatial_difference',
}.items():
    macro[label] = texnum(fiber[key], 3 if label in ('FiberDirect', 'FiberQuad') else 2)
macro['FiberMeanGap'] = percent(abs(fiber['quad_mean']/fiber['direct_mean']-1))
with np.load(ROOT/'results/fiber.npz') as raw_fiber:
    peak = 1000*fiber['nominal_input_peak_W']*np.max(np.abs(raw_fiber['terminal_profiles'])**2,axis=1)
    for label, value in zip(('FiberPeakMax','FiberPeakNominal','FiberPeakMin'),peak):
        macro[label] = f'{value:.2f}'
    relative_error = abs(raw_fiber['quadratic']/raw_fiber['direct']-1)
    for label, value in zip(('FiberMedianError','FiberTailError'),np.percentile(relative_error,[50,95])):
        macro[label] = percent(value)
    absolute_error = abs(raw_fiber['quadratic']-raw_fiber['direct'])
    for label, value in zip(('FiberAbsMedianError','FiberAbsTailError'),np.percentile(absolute_error,[50,95])):
        macro[label] = texnum(value, 2)
(ROOT/'results/macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+val+'}' for k,val in macro.items())+'\n',encoding='utf-8')

lines=[r'\begin{tabular}{rrrr}',r'\toprule',r'$a$ & $h$ & Relative $L^2$ error & $\tr H^{\rm al}$\\',r'\midrule']
for x in v['convergence']:
    if x['control'] not in (0.,best):continue
    lines.append(f"{x['control']:g} & {x['h']:g} & {texnum(x['nominal_l2'])} & {x['aligned_trace']:.7f} \\")
    lines[-1]+='\\'
lines.extend([r'\bottomrule',r'\end{tabular}'])
(ROOT/'results/table_convergence.tex').write_text('\n'.join(lines),encoding='utf-8')

lines=[r'\begin{tabular}{rrrrr}',r'\toprule',r' & \multicolumn{2}{c}{Baseline $a=0$} & \multicolumn{2}{c}{Selected $a=-0.65$}\\',r'$\epsilon$ & Direct mean (SE) & Quadratic & Direct mean (SE) & Quadratic\\',r'\midrule']
summary=[]
for eps in res['parameters']['epsilons']:
    nums=[]
    for a in (0.,best):
        vals=collect(a,eps);pred=eps**2*responses[str(a)]['aligned_trace']/3
        nums.extend([texnum(vals.mean(),3)+' ('+texnum(sem(vals),1)+')',texnum(pred,3)])
        raw=collect(a,eps,'raw_loss')
        H=np.array(responses[str(a)]['aligned_gram'])
        qvar=4/45*np.sum(np.diag(H)**2)+4/9*np.sum(np.triu(H,1)**2)
        realvars=[x['variance']['aligned_loss'] for x in res['records'] if x['control']==a and x['epsilon']==eps]
        summary.append({'a':a,'epsilon':eps,'mean':float(vals.mean()),'SE':float(sem(vals)),
            'quadratic_mean':pred,'raw_mean':float(raw.mean()),'quadratic_variance':eps**4*qvar,
            'mean_within_replicate_variance':float(np.mean(realvars))})
    lines.append(f'{eps:g} & '+ ' & '.join(nums)+r'\\')
lines.extend([r'\bottomrule',r'\end{tabular}'])
(ROOT/'results/table_risk.tex').write_text('\n'.join(lines),encoding='utf-8')

lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'Mode & $\partial_j\log E(L)$ & $\partial_j t_c(L)$ & Leading mean-loss share\\',r'\midrule']
names=['Dispersion','Nonlinearity','Net gain','Zero-integral gain']
for i,name in enumerate(names):
    lines.append(f"{name} & {[0,0,1,0][i]} & {[.45,0,0,0][i]:g} & {percent(responses['0.0']['leading_shares'][i])}"+r'\\')
lines.extend([r'\bottomrule',r'\end{tabular}'])
(ROOT/'results/table_shares.tex').write_text('\n'.join(lines),encoding='utf-8')

summary={'aggregated':summary,'paired_improvement_at_0.1':{'difference':float(dif.mean()),'SE':float(sem(dif)),
 'relative_percent':float(100*dif.mean()/base.mean()),
 'q05':float(np.quantile(paired_sample_differences, 0.05)),
 'median':float(np.quantile(paired_sample_differences, 0.50)),
 'q95':float(np.quantile(paired_sample_differences, 0.95)),
 'fraction_selected_worse':float(np.mean(paired_sample_differences < 0)),
 'baseline_variance_replicate_sd':float(np.std([
     x['variance']['aligned_loss'] for x in res['records']
     if x['control'] == 0.0 and x['epsilon'] == 0.1
 ], ddof=1))},
 'pce_comparison':pce,'macros':macro}
(ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

# Bibliography order is derived from first appearance, not metadata insertion order.
source=(ROOT/'main.tex').read_text(encoding='utf-8')
keys=[]
for group in re.findall(r'\\cite\w*\{([^}]+)\}',source):
    for key in group.split(','):
        if key not in keys:keys.append(key)
data=json.loads((ROOT/'references_verified.json').read_text(encoding='utf-8'))
def clean(text):
    text=html.unescape(text).replace('\u2019',"'").replace('\u2013','--').replace('\u2014','---')
    return text.replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')
year_override={'suslov2012':2012}
bib=[r'\begin{thebibliography}{99}']
years=[]
for i,k in enumerate(keys):
    d=data[k]
    authors=', '.join(clean((a.get('given','')+' '+a.get('family','')).strip()) for a in d['authors'])
    year=year_override.get(k,(d.get('print') or d['published'])['date-parts'][0][0]);years.append(year)
    title=d['title']
    if k=='luthen2022':title='Automatic selection of basis-adaptive sparse polynomial chaos expansions for engineering applications'
    journal=d['journal'][-1] if k=='owen2022' else d['journal'][0]
    if d.get('book_metadata'):
        title, edition = title.rsplit(', ', 1)
        ref = authors + r'. \emph{' + clean(title) + '}, ' + edition + ', '
        ref += clean(journal) + f' ({year})'
    elif d.get('url') and not d.get('doi'):
        ref = authors + '. ' + clean(title) + '. ' + clean(journal) + f' ({year})'
    else:
        ref=authors+'. '+clean(title)+'. '+r'\emph{'+clean(journal)+'}'
        if d['volume']:
            ref+=' '+clean(d['volume'])
        ref+=f' ({year})'
        pages=d['article_number'] or d['page']
        if pages:
            ref+=', '+clean(pages).replace('-', '--')
    if d.get('doi'):
        ref+=r'. \href{https://doi.org/'+d['doi']+'}{doi:'+clean(d['doi'])+'}.'
    else:
        ref+=r'. \href{'+d['url']+'}{Official recommendation}.'
    lead=d['authors'][0].get('family','')
    short=clean(lead+(' et al.' if len(d['authors'])>2 else
        (' and '+d['authors'][1]['family'] if len(d['authors'])==2 else '')))
    bib.append(r'\bibitem['+short+f'({year})'+']{'+k+'} '+ref)
bib.append(r'\end{thebibliography}')
(ROOT/'references.tex').write_text('\n\n'.join(bib),encoding='utf-8')
audit={'citation_order':keys,'reference_count':len(keys),'references_2022_onwards':sum(y>=2022 for y in years),
 'all_citations_resolved':True,'unused_verified_entries':sorted(set(data)-set(keys))}
crossref_path=ROOT/'results'/'crossref_audit.json'
if crossref_path.is_file():
    cr=json.loads(crossref_path.read_text(encoding='utf-8'))
    audit.update({'crossref_checked':cr['checked'],'crossref_verified':cr['verified'],
      'crossref_all_verified':cr['all_verified'],'doi_link_count':cr['doi_link_count'],
      'crossref_checked_at_utc':cr['checked_at_utc'],
      'official_standard_count':len(cr.get('official_standards',[])),
      'all_references_verified':cr.get('all_references_verified',False)})
assert audit['reference_count']>=20 and audit['references_2022_onwards']>len(keys)/2
(ROOT/'results/reference_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(summary['paired_improvement_at_0.1'],indent=2))
print(json.dumps(audit,indent=2))
