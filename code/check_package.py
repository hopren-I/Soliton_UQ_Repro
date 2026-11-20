"""Audit reported outputs against the retained data and compiled manuscript."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
from itertools import product
from numpy.polynomial.legendre import legval

root=Path(__file__).resolve().parents[1]
res=root/'results'
study=json.loads((res/'study.json').read_text(encoding='utf-8'))
summary=json.loads((res/'summary.json').read_text(encoding='utf-8'))
par=study['parameters'];best=study['design']['best']['control']
source=(root/'main.tex').read_text(encoding='utf-8')
aux=(root/'main.aux').read_text(encoding='utf-8')
bibliography=(root/'references.tex').read_text(encoding='utf-8')
log=(root/'main.log').read_text(encoding='utf-8',errors='replace')
assert not re.search(r'undefined|multiply defined|Overfull|Rerun to get|LaTeX Error',log,re.I)
labels=re.findall(r'\\label\{((?:fig|tab):[^}]+)\}',source)
assert sum(x.startswith('fig:') for x in labels)==4
assert sum(x.startswith('tab:') for x in labels)==4
for label in labels:
    assert source.count(r'\label{'+label+'}')==1
    assert source.index(r'\ref{'+label+'}')<source.index(r'\label{'+label+'}')
    assert r'\newlabel{'+label+'}' in aux
for image in re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}',source):
    figure = root/image
    assert figure.is_file() and figure.with_suffix('.png').is_file()
citations=list(dict.fromkeys(k for g in re.findall(r'\\cite\w*\{([^}]+)\}',source) for k in g.split(',')))
assert citations==re.findall(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}',bibliography)
assert citations==re.findall(r'\\bibcite\{([^}]+)\}',aux)
crossref=json.loads((res/'crossref_audit.json').read_text(encoding='utf-8'))
refs=json.loads((res/'reference_audit.json').read_text(encoding='utf-8'))
doi_links=len(re.findall(r'\\href\{https://doi\.org/',bibliography))
assert crossref['checked']==crossref['verified']==crossref['doi_link_count']==doi_links
assert crossref['all_verified'] and crossref['all_references_verified']
assert refs['reference_count']==len(citations) and refs['references_2022_onwards']>len(citations)/2
assert citations==refs['citation_order']
assert {row['key'] for row in crossref['entries']+crossref['official_standards']}==set(citations)

errE=errT=0.; n=0; paired={}; differences=[]
for row in study['records']:
    a,eps,j=row['control'],row['epsilon'],row['rep']; n+=row['samples']
    with np.load(res/f'samples_r{j}_a{a:+.3f}_e{eps:.3f}.npz') as ar:
        xi=ar['xi']
        assert xi.shape==(2**par['power'],4) and np.max(np.abs(xi))<=1
        for key in row['mean']:
            np.testing.assert_allclose(np.mean(ar[key]),row['mean'][key],rtol=1e-13,atol=0)
            np.testing.assert_allclose(np.var(ar[key]),row['variance'][key],rtol=1e-13,atol=0)
        errE=max(errE,float(np.max(abs(ar['energy']/(.5*np.exp(eps*xi[:,2]))-1))))
        errT=max(errT,float(np.max(abs(ar['centre']-.3*(3+eps*xi[:,0]*3/2)))))
        pairkey=(eps,j)
        if pairkey in paired:
            x0,y0=paired[pairkey]
            np.testing.assert_array_equal(x0,xi)
            if eps==.1: differences.append(y0-ar['aligned_loss'])
        else: paired[pairkey]=(xi.copy(),ar['aligned_loss'].copy())
assert len(study['records'])==2*len(par['epsilons'])*par['replicates']
assert errE<2e-10 and errT<5e-5
pairedmean=np.array([x.mean() for x in differences]);pairstat=summary['paired_improvement_at_0.1']
np.testing.assert_allclose(pairedmean.mean(),pairstat['difference'],rtol=1e-12)
np.testing.assert_allclose(pairedmean.std(ddof=1)/np.sqrt(len(pairedmean)),pairstat['SE'],rtol=1e-12)
sample_differences=np.concatenate(differences)
np.testing.assert_allclose(np.quantile(sample_differences,[.05,.5,.95]),
                           [pairstat['q05'],pairstat['median'],pairstat['q95']],rtol=1e-12)
np.testing.assert_allclose(np.mean(sample_differences<0),pairstat['fraction_selected_worse'],rtol=0,atol=0)

pce=json.loads((res/'pce_comparison.json').read_text(encoding='utf-8'))
assert pce['basis']=='orthonormal Legendre, total degree 2' and pce['terms']==15
training_x=[];training_y=[];test_x=[];test_y=[]
for rep in range(4):
    with np.load(res/f'samples_r{rep}_a+0.000_e0.100.npz') as ar:
        (training_x if rep<2 else test_x).append(ar['xi'])
        (training_y if rep<2 else test_y).append(ar['aligned_loss'])
training_x=np.concatenate(training_x);training_y=np.concatenate(training_y)
test_x=np.concatenate(test_x);test_y=np.concatenate(test_y)
indices=[m for m in product(range(3),repeat=4) if sum(m)<=2]
def legendre_design(x):
    design=np.ones((len(x),len(indices)))
    for j,multi in enumerate(indices):
        for k,degree in enumerate(multi):
            coeff=np.zeros(degree+1);coeff[degree]=1.
            design[:,j]*=np.sqrt(2*degree+1)*legval(x[:,k],coeff)
    return design
coefficients,*_=np.linalg.lstsq(legendre_design(training_x),training_y,rcond=None)
pce_prediction=legendre_design(test_x)@coefficients
constant=indices.index((0,0,0,0))
np.testing.assert_allclose(pce['test_rmse'],np.sqrt(np.mean((pce_prediction-test_y)**2)),rtol=1e-12)
np.testing.assert_allclose(pce['test_mean'],np.mean(test_y),rtol=1e-12)
np.testing.assert_allclose(pce['test_variance'],np.var(test_y,ddof=1),rtol=1e-12)
np.testing.assert_allclose(pce['pce_mean'],coefficients[constant],rtol=1e-12)
np.testing.assert_allclose(pce['pce_variance'],np.sum(coefficients**2)-coefficients[constant]**2,rtol=1e-12)

fiber=json.loads((res/'fiber.json').read_text(encoding='utf-8'))
extensions=json.loads((res/'extensions.json').read_text(encoding='utf-8'))
redistribution=extensions['redistribution_only']
with np.load(res/'redistribution_quadrature.npz') as ar:
    for row in redistribution:
        a=row['control']; tag=f'{a:+.2f}'
        for order in (24,48):
            loss=ar[f'loss{order}_a{tag}']
            mean=float(ar[f'weights{order}']@loss)
            np.testing.assert_allclose(mean,row['mean' if order==48 else 'mean_24'],rtol=1e-13)
    np.testing.assert_allclose(redistribution[1]['mean']/redistribution[0]['mean']-1,
                               extensions['redistribution_change'],rtol=1e-13)
    assert max(row['relative_quadrature_change'] for row in redistribution)<1e-3
with np.load(res/'fiber.npz') as ar:
    assert ar['xi'].shape==(512,4)
    assert np.max(abs(ar['xi']))<=1
    H=np.asarray(fiber['H'])
    assert np.linalg.eigvalsh(H).min()>-1e-14
    np.testing.assert_allclose(np.einsum('ni,ij,nj->n',ar['xi'],H,ar['xi']),
                               ar['quadratic'],rtol=1e-13,atol=1e-17)
    np.testing.assert_allclose(fiber['quad_mean'],np.trace(H)/3,rtol=1e-13)
    fiber_absolute_errors=np.abs(ar['quadratic']-ar['direct'])
    rep_means=ar['direct'].reshape(4,128).mean(axis=1)
    np.testing.assert_allclose(rep_means,fiber['replicate_means'],rtol=1e-13)
    np.testing.assert_allclose(rep_means.mean(),fiber['direct_mean'],rtol=1e-13)
    np.testing.assert_allclose(rep_means.std(ddof=1)/2,fiber['direct_SE'],rtol=1e-13)
    np.testing.assert_allclose(ar['sweep_loss'].max(),fiber['gain_only_max_loss'],rtol=1e-13)
    optical_depth=np.log(10)/10*fiber['loss_db_per_km']*fiber['length_km']
    E0=2*fiber['t0_ps']*fiber['nominal_input_peak_W']
    expected=E0*np.exp(-optical_depth*(1+fiber['tolerance'][2]*ar['xi'][:,2]))
    np.testing.assert_allclose(ar['energy'],expected,rtol=2e-12,atol=0)
    np.testing.assert_allclose(ar['sweep_energy'],E0*np.exp(-optical_depth),rtol=2e-12,atol=0)
    assert np.max(abs(ar['centroid_ps']))<1e-8
assert fiber['maximum_independent_RK4_field_difference']<5e-6
assert fiber['maximum_gauge_field_difference']<1e-13
assert redistribution[1]['mean']<redistribution[0]['mean']

with np.load(res/'relief_data.npz') as raw:
    direct,quad,x=raw['risk_direct'].astype(float),raw['risk_quadratic'].astype(float),raw['risk_grid']
    departure=np.divide(direct-quad,quad,out=np.full_like(quad,np.nan),where=quad>0)
    imax,jmax=np.unravel_index(np.nanargmax(departure),departure.shape)
    np.testing.assert_allclose([x[imax],x[jmax]],extensions['max_position'],rtol=0,atol=0)
    np.testing.assert_allclose(departure[imax,jmax],extensions['max_relative'],rtol=1e-13)
base_net,selected_net=extensions['net_gain_only']
np.testing.assert_allclose(selected_net['mean']/base_net['mean']-1,
                           extensions['net_gain_change'],rtol=1e-13)
assert selected_net['mean']>base_net['mean'] and pairstat['difference']>0

lean=(res/'lean_check.log').read_text(encoding='utf-8-sig')
theorems=re.findall(r"^'SolitonUQ\.([^']+)' depends on axioms: \[([^\]]+)\]$",lean,re.M)
declared=re.findall(r'^theorem (\w+)',(root/'formal/UQ.lean').read_text(encoding='utf-8'),re.M)
assert [x[0] for x in theorems]==declared
assert not re.search(r'error:|warning:|sorryAx',lean)
assert all(set(a.split(', '))<={'propext','Classical.choice','Quot.sound'} for _,a in theorems)
audit={'pdf_sha256':hashlib.sha256((root/'main.pdf').read_bytes()).hexdigest(),
    'figures':4,'tables':4,'citations':len(citations),'crossref_verified':crossref['verified'],
    'doi_link_count':crossref['doi_link_count'],'verified_lean_theorems':len(theorems),
    'official_standards_verified':len(crossref['official_standards']),
    'fiber_samples':512,'fiber_maximum_relative_energy_error':fiber['maximum_energy_relative_error'],
    'fiber_quad_mean_relative_difference':abs(fiber['quad_mean']/fiber['direct_mean']-1),
    'fiber_RK4_maximum_field_difference':fiber['maximum_independent_RK4_field_difference'],
    'fiber_gauge_maximum_field_difference':fiber['maximum_gauge_field_difference'],
    'symbolic_checks':json.loads((res/'symbolic_audit.json').read_text())['passed'],
    'paired_sample_fraction_improved_at_0.1':float(np.mean(np.concatenate(differences)>0)),
    'pce_degree':2,'pce_terms':pce['terms'],'pce_holdout_rmse':pce['test_rmse'],
    'pce_holdout_mean_relative_error':pce['pce_mean_relative_error'],
    'pce_holdout_variance_relative_error':pce['pce_variance_relative_error'],
    'redistribution_direct_relative_quadrature_change':max(row['relative_quadrature_change'] for row in redistribution),
    'redistribution_direct_mean_change':extensions['redistribution_change'],
    'fiber_absolute_prediction_error_median':float(np.median(fiber_absolute_errors)),
    'fiber_absolute_prediction_error_q95':float(np.quantile(fiber_absolute_errors,.95)),
    'main_propagations':n,'maximum_relative_energy_error':errE,'maximum_absolute_centroid_error':errT}
(res/'package_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
print(json.dumps(audit,indent=2))
