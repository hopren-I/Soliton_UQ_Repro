"""Recompute one-mode schedule comparisons and audit the archived risk map."""
import json
from pathlib import Path
import numpy as np
from study import exact, propagate, quantities, L

ROOT = Path(__file__).resolve().parents[1]
R = ROOT/'results'
s = json.loads((R/'study.json').read_text(encoding='utf-8'))
raw = np.load(R/'relief_data.npz')
coords = raw['risk_grid']
q = raw['risk_quadratic'].astype(float)
d = raw['risk_direct'].astype(float)
err = np.divide(d,q,where=q>0,out=np.full(q.shape,np.nan))-1
loc = np.unravel_index(np.nanargmax(np.abs(err)), err.shape)
corners = np.abs(err[np.ix_([0,-1],[0,-1])])
selected = s['design']['best']['control']
diag = [np.diag(s['responses'][str(a)]['aligned_gram']) for a in (0.,selected)]
nodes, weights = np.polynomial.legendre.leggauss(24)
record = []
for a,hdiag in zip((0.,selected),diag):
    x = np.zeros((len(nodes),4)); x[:,2] = nodes
    t,dt,k,u = propagate(x,.1,control=a,step=.0025)
    direct = quantities(u,t,dt,k,x,.1,exact(t,L,a))['aligned_loss']
    record.append({'control':a,'mean':float(weights@direct/2),'quadratic_mean':float(.1**2*hdiag[2]/3)})

redistribution = {}
quadrature_data = {}
for order in (24,48):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    weights = weights/2
    x = np.zeros((len(nodes),4)); x[:,3] = nodes
    quadrature_data[f'xi{order}'] = x
    quadrature_data[f'weights{order}'] = weights
    for a,hdiag in zip((0.,selected),diag):
        t,dt,k,u = propagate(x,.1,control=a,step=.0025)
        direct = quantities(u,t,dt,k,x,.1,exact(t,L,a))['aligned_loss']
        quadrature_data[f'loss{order}_a{a:+.2f}'] = direct
        mean = float(weights@direct)
        redistribution.setdefault(str(a),{})[str(order)] = mean
        if order == 48:
            redistribution[str(a)]['quadratic_mean'] = float(.1**2*hdiag[3]/3)

np.savez_compressed(R/'redistribution_quadrature.npz',**quadrature_data)
redistribution_rows = []
for a in (0.,selected):
    values = redistribution[str(a)]
    redistribution_rows.append({
        'control':a,
        'mean':values['48'],
        'mean_24':values['24'],
        'quadratic_mean':values['quadratic_mean'],
        'relative_quadrature_change':abs(values['48']/values['24']-1),
    })
result = {'max_relative':float(err[loc]),'max_position':[float(coords[i]) for i in loc],
          'corner_absolute_range':[float(corners.min()),float(corners.max())],
          'net_gain_only':record,'net_gain_change':record[1]['mean']/record[0]['mean']-1,
          'redistribution_only':redistribution_rows,
          'redistribution_change':redistribution_rows[1]['mean']/redistribution_rows[0]['mean']-1,
          'mode_diagonals':np.array(diag).tolist()}
(R/'extensions.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
