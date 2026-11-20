"""Reproducible finite-dimensional UQ study of managed cubic NLSE propagation.

All input distributions are synthetic uniform tolerances, not measured data.
The tangent solver differentiates the implemented symmetric splitting step.
An independent RK4 tangent calculation and mesh refinement check it below.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
L = 3.0
CARRIER = 0.3
DIM = 4
NAMES = ["dispersion", "nonlinearity", "net gain", "zero-integral gain"]


def sech(x):
    return 2 * np.exp(-np.abs(x)) / (1 + np.exp(-2 * np.abs(x)))


def profile(z, control):
    b = 1 + control * np.sin(2 * np.pi * z / L)
    w = 1 + z + control * L / (2 * np.pi) * (1 - np.cos(2 * np.pi * z / L))
    return b, w


def exact(t, z, control):
    _, w = profile(z, control)
    zz = 1 - 1 / w
    return (sech(t / w - CARRIER * zz) / w
            * np.exp(1j * (CARRIER * t / w + (1-CARRIER**2)*zz/2 + t*t/(2*w))))


def setup(size=1024, half_window=64., step=0.005, control=0.):
    count = round(L / step)
    step = L / count
    dt = 2 * half_window / size
    t = -half_window + dt * np.arange(size)
    k = 2 * np.pi * np.fft.fftfreq(size, dt)
    edges = np.linspace(0, L, count + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    nodes, weights = np.polynomial.legendre.leggauss(6)

    def integrals(left, right):
        radius = (right-left)/2
        z = (left+right)[:, None]/2 + radius[:, None]*nodes
        b, _ = profile(z, control)
        int_b = radius * (b @ weights)
        int_fb = radius * ((b * np.sin(np.pi*z/L)**2) @ weights)
        int_fn = radius * ((b * np.sin(2*np.pi*z/L)**2) @ weights)
        _, wl = profile(left, control)
        _, wr = profile(right, control)
        int_g = -np.log(wr/wl)
        int_h = (right-left)/L
        int_hosc = np.sin(2*np.pi*right/L) - np.sin(2*np.pi*left/L)
        return np.array([int_b, int_g, int_fb, int_fn, int_h, int_hosc]).T

    return t, dt, k, edges, integrals(edges[:-1], mids), integrals(mids, edges[1:])


def propagate(samples, epsilon, *, size=1024, half_window=64., step=0.005,
              control=0., batch=64):
    """One fixed physical initial field for every realization and control."""
    samples = np.atleast_2d(samples)
    t, dt, k, _, left, right = setup(size, half_window, step, control)
    fields = []
    k2 = k*k
    for start in range(0, len(samples), batch):
        xi = epsilon * samples[start:start+batch]
        u = np.broadcast_to(exact(t, 0, control), (len(xi), size)).copy()
        for l, r in zip(left, right):
            ib = l[0] + xi[:, 0]*l[2]
            ig = l[1] + xi[:, 2]*l[4] + xi[:, 3]*l[5]
            u = np.fft.ifft(np.fft.fft(u, axis=-1)
                * np.exp(-0.5j*ib[:, None]*k2 + 0.5*ig[:, None]), axis=-1)
            ing = l[0]+r[0] + xi[:, 1]*(l[3]+r[3])
            u *= np.exp(1j * ing[:, None] * np.abs(u)**2)
            ib = r[0] + xi[:, 0]*r[2]
            ig = r[1] + xi[:, 2]*r[4] + xi[:, 3]*r[5]
            u = np.fft.ifft(np.fft.fft(u, axis=-1)
                * np.exp(-0.5j*ib[:, None]*k2 + 0.5*ig[:, None]), axis=-1)
        fields.append(u)
    return t, dt, k, np.concatenate(fields)


def tangent(*, size=1024, half_window=64., step=0.005, control=0.):
    t, dt, k, _, left, right = setup(size, half_window, step, control)
    u = exact(t, 0, control)
    v = np.zeros((DIM, size), complex)
    k2 = k*k

    def linear(u, v, row):
        mult = np.exp(-0.5j*k2*row[0] + row[1]/2)
        u = np.fft.ifft(np.fft.fft(u)*mult)
        v = np.fft.ifft(np.fft.fft(v, axis=-1)*mult, axis=-1)
        v[0] += np.fft.ifft(np.fft.fft(u)*(-0.5j*k2*row[2]))
        v[2] += u*row[4]/2
        v[3] += u*row[5]/2
        return u, v

    for l, r in zip(left, right):
        u, v = linear(u, v, l)
        intensity = np.abs(u)**2
        gamma_int = l[0]+r[0]
        derivative_phase = 2*gamma_int*np.real(np.conj(u)*v)
        derivative_phase[1] += (l[3]+r[3])*intensity
        mult = np.exp(1j*gamma_int*intensity)
        v = mult*(v+1j*u*derivative_phase)
        u = mult*u
        u, v = linear(u, v, r)
    return t, dt, k, u, v


def tangent_rk4(step=0.0005, size=1024, half_window=64., control=0.):
    """Independent Duhamel integral using RK4 for the linear tangent PDE.

    The exact nominal field drives the source, and a Fourier integrating factor
    removes the stiff nominal dispersive term. This is not a differentiated split.
    """
    t, dt, k, _, _, _ = setup(size, half_window, step, control)
    k2 = k*k

    def multiplier(z):
        _, w = profile(z, control)
        return np.exp(-0.5j*k2*(w-1))/np.sqrt(w)

    def rhs(z, y):
        b, w = profile(z, control)
        u = exact(t, z, control)
        mult = multiplier(z)
        v = np.fft.ifft(y*mult, axis=-1)
        force = 1j*b*(2*np.abs(u)**2*v+u*u*np.conj(v))
        utt = np.fft.ifft(-k2*np.fft.fft(u))
        force[0] += 0.5j*b*np.sin(np.pi*z/L)**2*utt
        force[1] += 1j*b*np.sin(2*np.pi*z/L)**2*np.abs(u)**2*u
        force[2] += u/(2*L)
        force[3] += np.pi/L*np.cos(2*np.pi*z/L)*u
        return np.fft.fft(force, axis=-1)/mult

    y = np.zeros((DIM, size), complex)
    count = round(L/step)
    h = L/count
    for j in range(count):
        z = j*h
        a = rhs(z, y)
        b = rhs(z+h/2, y+h*a/2)
        c = rhs(z+h/2, y+h*b/2)
        d = rhs(z+h, y+h*c)
        y += h/6*(a+2*b+2*c+d)
    return np.fft.ifft(y*multiplier(L), axis=-1)


def response(u, v, t, dt, k):
    energy = dt*np.sum(np.abs(u)**2)
    norm2 = np.sum(np.abs(u)**2)
    coeff = (v @ np.conj(u))/norm2
    proj = v-coeff[:, None]*u
    gram = np.real(np.conj(proj) @ proj.T)/norm2
    va = v.copy()
    ut = np.fft.ifft(1j*k*np.fft.fft(u))
    va[0] += CARRIER*L/2*ut
    ca = (va @ np.conj(u))/norm2
    pa = va-ca[:, None]*u
    aligned = np.real(np.conj(pa) @ pa.T)/norm2
    energy_gradient = 2*dt*np.real(v @ np.conj(u))/energy
    tc = dt*np.sum(t*np.abs(u)**2)/energy
    centre_gradient = 2*dt*np.real((v*np.conj(u)) @ (t-tc))/energy
    variance_t = dt*np.sum((t-tc)**2*np.abs(u)**2)/energy
    width = np.sqrt(variance_t)
    width_gradient = dt*np.real((v*np.conj(u)) @ ((t-tc)**2-variance_t))/(energy*width)
    return {"gram":gram.tolist(), "aligned_gram":aligned.tolist(),
            "energy_gradient":energy_gradient.tolist(),
            "centre_gradient":centre_gradient.tolist(),
            "width_gradient":width_gradient.tolist(), "width":float(width),
            "raw_trace":float(np.trace(gram)), "aligned_trace":float(np.trace(aligned)),
            "aligned_eigenvalues":np.linalg.eigvalsh(aligned).tolist(),
            "leading_shares":(np.diag(aligned)/np.trace(aligned)).tolist()}


def quantities(u, t, dt, k, xi, epsilon, reference):
    energy = dt*np.sum(np.abs(u)**2, axis=-1)
    norm2 = np.sum(np.abs(u)**2, axis=-1)
    tc = dt*np.sum(t*np.abs(u)**2, axis=-1)/energy
    width = np.sqrt(dt*np.sum((t-tc[:,None])**2*np.abs(u)**2,axis=-1)/energy)
    refnorm = np.sum(np.abs(reference)**2)

    def loss(v):
        # Orthogonal residual is stable near F=1; avoid 1 - two nearly equal numbers.
        coeff = (v @ np.conj(reference))/refnorm
        orth = v-coeff[:,None]*reference
        return np.sum(np.abs(orth)**2,axis=-1)/np.sum(np.abs(v)**2,axis=-1)

    raw = loss(u)
    shift = epsilon*xi[:,0]*CARRIER*L/2
    aligned_u = np.fft.ifft(np.fft.fft(u,axis=-1)*np.exp(1j*shift[:,None]*k),axis=-1)
    aligned = loss(aligned_u)
    return {"energy":energy,"centre":tc,"width":width,"raw_loss":raw,"aligned_loss":aligned,
            "peak":np.max(np.abs(u)**2,axis=-1)}


def sx(x):
    return float(np.sinh(x)/x) if x else 1.


def validate():
    rows=[]
    for control in (0., -0.65, 0.65):
        for h in (0.01,0.005,0.0025):
            t,dt,k,u,v=tangent(step=h,control=control)
            e=np.linalg.norm(u-exact(t,L,control))/np.linalg.norm(u)
            resp=response(u,v,t,dt,k)
            rows.append({"control":control,"h":h,"nominal_l2":float(e),
                "aligned_trace":resp["aligned_trace"],
                "energy_gradient_error":float(np.max(np.abs(np.array(resp["energy_gradient"])-[0,0,1,0]))),
                "centre_gradient_error":float(np.max(np.abs(np.array(resp["centre_gradient"])-[CARRIER*L/2,0,0,0])))})
    t,dt,k,u,v=tangent(step=0.0025)
    vrk=tangent_rk4()
    err=np.linalg.norm(v-vrk,axis=1)/np.linalg.norm(vrk,axis=1)
    _,_,_,uf,vf=tangent(size=2048,half_window=80.,step=0.0025)
    tf,df,kf,_,_,_=setup(2048,80.,0.0025)
    r0=response(u,v,t,dt,k)
    rf=response(uf,vf,tf,df,kf)
    spatial=abs(r0['aligned_trace']-rf['aligned_trace'])/rf['aligned_trace']
    # Discrete tangent must match central finite differences of the propagation code.
    eye=np.eye(DIM)
    _,_,_,fields=propagate(np.concatenate([eye,-eye]),1e-4,step=0.0025)
    fd=(fields[:DIM]-fields[DIM:])/(2e-4)
    fd_error=np.linalg.norm(v-fd,axis=1)/np.linalg.norm(v,axis=1)
    assert max(err)<3e-4,("independent tangent",err)
    assert max(fd_error)<2e-6,("discrete tangent FD",fd_error)
    assert spatial<5e-5,("spatial domain",spatial)
    assert max(row['energy_gradient_error'] for row in rows)<2e-10
    assert max(row['centre_gradient_error'] for row in rows)<3e-5
    return {"convergence":rows,"rk4_tangent_relative_errors":err.tolist(),
            "finite_difference_tangent_errors":fd_error.tolist(),"spatial_trace_change":float(spatial)}


def design():
    out=[]
    for a in np.linspace(-0.65,0.65,27):
        t,dt,k,u,v=tangent(control=float(a))
        rr=response(u,v,t,dt,k)
        z=np.linspace(0,L,2001)
        b,w=profile(z,a)
        out.append({"control":float(a),"aligned_trace":rr['aligned_trace'],
                    "raw_trace":rr['raw_trace'],"max_loss_rate":float(np.max(b/w))})
    feasible=[r for r in out if r['max_loss_rate']<=1.25]
    best=min(feasible,key=lambda row:row['aligned_trace'])
    return {"grid":out,"best":best,"constraint":"0.35 <= beta=gamma <= 1.65; |g0| <= 1.25"}


def study(power=8, replicates=4, epsilons=(0.025,0.05,0.1,0.2)):
    start=perf_counter()
    ROOT.joinpath('results').mkdir(exist_ok=True)
    validation=validate()
    print('validation complete',flush=True)
    des=design()
    best=des['best']['control']
    print('selected design',best,flush=True)
    responses={}
    references={}
    for a in dict.fromkeys([0.,best]):
        t,dt,k,u,v=tangent(control=a,step=0.0025)
        references[str(a)]=u
        responses[str(a)]=response(u,v,t,dt,k)
    records=[]
    # Independent randomized Sobol replicates. Design used only tangent calculations.
    for rep in range(replicates):
        xi=2*qmc.Sobol(DIM,scramble=True,seed=20261119+rep).random_base2(power)-1
        for a in dict.fromkeys([0.,best]):
            for eps in epsilons:
                t,dt,k,u=propagate(xi,eps,control=a)
                out=quantities(u,t,dt,k,xi,eps,references[str(a)])
                eref=2/(1+L)*np.exp(eps*xi[:,2])
                tref=CARRIER*(L+eps*xi[:,0]*L/2)
                errE=float(np.max(np.abs(out['energy']/eref-1)))
                errT=float(np.max(np.abs(out['centre']-tref)))
                assert errE<2e-10,errE
                assert errT<5e-5,errT
                means={q:float(np.mean(val)) for q,val in out.items()}
                variances={q:float(np.var(val)) for q,val in out.items()}
                records.append({'rep':rep,'samples':len(xi),'control':a,'epsilon':eps,
                    'mean':means,'variance':variances,
                    'aligned_loss_q95':float(np.quantile(out['aligned_loss'],.95)),
                    'energy_law_max_error':errE,'centre_law_max_error':errT})
                np.savez_compressed(ROOT/'results'/f'samples_r{rep}_a{a:+.3f}_e{eps:.3f}.npz',
                                    xi=xi,**out)
                print(f'rep {rep} a {a:+.3f} epsilon {eps:g} aligned loss {means["aligned_loss"]:.6g}',flush=True)
    # Direct test: zero-integral gain perturbation, unchanged terminal energy.
    xi=np.zeros((41,DIM));xi[:,3]=np.linspace(-1,1,len(xi))
    eps=.2
    t,dt,k,u=propagate(xi,eps,control=0.)
    silent=quantities(u,t,dt,k,xi,eps,references['0.0'])
    # Additional PDE step refinement for worst fixed realization, independent of UQ sampling.
    corners=np.array([[1,1,1,1],[-1,1,-1,-1],[1,-1,-1,1]])
    refinement=[]
    for a in dict.fromkeys([0.,best]):
        _,_,_,coarse=propagate(corners,.2,control=a,step=.005)
        _,_,_,fine=propagate(corners,.2,control=a,step=.0025)
        refinement.append({'control':a,'relative_field_change':(np.linalg.norm(coarse-fine,axis=1)/np.linalg.norm(fine,axis=1)).tolist()})
    all_results={'parameters':{'L':L,'carrier':CARRIER,'dimension':DIM,'power':power,'replicates':replicates,
        'epsilons':list(epsilons),'N':1024,'half_window':64,'step':.005,'seed_base':20261119,
        'input_law':'independent uniform [-1,1], synthetic engineering tolerances'},
        'validation':validation,'design':des,'responses':responses,'records':records,
        'energy_silent':{'xi':xi[:,3].tolist(),**{q:val.tolist() for q,val in silent.items()}},
        'perturbed_refinement':refinement,'wall_seconds':perf_counter()-start}
    (ROOT/'results'/'study.json').write_text(json.dumps(all_results,indent=2),encoding='utf-8')
    make_figures(all_results)
    return all_results


def aggregate(res,a,eps,key='aligned_loss',kind='mean'):
    vals=np.array([r[kind][key] for r in res['records'] if np.isclose(r['control'],a) and r['epsilon']==eps])
    return float(vals.mean()),float(vals.std(ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else 0.


def make_figures(res):
    from figures import build
    build(res)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--power',type=int,default=8)
    p.add_argument('--replicates',type=int,default=4);p.add_argument('--figures-only',action='store_true')
    p.add_argument('--pilot',action='store_true');args=p.parse_args()
    if args.figures_only:make_figures(json.loads((ROOT/'results'/'study.json').read_text()))
    elif args.pilot:
        r=design();print(json.dumps(r,indent=2))
    else:
        r=study(args.power,args.replicates)
        print(json.dumps({'best':r['design']['best'],'wall_seconds':r['wall_seconds']},indent=2))
