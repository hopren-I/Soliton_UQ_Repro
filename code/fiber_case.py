"""Dimensioned 1550 nm G.652-style fiber experiment with independent RK4 checks."""
from pathlib import Path
import json

import numpy as np
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[1]
C = 299792458.0
WAVELENGTH_NM = 1550.0
DISPERSION = 17.0                       # ps / (nm km), ITU-T G.652 (2024), Table I.1
LOSS_DB = 0.275                          # dB / km, typical link, same table
BETA2 = -WAVELENGTH_NM**2*1e-9/(2*np.pi*C)*DISPERSION*1e12  # ps^2/km
GAMMA = 1.30                             # W^-1 km^-1, representative silica SMF
T0 = 8.0                                # ps, sech width parameter
LENGTH = 12.0                           # km
LD = T0**2/abs(BETA2)
P0 = abs(BETA2)/(GAMMA*T0**2)
X = LENGTH/LD
LOSS = LOSS_DB*np.log(10)/10              # power attenuation, km^-1
TOL = np.array([.02,.05,.03,.25])         # synthetic, no population inference
GRID = 1024
BOUND = 32.


def grid(size, bound):
    tau = np.linspace(-bound,bound,size,endpoint=False)
    k = 2*np.pi*np.fft.fftfreq(size,tau[1]-tau[0])
    return tau,k,1/np.cosh(tau)


def primitives(x, xi):
    """Integrals in x=z/LD of dispersion, power gain and Kerr coefficient."""
    d = np.asarray(xi)*TOL
    phi = 2*np.pi*x/X
    beta = x*(1+d[:,0])
    gain = -LOSS*LD*(x*(1+d[:,2])+d[:,3]*X/(2*np.pi)*np.sin(phi))
    gamma = x*(1+d[:,1])
    return beta,gain,gamma


def solve(xi, *, steps=800, size=GRID, bound=BOUND, snapshots=0, batch=48):
    """Symmetric Fourier splitting with exact coefficient integrals."""
    if snapshots and (steps%snapshots or len(xi)>batch):
        raise ValueError('Use one batch and an evenly divisible snapshot count.')
    tau,k,u0 = grid(size,bound)
    edges = np.linspace(0,X,steps+1)
    fields = []
    for start in range(0,len(xi),batch):
        chunk = xi[start:start+batch]
        u = np.broadcast_to(u0,(len(chunk),size)).astype(complex).copy()
        trace = [u.copy()] if snapshots else []
        left = primitives(0.,chunk)
        for i in range(steps):
            right = primitives(edges[i+1],chunk)
            mid = primitives((edges[i]+edges[i+1])/2,chunk)
            for begin,end in [(left,mid)]:
                fac = -.5j*(end[0]-begin[0])[:,None]*k**2+.5*(end[1]-begin[1])[:,None]
                u = np.fft.ifft(np.fft.fft(u,axis=1)*np.exp(fac),axis=1)
            u *= np.exp(1j*(right[2]-left[2])[:,None]*np.abs(u)**2)
            fac = -.5j*(right[0]-mid[0])[:,None]*k**2+.5*(right[1]-mid[1])[:,None]
            u = np.fft.ifft(np.fft.fft(u,axis=1)*np.exp(fac),axis=1)
            left = right
            if snapshots and (i+1)%(steps//snapshots)==0:
                trace.append(u.copy())
        fields.append(u)
    zz = np.linspace(0,LENGTH,snapshots+1) if snapshots else None
    return tau,np.concatenate(fields),zz,np.stack(trace,axis=1) if snapshots else None


def independent_rk4(xi, *, steps=3200, size=GRID, bound=BOUND, compensate=False):
    """Fourier interaction-picture RK4 on the full field with optional gauge test."""
    tau,k,u0 = grid(size,bound)
    d = xi*TOL
    h = X/steps

    def factors(x):
        b,g,_ = primitives(x,xi)
        p = np.exp(-.5j*b[:,None]*k**2+.5*g[:,None])
        n = 1+d[:,1]
        if compensate:
            extra_gain = -LOSS*LD*d[:,3]*X/(2*np.pi)*np.sin(2*np.pi*x/X)
            n = n*np.exp(-extra_gain)
        return p,n

    def rhs(y,p,n):
        u = np.fft.ifft(y*p,axis=1)
        return 1j*n[:,None]*np.fft.fft(np.abs(u)**2*u,axis=1)/p

    y = np.broadcast_to(np.fft.fft(u0),(len(xi),size)).copy()
    left = factors(0.)
    for i in range(steps):
        mid = factors((i+.5)*h)
        right = factors((i+1)*h)
        a = rhs(y,*left)
        b = rhs(y+h*a/2,*mid)
        c = rhs(y+h*b/2,*mid)
        d4 = rhs(y+h*c,*right)
        y += h/6*(a+2*b+2*c+d4)
        left = right
    return np.fft.ifft(y*left[0],axis=1)


def risk(u,s):
    projection = (u@np.conj(s))/np.vdot(s,s)
    return np.sum(np.abs(u-projection[:,None]*s)**2,axis=1)/np.sum(np.abs(u)**2,axis=1)


def summarize():
    # Refined central tangents on a common physical input and nominal output.
    eye = np.eye(4)
    norm_step = 1e-3
    probes = np.vstack((np.zeros((1,4)),eye*norm_step,-eye*norm_step))
    tau,fields,_,_ = solve(probes,steps=1600)
    reference = fields[0]
    v = (fields[1:5]-fields[5:])/(2*norm_step)
    orth = v-((v@np.conj(reference))/np.vdot(reference,reference))[:,None]*reference
    H = (np.conj(orth)@orth.T).real/np.vdot(reference,reference).real
    mean_quad = np.trace(H)/3

    # Four independently scrambled nets. The sweep is reserved for diagnostics.
    xi = np.vstack([2*qmc.Sobol(d=4,scramble=True,seed=20261120+j).random_base2(7)-1 for j in range(4)])
    _,samples,_,_ = solve(xi)
    losses = risk(samples,reference)
    loss_quad = np.einsum('ni,ij,nj->n',xi,H,xi)
    dt = tau[1]-tau[0]
    energies = P0*T0*dt*np.sum(np.abs(samples)**2,axis=1)   # W ps = pJ
    centres = T0*np.sum(tau*np.abs(samples)**2,axis=1)/np.sum(np.abs(samples)**2,axis=1)
    expected = 2*P0*T0*np.exp(-LOSS*LENGTH*(1+TOL[2]*xi[:,2]))
    replicate = losses.reshape(4,128).mean(axis=1)

    sweep = np.zeros((41,4))
    sweep[:,3] = np.linspace(-1,1,len(sweep))
    _,sweep_fields,_,_ = solve(sweep,steps=1600)
    scan = risk(sweep_fields,reference)
    _,profiles,z,snapshots = solve(sweep[[0,20,40]],steps=1600,snapshots=100)
    independent = independent_rk4(sweep[[0,20,40]])
    rk4_check = np.linalg.norm(profiles-independent,axis=1)/np.linalg.norm(independent,axis=1)
    half = independent_rk4(sweep[[0,20,40]],steps=6400)
    rk4_refine = np.linalg.norm(independent-half,axis=1)/np.linalg.norm(half,axis=1)
    compensated = independent_rk4(sweep[[0,40]],compensate=True)
    gauge_check = np.linalg.norm(compensated-independent[1],axis=1)/np.linalg.norm(independent[1])
    representative = np.vstack((sweep[[0,20,40]],[[1.,-1.,1.,-1.],[-1.,1.,-1.,1.]]))
    _,coarse,_,_ = solve(representative)
    _,fine,_,_ = solve(representative,steps=1600)
    _,space,_,_ = solve(representative,steps=1600,size=2048,bound=48.)
    t2,_,_ = grid(2048,48.)
    # Compare both grids after interpolation of the high-resolution Fourier field.
    global_fourier = np.fft.fft(space,axis=1)/len(t2)
    k2 = 2*np.pi*np.fft.fftfreq(len(t2),t2[1]-t2[0])
    spatial_on_base = global_fourier@np.exp(1j*k2[:,None]*(tau-t2[0]))
    time_err = np.linalg.norm(fine-coarse,axis=1)/np.linalg.norm(fine,axis=1)
    spatial_err = np.linalg.norm(fine-spatial_on_base,axis=1)/np.linalg.norm(fine,axis=1)
    energy_error = np.max(abs(energies/expected-1))
    assert energy_error<1e-10 and max(abs(centres))<1e-8
    assert max(rk4_check)<5e-6 and max(rk4_refine)<1e-8 and max(gauge_check)<1e-8
    assert max(time_err)<1e-5 and max(spatial_err)<1e-6

    result = dict(wavelength_nm=WAVELENGTH_NM,dispersion=DISPERSION,beta2=BETA2,
                  gamma=GAMMA,loss_db_per_km=LOSS_DB,length_km=LENGTH,t0_ps=T0,
                  dispersion_length_km=LD,nominal_input_peak_W=P0,input_fwhm_ps=2*np.arccosh(np.sqrt(2))*T0,
                  nominal_output_energy_pJ=2*P0*T0*np.exp(-LOSS*LENGTH),input_energy_pJ=2*P0*T0,
                  tolerance=TOL.tolist(),H=H.tolist(),direct_mean=float(np.mean(losses)),
                  direct_SE=float(np.std(replicate,ddof=1)/2),quad_mean=float(mean_quad),
                  replicate_means=replicate.tolist(),gain_only_max_loss=float(max(scan)),
                  maximum_energy_relative_error=float(energy_error),maximum_centroid_ps=float(max(abs(centres))),
                  maximum_independent_RK4_field_difference=float(max(rk4_check)),
                  maximum_RK4_step_halving_difference=float(max(rk4_refine)),
                  maximum_split_step_halving_difference=float(max(time_err)),
                  maximum_spatial_difference=float(max(spatial_err)),
                  maximum_gauge_field_difference=float(max(gauge_check)))
    (ROOT/'results'/'fiber.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    keep = np.abs(tau*T0)<=64.
    np.savez_compressed(ROOT/'results'/'fiber.npz',xi=xi,energy=energies,centroid_ps=centres,
                        direct=losses,quadratic=loss_quad,gain_sweep=sweep[:,3],sweep_loss=scan,
                        z_km=z,t_ps=tau[keep]*T0,fields=snapshots[:,:,keep],
                        terminal_profiles=profiles[:,keep],sweep_energy=P0*T0*dt*np.sum(np.abs(sweep_fields)**2,axis=1))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    summarize()
