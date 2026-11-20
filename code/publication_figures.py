"""Publication figures for the managed-pulse and physical-scale studies."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT/'results'
OUT = ROOT/'figures'
BLUE = '#185580'
TEAL = '#008175'
ORANGE = '#C65321'
INK = '#152A3B'
MUTE = '#637784'
GRID = '#E3EBEF'
WHITE = '#FFFFFF'


def load(name):
    return json.loads((R/name).read_text(encoding='utf-8'))


def panels(height=4.55):
    return plt.subplots(2,2,figsize=(6.65,height),layout='constrained')


def head(ax,letter,title):
    ax.set_title(f'({letter})  {title}',loc='left',fontsize=10,fontweight='medium',pad=8)


def axes(ax,x,y,*,grid=True):
    ax.set(xlabel=x,ylabel=y)
    if grid:
        ax.grid(axis='y',color=GRID,lw=.65)
    ax.set_axisbelow(True)


def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),dpi=380,facecolor=WHITE)
    fig.savefig(OUT/(name+'.png'),dpi=600,facecolor=WHITE)
    plt.close(fig)


def bar(fig,ax,colour,label,ticks=None):
    cb=fig.colorbar(colour,ax=ax,fraction=.053,pad=.025,ticks=ticks)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2,labelsize=7.3)
    cb.set_label(label,labelpad=5,fontsize=8)
    return cb


def terms(ax,items,loc='best',**kwargs):
    ax.legend(handles=items,loc=loc,frameon=False,labelspacing=.45,**kwargs)


def synthetic_mechanism(study,raw):
    fig,aa=panels()
    a,b,c,d=aa.ravel()
    z,t=raw['propagation_z'],raw['propagation_t']
    int0=raw['propagation_nominal'].astype(float)
    diff=raw['propagation_difference'].astype(float)
    log_ratio=.2*np.sin(2*np.pi*z/3)
    shape=(int0+diff)*np.exp(-log_ratio[:,None])-int0
    area=np.abs(t)<=12
    im=a.pcolormesh(t[area],z,int0[:,area],shading='nearest',cmap='cividis',
                    vmin=0,vmax=1,rasterized=True)
    centre=.3*z
    halfwidth=(1+z)*np.arccosh(np.sqrt(2))
    a.plot(centre-halfwidth,z,ls='--',color=WHITE,lw=.8)
    a.plot(centre+halfwidth,z,ls='--',color=WHITE,lw=.8)
    bar(fig,a,im,r'$|u_0|^2$',[0,.25,.5,.75,1])
    head(a,'a','Nominal pulse propagation')
    axes(a,'Retarded time, $t$','Distance, $z$',grid=False)
    lim=1e3*np.max(np.abs(shape))
    im=b.pcolormesh(t[area],z,1e3*shape[:,area],shading='nearest',cmap='RdBu_r',
                    norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),rasterized=True)
    b.contour(t[area],z,1e3*shape[:,area],levels=np.array([-.66,-.33,.33,.66])*lim,
              colors=INK,linewidths=.38,alpha=.36)
    bar(fig,b,im,r'Intensity change ($\times 10^{-3}$)')
    head(b,'b','Energy-scaled intensity change')
    axes(b,'Retarded time, $t$','Distance, $z$',grid=False)
    for ax in (a,b): ax.set(xlim=(-12,12),ylim=(0,3),xticks=[-12,-6,0,6,12],yticks=[0,1,2,3])

    p=c.plot(z,np.exp(log_ratio),color=ORANGE,label=r'$\epsilon\xi_4=+0.2$')[0]
    m=c.plot(z,np.exp(-log_ratio),color=BLUE,label=r'$\epsilon\xi_4=-0.2$')[0]
    c.axhline(1,color=MUTE,lw=.85,ls='--')
    c.scatter([3],[1],s=19,facecolor=WHITE,edgecolor=INK,zorder=5,clip_on=False)
    terms(c,[p,m],loc='upper center',ncol=2,columnspacing=1.1)
    head(c,'c','Energy during propagation')
    axes(c,'Distance, $z$',r'$E(z)/E_0(z)$')
    c.set(xlim=(0,3),xticks=[0,1,2,3],ylim=(.77,1.35))
    s=study['energy_silent']
    x=.2*np.asarray(s['xi'])
    h44=study['responses']['0.0']['aligned_gram'][3][3]
    p=d.plot(x,1e4*np.asarray(s['aligned_loss']),color=TEAL,label='Direct propagation')[0]
    q=d.plot(x,1e4*x*x*h44,color=MUTE,ls='--',label='Quadratic response')[0]
    d.scatter(x[::10],1e4*np.asarray(s['aligned_loss'])[::10],s=11,color=TEAL,zorder=3)
    terms(d,[p,q],loc='upper center')
    head(d,'d','Loss at fixed output energy')
    axes(d,r'Gain error amplitude, $\epsilon\xi_4$',r'Infidelity ($\times 10^{-4}$)')
    d.set(ylim=(0,5.2),xticks=[-.2,-.1,0,.1,.2])
    save(fig,'managed_energy_waveform')


def quadratic(study,raw,extension):
    fig,aa=panels()
    a,b,c,d=aa.ravel()
    eps=np.asarray(study['parameters']['epsilons'])
    chosen=study['design']['best']['control']
    legend=[]
    for control,color,marker,lbl in [(0.,BLUE,'o','Baseline'),(chosen,ORANGE,'s','Selected')]:
        samples=np.array([[row['mean']['aligned_loss'] for row in study['records']
                            if row['control']==control and row['epsilon']==e] for e in eps])
        scale=eps**2
        ebar=a.errorbar(eps,samples.mean(axis=1)/scale,
                        yerr=2*samples.std(axis=1,ddof=1)/np.sqrt(samples.shape[1])/scale,
                        color=color,marker=marker,ms=4,capthick=.8,capsize=2,label=lbl)
        h=np.trace(study['responses'][str(control)]['aligned_gram'])/3
        a.axhline(h,lw=.9,color=color,ls='--',alpha=.72)
        legend.append(ebar)
    terms(a,legend,loc='lower left')
    head(a,'a','Range of the quadratic mean')
    axes(a,r'Uncertainty amplitude, $\epsilon$',r'$\mathbb{E}[1-F]/\epsilon^2$')
    a.set(xticks=eps,xticklabels=['.025','.05','.10','.20'],ylim=(.352,.412))

    H=np.asarray(study['responses']['0.0']['aligned_gram'])
    sd=np.sqrt(np.diag(H))
    corr=H/(sd[:,None]*sd[None,:])
    # All off-diagonal correlations are positive in this specific experiment.
    palette=plt.get_cmap('Blues')
    im=b.imshow(corr,cmap=palette,vmin=0,vmax=1,aspect='auto',interpolation='none')
    for i in range(4):
        for j in range(4):
            b.text(j,i,f'{corr[i,j]:.2f}',ha='center',va='center',
                   color=WHITE if corr[i,j]>.72 else INK,fontsize=8.2)
    for pos in [-.5,.5,1.5,2.5,3.5]:
        b.axhline(pos,color=WHITE,lw=.65)
        b.axvline(pos,color=WHITE,lw=.65)
    b.set(xticks=range(4),yticks=range(4),xticklabels=[r'$\beta$',r'$\gamma$',r'$h_0$',r'$h_1$'],
          yticklabels=[r'$\beta$',r'$\gamma$',r'$h_0$',r'$h_1$'])
    b.tick_params(length=0)
    for spine in b.spines.values(): spine.set_visible(False)
    bar(fig,b,im,'Normalized inner product',[0,.5,1])
    head(b,'b','Projected response correlations')
    x=raw['risk_grid']
    loss=1e3*raw['risk_direct']
    q=1e3*raw['risk_quadratic']
    im=c.pcolormesh(x,x,loss.T,shading='nearest',cmap='cividis',vmin=0,
                    vmax=float(np.max(loss)),rasterized=True)
    xg,yg=np.meshgrid(x,x)
    c.contour(xg,yg,q.T,levels=[.1,1,3,5,8,11],colors=WHITE,linewidths=.65,
              linestyles='dashed',alpha=.85)
    bar(fig,c,im,r'Direct loss ($\times 10^{-3}$)',[0,3,6,9,12])
    head(c,'c','Direct loss, quadratic contours')
    error=100*np.divide(loss-q,q,where=q>0,out=np.full(q.shape,np.nan))
    emax=13
    cmap=plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#CFD6D9')
    im=d.pcolormesh(x,x,error.T,shading='nearest',cmap=cmap,
                    norm=TwoSlopeNorm(vmin=-emax,vcenter=0,vmax=emax),rasterized=True)
    d.contour(xg,yg,error.T,levels=[0],colors=INK,linewidths=.65)
    x0,y0=extension['max_position']
    d.scatter([x0],[y0],s=32,marker='o',facecolors='none',edgecolors=INK,lw=1,clip_on=False)
    d.annotate(f"{100*extension['max_relative']:+.2f}%",(x0,y0),xytext=(.35,.73),
               ha='left',fontsize=8,arrowprops={'arrowstyle':'-','color':INK,'lw':.7},
               bbox={'facecolor':WHITE,'edgecolor':'none','alpha':.88,'pad':1.5})
    bar(fig,d,im,'Relative error (%)',[-12,-6,0,6,12])
    head(d,'d','Relative prediction error')
    for ax in (c,d):
        axes(ax,r'Dispersion error, $\xi_1$',r'Gain redistribution, $\xi_4$',grid=False)
        ax.set(xlim=(-1,1),ylim=(-1,1),xticks=[-1,-.5,0,.5,1],yticks=[-1,-.5,0,.5,1])
    save(fig,'quadratic_risk')


def schedule(study,extension,summary):
    fig=plt.figure(figsize=(6.65,4.2),layout='constrained')
    gs=fig.add_gridspec(2,2,height_ratios=[1.18,.83])
    a=fig.add_subplot(gs[0,0]); b=fig.add_subplot(gs[0,1]); c=fig.add_subplot(gs[1,:])
    best=study['design']['best']['control']
    grid=study['design']['grid']
    xx=[r['control'] for r in grid]
    jj=[r['aligned_trace']/3 for r in grid]
    a.plot(xx,jj,color=BLUE,lw=1.7)
    a.scatter(xx,jj,s=7,color=BLUE)
    a.scatter([best],[study['design']['best']['aligned_trace']/3],facecolor=ORANGE,
              edgecolor=WHITE,linewidth=.6,s=45,zorder=5)
    a.scatter([0],[study['responses']['0.0']['aligned_trace']/3],facecolor=BLUE,
              edgecolor=WHITE,linewidth=.6,s=35,zorder=5)
    head(a,'a','Objective on 27 schedules')
    axes(a,r'Schedule parameter, $a$',r'$\mathrm{tr}(H^{\mathrm{al}})/3$')
    a.set(xticks=[-.65,-.325,0,.325,.65],xticklabels=['−0.65','−0.325','0','0.325','0.65'])
    changes=[]
    for i in range(study['parameters']['replicates']):
        with np.load(R/f'samples_r{i}_a+0.000_e0.100.npz') as a0, np.load(R/f'samples_r{i}_a{best:+.3f}_e0.100.npz') as a1:
            changes.extend(a0['aligned_loss']-a1['aligned_loss'])
    changes=np.sort(np.asarray(changes)*1e4)
    probability=(np.arange(len(changes))+1)/len(changes)
    b.plot(changes,probability,color=TEAL)
    b.axvline(0,color=MUTE,ls='--',lw=.8)
    frac=np.mean(changes>0)
    ymax=1-frac
    b.fill_between(changes,probability,1,where=changes>=0,color='#E7F3EF',zorder=-1)
    b.scatter([0],[ymax],s=16,color=TEAL,zorder=3)
    b.text(.965,.12,f'{frac:.1%} improve',ha='right',transform=b.transAxes,fontsize=8,color=TEAL)
    head(b,'b','Paired loss decrease at $\\epsilon=0.1$')
    axes(b,r'Baseline minus selected ($\times 10^{-4}$)','Cumulative probability')
    b.set(ylim=(0,1.02),yticks=[0,.25,.5,.75,1])

    xx=np.arange(3)
    yy=[100*extension['redistribution_change'],
        -summary['paired_improvement_at_0.1']['relative_percent'],
        100*extension['net_gain_change']]
    labs=['Redistribution only\n(direct)', 'Four modes\n(direct)', 'Net gain only\n(direct)']
    c.axvline(0,color=MUTE,lw=.85)
    c.barh(xx,yy,color=[TEAL,TEAL,ORANGE],height=.52,alpha=.9)
    for i,y in enumerate(yy):
        c.text(y+(-2.5 if y<0 else 2.5),i,f'{y:+.1f}%',va='center',
               ha='right' if y<0 else 'left',color=TEAL if y<0 else ORANGE,fontsize=8)
    c.set(yticks=xx,yticklabels=labs,xlim=(-98,110),ylim=(-.55,2.55),xticks=[-75,-50,-25,0,25,50,75,100])
    c.tick_params(axis='y',length=0)
    c.spines['left'].set_visible(False)
    head(c,'c','Direct mean loss by error source')
    axes(c,'Change in mean loss from baseline (%)','',grid=False)
    c.grid(axis='x',lw=.65,color=GRID)
    save(fig,'schedule_tradeoffs')


def fibre(case,data):
    fig,aa=panels()
    a,b,c,d=aa.ravel()
    z=data['z_km']
    t=data['t_ps']
    profile=case['loss_db_per_km']
    swing=case['tolerance'][3]
    curves=[]
    for sign,col in [(-1,BLUE),(0,MUTE),(1,ORANGE)]:
        curve=a.plot(z,profile*(1+sign*swing*np.cos(2*np.pi*z/z[-1])),
                    color=col,ls='--' if sign==0 else '-',label=rf'$\xi_4={sign:+d}$' if sign else 'Nominal')[0]
        curves.append(curve)
    terms(a,curves,loc='upper center',ncol=3,columnspacing=0.9,handlelength=1.6)
    head(a,'a','Profiles at fixed total attenuation')
    axes(a,'Distance (km)','Attenuation (dB/km)')
    a.set(xlim=(0,12),xticks=[0,3,6,9,12],ylim=(.18,.398))

    fields=data['fields']
    p0=case['nominal_input_peak_W']
    nominal=np.abs(fields[1])**2
    shifted=np.abs(fields[2])**2
    area=np.abs(t)<=52
    loge=-(np.log(10)/10)*profile*swing*z[-1]/(2*np.pi)*np.sin(2*np.pi*z/z[-1])
    dzmap=1e3*p0*(shifted*np.exp(-loge[:,None])-nominal)
    bound=np.max(np.abs(dzmap))
    im=b.pcolormesh(z,t[area],dzmap[:,area].T,shading='nearest',rasterized=True,
                    cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound))
    bar(fig,b,im,'Power difference (mW)')
    b.set(xlim=(0,12),ylim=(-52,52),xticks=[0,3,6,9,12],yticks=[-50,-25,0,25,50])
    head(b,'b','Energy-scaled power change')
    axes(b,'Distance (km)','Retarded time (ps)',grid=False)

    x=data['gain_sweep']
    y=data['sweep_loss']
    exp=int(np.floor(np.log10(np.max(y))))
    m=10.**(-exp)
    l=c.plot(x,m*y,color=TEAL,label='Direct propagation')[0]
    H=np.asarray(case['H'])
    q=c.plot(x,m*H[3,3]*x*x,color=MUTE,ls='--',label='Quadratic response')[0]
    c.scatter(x[::10],m*y[::10],color=TEAL,s=10,zorder=4)
    terms(c,[l,q],loc='upper center')
    c.set(xlim=(-1.03,1.03),xticks=[-1,-.5,0,.5,1],ylim=(0,m*max(y)*1.37))
    head(c,'c',f"Waveform loss at {case['nominal_output_energy_pJ']:.2f} pJ")
    axes(c,r'Loss redistribution, $\xi_4$',rf'Infidelity ($\times 10^{{{exp}}}$)')

    direct=data['direct']; tangent=data['quadratic']
    lim=(min(tangent.min(),direct.min())*.85,max(tangent.max(),direct.max())*1.22)
    d.plot(lim,lim,color=MUTE,ls='--',lw=.9)
    d.scatter(tangent,direct,alpha=.46,color=BLUE,s=9,ec='none',rasterized=True)
    head(d,'d','Quadratic prediction, 512 samples')
    axes(d,'Quadratic infidelity','Direct infidelity')
    d.set(xscale='log',yscale='log',xlim=lim,ylim=lim)
    d.text(.97,.08,'Dashed: equality',transform=d.transAxes,ha='right',color=MUTE,fontsize=8)
    save(fig,'dimensioned_fiber')


def main():
    plt.rcParams.update({
        'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','font.size':8.2,
        'axes.labelsize':8.6,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.65,'axes.edgecolor':MUTE,'axes.labelcolor':INK,
        'text.color':INK,'xtick.color':INK,'ytick.color':INK,'xtick.labelsize':7.7,
        'ytick.labelsize':7.7,'lines.linewidth':1.45,'legend.fontsize':7.3,
        'xtick.major.size':3,'ytick.major.size':3,'xtick.major.width':.65,
        'ytick.major.width':.65,'pdf.fonttype':42,'ps.fonttype':42,
    })
    study=load('study.json')
    extension=load('extensions.json')
    with np.load(R/'relief_data.npz') as raw:
        synthetic_mechanism(study,raw)
        quadratic(study,raw,extension)
    schedule(study,extension,load('summary.json'))
    with np.load(R/'fiber.npz') as raw:
        fibre(load('fiber.json'),raw)
    print('Created four publication figures as PDF and 600 dpi PNG.')


if __name__=='__main__':
    main()
