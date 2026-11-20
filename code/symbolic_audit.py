"""Independent exact-arithmetic derivations; no numerical study is imported."""
import json
from pathlib import Path
import sympy as s

checks = []
def zero(name, expr):
    reduced = s.simplify(s.expand(expr).rewrite(s.exp))
    if reduced != 0:
        raise AssertionError((name, reduced))
    checks.append({'claim': name, 'residual': '0'})
    print(name + ': 0', flush=True)

z,t,x,y,e=s.symbols('z t tau Z epsilon', real=True)
A,W,B,Z,b,n,g=(s.Function(k,real=True)(z) for k in ('A','W','B','Z','beta','gamma','g'))
q=s.Function('Q')(x,y)
at={x:t/W,y:Z}
Q=q.subs(at,simultaneous=True)
theta=B*t*t/2
def residual(u,beta,gamma,gain):
    return s.I*s.diff(u,z)+beta*s.diff(u,t,2)/2+gamma*u**2*s.conjugate(u)-s.I*gain*u/2
u=A*s.exp(s.I*theta)*Q
R=residual(u,b,n,g)*s.exp(-s.I*theta)/A
collected=(s.I*s.diff(Z,z)*s.diff(q,y).subs(at,simultaneous=True)
  +b/(2*W**2)*s.diff(q,x,2).subs(at,simultaneous=True)+n*A*A*Q**2*s.conjugate(Q)
  +s.I*t*(b*B/W-s.diff(W,z)/W**2)*s.diff(q,x).subs(at,simultaneous=True)
  +(-t*t*(s.diff(B,z)+b*B*B)/2+s.I*(s.diff(A,z)/A+b*B/2-g/2))*Q)
zero('General transformed PDE by direct differentiation',R-collected)
r=s.Function('ratio',positive=True)(z)
D=s.Function('D',real=True)(z)
gstar=s.diff(r,z)/r-b/W
rules={B:1/W,A:s.sqrt(r)/W,n:b/r,g:gstar+D}
slopes={s.diff(W,z):b,s.diff(Z,z):b/W**2}
reduced=b/W**2*(s.I*s.diff(q,y).subs(at,simultaneous=True)
    +s.diff(q,x,2).subs(at,simultaneous=True)/2+Q**2*s.conjugate(Q))-s.I*D*Q/2
zero('Similarity reduction with arbitrary constraint defect',R.subs(rules,simultaneous=True).doit().subs(slopes)-reduced)

fb,fn,h,dw=(s.Function(k,real=True)(z) for k in ('fb','fn','h','dw'))
be=b*(1+e*fb); ne=n*(1+e*fn); we=W+e*dw
de=g+e*h-(s.diff(be,z)/be-s.diff(ne,z)/ne-be/we)
slope=s.diff(de,e).subs(e,0)
zero('First variation of multiplicative constraint defect',slope-(h-s.diff(fb,z)+s.diff(fn,z)+b/W*(fb-dw/W)))

a=s.Function('a',positive=True)(z)
U=s.Function('u')(z,t)
zero('Exact amplitude gauge transformation',residual(a*U,b,n/a**2,g+2*s.diff(a,z)/a)-a*residual(U,b,n,g))

# Independent formal jet variables: U and V are complex; barred jets are separate.
u,v,ub,vb=s.symbols('u v ubar vbar')
zero('Cubic directional expansion',(u+e*v)**2*(ub+e*vb)-u*u*ub-e*(2*u*ub*v+u*u*vb)
  -e**2*(v*v*ub+2*u*v*vb)-e**3*v*v*vb)

# The local conservation identities are differentiated before integrating on R.
U=s.Function('u')(z,t); V=s.Function('ubar')(z,t)
rhs=s.I*b*s.diff(U,t,2)/2+s.I*n*U*U*V+g*U/2
rhsb=-s.I*b*s.diff(V,t,2)/2-s.I*n*V*V*U+g*V/2
rho=U*V
p=(V*s.diff(U,t)-U*s.diff(V,t))/(2*s.I)
fluxp=b/2*(s.diff(U,t)*s.diff(V,t)-(V*s.diff(U,t,2)+U*s.diff(V,t,2))/2)-n*rho**2/2
def zrule(expr):
    return expr.doit().subs({s.diff(U,z):rhs,s.diff(V,z):rhsb,
       s.diff(U,z,t):s.diff(rhs,t),s.diff(V,z,t):s.diff(rhsb,t)})
zero('Local energy balance',zrule(s.diff(rho,z))+b*s.diff(p,t)-g*rho)
zero('Local momentum balance',zrule(s.diff(p,z))+s.diff(fluxp,t)-g*p)
zero('Local first-moment balance',zrule(s.diff(t*rho,z))+s.diff(b*t*p,t)-b*p-g*t*rho)

L,alpha,c=s.symbols('L alpha c',positive=True)
bb=1+alpha*s.sin(2*s.pi*z/L)
ww=1+z+alpha*L*(1-s.cos(2*s.pi*z/L))/(2*s.pi)
zero('Schedule W derivative',s.diff(ww,z)-bb)
zero('Common endpoint width',ww.subs(z,L)-(1+L))
zero('Dispersion error integral independent of schedule',s.integrate(bb*s.sin(s.pi*z/L)**2,(z,0,L))-L/2)
zero('Zero-integral gain mode',s.integrate(2*s.pi/L*s.cos(2*s.pi*z/L),(z,0,L)))
eta,vel=s.symbols('eta vel',positive=True)
S=eta*s.sech(eta*(x-vel*y))*s.exp(s.I*(vel*x+(eta**2-vel**2)*y/2))
zero('Nominal focusing NLSE soliton',s.I*s.diff(S,y)+s.diff(S,x,2)/2+S*S*s.conjugate(S))

xx,k=s.symbols('xi k',real=True,nonzero=True)
zero('Uniform exponential moment',s.integrate(s.exp(k*xx)/2,(xx,-1,1))-(s.exp(k)-s.exp(-k))/(2*k))
zero('Uniform second moment',s.integrate(xx**2/2,(xx,-1,1))-s.Rational(1,3))
zero('Uniform fourth moment',s.integrate(xx**4/2,(xx,-1,1))-s.Rational(1,5))
xis=s.symbols('xi0:4',real=True)
H=s.Matrix(4,4,lambda i,j:s.Symbol('H'+str(min(i,j))+str(max(i,j)),real=True))
quad=(s.Matrix(xis).T*H*s.Matrix(xis))[0]
def uniform_expect(poly):
    ans=0
    for powers,coef in s.Poly(poly,*xis).terms():
        moment=s.prod(0 if power%2 else s.Rational(1,power+1) for power in powers)
        ans+=coef*moment
    return s.expand(ans)
mean=s.trace(H)/3
variance=s.Rational(4,45)*sum(H[i,i]**2 for i in range(4))+s.Rational(4,9)*sum(H[i,j]**2 for i in range(4) for j in range(i+1,4))
zero('Four-dimensional uniform quadratic mean',uniform_expect(quad)-mean)
zero('Four-dimensional uniform quadratic variance',uniform_expect(quad**2)-mean**2-variance)

# Fidelity is also proved in a general complex inner-product space by Lean.
nr,ns,ip=s.symbols('normR2 normS2 innerAbs2',positive=True)
zero('Projection fidelity scalar identity',1-ip/(ns*(nr+ip/ns))-nr/(nr+ip/ns))
out={'sympy':s.__version__,'checks':checks,'passed':len(checks),
 'scope':'Exact local identities; decay, PDE solution regularity and asymptotic remainder assumptions remain analytic.'}
dest=Path(__file__).resolve().parents[1]/'results'
dest.mkdir(exist_ok=True)
(dest/'symbolic_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
