"""FP32 Lorentz optimizer; curvature derivatives include the point homothety."""
import math
import torch
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import lorentz as L

class Geometry(torch.nn.Module):
    def __init__(self, shapes, method, device, lr=.1, rho=.001, beta=.9, gamma=1., radius=4., init_K=1., project_tangent=True, log_clip=.01, curvature_every=1, k_min=None):
        super().__init__()
        self.method,self.lr,self.rho,self.beta,self.gamma,self.radius=method,lr,rho,beta,gamma,radius
        self.project_tangent=project_tangent; self.log_clip=log_clip; self.curvature_every=curvature_every; self.epoch=0; self.last_curv_epoch=-1
        self.K_min=min(.1,.1*init_K) if k_min is None else k_min
        if not 0 < self.K_min < init_K:raise ValueError('Require 0 < K_min < initial K')
        c0=math.log(math.expm1(init_K-self.K_min))
        self.c=torch.nn.Parameter(torch.full((len(shapes),),c0,device=device),requires_grad=method in ['B4','B5','A5','shared'])
        self.register_buffer('point_K',torch.full((len(shapes),),init_K,device=device))
        self.points=torch.nn.ParameterList()
        for n,d in shapes:
            v=torch.cat([torch.zeros(n,1,device=device),.03*torch.randn(n,d,device=device)],-1)
            self.points.append(torch.nn.Parameter(L.exp_map(L.origin(d,init_K,device=device).expand(n,d+1),v,init_K)))
        self.m=[torch.zeros_like(p) for p in self.points]
        self.last={}
    def K(self):
        k=self.K_min+F.softplus(self.c)
        return k.mean().expand_as(k) if self.method=='shared' else k
    def chart_points(self):
        k=self.K()
        # At current K the forward value equals p; d/dK follows x(K)=q/sqrt(K).
        return [p*torch.sqrt(self.point_K[i]/k[i]) for i,p in enumerate(self.points)]
    @torch.no_grad()
    def step(self):
        ks=self.K().detach(); stats=[]
        newc=self.c.detach().clone()
        if self.c.grad is not None and (self.curvature_every==1 or ((self.epoch+1)%self.curvature_every==0 and self.last_curv_epoch!=self.epoch)):
            newc-=self.rho*self.c.grad
            self.last_curv_epoch=self.epoch
        nk=self.K_min+F.softplus(newc)
        if self.log_clip is not None:nk=torch.maximum(torch.minimum(nk,ks*math.exp(self.log_clip)),ks*math.exp(-self.log_clip))
        nk=nk.clamp(self.K_min+1e-6,5.)
        if self.method=='shared': nk=nk.mean().expand_as(nk)
        for i,p in enumerate(self.points):
            if p.grad is None: continue
            k=ks[i]; g=L.egrad_to_rgrad(p,p.grad,k)
            gn=L.lorentz_norm(g,True); g*=torch.clamp(10./gn,max=1.)
            momentum=self.method in ['B3','B5','A5','shared','B35']
            m=self.beta*self.m[i]+(1-self.beta)*g if momentum else g
            m=L.project_to_tangent(p,m,k) if self.project_tangent else m
            mn=L.lorentz_norm(m,True)
            normalized=self.method in ['B5','A5','shared','B35']
            eta=self.lr/(1+self.gamma*torch.sqrt(k)*mn) if normalized else self.lr
            delta=torch.sqrt(k)*mn*eta
            # Explicit geodesic trust cap is recorded; applied equally across methods.
            eta=eta*torch.clamp(.25/delta.clamp_min(1e-12),max=1.)
            xn=L.exp_map(p,-eta*m,k)
            mt=L.parallel_transport(p,xn,m,k)
            scale=torch.sqrt(k/nk[i]); xn*=scale
            if self.method!='A5': mt*=scale
            xn=L.project_to_manifold(xn,nk[i])
            r=L.dist(xn,L.origin(xn.shape[-1]-1,nk[i],device=p.device).expand_as(xn),nk[i])
            over=r>self.radius/torch.sqrt(nk[i])
            if over.any():
                v=L.log_map(L.origin(xn.shape[-1]-1,nk[i],device=p.device).expand_as(xn),xn,nk[i])
                clipped=L.exp_map(L.origin(xn.shape[-1]-1,nk[i],device=p.device).expand_as(xn),v*(self.radius/torch.sqrt(nk[i])/r.clamp_min(1e-8)).clamp(max=1).unsqueeze(-1),nk[i])
                mt=L.parallel_transport(xn,clipped,mt,nk[i]); xn=clipped
            self.m[i]=(L.project_to_tangent(xn,mt,nk[i]) if self.project_tangent else mt).detach()
            p.copy_(xn)
            stats.append((float(gn.mean()),float(mn.mean()),float(delta.max()),float(over.float().mean()),float((L.lorentz_inner(xn,xn)+1/nk[i]).abs().max()),float(L.lorentz_inner(xn,self.m[i]).abs().max()),float(r.mean())))
        self.c.copy_(torch.log(torch.expm1(nk-self.K_min)))
        self.point_K.copy_(nk)
        self.last={'grad_norm':max([s[0] for s in stats],default=0),'momentum_norm':max([s[1] for s in stats],default=0),'exp_argument':max([s[2] for s in stats],default=0),'radius_clipped_fraction':max([s[3] for s in stats],default=0),'manifold_residual':max([s[4] for s in stats],default=0),'tangent_residual':max([s[5] for s in stats],default=0),'mean_radius':max([s[6] for s in stats],default=0)}
    def zero_grad(self):
        for p in self.parameters(): p.grad=None

def origin_embed(v,k):
    # Fixed canonical tangent feature -> H_-1 -> homothety to H_-k.
    v=v/(1+v.norm(dim=-1,keepdim=True))
    full=torch.cat([torch.zeros_like(v[...,:1]),v],-1)
    o=torch.zeros_like(full); o[...,0]=1
    return L.exp_map(o,full,1.)/torch.sqrt(k)

class Head(torch.nn.Module):
    def __init__(self,dim,classes,method,device,**kw):
        super().__init__(); self.geo=Geometry([(1,dim)]*classes,method,device,**kw)
        self.euclidean=torch.nn.Linear(dim,classes).to(device) if method=='E' else None
    def forward(self,x):
        if self.euclidean is not None: return self.euclidean(x)
        ks=self.geo.K(); ps=self.geo.chart_points()
        return torch.stack([-L.dist(origin_embed(x,ks[i]),p.expand(len(x),-1),ks[i]).square() for i,p in enumerate(ps)],-1)
