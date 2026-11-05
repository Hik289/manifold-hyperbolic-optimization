"""Mechanistic checks: tangent constraints, PT, total curvature derivative, A5 contrast."""
import json,torch
from geometry import Geometry,L

def main():
    torch.manual_seed(42);device='cuda' if torch.cuda.is_available() else 'cpu'
    g=Geometry([(8,3)],'B5',device)
    p=g.points[0];v=L.project_to_tangent(p,torch.randn_like(p)*.01,g.K()[0]);y=L.exp_map(p,v,g.K()[0]);pt=L.parallel_transport(p,y,v,g.K()[0])
    errors={'exp_constraint':float((L.lorentz_inner(y,y)+1/g.K()[0]).abs().max()),'pt_tangency':float(L.lorentz_inner(y,pt).abs().max()),'pt_norm':float((L.lorentz_norm(pt)-L.lorentz_norm(v)).abs().max())}
    def objective():
        x=g.chart_points()[0];return x[:,1:].square().sum()+g.K().square().sum()
    loss=objective();loss.backward();analytic=float(g.c.grad[0]);base=float(g.c[0]);eps=.001
    with torch.no_grad():
        g.c[0]=base+eps;plus=float(objective());g.c[0]=base-eps;minus=float(objective());g.c[0]=base
    numeric=(plus-minus)/(2*eps);errors['curvature_finite_difference_relative_error']=abs(analytic-numeric)/max(abs(numeric),1e-6)
    before=p.detach().clone();g.step();errors.update(g.last)
    assert not torch.equal(before,g.points[0]),'No actual manifold update'
    assert errors['curvature_finite_difference_relative_error']<.002,errors
    assert errors['exp_constraint']<1e-5 and errors['pt_tangency']<1e-5 and errors['pt_norm']<1e-5,errors
    # Isolate transport: identical states, gradients, momentum, and step rule.
    b=Geometry([(8,3)],'B5',device,rho=.1);h=Geometry([(8,3)],'A5',device,rho=.1);h.load_state_dict(b.state_dict())
    for obj in [b,h]:
        pp=obj.chart_points()[0];ll=pp[:,1:].square().sum()+obj.K().sum();ll.backward();obj.step()
    assert torch.allclose(b.points[0],h.points[0]),'Point steps must agree in first transport ablation step'
    assert not torch.allclose(b.m[0],h.m[0],atol=1e-10,rtol=1e-5),'A5 and B5 must differ in momentum'
    errors['B5_A5_momentum_difference']=float((b.m[0]-h.m[0]).norm())
    print(json.dumps({'status':'passed','device':device,**errors},indent=2))
if __name__=='__main__':main()
