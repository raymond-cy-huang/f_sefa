"""Local soft Riemannian refinement of the mapped F-SeFa Smile direction."""
import csv, math, os, sys, time
import numpy as np
import torch
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),"..")); EXP=os.path.dirname(__file__)
sys.path.extend([ROOT,os.path.join(ROOT,"experiments_ro_verification")])
from config import Config
from evaluation import save_csv,save_json,save_traversal_grid
from geometry import legendre_gram_diagonal
from objective import GANImageObjective
from optimizer import optimize_sphere
from utils import factorize_weight,load_generator
RANKS={"gender":1,"age":3,"pose":7,"glasses":10,"smile":18}; NUIS=("gender","age","pose","glasses")
def norm(v,g): return v/np.sqrt(max(np.dot(v*g,v),1e-30))
def ih(a,b,g):
 a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64); g=np.asarray(g,dtype=np.float64); return float(np.dot(a*g,b))
def anchors(gen,n,seed):
 d=next(gen.parameters()).device; r=torch.Generator(device=d); r.manual_seed(seed); z=torch.randn(n,gen.z_space_dim,generator=r,device=d)
 with torch.no_grad(): w=gen.truncation(gen.mapping(z)["w"],trunc_psi=.7,trunc_layers=8)
 return (w[:,0,:] if w.ndim==3 else w).detach().cpu().float()
def dirs(gen):
 _,b,e=factorize_weight(gen,"all"); b=np.real(b).astype(float); e=np.real(e).astype(float); o=np.argsort(e)[::-1]; g=legendre_gram_diagonal(b.shape[1]); out={}; meta={}
 for n,r in RANKS.items():
  i=int(o[r]); out[n]=norm(b[i].reshape(-1),g); meta[n]={"rank":r,"source_index":i,"eigenvalue":float(e[i])}
 return out,meta,g
def evaluator(gen,q,eps,res,micro): return GANImageObjective(gen,q,eps,0.,res,micro)
def get_response(gen,direction,aset,cfg):
 dev=next(gen.parameters()).device; q=torch.eye(cfg.dimension,device=dev); ev=evaluator(gen,q,cfg.epsilon_h,cfg.resolution,cfg.microbatch); _,x=ev.evaluate(torch.as_tensor(direction,dtype=torch.float32,device=dev),aset,True); return x
def template(x):
 x=x.reshape(len(x),3,-1); x=x-x.mean(2,keepdims=True); x=x.reshape(len(x),-1); return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def amp(x,t): return np.sum(x*t,1)/np.sqrt(x.shape[1])
class LocalObjective(GANImageObjective):
 def __init__(self,gen,q,cfg,ts,sc,x0,tau,beta,gamma):
  super().__init__(gen,q,cfg.epsilon_h,0.,cfg.resolution,cfg.microbatch); self.ts={k:torch.as_tensor(v,dtype=torch.float32,device=self.device) for k,v in ts.items()}; self.sc=sc; self.x0=torch.as_tensor(x0,dtype=torch.float32,device=self.device); self.tau=tau; self.beta=beta; self.gamma=gamma; self.eta=cfg.eta; self.lw=torch.tensor([.2126,.7152,.0722],device=self.device).view(1,3)
 def terms(self,f,s,p):
  z=(f-f.mean((2,3),keepdim=True)).flatten(1); sm=(z*self.ts["smile"]).sum(1)/math.sqrt(z.shape[1]); leaks=[]; out={}
  for n in NUIS:
   a=(z*self.ts[n]).sum(1)/math.sqrt(z.shape[1]); out[n+"_leakage"]=(a*a).mean()/self.sc[n]; leaks.append(out[n+"_leakage"])
  lum=(f.mean((2,3))*self.lw).sum(1); out.update(smile_mean=sm.mean()/self.sc["smile"],smile_consistency=(sm>0).float().mean(),known_leakage=torch.stack(leaks).mean(),luminance_leakage=(lum*lum).mean()/self.sc["luminance"],cosine_to_fsefa=torch.dot(p,self.x0),locality_penalty=torch.relu(self.tau-torch.dot(p,self.x0))**2,curvature=(s*s).flatten(1).mean())
  out["objective"]=out["smile_mean"]-self.beta*out["known_leakage"]-self.gamma*out["luminance_leakage"]-self.eta*out["locality_penalty"]; return out
 def value_and_gradient(self,p,a):
  p=p.detach().to(self.device).clone(); p.requires_grad_(True); keys=None; total={}
  for st in range(0,len(a),self.microbatch):
   en=min(st+self.microbatch,len(a)); w=(en-st)/len(a); c=a.coefficients[st:en].to(self.device); b=a.base_low_resolution[st:en].to(self.device); _,_,f,s=self._batch_terms(p,c,b); z=self.terms(f,s,p); keys=keys or tuple(z); (w*z["objective"]).backward()
   for k in keys: total[k]=total.get(k,0)+w*float(z[k].detach().cpu())
  return total,p.grad.detach()
 @torch.no_grad()
 def evaluate(self,p,a,collect_responses=False):
  total={}; rs=[]
  for st in range(0,len(a),self.microbatch):
   en=min(st+self.microbatch,len(a)); w=(en-st)/len(a); c=a.coefficients[st:en].to(self.device); b=a.base_low_resolution[st:en].to(self.device); _,_,f,s=self._batch_terms(p,c,b); z=self.terms(f,s,p)
   for k,v in z.items(): total[k]=total.get(k,0)+w*float(v.cpu())
   if collect_responses: rs.append(f.flatten(1).cpu().numpy())
  return total,np.concatenate(rs) if rs else None
class Geo:
 def __init__(self,q,g): self.q=q; self.g=g
 def constraint_errors(self,d):
  n=np.dot(d*self.g,d); return {"hilbert_norm_squared":float(n),"hilbert_norm_error":abs(float(n)-1),"orthogonality_l2":0.,"orthogonality_max_abs":0.}
def main():
 cfg=Config(); t0=time.perf_counter(); os.makedirs(cfg.output_dir,exist_ok=True); [os.makedirs(os.path.join(cfg.output_dir,x),exist_ok=True) for x in ("directions","traversals","metrics")]; gen=load_generator(cfg.model); dev=next(gen.parameters()).device; ds,meta,g=dirs(gen); gram=np.asarray(g,dtype=np.float64); d0=ds["smile"]; q=torch.diag(torch.as_tensor(1/np.sqrt(gram),dtype=torch.float32,device=dev)); x0=np.sqrt(gram)*d0; geo=Geo(q.cpu().double().numpy(),gram)
 allw=anchors(gen,cfg.num_search+cfg.num_validation+cfg.num_test,cfg.seed); sb=allw[:cfg.num_search]; vb=allw[cfg.num_search:cfg.num_search+cfg.num_validation]; tb=allw[cfg.num_search+cfg.num_validation:]; base=evaluator(gen,q,cfg.epsilon_h,cfg.resolution,cfg.microbatch); sa=base.prepare_anchors("search",sb); va=base.prepare_anchors("validation",vb); ta=base.prepare_anchors("test",tb)
 ts={n:template(get_response(gen,d,sa,cfg)) for n,d in ds.items()}; sc={n:max(np.mean(amp(get_response(gen,d,sa,cfg),ts[n])**2),1e-8) for n,d in ds.items()}; xref=get_response(gen,d0,sa,cfg).reshape(len(sa),-1); lum=get_response(gen,d0,sa,cfg).reshape(len(sa),3,-1).mean(2)@np.array([.2126,.7152,.0722]); sc["luminance"]=max(np.mean(lum**2),1e-8)
 settings=[("tau_"+str(tau).replace(".",""),tau,1.,.5) for tau in cfg.taus]+[("ablation_beta0",.9,0.,.5),("ablation_gamma0",.9,1.,0.)]; rows=[]
 for name,tau,beta,gamma in settings:
  obj=LocalObjective(gen,q,cfg,ts,sc,x0,tau,beta,gamma); cand=[]
  for k in range(cfg.num_candidates):
   r=np.random.default_rng(cfg.seed+k+int(tau*1000)); z=torch.as_tensor(r.normal(size=cfg.dimension),dtype=torch.float32,device=dev); xt=torch.as_tensor(x0,dtype=torch.float32,device=dev); z-=xt*torch.dot(xt,z); z/=torch.linalg.vector_norm(z); a=[.02,.05,.1,.15][k%4]; p=torch.cos(torch.tensor(a,device=dev))*xt+torch.sin(torch.tensor(a,device=dev))*z; opt,h,rt=optimize_sphere(obj,sa,p,geo,cfg.learning_rate,cfg.min_learning_rate,1e-4,cfg.line_search_steps,cfg.max_iterations,cfg.gradient_tolerance); d=(q@opt).detach().cpu().double().numpy(); vm,vf=obj.evaluate(opt,va,True); tm,tf=obj.evaluate(opt,ta,True); r={"setting":name,"candidate_id":k,"tau":tau,"beta":beta,"gamma":gamma,"iterations":len(h)-1,"runtime":rt,"objective":vm["objective"],"smile_mean":vm["smile_mean"],"smile_consistency":vm["smile_consistency"],"known_leakage":vm["known_leakage"],"luminance_leakage":vm["luminance_leakage"],"cos_h":ih(d,d0,gram),"angle":np.degrees(np.arccos(np.clip(ih(d,d0,gram),-1,1))),"norm_error":abs(ih(d,d,gram)-1),"gradient_norm":h[-1]["riemannian_gradient_norm"]}; cand.append((r,d)); rows.append(r); save_csv(os.path.join(cfg.output_dir,"metrics","progress.csv"),rows); save_csv(os.path.join(cfg.output_dir,name+"_candidates.csv"),[x[0] for x in cand]); print(f"{name} candidate {k+1}/{cfg.num_candidates}",flush=True)
  cand.sort(key=lambda x:(x[0]["known_leakage"]+.5*x[0]["luminance_leakage"],-x[0]["smile_mean"])); best,d=cand[0]; np.savez_compressed(os.path.join(cfg.output_dir,"directions",name+"_best.npz"),direction=d,gram_diagonal=g); save_json(os.path.join(cfg.output_dir,name+"_manifest.json"),{"setting":name,"best_candidate":best}); print("Completed",name,"candidate",best["candidate_id"],flush=True)
  dist=[cfg.epsilon_h*i for i in (-3,-2,-1,0,1,2,3)]; grid=[]; ev=evaluator(gen,torch.eye(cfg.dimension,device=dev),cfg.epsilon_h,cfg.resolution,cfg.microbatch)
  for i in range(min(8,len(ta))):
   for label,direction in (("SeFa-Smile",ds["smile"]),("F-SeFa-Smile",d0),("Local-RO-best",d)): grid.append((f"seed {i:03d} / {label}",ev.render_traversal(torch.as_tensor(direction,dtype=torch.float32,device=dev),ta.coefficients[i],dist)))
  save_traversal_grid(os.path.join(cfg.output_dir,"traversals",name+".png"),grid,dist)
 save_csv(os.path.join(cfg.output_dir,"metrics","all_candidates.csv"),rows); save_csv(os.path.join(cfg.output_dir,"metrics","final_summary.csv"),[r for r in rows if r["candidate_id"]==min(x["candidate_id"] for x in rows if x["setting"]==r["setting"])]); save_json(os.path.join(cfg.output_dir,"manifest.json"),{"experiment":"local-smile-disentanglement","metadata":meta,"runtime":time.perf_counter()-t0,"settings":[x[0] for x in settings]}); open(os.path.join(cfg.output_dir,"SUMMARY.md"),"w",encoding="utf8").write("# Local Smile Disentanglement\n\nSee metrics/final_summary.csv and traversals/.\n")
if __name__=="__main__": main()
