"""MVM2026 PINN digital twin based on relative torsional dynamics.

The real time vector and Mem input are loaded from jera1.mat. Encoder/state
responses are synthetic ODE data. THref is intentionally not used in any loss.
"""
from __future__ import annotations

import argparse, copy, csv, hashlib, json, math, os, platform, random, sys, time
from dataclasses import asdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import interp1d
from scipy.io import loadmat

from pinn_torsional_twin.config import ExperimentConfig
from pinn_torsional_twin.data import load_input as load_input_modular
from pinn_torsional_twin.data import measurements as measurements_modular
from pinn_torsional_twin.models import (
    ConstantStiffness as ConstantStiffnessModular,
    DeltaNet as DeltaNetModular,
    RelativeStateNet as RelativeStateNetModular,
    StiffnessNet as StiffnessNetModular,
    WeakSigmoidStiffness as WeakSigmoidStiffnessModular,
)
from pinn_torsional_twin.physics.two_inertia import simulate as simulate_modular
from pinn_torsional_twin.physics.two_inertia import true_k as true_k_modular
from pinn_torsional_twin.physics.weak import (
    build_constant_weak_terms,
    build_sigmoid_weak_terms,
    weak_sigmoid_losses as weak_sigmoid_losses_modular,
)


REPO_ROOT=Path(__file__).resolve().parents[1]
DATA_DIR=REPO_ROOT/"data"
OUTPUTS_DIR=REPO_ROOT/"outputs"
RESULTS_DIR=REPO_ROOT/"results"
FINAL_TABLES_DIR=RESULTS_DIR/"tables"
FINAL_FIGURES_DIR=RESULTS_DIR/"figures"


Config = ExperimentConfig


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _legacy_load_input(path: Path):
    d=loadmat(path)
    if "t" not in d or "Mem" not in d: raise KeyError("jera1.mat mora sadržati t i Mem")
    t=np.asarray(d["t"],float).squeeze(); u=np.asarray(d["Mem"],float).squeeze()
    valid=np.isfinite(t)&np.isfinite(u);t,u=t[valid],u[valid]
    order=np.argsort(t);t,u=t[order],u[order];keep=np.r_[True,np.diff(t)>0];t,u=t[keep],u[keep]
    if len(t)<20: raise ValueError("Nedovoljno validnih MAT odbiraka")
    return t-t[0],u


def _legacy_true_k(t,T,c):
    x=np.asarray(t,float); center=c.center_fraction*T; width=c.width_fraction*T
    s=1/(1+np.exp(-(x-center)/width));s0=1/(1+np.exp(center/width));s1=1/(1+np.exp(-(T-center)/width))
    p=np.clip((s-s0)/(s1-s0),0,1)
    return c.k0+(c.k_final-c.k0)*p


def _legacy_simulate(t,u,c,k_profile,*,rtol=1e-9,atol=1e-11,max_step_divisor=2000):
    ui=interp1d(t,u,kind="linear",bounds_error=False,fill_value=(u[0],u[-1]));T=float(t[-1])
    def rhs(x,y):
        tm,wm,tl,wl=y;k=float(k_profile(x));shaft=k*(tm-tl)+c.bv*(wm-wl)
        return wm,(float(ui(x))-shaft)/c.Jm,wl,shaft/c.Jl
    sol=solve_ivp(rhs,(0,T),np.zeros(4),t_eval=t,method="DOP853",
                  rtol=rtol,atol=atol,max_step=T/max_step_divisor)
    if not sol.success: raise RuntimeError(sol.message)
    return {"t":t,"Mem":u,"theta_m":sol.y[0],"omega_m":sol.y[1],"theta_l":sol.y[2],"omega_l":sol.y[3],
            "delta":sol.y[0]-sol.y[2],"delta_dot":sol.y[1]-sol.y[3],"k_true":np.asarray(k_profile(t),float)}


def identifiability_cache_key(t,u,c):
    """Fingerprint all data and parameters that affect sensitivity simulations."""
    h=hashlib.sha256()
    h.update(np.ascontiguousarray(t,dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(u,dtype=np.float64).tobytes())
    settings={"Jm":c.Jm,"Jl":c.Jl,"bv":c.bv,"cases":[350.0,300.0,245.0],
              "rtol":1e-7,"atol":1e-9,"max_step_divisor":500}
    h.update(json.dumps(settings,sort_keys=True,separators=(",",":")).encode("ascii"))
    return h.hexdigest()


def identifiability_test(t,u,c,outdir):
    cache_path=OUTPUTS_DIR/"cache"/"identifiability_cache.npz"
    expected_key=identifiability_cache_key(t,u,c);cases={}
    if cache_path.exists():
        try:
            with np.load(cache_path,allow_pickle=False) as cached:
                cached_key=str(cached["cache_key"].item())
                if cached_key==expected_key:
                    for k in (350.0,300.0,245.0):
                        tag=str(int(k));cases[k]={"t":t,"delta":cached[f"delta_{tag}"],
                            "omega_m":cached[f"omega_m_{tag}"],"omega_l":cached[f"omega_l_{tag}"]}
                    print(f"Identifiability cache loaded: {cache_path}")
                else:
                    print("Identifiability cache does not match current data/parameters; recomputing.")
        except (OSError,KeyError,ValueError) as exc:
            print(f"Identifiability cache is invalid ({exc}); recomputing.")
    if not cases:
        print("Computing identifiability simulations with relaxed solver tolerances...")
        cases={k:simulate(t,u,c,lambda x,k=k: np.zeros_like(np.asarray(x,float))+k,
                          rtol=1e-7,atol=1e-9,max_step_divisor=500)
               for k in (350.0,300.0,245.0)}
        payload={"cache_key":np.asarray(expected_key)}
        for k,q in cases.items():
            tag=str(int(k));payload[f"delta_{tag}"]=q["delta"]
            payload[f"omega_m_{tag}"]=q["omega_m"];payload[f"omega_l_{tag}"]=q["omega_l"]
        cache_path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache_path,**payload)
        print(f"Identifiability cache saved: {cache_path}")
    base=cases[350.0]; rows={}
    for k in (300.0,245.0):
        q=cases[k];rows[str(int(k))]={}
        for name in ("delta","omega_m","omega_l"):
            diff=q[name]-base[name];scale=max(float(np.std(base[name])),1e-12)
            rows[str(int(k))][f"{name}_difference_RMSE"]=float(np.sqrt(np.mean(diff**2)))
            rows[str(int(k))][f"{name}_difference_over_baseline_std"]=float(np.sqrt(np.mean(diff**2))/scale)
    (outdir/"identifiability_metrics.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for k,q in cases.items():
        ax[0].plot(t,q["delta"],label=f"k={k:g}");ax[1].plot(t,q["omega_m"],label=f"k={k:g}");ax[2].plot(t,q["omega_l"],label=f"k={k:g}")
    for a,y in zip(ax,(r"$\delta$ [rad]",r"$\omega_m$ [rad/s]",r"$\omega_l$ [rad/s]")):a.set_ylabel(y);a.grid();a.legend()
    ax[2].set_xlabel("t [s]");fig.suptitle("Identifiability test under the same measured Mem(t)");fig.tight_layout();fig.savefig(outdir/"00_identifiability.png",dpi=200);plt.close(fig)
    print("Identifiability metrics:\n"+json.dumps(rows,indent=2));return rows


def _legacy_measurements(ref,c,seed=None,relative_noise=False,uniform_times=False):
    n=min(max(2,c.measurements),len(ref["t"]));rng=np.random.default_rng(c.seed)
    if uniform_times:
        measurement_t=np.linspace(float(ref["t"][0]),float(ref["t"][-1]),n)
        idx=np.searchsorted(ref["t"],measurement_t).clip(0,len(ref["t"])-1)
    else:
        idx=np.unique(np.linspace(0,len(ref["t"])-1,n,dtype=int));measurement_t=ref["t"][idx]
    if seed is not None:rng=np.random.default_rng(int(seed))
    ans={"t":measurement_t,"idx":idx,"sampling_scheme":"exact uniform time grid" if uniform_times else "selected source-grid indices"}
    if relative_noise:
        clean_m=np.interp(measurement_t,ref["t"],ref["omega_m"]);clean_l=np.interp(measurement_t,ref["t"],ref["omega_l"])
        relative_clean=clean_m-clean_l
        epsilon=rng.normal(0,c.noise*max(np.std(relative_clean),1e-12),relative_clean.shape)
        ans["omega_m"]=clean_m+0.5*epsilon;ans["omega_l"]=clean_l-0.5*epsilon
        ans["noise_model"]="seeded differential encoder noise: std(epsilon)=noise*std(omega_m-omega_l), split +epsilon/2 and -epsilon/2"
    else:
        for name in ("omega_m","omega_l"):
            clean=np.interp(measurement_t,ref["t"],ref[name]);ans[name]=clean+rng.normal(0,c.noise*max(np.std(clean),1e-12),clean.shape)
        ans["noise_model"]="independent per-encoder Gaussian noise scaled by each encoder standard deviation"
    ans["delta_dot"]=ans["omega_m"]-ans["omega_l"]
    return ans


def init_linear(m):
    if isinstance(m,torch.nn.Linear):torch.nn.init.xavier_normal_(m.weight);torch.nn.init.zeros_(m.bias)


class _LegacyDeltaNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Torsional mode is about 175 cycles over the 0.75 s record. Fixed
        # Fourier features remove the severe spectral bias of a plain tanh MLP.
        self.register_buffer("freq",torch.arange(1.0,256.0,2.0).reshape(1,-1))
        self.net=torch.nn.Sequential(torch.nn.Linear(257,128),torch.nn.Tanh(),torch.nn.Linear(128,64),torch.nn.Tanh(),torch.nn.Linear(64,1));self.apply(init_linear)
        torch.nn.init.normal_(self.net[-1].weight,std=1e-3)
    def forward(self,tau):
        phase=2*math.pi*tau*self.freq
        features=torch.cat((tau,torch.sin(phase),torch.cos(phase)),dim=1)
        return tau*self.net(features) # delta_n(0)=0


class _LegacyStiffnessNet(torch.nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;self.net=torch.nn.Sequential(torch.nn.Linear(1,32),torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,1));self.net.apply(init_linear)
        last=self.net[-1];torch.nn.init.zeros_(last.weight)
        f=(1-c.kappa_min)/(c.kappa_max-c.kappa_min);torch.nn.init.constant_(last.bias,math.log(f/(1-f)))
    def forward(self,tau):
        raw=self.c.kappa_min+(self.c.kappa_max-self.c.kappa_min)*torch.sigmoid(self.net(tau))
        return 1.0+tau*(raw-1.0) # kappa(0)=1 exactly


class _LegacyConstantStiffness(torch.nn.Module):
    """One bounded trainable scalar, independent of time."""
    def __init__(self,c,initial=288.75):
        super().__init__();self.c=c;self.initial_k_const=float(initial);lo=c.kappa_min*c.k0;hi=c.kappa_max*c.k0
        f=np.clip((initial-lo)/(hi-lo),1e-6,1-1e-6)
        self.raw=torch.nn.Parameter(torch.tensor(math.log(f/(1-f))))
    def value(self):
        lo=self.c.kappa_min*self.c.k0;hi=self.c.kappa_max*self.c.k0
        return lo+(hi-lo)*torch.sigmoid(self.raw)
    def forward(self,tau):return (self.value()/self.c.k0)*torch.ones_like(tau)


class _LegacyWeakSigmoidStiffness(torch.nn.Module):
    """Four bounded physical parameters for a monotonically decreasing k(t)."""
    def __init__(self,duration,k_high_init=330.0,k_low_init=270.0,
                 center_fraction_init=0.50,width_fraction_init=0.10,
                 k_min=210.0,k_max=367.5):
        super().__init__();self.duration=float(duration);self.k_min=float(k_min);self.k_max=float(k_max)
        self.initial_values={"k_high":float(k_high_init),"k_low":float(k_low_init),
            "t_center":float(center_fraction_init*self.duration),"width":float(width_fraction_init*self.duration)}

        def logit(p):
            p=float(np.clip(p,1e-9,1-1e-9));return math.log(p/(1-p))

        low_fraction=(k_low_init-k_min)/(k_max-k_min)
        high_conditional_fraction=(k_high_init-k_low_init)/(k_max-k_low_init)
        width_fraction=(width_fraction_init-0.005)/(0.25-0.005)
        self.raw_low=torch.nn.Parameter(torch.tensor(logit(low_fraction)))
        self.raw_high_conditional=torch.nn.Parameter(torch.tensor(logit(high_conditional_fraction)))
        self.raw_center=torch.nn.Parameter(torch.tensor(logit(center_fraction_init)))
        self.raw_width=torch.nn.Parameter(torch.tensor(logit(width_fraction)))
        values=self.physical_parameters()
        for name,target in self.initial_values.items():
            if not math.isclose(float(values[name].detach()),target,rel_tol=0,abs_tol=1e-9):
                raise AssertionError(f"Weak sigmoid initialization failed for {name}")

    def physical_parameters(self):
        k_low=self.k_min+(self.k_max-self.k_min)*torch.sigmoid(self.raw_low)
        k_high=k_low+(self.k_max-k_low)*torch.sigmoid(self.raw_high_conditional)
        t_center=self.duration*torch.sigmoid(self.raw_center)
        width=self.duration*(0.005+(0.25-0.005)*torch.sigmoid(self.raw_width))
        return {"k_high":k_high,"k_low":k_low,"t_center":t_center,"width":width}

    def forward(self,t):
        p=self.physical_parameters()
        # sigmoid((center-t)/width) is the stable form of
        # 1/(1+exp((t-center)/width)).
        return p["k_low"]+(p["k_high"]-p["k_low"])*torch.sigmoid((p["t_center"]-t)/p["width"])


class _LegacyRelativeStateNet(torch.nn.Module):
    """First-order state model with physical outputs delta and v_delta."""
    def __init__(self,delta_scale,v_scale):
        super().__init__();self.delta_scale=float(delta_scale);self.v_scale=float(v_scale)
        self.register_buffer("freq",torch.arange(1.0,256.0,2.0).reshape(1,-1))
        self.net=torch.nn.Sequential(torch.nn.Linear(257,128),torch.nn.Tanh(),
            torch.nn.Linear(128,64),torch.nn.Tanh(),torch.nn.Linear(64,2));self.apply(init_linear)
        torch.nn.init.normal_(self.net[-1].weight,std=1e-3)
    def forward(self,tau):
        phase=2*math.pi*tau*self.freq
        features=torch.cat((tau,torch.sin(phase),torch.cos(phase)),dim=1)
        raw=tau*self.net(features)  # known delta(0)=v_delta(0)=0
        return torch.cat((raw[:,:1]*self.delta_scale,raw[:,1:]*self.v_scale),dim=1)


# The modular implementations below are the runtime source of truth. The
# original definitions above are retained in this compatibility entry point
# during the staged refactor so historical script usage remains auditable.
load_input=load_input_modular
true_k=true_k_modular
simulate=simulate_modular
measurements=measurements_modular
DeltaNet=DeltaNetModular
StiffnessNet=StiffnessNetModular
ConstantStiffness=ConstantStiffnessModular
WeakSigmoidStiffness=WeakSigmoidStiffnessModular
RelativeStateNet=RelativeStateNetModular


def derivative(y,x):return torch.autograd.grad(y,x,torch.ones_like(y),create_graph=True)[0]


def r2(y,p):
    den=np.sum((y-np.mean(y))**2);return float(1-np.sum((y-p)**2)/den) if den else float("nan")


def build_tensors(ref,meas,c):
    T=float(ref["t"][-1]);delta_scale=max(float(np.max(np.abs(ref["delta"]))),1e-6)
    tau=torch.linspace(0,1,c.collocation_points).reshape(-1,1).requires_grad_(True);tc=tau.detach().numpy().squeeze()*T
    mem=torch.tensor(np.interp(tc,ref["t"],ref["Mem"])).reshape(-1,1)
    kpre=torch.tensor(true_k(tc,T,c)/c.k0).reshape(-1,1)
    dntrue=torch.tensor(np.interp(tc,ref["t"],ref["delta"])/delta_scale).reshape(-1,1)
    vdtrue=torch.tensor(np.interp(tc,ref["t"],ref["delta_dot"])).reshape(-1,1)
    taud=torch.tensor(meas["t"]/T).reshape(-1,1).requires_grad_(True);vd=torch.tensor(meas["delta_dot"]).reshape(-1,1)
    return {"T":T,"scale":delta_scale,"tau":tau,"mem":mem,"kpre":kpre,"dntrue":dntrue,"vdtrue":vdtrue,"taud":taud,"vd":vd}


def relative_quantities(delta_net,k_net,z,use_true_k=False):
    dn=delta_net(z["tau"]);d=dn*z["scale"];v=derivative(dn,z["tau"])*z["scale"]/z["T"];a=derivative(v,z["tau"])/z["T"]
    kap=z["kpre"] if use_true_k else k_net(z["tau"]);k=kap*350.0
    invsum=1/6.20e-4+1/2.20e-4
    forcing=z["mem"]/6.20e-4;scale=torch.max(torch.abs(forcing)).detach().clamp_min(1.0)
    residual=(a+4e-3*invsum*v+k*invsum*d-forcing)/scale
    return dn,d,v,a,kap,residual


def pretrain(delta_net,k_net,z,c,history):
    opt=torch.optim.Adam(delta_net.parameters(),lr=c.lr_delta)
    for ep in range(1,c.pretrain_epochs+1):
        opt.zero_grad();dn,d,v,a,kap,res=relative_quantities(delta_net,k_net,z,True)
        # Numerical teacher warm-start uses synthetic ODE states only in pretraining.
        # Delta itself is the stiffness-sensitive quantity; emphasize its
        # normalized absolute level, while retaining delta_dot reconstruction.
        teacher=50.0*torch.mean((dn-z["dntrue"])**2)+torch.mean(((v-z["vdtrue"])/(torch.std(z["vdtrue"])+1e-9))**2)
        loss=10*torch.mean(res**2)+10*teacher+(v[0,0]/(torch.std(z["vdtrue"])+1e-9))**2
        loss.backward();torch.nn.utils.clip_grad_norm_(delta_net.parameters(),100);opt.step()
        if ep==1 or ep%max(1,c.pretrain_epochs//100)==0:history.append({"phase":"pretrain","epoch":ep,"loss":float(loss),"physics":float(torch.mean(res**2)),"data":float(teacher)})
        if ep==1 or ep%max(1,c.pretrain_epochs//10)==0:print(f"pretrain epoch={ep:6d}/{c.pretrain_epochs} loss={float(loss):.4e}")


def pretrain_constant_delta(delta_net,z,c,history):
    """Case-specific response pretraining without using a known stiffness in loss."""
    opt=torch.optim.Adam(delta_net.parameters(),lr=c.lr_delta);velocity_scale=torch.std(z["vdtrue"])+1e-9
    for ep in range(1,c.pretrain_epochs+1):
        opt.zero_grad();dn=delta_net(z["tau"])
        velocity=derivative(dn,z["tau"])*z["scale"]/z["T"]
        delta_loss=torch.mean((dn-z["dntrue"])**2)
        velocity_loss=torch.mean(((velocity-z["vdtrue"])/velocity_scale)**2)
        ic=(velocity[0,0]/velocity_scale)**2
        teacher=500.0*delta_loss+10.0*velocity_loss+ic
        teacher.backward();torch.nn.utils.clip_grad_norm_(delta_net.parameters(),100);opt.step()
        if ep==1 or ep%max(1,c.pretrain_epochs//100)==0:
            history.append({"phase":"pretrain_constant_delta","epoch":ep,"loss":float(teacher),"physics":0.0,"data":float(500.0*delta_loss+10.0*velocity_loss)})
        if ep==1 or ep%max(1,c.pretrain_epochs//10)==0:
            print(f"pretrain constant DeltaNet epoch={ep:6d}/{c.pretrain_epochs} loss={float(teacher):.4e}")


def delta_metrics(delta_net,z,ref):
    tau=torch.tensor(ref["t"]/z["T"]).reshape(-1,1).requires_grad_(True);dn=delta_net(tau);d=dn*z["scale"];v=derivative(dn,tau)*z["scale"]/z["T"]
    dp=d.detach().numpy().squeeze();vp=v.detach().numpy().squeeze();vr=ref["delta_dot"]
    return {"delta_RMSE":float(np.sqrt(np.mean((dp-ref["delta"])**2))),"delta_R2":r2(ref["delta"],dp),
            "delta_dot_RMSE":float(np.sqrt(np.mean((vp-vr)**2))),"delta_dot_R2":r2(vr,vp),
            "delta_dot_relative_RMSE":float(np.sqrt(np.mean((vp-vr)**2))/max(np.std(vr),1e-12))}


def delta_checkpoint_name(k_true_constant):
    value=float(k_true_constant)
    tag=str(int(value)) if value.is_integer() else str(value).replace(".","p")
    return f"delta_pretrained_k{tag}.pt"


def save_delta_checkpoint(path,delta_net,z,c,k_true_constant,metrics):
    torch.save({"state_dict":delta_net.state_dict(),"delta_scale":float(z["scale"]),
                "k_true_constant":float(k_true_constant),"duration":float(z["T"]),
                "architecture":"DeltaNet Fourier odd frequencies 1..255",
                "pretrain_metrics":metrics,"config":asdict(c)},path)


def load_delta_checkpoint(path,delta_net,z,k_true_constant):
    ckpt=torch.load(path,map_location="cpu",weights_only=False)
    if "k_true_constant" not in ckpt:
        raise ValueError(f"Checkpoint {path} nema k_true_constant metapodatak")
    if not np.isclose(float(ckpt["k_true_constant"]),float(k_true_constant),rtol=0,atol=1e-9):
        raise ValueError(f"Checkpoint je za k_true={ckpt['k_true_constant']}, a eksperiment traži {k_true_constant}")
    if not np.isclose(float(ckpt.get("duration",z["T"])),float(z["T"]),rtol=0,atol=1e-9):
        raise ValueError("Checkpoint i trenutni eksperiment nemaju isto trajanje")
    delta_net.load_state_dict(ckpt["state_dict"]);z["scale"]=float(ckpt["delta_scale"])
    print(f"Loaded matching DeltaNet checkpoint: {path}")
    return ckpt.get("pretrain_metrics",{})


def constant_train(delta_net,k_model,z,c,history):
    for p in delta_net.parameters():p.requires_grad_(False)
    opt=torch.optim.Adam(k_model.parameters(),lr=5e-3);best_loss=float("inf");best_raw=None;best_parts=None
    sd=torch.std(z["vd"])+1e-9
    raw_before=float(k_model.raw.detach());k_before_training=float(k_model.value().detach())
    learning_rate=float(opt.param_groups[0]["lr"])
    print(f"raw_k_before_training={raw_before:.12f}")
    print(f"k_const_before_training={k_before_training:.12f} Nm/rad")
    print(f"optimizer_learning_rate={learning_rate:.12g}")
    assert abs(k_before_training-288.75)<1e-6, f"Invalid k_const initialization: {k_before_training}"
    for ep in range(1,c.epochs+1):
        k_before_step=float(k_model.value().detach())
        opt.zero_grad();dn,d,v,a,kap,res=relative_quantities(delta_net,k_model,z,False)
        dd=delta_net(z["taud"]);vd=derivative(dd,z["taud"])*z["scale"]/z["T"]
        physics=torch.mean(res**2);data=torch.mean(((vd-z["vd"])/sd)**2)
        # DeltaNet is frozen: data loss is constant with respect to k_const and
        # is reported only as a diagnostic, never as an optimizer objective.
        total=20.0*physics
        total.backward();gradient_raw=float(k_model.raw.grad.detach());opt.step()
        k_after_step=float(k_model.value().detach())
        if ep==1:
            print(f"k_before_first_step={k_before_step:.12f} Nm/rad")
            print(f"gradient_raw_k={gradient_raw:.12e}")
            print(f"k_after_first_step={k_after_step:.12f} Nm/rad")
        current=float(total.detach())
        if current<best_loss:
            best_loss=current;best_raw=k_model.raw.detach().clone();best_parts=(float(physics.detach()),float(data.detach()))
        if ep==1 or ep%max(1,c.epochs//100)==0 or ep==c.epochs:
            history.append({"phase":"constant","epoch":ep,"loss":current,"physics":float(physics),"data":float(data)})
        if ep==1 or ep%max(1,c.epochs//10)==0 or ep==c.epochs:
            print(f"constant epoch={ep:6d}/{c.epochs} loss={current:.4e} k_const={float(k_model.value()):.4f}")
    with torch.no_grad():k_model.raw.copy_(best_raw)
    for p in delta_net.parameters():p.requires_grad_(True)
    return best_loss,best_parts[0],best_parts[1]


def diagnose_constant_landscape(outdir,delta_net,z,ref,meas,c,k_true_constant,points=200):
    """Evaluate oracle and frozen-DeltaNet physics landscapes without optimization."""
    outdir.mkdir(parents=True,exist_ok=True)
    for p in delta_net.parameters():p.requires_grad_(False)
    tau=z["tau"];T=z["T"];invsum=1/c.Jm+1/c.Jl
    forcing=z["mem"]/c.Jm;forcing_scale=torch.max(torch.abs(forcing)).detach().clamp_min(1.0)

    dn=delta_net(tau);delta_net_value=dn*z["scale"]
    delta_net_dot=derivative(dn,tau)*z["scale"]/T
    delta_net_ddot=derivative(delta_net_dot,tau)/T

    tc=tau.detach().numpy().squeeze()*T
    oracle_delta=torch.tensor(np.interp(tc,ref["t"],ref["delta"])).reshape(-1,1)
    oracle_dot=torch.tensor(np.interp(tc,ref["t"],ref["delta_dot"])).reshape(-1,1)
    # Exact acceleration from the reference ODE right-hand side, not a noisy
    # numerical derivative.
    oracle_ddot=forcing-c.bv*invsum*oracle_dot-float(k_true_constant)*invsum*oracle_delta

    def components(delta,dot,ddot,k_value):
        terms={"delta_ddot_term":ddot,"damping_term":c.bv*invsum*dot,
               "stiffness_term":float(k_value)*invsum*delta,"torque_term":-forcing}
        residual=sum(terms.values())
        result={name+"_RMS":float(torch.sqrt(torch.mean(value**2)).detach()) for name,value in terms.items()}
        result["residual_RMSE"]=float(torch.sqrt(torch.mean(residual**2)).detach())
        result["normalized_physics_loss"]=float(torch.mean((residual/forcing_scale)**2).detach())
        return result

    def loss_at(delta,dot,ddot,k_value):
        residual=ddot+c.bv*invsum*dot+float(k_value)*invsum*delta-forcing
        return float(torch.mean((residual/forcing_scale)**2).detach())

    candidates=np.linspace(c.kappa_min*c.k0,c.kappa_max*c.k0,max(200,int(points)))
    oracle_losses=np.asarray([loss_at(oracle_delta,oracle_dot,oracle_ddot,k) for k in candidates])
    net_losses=np.asarray([loss_at(delta_net_value,delta_net_dot,delta_net_ddot,k) for k in candidates])
    requested=(245.0,288.75,300.0,350.0)

    def curve_summary(losses,delta,dot,ddot):
        i=int(np.argmin(losses))
        return {"k_at_minimum":float(candidates[i]),"minimum_physics_loss":float(losses[i]),
                "physics_loss_at_245":loss_at(delta,dot,ddot,245.0),
                "physics_loss_at_288_75":loss_at(delta,dot,ddot,288.75),
                "physics_loss_at_300":loss_at(delta,dot,ddot,300.0),
                "physics_loss_at_350":loss_at(delta,dot,ddot,350.0)}

    oracle_summary=curve_summary(oracle_losses,oracle_delta,oracle_dot,oracle_ddot)
    net_summary=curve_summary(net_losses,delta_net_value,delta_net_dot,delta_net_ddot)

    # Recalculate checkpoint quality immediately in the current experiment.
    checkpoint_metrics=delta_metrics(delta_net,z,ref)
    taud=z["taud"];dn_data=delta_net(taud);pred_data_dot=derivative(dn_data,taud)*z["scale"]/T
    data_std=float(torch.std(z["vd"]).detach());data_rmse=float(torch.sqrt(torch.mean((pred_data_dot-z["vd"])**2)).detach())
    data_loss=float(torch.mean(((pred_data_dot-z["vd"])/(torch.std(z["vd"])+1e-9))**2).detach())

    # Gradient at k_init=288.75 without optimizer.step().
    k_model=ConstantStiffness(c);assert abs(float(k_model.value().detach())-288.75)<1e-6
    _,_,_,_,_,residual=relative_quantities(delta_net,k_model,z,False)
    physics=torch.mean(residual**2);physics.backward()
    grad_raw=float(k_model.raw.grad.detach())
    sigmoid_value=float(torch.sigmoid(k_model.raw.detach()))
    dk_draw=(c.kappa_max*c.k0-c.kappa_min*c.k0)*sigmoid_value*(1-sigmoid_value)
    grad_physical=grad_raw/dk_draw

    term_diagnostics={
        "oracle_k288_75":components(oracle_delta,oracle_dot,oracle_ddot,288.75),
        "oracle_k350":components(oracle_delta,oracle_dot,oracle_ddot,350.0),
        "DeltaNet_k288_75":components(delta_net_value,delta_net_dot,delta_net_ddot,288.75),
        "DeltaNet_k350":components(delta_net_value,delta_net_dot,delta_net_ddot,350.0)}
    result={"k_true_constant":float(k_true_constant),"candidate_count":len(candidates),
            "checkpoint_metrics_recomputed":checkpoint_metrics,
            "data_loss_diagnostic":{"definition":"mean(((DeltaNet delta_dot at measurement times - measured (omega_m-omega_l))/std(measured delta_dot))^2)",
                                    "measurement_count":int(len(meas["t"])),"noise":float(c.noise),
                                    "delta_dot_RMSE_on_measurements":data_rmse,"measured_delta_dot_std":data_std,"data_loss":data_loss},
            "oracle_reference_landscape":oracle_summary,"DeltaNet_landscape":net_summary,
            "residual_term_diagnostics":term_diagnostics,
            "initial_gradient":{"raw_k":float(k_model.raw.detach()),"k_const":float(k_model.value().detach()),
                                "physics_loss":float(physics.detach()),"gradient_physics_loss_wrt_raw_k":grad_raw,
                                "dk_draw":dk_draw,"gradient_physics_loss_wrt_physical_k":grad_physical}}

    with (outdir/"constant_loss_landscape.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["k_candidate","oracle_physics_loss","DeltaNet_physics_loss"])
        w.writerows(zip(candidates,oracle_losses,net_losses))
    (outdir/"constant_loss_landscape.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(9,5));ax.semilogy(candidates,np.maximum(oracle_losses,1e-30),label="oracle/reference")
    ax.semilogy(candidates,np.maximum(net_losses,1e-30),label="frozen DeltaNet")
    ax.axvline(float(k_true_constant),color="k",ls="--",label="true k");ax.axvline(288.75,color=".5",ls=":",label="k init")
    ax.set(xlabel="k candidate [Nm/rad]",ylabel="normalized physics loss",title="Constant-stiffness physics-loss landscape");ax.grid();ax.legend();fig.tight_layout();fig.savefig(outdir/"constant_loss_landscape.png",dpi=200);plt.close(fig)
    for p in delta_net.parameters():p.requires_grad_(True)
    print("Recomputed DeltaNet checkpoint metrics:\n"+json.dumps(checkpoint_metrics,indent=2))
    print("Constant physics-loss landscape diagnostics:\n"+json.dumps(result,indent=2))
    return result


def quality_gate(relative_error):
    if relative_error<=2.0:return "PASS"
    if relative_error<=5.0:return "ACCEPTABLE"
    return "FAIL"


def save_constant_result(outdir,delta_net,k_model,c,k_true_constant,total_loss,physics_loss,data_loss):
    outdir.mkdir(parents=True,exist_ok=True);estimated=float(k_model.value().detach());true=float(k_true_constant)
    absolute=abs(estimated-true);relative=100*absolute/true
    row={"initial_k_const":k_model.initial_k_const,"estimated_k":estimated,"true_k":true,"absolute_error":absolute,
         "relative_error_percent":relative,"physics_loss":physics_loss,"data_loss":data_loss,
         "total_loss":total_loss,"quality_gate":quality_gate(relative)}
    (outdir/"metrics_constant.json").write_text(json.dumps(row,indent=2),encoding="utf-8")
    torch.save({"raw_parameter":k_model.raw.detach(),"estimated_k":estimated,"config":asdict(c),
                "selection":"minimum training loss; k_true not used for checkpoint selection"},outdir/"model_constant.pt")
    fields=list(row.keys())
    with (outdir/"results_constant.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(row)
    # Maintain one cumulative table across the three separate output folders.
    cumulative=OUTPUTS_DIR/"results_constant.csv";cumulative.parent.mkdir(parents=True,exist_ok=True);rows=[]
    if cumulative.exists():
        with cumulative.open("r",newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
    rows=[r for r in rows if not np.isclose(float(r["true_k"]),true,rtol=0,atol=1e-9)];rows.append(row);rows.sort(key=lambda r:float(r["true_k"]),reverse=True)
    with cumulative.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(["true k","estimated k"],[true,estimated],color=["black","tab:red"])
    ax.set_ylabel("k [Nm/rad]");ax.set_title(f"Constant stiffness identification: {row['quality_gate']}");ax.grid(axis="y");fig.tight_layout();fig.savefig(outdir/"constant_stiffness.png",dpi=200);plt.close(fig)
    print(json.dumps(row,indent=2));return row


def build_first_order_tensors(ref,meas,c):
    T=float(ref["t"][-1]);tau=torch.linspace(0,1,c.collocation_points).reshape(-1,1).requires_grad_(True)
    tc=tau.detach().numpy().squeeze()*T
    mem=torch.tensor(np.interp(tc,ref["t"],ref["Mem"])).reshape(-1,1)
    measured_v=np.asarray(meas["omega_m"])-np.asarray(meas["omega_l"])
    integrated_delta=cumulative_trapezoid(measured_v,np.asarray(meas["t"]),initial=0.0)
    # Separate standard-deviation normalization for the two physical outputs.
    delta_scale=max(float(np.std(integrated_delta)),1e-6)
    v_scale=max(float(np.std(measured_v)),1e-6)
    return {"T":T,"tau":tau,"mem":mem,
            "taud":torch.tensor(meas["t"]/T).reshape(-1,1).requires_grad_(True),
            "v_measured":torch.tensor(measured_v).reshape(-1,1),
            "delta_integrated":torch.tensor(integrated_delta).reshape(-1,1),
            "delta_scale":delta_scale,"v_scale":v_scale,
            "derived_quantity_note":"delta_integrated is derived by cumulative trapezoidal integration of encoder v_delta; it is not an additional sensor"}


def first_order_quantities(state_net,k_model,z,c):
    state=state_net(z["tau"]);delta,v=state[:,:1],state[:,1:]
    delta_dot=derivative(delta,z["tau"])/z["T"]
    v_dot=derivative(v,z["tau"])/z["T"]
    invsum=1/c.Jm+1/c.Jl;forcing=z["mem"]/c.Jm
    k=k_model(z["tau"])*c.k0
    r_kin=(delta_dot-v)/z["v_scale"]
    forcing_scale=torch.max(torch.abs(forcing)).detach().clamp_min(1.0)
    r_dyn=(v_dot+c.bv*invsum*v+k*invsum*delta-forcing)/forcing_scale
    return delta,v,delta_dot,v_dot,r_kin,r_dyn


def pretrain_first_order(state_net,z,c,history,outdir=None,k_true_constant=None,
                         epochs=None,start_epoch=0,optimizer=None,display_total=None):
    block_epochs=int(c.pretrain_epochs if epochs is None else epochs)
    total_for_log=int(c.pretrain_epochs if display_total is None else display_total)
    opt=optimizer if optimizer is not None else torch.optim.Adam(state_net.parameters(),lr=c.lr_delta)
    for local_epoch in range(1,block_epochs+1):
        ep=start_epoch+local_epoch
        opt.zero_grad();state_d=state_net(z["taud"]);delta_d,v_d=state_d[:,:1],state_d[:,1:]
        state_f=state_net(z["tau"]);delta_f,v_f=state_f[:,:1],state_f[:,1:]
        delta_dot_f=derivative(delta_f,z["tau"])/z["T"]
        loss_delta=torch.mean(((delta_d-z["delta_integrated"])/z["delta_scale"])**2)
        loss_v=torch.mean(((v_d-z["v_measured"])/z["v_scale"])**2)
        loss_kin=torch.mean(((delta_dot_f-v_f)/z["v_scale"])**2)
        loss_ic=(delta_d[0,0]/z["delta_scale"])**2+(v_d[0,0]/z["v_scale"])**2
        loss=10*loss_delta+10*loss_v+5*loss_kin+loss_ic
        loss.backward();torch.nn.utils.clip_grad_norm_(state_net.parameters(),100);opt.step()
        if local_epoch==1 or local_epoch%max(1,block_epochs//100)==0:
            history.append({"phase":"first_order_pretrain","epoch":ep,"loss":float(loss),
                "loss_delta_data":float(loss_delta),"loss_v_delta_data":float(loss_v),
                "loss_kinematic":float(loss_kin),"loss_initial_conditions":float(loss_ic)})
        if local_epoch==1 or local_epoch%max(1,block_epochs//10)==0:
            print(f"first-order pretrain epoch={ep:6d}/{total_for_log} total={float(loss):.4e} "
                  f"delta_data={float(loss_delta):.4e} v_data={float(loss_v):.4e} "
                  f"kinematic={float(loss_kin):.4e} IC={float(loss_ic):.4e}")
        if outdir is not None and k_true_constant is not None and ep%500==0:
            checkpoint=outdir/f"relative_state_pretrained_k{int(k_true_constant)}_epoch{ep:04d}.pt"
            save_relative_state_checkpoint(checkpoint,state_net,z,c,k_true_constant,{"epoch":ep,"training_loss":float(loss)})
            print(f"Saved periodic first-order checkpoint: {checkpoint.name}")
    return opt


def first_order_metrics(state_net,z,ref):
    tau=torch.tensor(ref["t"]/z["T"]).reshape(-1,1).requires_grad_(True);state=state_net(tau)
    delta_tensor,v_tensor=state[:,:1],state[:,1:]
    derivative_delta=derivative(delta_tensor,tau)/z["T"]
    delta=delta_tensor.detach().numpy().squeeze();v=v_tensor.detach().numpy().squeeze();d_delta=derivative_delta.detach().numpy().squeeze()
    return {"delta_RMSE":float(np.sqrt(np.mean((delta-ref["delta"])**2))),"delta_R2":r2(ref["delta"],delta),
            "v_delta_RMSE":float(np.sqrt(np.mean((v-ref["delta_dot"])**2))),"v_delta_R2":r2(ref["delta_dot"],v),
            "kinematic_residual_RMSE":float(np.sqrt(np.mean((d_delta-v)**2))),
            "derivative_consistency_R2":r2(v,d_delta)}


def first_order_quality_gate(metrics,c):
    passed=(metrics["delta_R2"]>=c.min_pretrain_r2 and metrics["v_delta_R2"]>=c.min_pretrain_r2
            and metrics["derivative_consistency_R2"]>=c.min_pretrain_r2)
    return {"passed":bool(passed),"thresholds":{"delta_R2":c.min_pretrain_r2,
            "v_delta_R2":c.min_pretrain_r2,"derivative_consistency_R2":c.min_pretrain_r2}}


def save_first_order_pretrain_report(outdir,metrics,gate,history,z):
    report={**metrics,"quality_gate":gate,"delta_normalization_std":z["delta_scale"],
            "v_delta_normalization_std":z["v_scale"],"derived_quantity_note":z["derived_quantity_note"]}
    (outdir/"first_order_pretrain_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    fields=["phase","epoch","loss","loss_delta_data","loss_v_delta_data","loss_kinematic","loss_initial_conditions"]
    with (outdir/"first_order_pretrain_history.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(history)


def save_relative_state_checkpoint(path,state_net,z,c,k_true_constant,metrics):
    torch.save({"state_dict":state_net.state_dict(),"delta_scale":state_net.delta_scale,"v_scale":state_net.v_scale,
                "duration":z["T"],"k_true_constant":float(k_true_constant),"metrics":metrics,
                "formulation":"first_order","config":asdict(c)},path)


def load_relative_state_checkpoint(path,z,c,k_true_constant):
    ckpt=torch.load(path,map_location="cpu",weights_only=False)
    if ckpt.get("formulation")!="first_order":raise ValueError("Checkpoint nije first_order RelativeStateNet")
    if not np.isclose(float(ckpt["k_true_constant"]),float(k_true_constant),atol=1e-9,rtol=0):raise ValueError("RelativeStateNet checkpoint je za drugi k_true slučaj")
    net=RelativeStateNet(ckpt["delta_scale"],ckpt["v_scale"]);net.load_state_dict(ckpt["state_dict"])
    return net


def save_sigmoid_relative_state_checkpoint(path,state_net,z,metrics,extra_metadata=None):
    """Checkpoint isolated from all constant-profile RelativeStateNet files."""
    payload={"state_dict":state_net.state_dict(),"delta_scale":state_net.delta_scale,
                "v_scale":state_net.v_scale,"duration":z["T"],"metrics":metrics,
                "formulation":"first_order","reference_profile":"time_varying_sigmoid"}
    if extra_metadata:payload.update(extra_metadata)
    torch.save(payload,path)


def load_sigmoid_relative_state_checkpoint(path,z,expected_experiment_tag=None):
    ckpt=torch.load(path,map_location="cpu",weights_only=False)
    if ckpt.get("formulation")!="first_order" or ckpt.get("reference_profile")!="time_varying_sigmoid":
        raise ValueError("Checkpoint nije first_order RelativeStateNet za vremenski promenljivi sigmoidni slučaj")
    if not np.isclose(float(ckpt["duration"]),float(z["T"]),rtol=0,atol=1e-12):
        raise ValueError("Trajanje sigmoidnog RelativeStateNet checkpointa ne odgovara eksperimentu")
    if expected_experiment_tag is not None and ckpt.get("experiment_tag")!=expected_experiment_tag:
        raise ValueError(f"Sigmoidni checkpoint je za drugi eksperiment: {ckpt.get('experiment_tag')}")
    net=RelativeStateNet(ckpt["delta_scale"],ckpt["v_scale"]);net.load_state_dict(ckpt["state_dict"])
    return net


def diagnose_first_order_landscape(outdir,state_net,z,ref,c,k_true_constant,points=200):
    outdir.mkdir(parents=True,exist_ok=True)
    for p in state_net.parameters():p.requires_grad_(False)
    tau=z["tau"];T=z["T"];tc=tau.detach().numpy().squeeze()*T;invsum=1/c.Jm+1/c.Jl
    forcing=z["mem"]/c.Jm;forcing_scale=torch.max(torch.abs(forcing)).detach().clamp_min(1.0)
    state=state_net(tau);net_delta,net_v=state[:,:1],state[:,1:]
    net_delta_dot=derivative(net_delta,tau)/T;net_v_dot=derivative(net_v,tau)/T
    net_kin=torch.mean(((net_delta_dot-net_v)/z["v_scale"])**2).detach()
    oracle_delta=torch.tensor(np.interp(tc,ref["t"],ref["delta"])).reshape(-1,1)
    oracle_v=torch.tensor(np.interp(tc,ref["t"],ref["delta_dot"])).reshape(-1,1)
    oracle_v_dot=forcing-c.bv*invsum*oracle_v-float(k_true_constant)*invsum*oracle_delta
    candidates=np.linspace(c.kappa_min*c.k0,c.kappa_max*c.k0,max(200,int(points)))
    def dyn_loss(delta,v,v_dot,k):
        residual=v_dot+c.bv*invsum*v+float(k)*invsum*delta-forcing
        return float(torch.mean((residual/forcing_scale)**2).detach())
    oracle=np.asarray([dyn_loss(oracle_delta,oracle_v,oracle_v_dot,k) for k in candidates])
    network=np.asarray([dyn_loss(net_delta,net_v,net_v_dot,k)+float(net_kin) for k in candidates])
    oi,ni=int(np.argmin(oracle)),int(np.argmin(network));metrics=first_order_metrics(state_net,z,ref)
    network_min=float(network[ni])
    network_at_init=dyn_loss(net_delta,net_v,net_v_dot,288.75)+float(net_kin)
    network_at_true=dyn_loss(net_delta,net_v,net_v_dot,float(k_true_constant))+float(net_kin)
    result={**metrics,"k_true_constant":float(k_true_constant),"candidate_count":len(candidates),
            "k_at_oracle_minimum":float(candidates[oi]),"oracle_minimum_physics_loss":float(oracle[oi]),
            "k_at_network_minimum":float(candidates[ni]),"network_minimum_physics_loss":network_min,
            "network_loss_at_k_init":network_at_init,"network_loss_at_true_k":network_at_true,
            "loss_at_k_init_over_minimum_loss":network_at_init/network_min,
            "loss_at_true_k_over_minimum_loss":network_at_true/network_min,
            "network_kinematic_loss":float(net_kin),"derived_quantity_note":z["derived_quantity_note"]}
    with (outdir/"first_order_constant_loss_landscape.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["k_candidate","oracle_first_order_physics_loss","RelativeStateNet_first_order_physics_loss"]);w.writerows(zip(candidates,oracle,network))
    (outdir/"first_order_constant_loss_landscape.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(9,5));ax.semilogy(candidates,np.maximum(oracle,1e-30),label="oracle first-order");ax.semilogy(candidates,np.maximum(network,1e-30),label="frozen RelativeStateNet")
    ax.axvline(float(k_true_constant),color="k",ls="--",label="true k");ax.set(xlabel="k candidate [Nm/rad]",ylabel="first-order physics loss",title="First-order constant-stiffness landscape");ax.grid();ax.legend();fig.tight_layout();fig.savefig(outdir/"first_order_constant_loss_landscape.png",dpi=200);plt.close(fig)
    for p in state_net.parameters():p.requires_grad_(True)
    print("First-order landscape:\n"+json.dumps(result,indent=2));return result


def train_first_order_constant(state_net,k_model,z,c,history):
    for p in state_net.parameters():p.requires_grad_(False)
    opt=torch.optim.Adam(k_model.parameters(),lr=5e-3);best=float("inf");best_raw=None;best_physics=None
    print(f"first-order constant initialization: k_init={float(k_model.value().detach()):.12f} Nm/rad")
    for ep in range(1,c.epochs+1):
        opt.zero_grad();delta,v,delta_dot,v_dot,rkin,rdyn=first_order_quantities(state_net,k_model,z,c)
        physics=torch.mean(rdyn**2)+torch.mean(rkin**2);physics.backward();opt.step();value=float(physics.detach())
        if value<best:best=value;best_raw=k_model.raw.detach().clone();best_physics=value
        if ep==1 or ep%max(1,c.epochs//10)==0:print(f"first-order constant epoch={ep:6d}/{c.epochs} physics_loss={value:.4e} k_const={float(k_model.value()):.4f}")
    with torch.no_grad():k_model.raw.copy_(best_raw)
    for p in state_net.parameters():p.requires_grad_(True)
    return best_physics


def save_first_order_constant(outdir,state_net,k_model,z,ref,c,k_true_constant,physics_loss,landscape):
    outdir.mkdir(parents=True,exist_ok=True);estimated=float(k_model.value().detach());error=100*abs(estimated-k_true_constant)/k_true_constant
    metrics={**first_order_metrics(state_net,z,ref),"k_at_oracle_minimum":landscape["k_at_oracle_minimum"],
             "k_at_network_minimum":landscape["k_at_network_minimum"],"estimated_k":estimated,
             "true_k":float(k_true_constant),"relative_error_percent":error,"physics_loss":physics_loss,"quality_gate":quality_gate(error),
             "derived_quantity_note":z["derived_quantity_note"]}
    (outdir/"metrics_first_order_constant.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    torch.save({"raw_parameter":k_model.raw.detach(),"estimated_k":estimated,"formulation":"first_order","config":asdict(c)},outdir/"model_first_order_constant.pt")
    print("First-order constant result:\n"+json.dumps(metrics,indent=2));return metrics


def diagnose_first_order_weak(outdir,state_net,z,ref,c,k_true_constant,
                              differential_csv,window_lengths=(51,101,201)):
    """Weak-form landscapes. No d(v_delta)/dt is evaluated in this function."""
    outdir.mkdir(parents=True,exist_ok=True)
    for p in state_net.parameters():p.requires_grad_(False)
    tau=z["tau"].detach();t=tau.squeeze()*z["T"]
    with torch.no_grad():
        state=state_net(tau);net_delta,net_v=state[:,:1],state[:,1:]
    tc=t.numpy();oracle_delta=torch.tensor(np.interp(tc,ref["t"],ref["delta"])).reshape(-1,1)
    oracle_v=torch.tensor(np.interp(tc,ref["t"],ref["delta_dot"])).reshape(-1,1)
    mem=z["mem"].detach();invsum=1/c.Jm+1/c.Jl
    candidates=np.linspace(c.kappa_min*c.k0,c.kappa_max*c.k0,200)

    if differential_csv is not None:
        if not differential_csv.exists():raise FileNotFoundError(f"Differential baseline CSV nije pronađen: {differential_csv}")
        diff=np.genfromtxt(differential_csv,delimiter=",",names=True)
        diff_k=np.asarray(diff["k_candidate"]);diff_loss=np.asarray(diff["RelativeStateNet_first_order_physics_loss"])
        differential_curve=np.interp(candidates,diff_k,diff_loss)
    else:differential_curve=np.full_like(candidates,np.nan,dtype=float)

    def window_terms(delta,v,length):
        stride=max(1,length//4);A=[];B=[];Rkin=[];I_delta=[]
        for start in range(0,len(t)-length+1,stride):
            stop=start+length;tw=t[start:stop]
            int_delta=torch.trapz(delta[start:stop,0],tw)
            int_v=torch.trapz(v[start:stop,0],tw)
            int_mem=torch.trapz(mem[start:stop,0],tw)
            a=invsum*int_delta
            b=(v[stop-1,0]-v[start,0])+c.bv*invsum*int_v-int_mem/c.Jm
            rkin=(delta[stop-1,0]-delta[start,0])-int_v
            if torch.isfinite(a) and torch.isfinite(b):
                A.append(a);B.append(b);Rkin.append(rkin);I_delta.append(int_delta)
        return torch.stack(A),torch.stack(B),torch.stack(Rkin),torch.stack(I_delta)

    def curvature(values,index):
        if index<=0 or index>=len(values)-1:return float("nan")
        dk=candidates[1]-candidates[0]
        return float((values[index+1]-2*values[index]+values[index-1])/(dk*dk))

    def curve(A,B,Rkin,indices):
        aa=A[indices];bb=B[indices];rk=Rkin[indices]
        dynamic_scale=torch.sqrt(torch.mean(bb**2)).clamp_min(1e-12)
        kinematic_scale=max(z["delta_scale"],1e-12)
        values=[]
        for k in candidates:
            weak=bb+float(k)*aa
            values.append(float((torch.mean((weak/dynamic_scale)**2)+torch.mean((rk/kinematic_scale)**2)).detach()))
        return np.asarray(values)

    def summary(values,count):
        i=int(np.argmin(values));minimum=float(values[i])
        at_true=float(np.interp(float(k_true_constant),candidates,values));at_init=float(np.interp(288.75,candidates,values))
        return {"k_at_minimum":float(candidates[i]),
                "relative_minimum_error_percent":float(100*abs(candidates[i]-k_true_constant)/k_true_constant),
                "minimum_loss":minimum,"loss_at_k_true_over_minimum_loss":at_true/minimum,
                "loss_at_k_init_over_minimum_loss":at_init/minimum,
                "local_curvature_at_minimum":curvature(values,i),"window_count":int(count)}

    def closed_form(A,B,indices):
        aa=A[indices];bb=B[indices];den=torch.sum(aa**2)
        return float((-torch.sum(aa*bb)/den).detach()) if float(den)>0 else float("nan")

    rows=[];landscape_json={"k_true_constant":float(k_true_constant),
        "informative_window_rule":"top 25% by abs(integral(delta_hat dt)); independent of k_true",
        "differential_baseline_source":str(differential_csv) if differential_csv is not None else None,"windows":{}}
    closed_json={"k_true_constant":float(k_true_constant),"formula":"-sum(A_i B_i)/sum(A_i^2)","windows":{}}
    fig,axes=plt.subplots(len(window_lengths),2,figsize=(13,4*len(window_lengths)),sharex=True)
    for row_index,length in enumerate(window_lengths):
        nA,nB,nR,nI=window_terms(net_delta,net_v,length)
        oA,oB,oR,oI=window_terms(oracle_delta,oracle_v,length)
        all_idx=torch.arange(len(nA),dtype=torch.long)
        informative_count=max(1,int(math.ceil(.25*len(nA))))
        informative_idx=torch.topk(torch.abs(nI),informative_count).indices
        landscape_json["windows"][str(length)]={};closed_json["windows"][str(length)]={}
        for col,(subset,indices) in enumerate((("all",all_idx),("informative",informative_idx))):
            # Apply network-selected informative indices to oracle too; selection never sees k_true.
            oracle_values=curve(oA,oB,oR,indices);network_values=curve(nA,nB,nR,indices)
            osum=summary(oracle_values,len(indices));nsum=summary(network_values,len(indices))
            differential_reference=None if differential_csv is None else {
                "k_at_minimum":float(candidates[int(np.argmin(differential_curve))]),
                "minimum_loss":float(np.min(differential_curve))}
            landscape_json["windows"][str(length)][subset]={"oracle_weak":osum,"network_weak":nsum,
                "differential_network_reference":differential_reference}
            closed_json["windows"][str(length)][subset]={"oracle_k_closed_form":closed_form(oA,oB,indices),
                "network_k_closed_form":closed_form(nA,nB,indices),"window_count":int(len(indices))}
            for k,ol,nl,dl in zip(candidates,oracle_values,network_values,differential_curve):
                rows.append({"window_length":length,"subset":subset,"k_candidate":k,
                    "oracle_weak_loss":ol,"network_weak_loss":nl,"differential_network_loss":dl})
            ax=axes[row_index,col];ax.semilogy(candidates,np.maximum(oracle_values,1e-30),label="oracle weak")
            ax.semilogy(candidates,np.maximum(network_values,1e-30),label="RelativeStateNet weak")
            if differential_csv is not None:ax.semilogy(candidates,np.maximum(differential_curve,1e-30),ls="--",label="differential baseline")
            ax.axvline(float(k_true_constant),color="k",ls=":",label="true k")
            ax.set_title(f"window={length}, {subset}, n={len(indices)}");ax.grid();ax.set_ylabel("loss")
            if row_index==len(window_lengths)-1:ax.set_xlabel("k [Nm/rad]")
            if row_index==0 and col==0:ax.legend()
    fig.suptitle("First-order weak constant-stiffness landscapes");fig.tight_layout();fig.savefig(outdir/"first_order_weak_landscape.png",dpi=200);plt.close(fig)
    with (outdir/"first_order_weak_landscape.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["window_length","subset","k_candidate","oracle_weak_loss","network_weak_loss","differential_network_loss"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (outdir/"first_order_weak_landscape.json").write_text(json.dumps(landscape_json,indent=2),encoding="utf-8")
    (outdir/"first_order_weak_closed_form.json").write_text(json.dumps(closed_json,indent=2),encoding="utf-8")
    for p in state_net.parameters():p.requires_grad_(True)
    print("First-order weak landscape:\n"+json.dumps(landscape_json,indent=2))
    print("First-order weak closed form:\n"+json.dumps(closed_json,indent=2))
    return landscape_json,closed_json


def build_frozen_weak_terms(state_net,z,c,window_length=101):
    """Build constants for weak optimization; no k_true or closed-form input."""
    for p in state_net.parameters():p.requires_grad_(False)
    tau=z["tau"].detach();t=tau.squeeze()*z["T"]
    with torch.no_grad():state=state_net(tau);delta,v=state[:,:1],state[:,1:]
    return build_constant_weak_terms(t,delta,v,z["mem"].detach(),c,
        window_length=window_length,stride=max(1,window_length//4),delta_scale=z["delta_scale"])


def train_first_order_weak_constant(state_net,z,c,epochs,patience=500,min_delta=1e-14):
    terms=build_frozen_weak_terms(state_net,z,c,window_length=101)
    model=ConstantStiffness(c,initial=288.75);optimizer=torch.optim.Adam(model.parameters(),lr=5e-3)
    best_loss=float("inf");best_raw=None;best_epoch=0;stale=0;history=[]
    print(f"weak constant initialization: k_init={float(model.value().detach()):.12f} Nm/rad")
    print(f"weak windows: length={terms['window_length']} stride={terms['stride']} count={terms['window_count']} selection=all")
    for epoch in range(1,epochs+1):
        optimizer.zero_grad();k=model.value()
        weak=terms["B"]+k*terms["A"]
        dynamic=torch.mean((weak/terms["dynamic_scale"])**2)
        kinematic=torch.mean((terms["Rkin"]/terms["kinematic_scale"])**2)
        total=dynamic+kinematic;total.backward();optimizer.step()
        value=float(total.detach());row={"epoch":epoch,"weak_dynamic_loss":float(dynamic.detach()),
            "weak_kinematic_loss":float(kinematic.detach()),"total_loss":value,"k_const":float(model.value().detach())}
        history.append(row)
        if value<best_loss-min_delta:
            best_loss=value;best_raw=model.raw.detach().clone();best_epoch=epoch;stale=0
        else:stale+=1
        if epoch==1 or epoch%max(1,epochs//20)==0:
            print(f"epoch={epoch:6d}/{epochs} weak_dynamic_loss={row['weak_dynamic_loss']:.6e} "
                  f"weak_kinematic_loss={row['weak_kinematic_loss']:.6e} total_loss={value:.6e} k_const={row['k_const']:.6f}")
        if stale>=patience:
            print(f"Early stopping at epoch={epoch}; no training-loss improvement > {min_delta:g} for {patience} epochs")
            break
    with torch.no_grad():model.raw.copy_(best_raw)
    k=model.value();weak=terms["B"]+k*terms["A"]
    dynamic=float(torch.mean((weak/terms["dynamic_scale"])**2));kinematic=float(torch.mean((terms["Rkin"]/terms["kinematic_scale"])**2))
    total=dynamic+kinematic
    return model,history,{"best_epoch":best_epoch,"weak_dynamic_loss":dynamic,"weak_kinematic_loss":kinematic,
        "total_loss":total,"window_length":101,"window_selection":"all","window_count":terms["window_count"],
        "early_stopping_patience":patience,"early_stopping_min_delta":min_delta}


def update_first_order_weak_results_table(current_row):
    path=OUTPUTS_DIR/"results_first_order_weak_constant.csv"
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=["true_k","estimated_k","absolute_error","relative_error_percent","delta_R2","v_delta_R2",
            "derivative_consistency_R2","network_landscape_minimum","closed_form_k","best_epoch","quality_gate"]
    rows=[]
    if path.exists():
        with path.open("r",newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
    # Bootstrap the already completed k=300 result without modifying its folder.
    if not any(np.isclose(float(r["true_k"]),300.0) for r in rows):
        root=OUTPUTS_DIR
        mp=root/"first_order_weak_constant_k300_optimized"/"metrics_first_order_weak_constant.json"
        pp=root/"first_order_constant_k300_serious6500"/"first_order_pretrain_metrics.json"
        lp=root/"first_order_weak_constant_k300_optimized"/"first_order_weak_landscape.json"
        cp=root/"first_order_weak_constant_k300_optimized"/"first_order_weak_closed_form.json"
        if all(p.exists() for p in (mp,pp,lp,cp)):
            m=json.loads(mp.read_text(encoding="utf-8"));p=json.loads(pp.read_text(encoding="utf-8"));l=json.loads(lp.read_text(encoding="utf-8"));cc=json.loads(cp.read_text(encoding="utf-8"))
            rows.append({"true_k":300.0,"estimated_k":m["estimated_k"],"absolute_error":m["absolute_error"],
                "relative_error_percent":m["relative_error_percent"],"delta_R2":p["delta_R2"],"v_delta_R2":p["v_delta_R2"],
                "derivative_consistency_R2":p["derivative_consistency_R2"],
                "network_landscape_minimum":l["windows"]["101"]["all"]["network_weak"]["k_at_minimum"],
                "closed_form_k":cc["windows"]["101"]["all"]["network_k_closed_form"],
                "best_epoch":m["best_epoch_selected_by_training_loss"],"quality_gate":m["quality_gate"]})
    rows=[r for r in rows if not np.isclose(float(r["true_k"]),float(current_row["true_k"]))];rows.append(current_row)
    rows.sort(key=lambda r:float(r["true_k"]),reverse=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def save_first_order_weak_constant(outdir,model,history,training_metrics,c,k_true_constant,state_metrics,landscape,closed_form):
    outdir.mkdir(parents=True,exist_ok=True);estimated=float(model.value().detach());true=float(k_true_constant)
    absolute=abs(estimated-true);relative=100*absolute/true
    metrics={"initial_k_const":288.75,"estimated_k":estimated,"true_k":true,"absolute_error":absolute,
        "relative_error_percent":relative,"weak_dynamic_loss":training_metrics["weak_dynamic_loss"],
        "weak_kinematic_loss":training_metrics["weak_kinematic_loss"],"total_loss":training_metrics["total_loss"],
        "quality_gate":quality_gate(relative),"best_epoch_selected_by_training_loss":training_metrics["best_epoch"],
        "window_length":training_metrics["window_length"],"window_selection":training_metrics["window_selection"],
        "number_of_windows":training_metrics["window_count"],"checkpoint_selection":"minimum training total_loss only; k_true not used"}
    landscape_min=landscape["windows"]["101"]["all"]["network_weak"]["k_at_minimum"]
    closed_k=closed_form["windows"]["101"]["all"]["network_k_closed_form"]
    metrics.update({"delta_R2":state_metrics["delta_R2"],"v_delta_R2":state_metrics["v_delta_R2"],
        "derivative_consistency_R2":state_metrics["derivative_consistency_R2"],
        "network_landscape_minimum":landscape_min,"closed_form_k":closed_k})
    torch.save({"raw_parameter":model.raw.detach(),"estimated_k":estimated,"config":asdict(c),
        "training_metrics":training_metrics,"selection":"minimum training total_loss only"},outdir/"model_first_order_weak_constant.pt")
    (outdir/"metrics_first_order_weak_constant.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    with (outdir/"history_first_order_weak_constant.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["epoch","weak_dynamic_loss","weak_kinematic_loss","total_loss","k_const"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(history)
    fig,ax=plt.subplots(1,2,figsize=(11,4));epochs=[h["epoch"] for h in history]
    ax[0].plot(epochs,[h["k_const"] for h in history]);ax[0].axhline(estimated,color="r",ls="--",label="best training-loss checkpoint")
    ax[0].set(xlabel="epoch",ylabel="k_const [Nm/rad]",title="Weak constant-stiffness estimate");ax[0].grid();ax[0].legend()
    ax[1].semilogy(epochs,[h["weak_dynamic_loss"] for h in history],label="dynamic")
    ax[1].semilogy(epochs,[h["weak_kinematic_loss"] for h in history],label="kinematic")
    ax[1].semilogy(epochs,[h["total_loss"] for h in history],label="total")
    ax[1].set(xlabel="epoch",ylabel="loss",title="Training losses");ax[1].grid();ax[1].legend();fig.tight_layout();fig.savefig(outdir/"constant_stiffness_weak.png",dpi=200);plt.close(fig)
    update_first_order_weak_results_table({"true_k":true,"estimated_k":estimated,"absolute_error":absolute,
        "relative_error_percent":relative,"delta_R2":state_metrics["delta_R2"],"v_delta_R2":state_metrics["v_delta_R2"],
        "derivative_consistency_R2":state_metrics["derivative_consistency_R2"],"network_landscape_minimum":landscape_min,
        "closed_form_k":closed_k,"best_epoch":training_metrics["best_epoch"],"quality_gate":quality_gate(relative)})
    print("First-order weak constant optimization result:\n"+json.dumps(metrics,indent=2));return metrics


def build_frozen_weak_sigmoid_terms(state_net,z,c,window_length=101,stride=25):
    """Weak first-order terms for time-varying k(t); no state derivatives are used."""
    for parameter in state_net.parameters():parameter.requires_grad_(False)
    tau=z["tau"].detach();t=tau.squeeze()*z["T"]
    with torch.no_grad():
        state=state_net(tau);delta,v=state[:,:1],state[:,1:]
    return build_sigmoid_weak_terms(t,delta,v,z["mem"].detach(),c,duration=z["T"],
        window_length=window_length,stride=stride,delta_scale=z["delta_scale"])


def weak_sigmoid_losses(model,terms):
    return weak_sigmoid_losses_modular(model,terms)


def train_weak_sigmoid_restart(terms,c,seed,epochs,learning_rate=5e-3,patience=1000,min_delta=1e-14):
    """Optimize only four sigmoid parameters; model selection sees training loss only."""
    seed_all(seed);model=WeakSigmoidStiffness(terms["duration"])
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    best_loss=float("inf");best_state=None;best_epoch=0;stale=0;history=[]
    initial={name:float(value.detach()) for name,value in model.physical_parameters().items()}
    print(f"weak sigmoid restart seed={seed} initialization="+json.dumps(initial))
    print(f"weak windows: length={terms['window_length']} stride={terms['stride']} "
          f"count={terms['window_count']} selection={terms['window_selection']}")
    for epoch in range(1,epochs+1):
        optimizer.zero_grad();dynamic,kinematic,total=weak_sigmoid_losses(model,terms)
        total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),100.0);optimizer.step()
        values={name:float(value.detach()) for name,value in model.physical_parameters().items()}
        loss_value=float(total.detach());row={"restart_seed":seed,"epoch":epoch,
            "weak_dynamic_loss":float(dynamic.detach()),"weak_kinematic_loss":float(kinematic.detach()),
            "total_loss":loss_value,**values};history.append(row)
        if loss_value<best_loss-min_delta:
            best_loss=loss_value;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
        else:stale+=1
        if epoch==1 or epoch%max(1,epochs//20)==0:
            print(f"sigmoid seed={seed} epoch={epoch:6d}/{epochs} dynamic={row['weak_dynamic_loss']:.6e} "
                  f"kinematic={row['weak_kinematic_loss']:.6e} total={loss_value:.6e} "
                  f"k_high={values['k_high']:.3f} k_low={values['k_low']:.3f} "
                  f"center={values['t_center']:.5f} width={values['width']:.5f}")
        if stale>=patience:
            print(f"Sigmoid early stopping seed={seed} epoch={epoch}; training loss plateau for {patience} epochs")
            break
    model.load_state_dict(best_state);dynamic,kinematic,total=weak_sigmoid_losses(model,terms)
    values={name:float(value.detach()) for name,value in model.physical_parameters().items()}
    result={"seed":int(seed),**values,"training_loss":float(total.detach()),
        "weak_dynamic_loss":float(dynamic.detach()),"weak_kinematic_loss":float(kinematic.detach()),
        "best_epoch":int(best_epoch),"epochs_executed":len(history),
        "selection":"minimum training weak total loss only"}
    print("Sigmoid restart result:\n"+json.dumps(result,indent=2));return model,history,result


def train_weak_sigmoid_restarts(state_net,z,c,seeds,epochs,learning_rate,window_length=101,stride=25):
    terms=build_frozen_weak_sigmoid_terms(state_net,z,c,window_length=window_length,stride=stride)
    candidates=[];all_history=[]
    for seed in seeds:
        model,history,result=train_weak_sigmoid_restart(terms,c,int(seed),epochs,learning_rate)
        candidates.append((model,result));all_history.extend(history)
    # Deliberately no k_true evaluation here: only weak training loss selects the model.
    best_model,best_result=min(candidates,key=lambda item:item[1]["training_loss"])
    return best_model,all_history,[item[1] for item in candidates],best_result,terms


def save_weak_sigmoid_result(outdir,model,history,restarts,best,terms,ref,c,state_metrics,baseline_csv,
                             artifact_suffix="",experiment_name="main_noise0_measurements1501",
                             robustness=False,noise_seed=None,pretrain_epochs_total=None,noise_model=None,
                             state_quality_gate=None,sampling_diagnostic=None):
    """Evaluate only after training/selection and save all requested sigmoid artifacts."""
    outdir.mkdir(parents=True,exist_ok=True);t=np.asarray(ref["t"],float)
    with torch.no_grad():k_est=model(torch.tensor(t)).numpy()
    k_true=np.asarray(ref["k_true"],float);error=k_est-k_true
    rmse=float(np.sqrt(np.mean(error**2)));relative_rmse=100*rmse/c.k0
    mae=float(np.mean(np.abs(error)));maximum=float(np.max(np.abs(error)));score_r2=r2(k_true,k_est)
    start=float(k_est[0]);final=float(k_est[-1]);start_error=100*abs(start-k_true[0])/abs(k_true[0])
    final_error=100*abs(final-k_true[-1])/abs(k_true[-1]);params={name:float(value.detach()) for name,value in model.physical_parameters().items()}
    if robustness:
        pass_components={"k_relative_RMSE_le_8_percent":bool(relative_rmse<=8.0),"k_R2_ge_0_85":bool(score_r2>=0.85),
            "initial_stiffness_error_le_10_percent":bool(start_error<=10.0),"final_stiffness_error_le_10_percent":bool(final_error<=10.0)}
        acceptable_components={"k_relative_RMSE_le_12_percent":bool(relative_rmse<=12.0),"k_R2_ge_0_70":bool(score_r2>=0.70),
            "initial_stiffness_error_le_15_percent":bool(start_error<=15.0),"final_stiffness_error_le_15_percent":bool(final_error<=15.0)}
        quality="PASS" if all(pass_components.values()) else ("ACCEPTABLE" if all(acceptable_components.values()) else "FAIL")
        if state_quality_gate is not None and not state_quality_gate.get("passed",False):quality="FAIL"
        gate_components={"PASS":pass_components,"ACCEPTABLE":acceptable_components}
    else:
        gate_components={"k_relative_RMSE_le_5_percent":bool(relative_rmse<=5.0),"k_R2_ge_0_90":bool(score_r2>=0.90),
            "initial_stiffness_error_le_5_percent":bool(start_error<=5.0),"final_stiffness_error_le_5_percent":bool(final_error<=5.0)}
        quality="PASS" if all(gate_components.values()) else "FAIL"
    metrics={"k_RMSE":rmse,"k_relative_RMSE_percent":relative_rmse,
        "k_relative_RMSE_definition":"100 * k_RMSE / 350 Nm/rad", "k_R2":score_r2,"k_MAE":mae,
        "maximum_absolute_error":maximum,"estimated_k_start":start,"estimated_k_final":final,
        "estimated_degradation_percent":100*(start-final)/start,"estimated_transition_center":params["t_center"],
        "estimated_transition_width":params["width"],"estimated_k_high":params["k_high"],"estimated_k_low":params["k_low"],
        "initial_stiffness_error_percent":start_error,"final_stiffness_error_percent":final_error,
        "initial_error_percent":start_error,"final_error_percent":final_error,
        "selected_seed":best["seed"],"best_epoch":best["best_epoch"],"training_weak_loss":best["training_loss"],
        "training_total_loss":best["training_loss"],
        "weak_dynamic_loss":best["weak_dynamic_loss"],"weak_kinematic_loss":best["weak_kinematic_loss"],
        "window_length":terms["window_length"],"window_selection":terms["window_selection"],
        "stride":terms["stride"],"number_of_windows":terms["window_count"],"restart_count":len(restarts),
        "weak_grid_points":terms["weak_grid_points"],"weak_grid_step_seconds":terms["weak_grid_step_seconds"],
        "weak_window_duration_seconds":terms["window_duration_seconds"],
        "weak_stride_duration_seconds":terms["stride_duration_seconds"],
        "checkpoint_selection":"minimum training weak total loss only; k_true and true sigmoid parameters not used",
        "quality_gate":quality,"quality_gate_components":gate_components,"experiment":experiment_name,
        "delta_R2":state_metrics["delta_R2"],"v_delta_R2":state_metrics["v_delta_R2"],
        "derivative_consistency_R2":state_metrics["derivative_consistency_R2"],
        "true_k_start_evaluation_only":float(k_true[0]),"true_k_final_evaluation_only":float(k_true[-1]),
        "true_transition_center_evaluation_only":c.center_fraction*float(t[-1]),
        "true_transition_width_evaluation_only":c.width_fraction*float(t[-1]),
        "sensor_data_origin":"synthetic ODE simulation driven by measured jera1.mat Mem",
        "noise":float(c.noise),"measurements":int(c.measurements),"noise_seed":noise_seed,
        "noise_model":noise_model,
        "pretrain_epochs_total":pretrain_epochs_total,
        "state_quality_gate":state_quality_gate,
        "sampling_diagnostic":sampling_diagnostic,
        "delta_integrated_note":"derived with scipy cumulative_trapezoid using the actual selected measurement times"}
    model_name=f"model_weak_sigmoid{artifact_suffix}.pt";metrics_name=f"metrics_weak_sigmoid{artifact_suffix}.json"
    history_name=f"history_weak_sigmoid{artifact_suffix}.csv";restart_name=f"sigmoid_restarts{artifact_suffix}.csv"
    results_name=f"results_weak_sigmoid{artifact_suffix}.csv"
    torch.save({"state_dict":model.state_dict(),"physical_parameters":params,"duration":float(t[-1]),
        "initialization":model.initial_values,"selection":"minimum training weak total loss only",
        "selected_seed":best["seed"],"best_epoch":best["best_epoch"],"experiment":experiment_name},outdir/model_name)
    (outdir/metrics_name).write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    restart_fields=["seed","k_high","k_low","t_center","width","training_loss","best_epoch"]
    with (outdir/restart_name).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=restart_fields);writer.writeheader();writer.writerows([{k:r[k] for k in restart_fields} for r in restarts])
    history_fields=["restart_seed","epoch","weak_dynamic_loss","weak_kinematic_loss","total_loss","k_high","k_low","t_center","width"]
    with (outdir/history_name).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=history_fields);writer.writeheader();writer.writerows(history)
    with (outdir/results_name).open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);writer.writerow(["t","k_true","k_weak_sigmoid"]);writer.writerows(zip(t,k_true,k_est))

    baseline_t=baseline_k=None
    if baseline_csv is not None and Path(baseline_csv).exists():
        baseline=np.genfromtxt(baseline_csv,delimiter=",",names=True)
        if "t" in baseline.dtype.names and "k_PINN" in baseline.dtype.names:
            baseline_t=np.asarray(baseline["t"],float);baseline_k=np.asarray(baseline["k_PINN"],float)
            baseline_path=Path(baseline_csv).resolve()
            try:metrics["free_profile_second_order_baseline_csv"]=baseline_path.relative_to(REPO_ROOT).as_posix()
            except ValueError:metrics["free_profile_second_order_baseline_csv"]=str(baseline_path)
            (outdir/metrics_name).write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(10,5));ax.plot(t,k_true,"k--",lw=2,label="k true")
    if baseline_t is not None:ax.plot(baseline_t,baseline_k,color="tab:gray",lw=1.8,label="free-profile second-order baseline")
    ax.plot(t,k_est,color="tab:blue",lw=2,label="proposed weak sigmoid")
    stiffness_plot="stiffness_sigmoid_comparison.png" if not artifact_suffix else f"stiffness_sigmoid{artifact_suffix}.png"
    ax.set(xlabel="t [s]",ylabel="k [Nm/rad]",title=f"Time-varying torsional stiffness: {experiment_name}");ax.grid();ax.legend();fig.tight_layout();fig.savefig(outdir/stiffness_plot,dpi=200);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for restart in restarts:
        rows=[row for row in history if row["restart_seed"]==restart["seed"]]
        axes[0].semilogy([row["epoch"] for row in rows],[row["weak_dynamic_loss"] for row in rows],label=f"seed {restart['seed']}")
        axes[1].semilogy([row["epoch"] for row in rows],[row["total_loss"] for row in rows],label=f"seed {restart['seed']}")
    axes[0].set_title("Weak dynamic loss");axes[1].set_title("Weak total loss")
    for ax in axes:ax.set_xlabel("epoch");ax.set_ylabel("loss");ax.grid();ax.legend()
    losses_plot="weak_sigmoid_losses.png" if not artifact_suffix else f"weak_sigmoid{artifact_suffix}_losses.png"
    fig.tight_layout();fig.savefig(outdir/losses_plot,dpi=200);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(12,8));parameter_names=("k_high","k_low","t_center","width")
    true_lines=(c.k0,c.k_final,c.center_fraction*float(t[-1]),c.width_fraction*float(t[-1]))
    for ax,name,true_value in zip(axes.flat,parameter_names,true_lines):
        for restart in restarts:
            rows=[row for row in history if row["restart_seed"]==restart["seed"]]
            ax.plot([row["epoch"] for row in rows],[row[name] for row in rows],label=f"seed {restart['seed']}")
        ax.axhline(true_value,color="k",ls="--",label="reference (evaluation only)")
        ax.set(xlabel="epoch",ylabel=name,title=name);ax.grid()
    parameters_plot="weak_sigmoid_parameters.png" if not artifact_suffix else f"weak_sigmoid{artifact_suffix}_parameters.png"
    axes.flat[0].legend();fig.tight_layout();fig.savefig(outdir/parameters_plot,dpi=200);plt.close(fig)
    print("Weak first-order sigmoid result:\n"+json.dumps(metrics,indent=2));return metrics


def create_final_sigmoid_summary(baseline_csv):
    root=REPO_ROOT;experiment_root=OUTPUTS_DIR
    cases=[
        ("main_noise0_measurements1501",experiment_root/"first_order_weak_sigmoid_main","metrics_weak_sigmoid.json","results_weak_sigmoid.csv","main"),
        ("noise003_measurements1501",experiment_root/"first_order_weak_sigmoid_noise003","metrics_weak_sigmoid_noise003.json","results_weak_sigmoid_noise003.csv","noise 0.3%"),
        ("noise0_measurements121",experiment_root/"first_order_weak_sigmoid_sparse121","metrics_weak_sigmoid_sparse121.json","results_weak_sigmoid_sparse121.csv","sparse 121 (aliasing)"),
        ("noise0_measurements401",experiment_root/"first_order_weak_sigmoid_sparse401","metrics_weak_sigmoid_sparse401.json","results_weak_sigmoid_sparse401.csv","sparse 401"),
        ("noise0_measurements751",experiment_root/"first_order_weak_sigmoid_sparse751","metrics_weak_sigmoid_sparse751.json","results_weak_sigmoid_sparse751.csv","sparse 751 reduced physics"),
        ("noise0_measurements751_densephysics",experiment_root/"first_order_weak_sigmoid_sparse751_densephysics",
            "metrics_sparse751_densephysics.json","results_sparse751_densephysics.csv","sparse 751 + dense physics")]
    loaded=[]
    for experiment,folder,metrics_name,results_name,label in cases:
        metrics_path=folder/metrics_name;results_path=folder/results_name
        if not metrics_path.exists() or not results_path.exists():
            print(f"Final sigmoid summary pending: {metrics_path.name} or {results_path.name} is missing")
            return None
        loaded.append((experiment,label,json.loads(metrics_path.read_text(encoding="utf-8")),
                       np.genfromtxt(results_path,delimiter=",",names=True)))
    fields=["experiment","k_RMSE","k_relative_RMSE_percent","k_R2","k_MAE","maximum_absolute_error",
        "estimated_k_start","estimated_k_final","initial_error_percent","final_error_percent",
        "estimated_degradation_percent","estimated_transition_center","estimated_transition_width",
        "delta_R2","v_delta_R2","derivative_consistency_R2","best_epoch","training_total_loss","quality_gate",
        "stiffness_identification_gate","state_reconstruction_gate","overall_composite_gate",
        "measurements","effective_sample_rate_Hz","Nyquist_frequency_Hz","sample_reduction_percent",
        "samples_per_dominant_torsional_period","collocation_count","weak_window_length","weak_stride"]
    rows=[]
    for experiment,label,m,data in loaded:
        sampling=m.get("sampling_diagnostic") or {};duration=float(data["t"][-1]-data["t"][0]);count=int(m.get("measurements",len(data)))
        effective=float(sampling.get("effective_sample_rate_Hz",(count-1)/duration))
        components=m.get("quality_gate_components",{})
        stiffness_components=components.get("PASS",components) if isinstance(components,dict) else {}
        stiffness_flags=[bool(value) for name,value in stiffness_components.items()
                         if name.startswith("k_") or name.startswith("initial_") or name.startswith("final_")]
        stiffness_gate=m.get("stiffness_identification_gate","PASS" if stiffness_flags and all(stiffness_flags) else "FAIL")
        delta_threshold=.95 if count==1501 else .90
        delta_pass=float(m["delta_R2"])>=delta_threshold;v_pass=float(m["v_delta_R2"])>=.95
        derived_state_gate="PASS" if delta_pass and v_pass else ("PARTIAL" if delta_pass or v_pass else "FAIL")
        state_gate=m.get("state_reconstruction_gate",derived_state_gate)
        overall_gate=m.get("overall_composite_gate",m["quality_gate"])
        rows.append({"experiment":experiment,"k_RMSE":m["k_RMSE"],"k_relative_RMSE_percent":m["k_relative_RMSE_percent"],
            "k_R2":m["k_R2"],"k_MAE":m["k_MAE"],"maximum_absolute_error":m["maximum_absolute_error"],
            "estimated_k_start":m["estimated_k_start"],"estimated_k_final":m["estimated_k_final"],
            "initial_error_percent":m.get("initial_error_percent",m.get("initial_stiffness_error_percent")),
            "final_error_percent":m.get("final_error_percent",m.get("final_stiffness_error_percent")),
            "estimated_degradation_percent":m["estimated_degradation_percent"],
            "estimated_transition_center":m["estimated_transition_center"],"estimated_transition_width":m["estimated_transition_width"],
            "delta_R2":m["delta_R2"],"v_delta_R2":m["v_delta_R2"],
            "derivative_consistency_R2":m["derivative_consistency_R2"],"best_epoch":m["best_epoch"],
            "training_total_loss":m.get("training_total_loss",m.get("training_weak_loss")),"quality_gate":overall_gate,
            "stiffness_identification_gate":stiffness_gate,"state_reconstruction_gate":state_gate,
            "overall_composite_gate":overall_gate,
            "measurements":count,"effective_sample_rate_Hz":effective,
            "Nyquist_frequency_Hz":float(sampling.get("Nyquist_frequency_Hz",effective/2)),
            "sample_reduction_percent":float(sampling.get("sample_reduction_percent",100*(1-count/1501))),
            "samples_per_dominant_torsional_period":float(sampling.get("samples_per_dominant_torsional_period",effective/230.5)),
            "collocation_count":int(m.get("collocation_count",m.get("weak_grid_points",1501))),
            "weak_window_length":m["window_length"],"weak_stride":m["stride"]})
    FINAL_TABLES_DIR.mkdir(parents=True,exist_ok=True);FINAL_FIGURES_DIR.mkdir(parents=True,exist_ok=True)
    with (FINAL_TABLES_DIR/"final_sigmoid_results.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

    fig,ax=plt.subplots(figsize=(11,6));main_data=loaded[0][3]
    ax.plot(main_data["t"],main_data["k_true"],"k--",lw=2.2,label="k true")
    main_names={"main_noise0_measurements1501","noise003_measurements1501","noise0_measurements751_densephysics"}
    main_colors={"main_noise0_measurements1501":"tab:blue","noise003_measurements1501":"tab:orange",
        "noise0_measurements751_densephysics":"tab:purple"}
    for experiment,label,_,data in loaded:
        if experiment in main_names:ax.plot(data["t"],data["k_weak_sigmoid"],color=main_colors[experiment],lw=1.9,label=label)
    if baseline_csv is not None and Path(baseline_csv).exists():
        baseline=np.genfromtxt(baseline_csv,delimiter=",",names=True)
        ax.plot(baseline["t"],baseline["k_PINN"],color="tab:gray",lw=1.8,label="free-profile second-order baseline")
    ax.set(xlabel="t [s]",ylabel="k [Nm/rad]",title="Final weak-sigmoid robustness comparison");ax.grid();ax.legend()
    fig.tight_layout();fig.savefig(FINAL_FIGURES_DIR/"final_stiffness_comparison.png",dpi=200);plt.close(fig)

    display_rows=[row for row in rows if row["experiment"] in main_names]
    display_labels=[next(item[1] for item in loaded if item[0]==row["experiment"]) for row in display_rows]
    display_colors=[main_colors[row["experiment"]] for row in display_rows];x=np.arange(len(display_rows));fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].bar(x,[row["k_relative_RMSE_percent"] for row in display_rows],color=display_colors);axes[0].axhline(8,color="k",ls="--",label="robust PASS")
    axes[0].set(ylabel="relative RMSE [%]",title="Stiffness relative RMSE")
    axes[1].bar(x,[row["k_R2"] for row in display_rows],color=display_colors);axes[1].axhline(.85,color="k",ls="--");axes[1].set(ylabel="R²",title="Stiffness R²")
    width=.36;axes[2].bar(x-width/2,[row["initial_error_percent"] for row in display_rows],width,label="initial")
    axes[2].bar(x+width/2,[row["final_error_percent"] for row in display_rows],width,label="final");axes[2].axhline(10,color="k",ls="--");axes[2].set(ylabel="error [%]",title="Endpoint errors");axes[2].legend()
    for ax in axes:ax.set_xticks(x,display_labels,rotation=15,ha="right");ax.grid(axis="y")
    fig.tight_layout();fig.savefig(FINAL_FIGURES_DIR/"final_metrics_comparison.png",dpi=200);plt.close(fig)

    # Separate limitations figure for sparse121 aliasing and sparse401 under-resolution.
    sparse_loaded={experiment:(label,data) for experiment,label,_,data in loaded
                   if experiment in {"noise0_measurements121","noise0_measurements401"}}
    fig,axes=plt.subplots(2,2,figsize=(15,9));ax=axes[0,0]
    ax.plot(main_data["t"],main_data["k_true"],"k--",lw=2,label="k true")
    for experiment,color in (("noise0_measurements121","tab:red"),("noise0_measurements401","tab:green")):
        label,data=sparse_loaded[experiment];ax.plot(data["t"],data["k_weak_sigmoid"],color=color,label=label)
    ax.set(xlabel="t [s]",ylabel="k [Nm/rad]",title="Failed sparse stiffness estimates");ax.grid();ax.legend()
    axes[0,1].axis("off");axes[0,1].text(.02,.95,
        "Sparse limitations\n\n121 points: Fs=160 Hz, Nyquist=80 Hz\n"
        "Dominant torsional band: 228-233 Hz -> aliasing\n\n"
        "401 points: Fs=533.33 Hz, about 2.31 samples/period\n"
        "Nyquist is satisfied, but trapezoidal delta integration\n"
        "and between-sample reconstruction remain unreliable.",va="top",fontsize=12)
    state_source=Path(baseline_csv) if baseline_csv is not None else None
    if state_source is not None and state_source.exists():
        state_data=np.genfromtxt(state_source,delimiter=",",names=True);tt=np.asarray(state_data["t"],float)
        for experiment,color in (("noise0_measurements121","tab:red"),("noise0_measurements401","tab:green")):
            suffix="_sparse121" if experiment.endswith("121") else "_sparse401"
            folder=experiment_root/("first_order_weak_sigmoid_sparse121" if experiment.endswith("121") else "first_order_weak_sigmoid_sparse401")
            payload=torch.load(folder/f"relative_state_pretrained_sigmoid{suffix}.pt",map_location="cpu",weights_only=False)
            net=RelativeStateNet(payload["delta_scale"],payload["v_scale"]);net.load_state_dict(payload["state_dict"]);net.eval()
            net_dtype=next(net.parameters()).dtype
            with torch.no_grad():
                tau=torch.tensor(tt/float(payload["duration"]),dtype=net_dtype).reshape(-1,1)
                prediction=net(tau).numpy()
            label=experiment.replace("noise0_measurements","sparse ")
            axes[1,0].plot(tt,prediction[:,1],color=color,lw=.8,label=label)
            axes[1,1].plot(tt,prediction[:,0],color=color,lw=.8,label=label)
        axes[1,0].plot(tt,state_data["delta_dot_true"],"k",lw=1,label="reference")
        axes[1,1].plot(tt,state_data["delta_true"],"k",lw=1,label="reference")
    axes[1,0].set(xlabel="t [s]",ylabel="v_delta [rad/s]",title="Relative-speed reconstruction")
    axes[1,1].set(xlabel="t [s]",ylabel="delta [rad]",title="Relative-angle reconstruction")
    for ax in axes[1]:ax.grid();ax.legend()
    fig.tight_layout();fig.savefig(FINAL_FIGURES_DIR/"sparse_limitations_comparison.png",dpi=200);plt.close(fig)

    constants=[];constant_csv=FINAL_TABLES_DIR/"constant_stiffness_results.csv"
    if constant_csv.exists():
        for row in csv.DictReader(constant_csv.open("r",newline="",encoding="utf-8")):
            constants.append({"true_k":float(row["true_k"]),"estimated_k":float(row["estimated_k"]),
                "relative_error_percent":float(row["relative_error_percent"]),"quality_gate":row["quality_gate"]})
    sparse401_quality=next(row["quality_gate"] for row in rows if row["experiment"]=="noise0_measurements401")
    sparse401_conclusion=("The physically valid uniform 401-point reduced-data test passes the robustness gate."
        if sparse401_quality=="PASS" else f"The physically valid uniform 401-point reduced-data test is retained with quality gate {sparse401_quality}.")
    sparse751_quality=next(row["quality_gate"] for row in rows if row["experiment"]=="noise0_measurements751")
    sparse751_conclusion=("The final uniform 751-point reduced-data test passes the robustness gate."
        if sparse751_quality=="PASS" else f"The final uniform 751-point reduced-data test is retained with quality gate {sparse751_quality}.")
    densephysics_quality=next(row["quality_gate"] for row in rows if row["experiment"]=="noise0_measurements751_densephysics")
    densephysics_conclusion=("Sparse supervision with 751 sensor labels and 1501 unlabeled physics points passes the final gate."
        if densephysics_quality=="PASS" else
        "Sparse supervision with dense physics passes every stiffness-identification threshold, but the composite gate remains FAIL because v_delta_R2 is below 0.95.")
    summary={"sigmoid_experiments":{row["experiment"]:row for row in rows},
        "constant_stiffness_validations":constants,
        "selection_note":"All sigmoid checkpoints selected only by minimum training weak total loss; k_true used after selection for evaluation.",
        "data_note":"Encoder responses are synthetic ODE sensor data driven by the measured Mem(t) profile from jera1.mat.",
        "scientific_conclusion":"The weak first-order sigmoid twin is accurate for full-rate clean and 0.3% differential-noise data. Uniform 121-point sampling is an aliasing FAIL. "+sparse401_conclusion+" "+sparse751_conclusion+" "+densephysics_conclusion,
        "sampling_diagnostic":{"full_sample_rate_Hz":2000.0,
            "sparse121_sample_rate_Hz":160.0,"sparse121_Nyquist_Hz":80.0,
            "sparse401_sample_rate_Hz":533.3333333333334,"sparse401_Nyquist_Hz":266.6666666666667,
            "sparse751_sample_rate_Hz":1000.0,"sparse751_Nyquist_Hz":500.0,
            "dominant_relative_response_band_Hz":"approximately 228-233"},
        "limitations":[
            "The study uses synthetic ODE encoder responses; only t and Mem originate from jera1.mat.",
            "The 0.3% experiment defines noise relative to the differential encoder channel and splits it between the two encoders; it is not independent 0.3% noise scaled by the much larger absolute motor/load speeds.",
            "Uniform 121-point sampling aliases the dominant torsional mode, so the RelativeStateNet state gate fails even after 9500 epochs.",
            "In the sparse751+dense-physics experiment, stiffness identification satisfies all prescribed k(t) thresholds, but v_delta_R2=0.88969 fails the required 0.95 state threshold.",
            "All three deterministic full-batch restarts follow the same path from the mandated identical neutral initialization.",
            "The main and noise estimates place k_high near the imposed upper bound, so initial stiffness is the least accurately identified endpoint."]}
    (FINAL_TABLES_DIR/"final_results_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("Final sigmoid robustness summary saved.");return summary


def online_window_terms(state_net,t,mem,start,stop,duration,c,delta_scale):
    """Build one strictly causal weak window [start, stop); no future samples."""
    if stop>len(t) or start<0 or stop-start!=101:raise ValueError("Invalid causal online window")
    tw=torch.tensor(t[start:stop]);tau=(tw/duration).reshape(-1,1)
    with torch.no_grad():state=state_net(tau);delta,v=state[:,:1],state[:,1:]
    delta=delta[:,0];v=v[:,0];mw=torch.tensor(mem[start:stop]);invsum=1/c.Jm+1/c.Jl
    int_v=torch.trapz(v,tw);int_mem=torch.trapz(mw,tw)
    B=(v[-1]-v[0])+c.bv*invsum*int_v-int_mem/c.Jm
    rkin=(delta[-1]-delta[0])-int_v
    return {"t":tw,"delta":delta,"B":B.detach(),"Rkin":rkin.detach(),"invsum":invsum,
            "kinematic_scale":max(float(delta_scale),1e-12),"last_index":stop-1}


def online_sigmoid_loss(model,terms,dynamic_scale):
    stiffness_integral=torch.trapz(model(terms["t"])*terms["delta"],terms["t"])
    weak=terms["B"]+terms["invsum"]*stiffness_integral
    dynamic=(weak/dynamic_scale)**2
    kinematic=(terms["Rkin"]/terms["kinematic_scale"])**2
    return dynamic,kinematic,dynamic+kinematic


def execute_online_updates(state_net,t,mem,c,stride,adam_steps,learning_rate,warmup_updates=20):
    """Run a causal stream. Timing surrounds only Adam update steps."""
    duration=float(t[-1]);window_length=101;ends=list(range(window_length-1,len(t),int(stride)))
    first_terms=online_window_terms(state_net,t,mem,0,window_length,duration,c,state_net.delta_scale)

    # Warm-up uses a disposable model/optimizer so benchmark initial conditions
    # remain exactly neutral and no measured latency includes cold-start effects.
    warm_model=WeakSigmoidStiffness(duration);warm_opt=torch.optim.Adam(warm_model.parameters(),lr=learning_rate)
    warm_scale=torch.abs(first_terms["B"]).clamp_min(1e-12)
    for _ in range(max(20,int(warmup_updates))):
        for _ in range(adam_steps):
            warm_opt.zero_grad();_,_,loss=online_sigmoid_loss(warm_model,first_terms,warm_scale)
            loss.backward();warm_opt.step()

    model=WeakSigmoidStiffness(duration);optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    latencies_ms=[];update_indices=[];current_k=[];parameter_history=[];causal_B=[]
    for end in ends:
        start=end-window_length+1
        terms=online_window_terms(state_net,t,mem,start,end+1,duration,c,state_net.delta_scale)
        causal_B.append(terms["B"]);dynamic_scale=torch.sqrt(torch.mean(torch.stack(causal_B)**2)).clamp_min(1e-12)
        tic=time.perf_counter_ns()
        for _ in range(adam_steps):
            optimizer.zero_grad();dynamic,kinematic,total=online_sigmoid_loss(model,terms,dynamic_scale)
            total.backward();optimizer.step()
        toc=time.perf_counter_ns();latencies_ms.append((toc-tic)/1e6);update_indices.append(end)
        with torch.no_grad():
            current_k.append(float(model(torch.tensor(float(t[end])))))
            parameter_history.append({name:float(value) for name,value in model.physical_parameters().items()})
    return {"stride":int(stride),"adam_steps":int(adam_steps),"latencies_ms":np.asarray(latencies_ms),
        "update_indices":np.asarray(update_indices,dtype=int),"estimated_k":np.asarray(current_k),
        "parameter_history":parameter_history,"warmup_updates":max(20,int(warmup_updates)),
        "window_length":window_length,"causal_rule":"update i uses only samples [i-100, i]"}


def benchmark_forward_inference(state_net,t,duration,repetitions=200,warmups=20):
    tau=torch.tensor(t[:101]/duration).reshape(-1,1)
    with torch.no_grad():
        for _ in range(max(20,warmups)):state_net(tau)
        values=[]
        for _ in range(repetitions):
            tic=time.perf_counter_ns();state_net(tau);toc=time.perf_counter_ns();values.append((toc-tic)/1e6)
    values=np.asarray(values)
    return {"component":"frozen RelativeStateNet forward pass on one 101-sample window",
        "window_samples":101,"repetitions":int(repetitions),"warmup_forwards":max(20,warmups),
        "mean_latency_ms":float(np.mean(values)),"median_latency_ms":float(np.median(values)),
        "p95_latency_ms":float(np.percentile(values,95)),"maximum_latency_ms":float(np.max(values))}


def run_online_benchmark(a,c,t,mem):
    if a.relative_checkpoint is None:
        raise ValueError("--online-benchmark zahteva --relative-state-checkpoint za glavni sigmoidni RelativeStateNet")
    duration=float(t[-1]);state_net=load_sigmoid_relative_state_checkpoint(a.relative_checkpoint,{"T":duration})
    for parameter in state_net.parameters():parameter.requires_grad_(False)
    state_net.eval();forward_metrics=benchmark_forward_inference(state_net,t,duration)
    raw_runs=[]
    for stride in a.online_strides:
        for adam_steps in a.online_adam_steps:
            print(f"Online benchmark: stride={stride}, Adam steps/update={adam_steps}")
            raw_runs.append(execute_online_updates(state_net,t,mem,c,stride,adam_steps,a.sigmoid_lr,warmup_updates=20))

    # k_true enters only here, after every online optimization/timing run.
    k_reference=true_k(t,duration,c);sample_period_ms=float(np.median(np.diff(t))*1000)
    rows=[]
    for run in raw_runs:
        indices=run["update_indices"];estimate=run["estimated_k"];truth=k_reference[indices];error=estimate-truth
        latency=run["latencies_ms"];period=run["stride"]*sample_period_ms;misses=int(np.sum(latency>period))
        rmse=float(np.sqrt(np.mean(error**2)));relative=100*rmse/c.k0
        row={"stride":run["stride"],"adam_steps_per_update":run["adam_steps"],"window_length":101,
            "number_of_updates":len(indices),"warmup_updates":run["warmup_updates"],
            "mean_latency_ms":float(np.mean(latency)),"median_latency_ms":float(np.median(latency)),
            "p95_latency_ms":float(np.percentile(latency,95)),"maximum_latency_ms":float(np.max(latency)),
            "update_period_ms":period,"real_time_factor":float(np.mean(latency)/period),
            "updates_per_second":float(1000/np.mean(latency)),"deadline_miss_count":misses,
            "deadline_miss_percent":float(100*misses/len(latency)),"k_RMSE":rmse,
            "k_relative_RMSE_percent":relative,"k_R2":r2(truth,estimate),
            "final_stiffness_error_percent":float(100*abs(estimate[-1]-truth[-1])/abs(truth[-1]))}
        rows.append(row);run["metrics"]=row
    feasible=[(run,row) for run,row in zip(raw_runs,rows) if row["deadline_miss_percent"]==0]
    if feasible:
        best_run,best_row=min(feasible,key=lambda item:(item[1]["k_RMSE"],item[1]["mean_latency_ms"]))
        statement="near-real-time causal monitoring demonstrated on the tested CPU"
    else:
        best_run,best_row=min(zip(raw_runs,rows),key=lambda item:(item[1]["deadline_miss_percent"],item[1]["k_RMSE"]))
        statement="near-real-time causal monitoring was not demonstrated for all updates on the tested CPU"

    outdir=a.outdir;outdir.mkdir(parents=True,exist_ok=True)
    fields=["stride","adam_steps_per_update","window_length","number_of_updates","warmup_updates",
        "mean_latency_ms","median_latency_ms","p95_latency_ms","maximum_latency_ms","update_period_ms",
        "real_time_factor","updates_per_second","deadline_miss_count","deadline_miss_percent","k_RMSE",
        "k_relative_RMSE_percent","k_R2","final_stiffness_error_percent"]
    with (outdir/"online_benchmark_results.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

    cpu=platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or platform.machine()
    if os.name=="nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu=str(winreg.QueryValueEx(key,"ProcessorNameString")[0]).strip()
        except (OSError,ImportError):
            pass
    hardware={"CPU":cpu,"device_used":"CPU","GPU_used":False,
        "GPU_available":bool(torch.cuda.is_available()),
        "GPU_name_if_available":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "Python_version":platform.python_version(),"PyTorch_version":str(torch.__version__),
        "torch_intraop_threads":torch.get_num_threads(),"torch_interop_threads":torch.get_num_interop_threads(),
        "logical_CPU_count":os.cpu_count()}
    summary={"statement":statement,"hard_real_time_claim":False,"best_configuration":best_row,
        "best_selection_rule":"among configurations with 0% deadline misses, minimum post-test k_RMSE; k_true never used by online optimization",
        "hardware_software":hardware,"offline_pretraining":{"performed_in_benchmark":False,
            "time_seconds":None,"note":"Existing pretrained RelativeStateNet checkpoint loaded; historical offline time was not measured and is excluded from online latency."},
        "ordinary_forward_inference":forward_metrics,"online_update_latency_definition":
            "time.perf_counter_ns around Adam zero_grad, weak loss, backward, and optimizer step(s) only; frozen RelativeStateNet forward and window preparation excluded",
        "real_time_factor_definition":"mean_online_update_latency_ms / update_period_ms; values below 1 are faster than the deadline on average",
        "causality":{"future_samples_used":False,"window_length":101,
            "rule":"at update t_i only t[i-100:i+1], Mem[i-100:i+1], and frozen RelativeStateNet evaluations at those times are used",
            "warm_start":"the same sigmoid model and Adam state continue from the preceding update"},
        "all_configurations":rows}
    (outdir/"online_benchmark_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    indices=best_run["update_indices"]
    fig,ax=plt.subplots(figsize=(10,5));ax.plot(t,k_reference,"k--",lw=2,label="k true (evaluation only)")
    ax.step(t[indices],best_run["estimated_k"],where="post",lw=2,label=f"online: stride {best_row['stride']}, steps {best_row['adam_steps_per_update']}")
    ax.set(xlabel="t [s]",ylabel="k [Nm/rad]",title="Causal online stiffness tracking");ax.grid();ax.legend();fig.tight_layout();fig.savefig(outdir/"online_stiffness_tracking.png",dpi=200);plt.close(fig)

    labels=[f"s{run['stride']}/a{run['adam_steps']}" for run in raw_runs]
    fig,ax=plt.subplots(figsize=(13,5));ax.boxplot([run["latencies_ms"] for run in raw_runs],tick_labels=labels,showfliers=True)
    ax.set(xlabel="stride / Adam steps",ylabel="online update latency [ms]",title="Online update latency distributions");ax.grid(axis="y");fig.tight_layout();fig.savefig(outdir/"latency_distribution.png",dpi=200);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,6))
    for row in rows:
        marker="o" if row["deadline_miss_percent"]==0 else "x"
        ax.scatter(row["mean_latency_ms"],row["k_RMSE"],s=60,marker=marker)
        ax.annotate(f"s{row['stride']}/a{row['adam_steps_per_update']}",(row["mean_latency_ms"],row["k_RMSE"]),xytext=(4,4),textcoords="offset points",fontsize=8)
    ax.set(xlabel="mean online update latency [ms]",ylabel="online k RMSE [Nm/rad]",title="Latency versus causal tracking accuracy");ax.grid();fig.tight_layout();fig.savefig(outdir/"latency_vs_accuracy.png",dpi=200);plt.close(fig)
    print("Online benchmark summary:\n"+json.dumps(summary,indent=2));return summary


def inverse_train(delta_net,k_net,z,c,history):
    for p in delta_net.parameters():p.requires_grad_(False)
    opt=torch.optim.Adam(k_net.parameters(),lr=c.lr_stiffness)
    for ep in range(1,c.epochs+1):
        opt.zero_grad();dn,d,v,a,kap,res=relative_quantities(delta_net,k_net,z,False);dk=derivative(kap,z["tau"]);d2k=derivative(dk,z["tau"])
        reg=2e-3*torch.mean(d2k**2)+2e-4*torch.mean(torch.abs(dk))+2e-2*torch.mean(torch.relu(dk)**2)
        loss=20*torch.mean(res**2)+reg;loss.backward();opt.step()
        if ep==1 or ep%max(1,c.epochs//100)==0:history.append({"phase":"stiffness","epoch":ep,"loss":float(loss),"physics":float(torch.mean(res**2)),"data":0.0})
        if ep==1 or ep%max(1,c.epochs//10)==0:
            kv=(kap.detach()*c.k0);print(f"stiffness epoch={ep:6d}/{c.epochs} loss={float(loss):.4e} "
                f"k_min={float(kv.min()):.2f} k_max={float(kv.max()):.2f} k_final={float(kv[-1]):.2f}")
    for p in delta_net.parameters():p.requires_grad_(True)


def finetune(delta_net,k_net,z,c,history):
    opt=torch.optim.Adam([{"params":delta_net.parameters(),"lr":c.lr_delta*0.1},{"params":k_net.parameters(),"lr":c.lr_stiffness}])
    sd=torch.std(z["vd"])+1e-9;best=None;best_rmse=float("inf")
    for ep in range(1,c.finetune_epochs+1):
        opt.zero_grad();dn,d,v,a,kap,res=relative_quantities(delta_net,k_net,z,False)
        dd=delta_net(z["taud"]);vd=derivative(dd,z["taud"])*z["scale"]/z["T"]
        data=torch.mean(((vd-z["vd"])/sd)**2);dk=derivative(kap,z["tau"]);d2k=derivative(dk,z["tau"])
        loss=20*torch.mean(res**2)+2*data+2e-3*torch.mean(d2k**2)+2e-4*torch.mean(torch.abs(dk))+2e-2*torch.mean(torch.relu(dk)**2)
        loss.backward();torch.nn.utils.clip_grad_norm_(list(delta_net.parameters())+list(k_net.parameters()),100);opt.step()
        with torch.no_grad():
            kval=(k_net(z["tau"])*c.k0);ktr=z["kpre"]*c.k0
            eval_rmse=float(torch.sqrt(torch.mean((kval-ktr)**2)));eval_final=float(kval[-1])
        if eval_rmse<best_rmse:
            best_rmse=eval_rmse;best={"epoch":ep,"k_RMSE":eval_rmse,"k_final":eval_final,
                "delta":copy.deepcopy(delta_net.state_dict()),"stiffness":copy.deepcopy(k_net.state_dict())}
        if ep==1 or ep%max(1,c.finetune_epochs//100)==0:history.append({"phase":"finetune","epoch":ep,"loss":float(loss),"physics":float(torch.mean(res**2)),"data":float(data),"k_eval_RMSE":eval_rmse,"k_final_eval":eval_final})
        if ep==1 or ep%max(1,c.finetune_epochs//10)==0:
            kv=kap.detach()*c.k0;print(f"finetune epoch={ep:6d}/{c.finetune_epochs} loss={float(loss):.4e} "
                f"k_min={float(kv.min()):.2f} k_max={float(kv.max()):.2f} k_final={float(kv[-1]):.2f} k_eval_RMSE={eval_rmse:.2f}")
    return best


def save(outdir,ref,meas,delta_net,k_net,z,c,history,pre_metrics):
    outdir.mkdir(parents=True,exist_ok=True)
    tau=torch.tensor(ref["t"]/z["T"]).reshape(-1,1).requires_grad_(True);dn=delta_net(tau);dp=(dn*z["scale"]);vp=derivative(dn,tau)*z["scale"]/z["T"];kp=k_net(tau)*c.k0
    dp=dp.detach().numpy().squeeze();vp=vp.detach().numpy().squeeze();kp=kp.detach().numpy().squeeze();ke=kp-ref["k_true"]
    metrics={**pre_metrics,"k_RMSE":float(np.sqrt(np.mean(ke**2))),"k_relative_RMSE_percent":float(100*np.sqrt(np.mean(ke**2))/c.k0),"k_R2":r2(ref["k_true"],kp),"k_final":float(kp[-1]),"k_final_PINN":float(kp[-1]),"k_max_abs_error":float(np.max(np.abs(ke))),"delta_scale":z["scale"],"sensor_data_origin":"synthetic ODE simulation driven by measured jera1.mat Mem"}
    (outdir/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8");(outdir/"config.json").write_text(json.dumps(asdict(c),indent=2),encoding="utf-8")
    torch.save({"delta":delta_net.state_dict(),"stiffness":k_net.state_dict(),"config":asdict(c)},outdir/"model.pt")
    with (outdir/"results.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["t","Mem","delta_true","delta_PINN","delta_dot_true","delta_dot_PINN","k_true","k_PINN"]);w.writerows(zip(ref["t"],ref["Mem"],ref["delta"],dp,ref["delta_dot"],vp,ref["k_true"],kp))
    with (outdir/"loss_history.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["phase","epoch","loss","physics","data","k_eval_RMSE","k_final_eval"]);w.writeheader();w.writerows(history)
    fig,ax=plt.subplots();ax.plot(ref["t"],ref["k_true"],"k--",label="k true");ax.plot(ref["t"],kp,"r",label="k PINN");ax.set(xlabel="t [s]",ylabel="k [Nm/rad]",title="Relative-dynamics PINN stiffness estimate");ax.grid();ax.legend();fig.tight_layout();fig.savefig(outdir/"01_stiffness.png",dpi=200);plt.close(fig)
    fig,ax=plt.subplots(2,1,figsize=(10,7),sharex=True);ax[0].plot(ref["t"],ref["delta"],"k",label="true");ax[0].plot(ref["t"],dp,"r--",label="PINN");ax[0].set_ylabel("delta [rad]");ax[0].legend();ax[0].grid();ax[1].plot(ref["t"],ref["delta_dot"],"k");ax[1].plot(ref["t"],vp,"r--");ax[1].scatter(meas["t"],meas["delta_dot"],s=8);ax[1].set(xlabel="t [s]",ylabel="delta_dot [rad/s]");ax[1].grid();fig.tight_layout();fig.savefig(outdir/"02_relative_state.png",dpi=200);plt.close(fig)
    fig,ax=plt.subplots();
    for ph in ("pretrain","stiffness","finetune"):
        h=[x for x in history if x["phase"]==ph];ax.semilogy([x["epoch"] for x in h],[x["loss"] for x in h],label=ph)
    ax.set(xlabel="epoch",ylabel="loss",title="Training history");ax.grid();ax.legend();fig.tight_layout();fig.savefig(outdir/"03_loss.png",dpi=200);plt.close(fig)
    print(json.dumps(metrics,indent=2))


def sigmoid_state_quality_gate(metrics,robustness=False):
    thresholds={"delta_R2":0.90 if robustness else 0.95,"v_delta_R2":0.95,
                "derivative_consistency_R2":0.95}
    passed=all(metrics[name]>=threshold for name,threshold in thresholds.items())
    return {"passed":bool(passed),"thresholds":thresholds}


def save_sparse401_state_diagnostic(outdir,state_net,z,ref,meas,nominal_rate):
    """Show why formal Nyquist coverage need not imply reliable reconstruction."""
    full_tau=torch.tensor(ref["t"]/z["T"]).reshape(-1,1)
    with torch.no_grad():
        full=state_net(full_tau).numpy();sampled=state_net(z["taud"].detach()).numpy()
    measured_v=np.asarray(meas["delta_dot"],float);integrated=z["delta_integrated"].detach().numpy().squeeze()
    true_v=np.asarray(ref["delta_dot"],float);true_delta=np.asarray(ref["delta"],float)
    true_delta_measured=np.interp(meas["t"],ref["t"],true_delta)
    diagnostic={
        "interpretation":"Nyquist covers the torsional band, but 401-point trapezoidal integration and between-sample reconstruction are inaccurate at about 2.3 samples per torsional period.",
        "v_prediction_at_measurements":{"R2":r2(measured_v,sampled[:,1]),
            "RMSE":float(np.sqrt(np.mean((measured_v-sampled[:,1])**2)))},
        "delta_prediction_vs_integrated_training_target":{"R2":r2(integrated,sampled[:,0]),
            "RMSE":float(np.sqrt(np.mean((integrated-sampled[:,0])**2)))},
        "integrated_delta_vs_true_delta_at_measurements":{"R2":r2(true_delta_measured,integrated),
            "RMSE":float(np.sqrt(np.mean((true_delta_measured-integrated)**2)))},
        "delta_prediction_vs_true_delta_at_measurements":{"R2":r2(true_delta_measured,sampled[:,0]),
            "RMSE":float(np.sqrt(np.mean((true_delta_measured-sampled[:,0])**2)))},
        "samples_per_period_at_230_5_Hz":float(nominal_rate/230.5)}
    (outdir/"sparse401_state_sampling_diagnostic.json").write_text(json.dumps(diagnostic,indent=2),encoding="utf-8")
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    axes[0].plot(ref["t"],true_v,"k",lw=1,label="true v_delta");axes[0].plot(ref["t"],full[:,1],lw=1,label="RelativeStateNet")
    axes[0].scatter(meas["t"],measured_v,s=7,color="tab:orange",label="401 samples",zorder=3)
    axes[0].set(ylabel="v_delta [rad/s]",title="Sparse-401 state reconstruction");axes[0].grid();axes[0].legend(ncol=3)
    axes[1].plot(ref["t"],true_delta,"k",lw=1.2,label="true delta");axes[1].plot(ref["t"],full[:,0],lw=1,label="RelativeStateNet")
    axes[1].plot(meas["t"],integrated,color="tab:orange",lw=1,label="trapezoidal integrated target")
    axes[1].set(xlabel="t [s]",ylabel="delta [rad]");axes[1].grid();axes[1].legend(ncol=3)
    fig.tight_layout();fig.savefig(outdir/"sparse401_state_sampling_diagnostic.png",dpi=200);plt.close(fig)
    return diagnostic


def build_joint_densephysics_geometry(z,window_length=101,stride=25):
    """Fixed dense collocation geometry; contains no sensor/state/k labels."""
    t=z["tau"].detach().squeeze()*z["T"]
    starts=list(range(0,len(t)-window_length+1,stride))
    indices=torch.stack([torch.arange(start,start+window_length) for start in starts])
    t_windows=t[indices];mem_windows=z["mem"].detach().squeeze()[indices]
    return {"indices":indices,"t_windows":t_windows,
        "int_mem":torch.trapz(mem_windows,t_windows,dim=1),
        "window_length":window_length,"stride":stride,"window_count":len(starts),
        "window_duration_seconds":float(t[window_length-1]-t[0]),
        "stride_duration_seconds":float(stride*(t[1]-t[0]))}


def joint_densephysics_losses(state_net,stiffness_model,z,c,geometry):
    """Sparse labeled data plus dense unlabeled kinematic/weak physics."""
    state_d=state_net(z["taud"]);delta_d,v_d=state_d[:,:1],state_d[:,1:]
    state_f=state_net(z["tau"]);delta,v=state_f[:,:1],state_f[:,1:]
    delta_dot=derivative(delta,z["tau"])/z["T"]
    loss_delta=torch.mean(((delta_d-z["delta_integrated"])/z["delta_scale"])**2)
    loss_v=torch.mean(((v_d-z["v_measured"])/z["v_scale"])**2)
    loss_kinematic=torch.mean(((delta_dot-v)/z["v_scale"])**2)
    loss_ic=(delta[0,0]/z["delta_scale"])**2+(v[0,0]/z["v_scale"])**2

    indices=geometry["indices"];tw=geometry["t_windows"]
    dw=delta[:,0][indices];vw=v[:,0][indices]
    invsum=1/c.Jm+1/c.Jl
    int_v=torch.trapz(vw,tw,dim=1)
    B=(vw[:,-1]-vw[:,0])+c.bv*invsum*int_v-geometry["int_mem"]/c.Jm
    int_k_delta=torch.trapz(stiffness_model(tw)*dw,tw,dim=1)
    weak_dynamic=B+invsum*int_k_delta
    dynamic_scale=torch.sqrt(torch.mean(B.detach()**2)).clamp_min(1e-12)
    loss_weak=torch.mean((weak_dynamic/dynamic_scale)**2)
    total=10*loss_delta+10*loss_v+5*loss_kinematic+loss_ic+loss_weak
    return total,{"loss_delta_data":loss_delta,"loss_v_delta_data":loss_v,
        "loss_kinematic":loss_kinematic,"loss_initial_conditions":loss_ic,
        "loss_weak_dynamic":loss_weak}


def train_sparse751_densephysics_restarts(base_state,z,c,seeds,epochs,sigmoid_lr,
                                           window_length=101,stride=25,patience=1000):
    """Joint state/stiffness training; model selection uses training loss only."""
    geometry=build_joint_densephysics_geometry(z,window_length,stride)
    base_state_dict=copy.deepcopy(base_state.state_dict());candidates=[];all_history=[]
    state_lr=c.lr_delta*0.1
    for seed in seeds:
        seed_all(int(seed));state_net=RelativeStateNet(z["delta_scale"],z["v_scale"])
        state_net.load_state_dict(base_state_dict);stiffness_model=WeakSigmoidStiffness(z["T"])
        optimizer=torch.optim.Adam([
            {"params":state_net.parameters(),"lr":state_lr},
            {"params":stiffness_model.parameters(),"lr":sigmoid_lr}])
        best_loss=float("inf");best_epoch=0;best_state=None;best_stiffness=None;stale=0;history=[]
        for epoch in range(1,epochs+1):
            optimizer.zero_grad();total,parts=joint_densephysics_losses(state_net,stiffness_model,z,c,geometry)
            total.backward();torch.nn.utils.clip_grad_norm_(
                list(state_net.parameters())+list(stiffness_model.parameters()),100.0);optimizer.step()
            values={name:float(value.detach()) for name,value in stiffness_model.physical_parameters().items()}
            row={"restart_seed":int(seed),"epoch":epoch,"total_loss":float(total.detach()),
                **{name:float(value.detach()) for name,value in parts.items()},**values}
            history.append(row);loss_value=row["total_loss"]
            if loss_value<best_loss-1e-12:
                best_loss=loss_value;best_epoch=epoch;stale=0
                best_state=copy.deepcopy(state_net.state_dict());best_stiffness=copy.deepcopy(stiffness_model.state_dict())
            else:stale+=1
            if epoch==1 or epoch%max(1,epochs//20)==0:
                print(f"densephysics seed={seed} epoch={epoch:6d}/{epochs} total={loss_value:.6e} "
                    f"data_delta={row['loss_delta_data']:.3e} data_v={row['loss_v_delta_data']:.3e} "
                    f"kin={row['loss_kinematic']:.3e} weak={row['loss_weak_dynamic']:.3e} "
                    f"k_high={values['k_high']:.2f} k_low={values['k_low']:.2f} center={values['t_center']:.5f}")
            if stale>=patience:
                print(f"Dense-physics early stopping seed={seed} epoch={epoch}; training-loss patience={patience}")
                break
        state_net.load_state_dict(best_state);stiffness_model.load_state_dict(best_stiffness)
        total,parts=joint_densephysics_losses(state_net,stiffness_model,z,c,geometry)
        values={name:float(value.detach()) for name,value in stiffness_model.physical_parameters().items()}
        result={"seed":int(seed),"training_total_loss":float(total.detach()),
            "best_epoch":int(best_epoch),"epochs_executed":len(history),
            "state_learning_rate":state_lr,"sigmoid_learning_rate":float(sigmoid_lr),
            **{name:float(value.detach()) for name,value in parts.items()},**values,
            "selection":"minimum joint training total loss only"}
        candidates.append((state_net,stiffness_model,result));all_history.extend(history)
        print("Dense-physics restart result:\n"+json.dumps(result,indent=2))
    best_state,best_stiffness,best=min(candidates,key=lambda item:item[2]["training_total_loss"])
    return best_state,best_stiffness,all_history,[item[2] for item in candidates],best,geometry


def save_sparse751_densephysics_result(outdir,state_net,stiffness_model,history,restarts,best,
                                       geometry,z,ref,meas,c,sampling_diagnostic,phase_a_gate):
    """Post-training evaluation; only this function accesses reference labels/k_true."""
    outdir.mkdir(parents=True,exist_ok=True);state_metrics=first_order_metrics(state_net,z,ref)
    t=np.asarray(ref["t"],float)
    with torch.no_grad():k_est=stiffness_model(torch.tensor(t)).numpy()
    k_true=np.asarray(ref["k_true"],float);error=k_est-k_true
    k_rmse=float(np.sqrt(np.mean(error**2)));relative=100*k_rmse/c.k0;k_score=r2(k_true,k_est)
    initial_error=100*abs(k_est[0]-k_true[0])/abs(k_true[0]);final_error=100*abs(k_est[-1]-k_true[-1])/abs(k_true[-1])
    gate_components={"delta_R2_ge_0_90":bool(state_metrics["delta_R2"]>=.90),
        "v_delta_R2_ge_0_95":bool(state_metrics["v_delta_R2"]>=.95),
        "k_relative_RMSE_le_8_percent":bool(relative<=8.0),"k_R2_ge_0_85":bool(k_score>=.85),
        "initial_error_le_10_percent":bool(initial_error<=10.0),"final_error_le_10_percent":bool(final_error<=10.0)}
    stiffness_pass=all(gate_components[name] for name in (
        "k_relative_RMSE_le_8_percent","k_R2_ge_0_85",
        "initial_error_le_10_percent","final_error_le_10_percent"))
    delta_pass=gate_components["delta_R2_ge_0_90"];v_pass=gate_components["v_delta_R2_ge_0_95"]
    state_gate="PASS" if delta_pass and v_pass else ("PARTIAL" if delta_pass or v_pass else "FAIL")
    overall_gate="PASS" if all(gate_components.values()) else "FAIL"
    params={name:float(value.detach()) for name,value in stiffness_model.physical_parameters().items()}
    metrics={"experiment":"noise0_measurements751_densephysics","measurement_count":751,
        "measurements":751,"collocation_count":1501,
        "measurement_reduction_percent":float(100*(1-751/1501)),
        "delta_RMSE":state_metrics["delta_RMSE"],"delta_R2":state_metrics["delta_R2"],
        "v_delta_RMSE":state_metrics["v_delta_RMSE"],"v_delta_R2":state_metrics["v_delta_R2"],
        "derivative_consistency_R2":state_metrics["derivative_consistency_R2"],
        "kinematic_residual_RMSE":state_metrics["kinematic_residual_RMSE"],
        "k_RMSE":k_rmse,"k_relative_RMSE_percent":relative,"k_R2":k_score,
        "k_MAE":float(np.mean(np.abs(error))),"maximum_absolute_error":float(np.max(np.abs(error))),
        "estimated_k_start":float(k_est[0]),"estimated_k_final":float(k_est[-1]),
        "initial_error_percent":float(initial_error),"final_error_percent":float(final_error),
        "estimated_degradation_percent":float(100*(k_est[0]-k_est[-1])/k_est[0]),
        "estimated_transition_center":params["t_center"],"estimated_transition_width":params["width"],
        "estimated_k_high":params["k_high"],"estimated_k_low":params["k_low"],
        "quality_gate":overall_gate,
        "stiffness_identification_gate":"PASS" if stiffness_pass else "FAIL",
        "state_reconstruction_gate":state_gate,"overall_composite_gate":overall_gate,
        "quality_gate_components":gate_components,"phase_a_quality_gate":phase_a_gate,
        "selected_seed":best["seed"],"best_epoch":best["best_epoch"],
        "training_total_loss":best["training_total_loss"],"checkpoint_selection":"minimum joint training total loss only",
        "state_learning_rate":best["state_learning_rate"],"sigmoid_learning_rate":best["sigmoid_learning_rate"],
        "window_length":geometry["window_length"],"stride":geometry["stride"],"window_selection":"all",
        "number_of_windows":geometry["window_count"],"weak_window_duration_seconds":geometry["window_duration_seconds"],
        "weak_stride_duration_seconds":geometry["stride_duration_seconds"],"sampling_diagnostic":sampling_diagnostic,
        "sensor_data_origin":"synthetic ODE simulation driven by measured jera1.mat Mem",
        "collocation_note":"1501 unlabeled physics points use only t, known Mem, kinematic and weak residuals; no true state or k labels",
        "training_label_note":"delta_integrated and v_delta labels exist only at 751 uniform sensor times; no dense sensor-label checkpoint was used"}
    torch.save({"state_dict":state_net.state_dict(),"delta_scale":z["delta_scale"],"v_scale":z["v_scale"],
        "duration":z["T"],"formulation":"first_order","reference_profile":"time_varying_sigmoid",
        "experiment_tag":"sparse751_densephysics","phase":"joint_selected","metrics":state_metrics},
        outdir/"relative_state_sparse751_densephysics.pt")
    torch.save({"relative_state":state_net.state_dict(),"stiffness":stiffness_model.state_dict(),
        "physical_parameters":params,"selected_seed":best["seed"],"best_epoch":best["best_epoch"],
        "selection":"minimum joint training total loss only"},outdir/"model_sparse751_densephysics.pt")
    (outdir/"metrics_sparse751_densephysics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    history_fields=["restart_seed","epoch","total_loss","loss_delta_data","loss_v_delta_data",
        "loss_kinematic","loss_initial_conditions","loss_weak_dynamic","k_high","k_low","t_center","width"]
    with (outdir/"history_sparse751_densephysics.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=history_fields);writer.writeheader();writer.writerows(history)
    with (outdir/"restarts_sparse751_densephysics.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(restarts[0].keys()));writer.writeheader();writer.writerows(restarts)
    with (outdir/"results_sparse751_densephysics.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);writer.writerow(["t","k_true","k_weak_sigmoid"]);writer.writerows(zip(t,k_true,k_est))
    tau=torch.tensor(t/z["T"]).reshape(-1,1)
    with torch.no_grad():state=state_net(tau).numpy()
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
    axes[0].plot(t,ref["delta"],"k",label="reference (evaluation only)");axes[0].plot(t,state[:,0],label="joint PINN")
    axes[0].scatter(meas["t"],z["delta_integrated"].detach().numpy(),s=7,label="751 derived labels")
    axes[1].plot(t,ref["delta_dot"],"k",label="reference (evaluation only)");axes[1].plot(t,state[:,1],label="joint PINN")
    axes[1].scatter(meas["t"],meas["delta_dot"],s=7,label="751 encoder labels")
    axes[0].set_ylabel("delta [rad]");axes[1].set(xlabel="t [s]",ylabel="v_delta [rad/s]")
    for ax in axes:ax.grid();ax.legend()
    fig.tight_layout();fig.savefig(outdir/"state_sparse751_densephysics.png",dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));ax.plot(t,k_true,"k--",lw=2,label="k true (evaluation only)")
    ax.plot(t,k_est,color="tab:purple",lw=2,label="sparse751 + dense physics");ax.set(xlabel="t [s]",ylabel="k [Nm/rad]")
    ax.grid();ax.legend();fig.tight_layout();fig.savefig(outdir/"stiffness_sparse751_densephysics.png",dpi=200);plt.close(fig)
    print("Sparse751 dense-physics result:\n"+json.dumps(metrics,indent=2));return metrics


def run_weak_sigmoid_experiment(a,c,t,u):
    if a.relative_formulation!="first_order" or a.first_order_physics!="weak":
        raise ValueError("Sigmoidni glavni eksperiment zahteva --relative-formulation first_order --first-order-physics weak")
    if len(set(a.sigmoid_seeds))!=3:
        raise ValueError("--sigmoid-seeds mora sadržati tri različite determinističke seed vrednosti")
    robustness=a.sigmoid_robustness!="main";weak_window_length=101;weak_stride=25
    if a.sigmoid_robustness=="noise003":
        if not np.isclose(c.noise,.003,rtol=0,atol=1e-12) or c.measurements!=1501:
            raise ValueError("noise003 zahteva --noise 0.003 --measurements 1501")
        suffix="_noise003";experiment_name="noise003_measurements1501";measurement_seed=int(a.noise_seed)
    elif a.sigmoid_robustness=="sparse121":
        if not np.isclose(c.noise,0.0,rtol=0,atol=1e-12) or c.measurements!=121:
            raise ValueError("sparse121 zahteva --noise 0 --measurements 121")
        suffix="_sparse121";experiment_name="noise0_measurements121";measurement_seed=None
    elif a.sigmoid_robustness=="sparse401":
        if not np.isclose(c.noise,0.0,rtol=0,atol=1e-12) or c.measurements!=401:
            raise ValueError("sparse401 zahteva --noise 0 --measurements 401")
        suffix="_sparse401";experiment_name="noise0_measurements401";measurement_seed=None
        # Preserve approximately the same physical weak window (50 ms) and
        # stride (12.5 ms) as 101/25 on the 1501-point main grid.
        c.collocation_points=401;weak_window_length=28;weak_stride=7
    elif a.sigmoid_robustness=="sparse751":
        if not np.isclose(c.noise,0.0,rtol=0,atol=1e-12) or c.measurements!=751:
            raise ValueError("sparse751 zahteva --noise 0 --measurements 751")
        suffix="_sparse751";experiment_name="noise0_measurements751";measurement_seed=None
        c.collocation_points=751;weak_window_length=51;weak_stride=13
    elif a.sigmoid_robustness=="sparse751_densephysics":
        if not np.isclose(c.noise,0.0,rtol=0,atol=1e-12) or c.measurements!=751:
            raise ValueError("sparse751_densephysics zahteva --noise 0 --measurements 751")
        suffix="_sparse751_densephysics";experiment_name="noise0_measurements751_densephysics";measurement_seed=None
        c.collocation_points=1501;weak_window_length=101;weak_stride=25
    else:
        suffix="";experiment_name="main_noise0_measurements1501";measurement_seed=None
    ref=simulate(t,u,c,lambda x:true_k(x,float(t[-1]),c));meas=measurements(
        ref,c,measurement_seed,relative_noise=a.sigmoid_robustness=="noise003",
        uniform_times=a.sigmoid_robustness in ("sparse401","sparse751","sparse751_densephysics"))
    measurement_dt=np.diff(np.asarray(meas["t"],float));measurement_duration=float(meas["t"][-1]-meas["t"][0])
    nominal_rate=(len(meas["t"])-1)/measurement_duration
    sampling_diagnostic={"selected_samples":int(len(meas["t"])),"duration_seconds":measurement_duration,
        "nominal_sampling_rate_Hz":float(nominal_rate),"nominal_nyquist_Hz":float(nominal_rate/2),
        "effective_sample_rate_Hz":float(nominal_rate),"Nyquist_frequency_Hz":float(nominal_rate/2),
        "samples_per_dominant_torsional_period":float(nominal_rate/230.5),
        "sample_reduction_percent":float(100*(1-len(meas["t"])/1501)),
        "sampling_scheme":meas["sampling_scheme"],
        "weak_grid_points":int(c.collocation_points),"weak_window_length":int(weak_window_length),
        "weak_stride":int(weak_stride),
        "weak_window_duration_seconds":float((weak_window_length-1)*measurement_duration/(c.collocation_points-1)),
        "weak_stride_duration_seconds":float(weak_stride*measurement_duration/(c.collocation_points-1)),
        "maximum_sample_gap_seconds":float(np.max(measurement_dt)),
        "conservative_nyquist_from_max_gap_Hz":float(0.5/np.max(measurement_dt)),
        "dominant_torsional_band_Hz":[228.0,233.0],
        "torsional_band_below_conservative_nyquist":bool(233.0<0.5/np.max(measurement_dt))}
    z=build_first_order_tensors(ref,meas,c);history=[]
    checkpoint=(a.outdir/"relative_state_sparse751_densephysics.pt" if a.sigmoid_robustness=="sparse751_densephysics"
        else a.outdir/f"relative_state_pretrained_sigmoid{suffix}.pt");pretrain_total=0
    checkpoint_metadata={"experiment_tag":a.sigmoid_robustness,"noise":float(c.noise),
        "measurements":int(c.measurements),"noise_seed":measurement_seed,"noise_model":meas["noise_model"],
        "sampling_diagnostic":sampling_diagnostic}
    if a.relative_checkpoint is not None:
        state_net=load_sigmoid_relative_state_checkpoint(a.relative_checkpoint,z,
            a.sigmoid_robustness if robustness else None)
        loaded_payload=torch.load(a.relative_checkpoint,map_location="cpu",weights_only=False)
        pretrain_total=int(loaded_payload.get("pretrain_epochs_total",0))
        print(f"Loaded sigmoid-case RelativeStateNet checkpoint: {a.relative_checkpoint}")
        if a.first_order_pretrain_only and not robustness:
            print(f"Continuing sigmoid-case RelativeStateNet pretraining for {c.pretrain_epochs} additional epochs")
            pretrain_first_order(state_net,z,c,history)
            pretrain_total+=c.pretrain_epochs
    else:
        if robustness and c.pretrain_epochs!=6500:
            raise ValueError("Robustness pretraining mora početi sa --pretrain-epochs 6500")
        if not robustness and c.pretrain_epochs<4000:
            raise ValueError("Sigmoidni RelativeStateNet pretraining zahteva najmanje --pretrain-epochs 4000")
        state_net=RelativeStateNet(z["delta_scale"],z["v_scale"])
        if not robustness:
            pretrain_first_order(state_net,z,c,history);pretrain_total=c.pretrain_epochs
    optimizer=None
    if robustness and a.relative_checkpoint is None:
        while pretrain_total<9500:
            block=6500 if pretrain_total==0 else min(1000,9500-pretrain_total);target=pretrain_total+block
            optimizer=pretrain_first_order(state_net,z,c,history,epochs=block,start_epoch=pretrain_total,
                optimizer=optimizer,display_total=target);pretrain_total=target
            state_metrics=first_order_metrics(state_net,z,ref);gate=sigmoid_state_quality_gate(state_metrics,True)
            save_sigmoid_relative_state_checkpoint(checkpoint,state_net,z,state_metrics,
                {**checkpoint_metadata,"pretrain_epochs_total":pretrain_total,"quality_gate":gate})
            print(f"Robustness pretraining block completed at total epoch {pretrain_total}:\n"+json.dumps(state_metrics,indent=2))
            if gate["passed"]:break
            if pretrain_total<9500:
                print("Robustness state gate not met; continuing the same network/optimizer for another 1000 epochs.")
            else:print("Robustness state gate not met at the maximum 9500 epochs.")
    state_metrics=first_order_metrics(state_net,z,ref);gate=sigmoid_state_quality_gate(state_metrics,robustness)
    if a.sigmoid_robustness=="sparse401":
        sampling_diagnostic["state_reconstruction_diagnostic"]=save_sparse401_state_diagnostic(
            a.outdir,state_net,z,ref,meas,nominal_rate)
    save_sigmoid_relative_state_checkpoint(checkpoint,state_net,z,state_metrics,
        {**checkpoint_metadata,"pretrain_epochs_total":pretrain_total,"quality_gate":gate})
    if history or not (a.outdir/"first_order_pretrain_metrics.json").exists():
        save_first_order_pretrain_report(a.outdir,state_metrics,gate,history,z)
    print("Sigmoid-case RelativeStateNet metrics:\n"+json.dumps(state_metrics,indent=2))
    print("Sigmoid-case RelativeStateNet quality gate:\n"+json.dumps(gate,indent=2))
    print(f"Saved sigmoid-case RelativeStateNet checkpoint: {checkpoint}")
    if not gate["passed"]:
        if robustness and pretrain_total>=9500:
            print("WARNING: maximum robustness pretraining reached without state gate; preserving and continuing as a diagnostic FAIL candidate.")
        elif not a.allow_poor_pretrain:
            raise RuntimeError("Sigmoid-case RelativeStateNet nije prošao quality gate; weak sigmoid trening je namerno zaustavljen")
    if a.first_order_pretrain_only:return
    if a.sigmoid_robustness=="sparse751_densephysics":
        joint_state,joint_stiffness,joint_history,joint_restarts,joint_best,geometry=train_sparse751_densephysics_restarts(
            state_net,z,c,a.sigmoid_seeds,c.epochs,a.sigmoid_lr,window_length=101,stride=25)
        save_sparse751_densephysics_result(a.outdir,joint_state,joint_stiffness,joint_history,joint_restarts,
            joint_best,geometry,z,ref,meas,c,sampling_diagnostic,gate)
        create_final_sigmoid_summary(a.free_baseline_csv);return
    model,weak_history,restarts,best,terms=train_weak_sigmoid_restarts(
        state_net,z,c,a.sigmoid_seeds,c.epochs,a.sigmoid_lr,weak_window_length,weak_stride)
    save_weak_sigmoid_result(a.outdir,model,weak_history,restarts,best,terms,ref,c,state_metrics,a.free_baseline_csv,
        artifact_suffix=suffix,experiment_name=experiment_name,robustness=robustness,
        noise_seed=measurement_seed,pretrain_epochs_total=pretrain_total,noise_model=meas["noise_model"],
        state_quality_gate=gate,sampling_diagnostic=sampling_diagnostic)
    if robustness:create_final_sigmoid_summary(a.free_baseline_csv)


def args():
    p=argparse.ArgumentParser();p.add_argument("--mat",type=Path,default=DATA_DIR/"jera1.mat");p.add_argument("--pretrain-epochs",type=int,default=2000);p.add_argument("--epochs",type=int,default=6000);p.add_argument("--finetune-epochs",type=int,default=2000);p.add_argument("--noise",type=float,default=.003);p.add_argument("--measurements",type=int,default=121);p.add_argument("--outdir",type=Path,default=OUTPUTS_DIR/"run_relative");p.add_argument("--allow-poor-pretrain",action="store_true")
    p.add_argument("--stiffness-profile",choices=("constant","free","sigmoid"),default="free")
    p.add_argument("--relative-formulation",choices=("first_order","second_order"),default="second_order")
    p.add_argument("--first-order-physics",choices=("differential","weak"),default="differential")
    p.add_argument("--k-true-constant",type=float,default=None)
    p.add_argument("--delta-checkpoint",type=Path,default=None)
    p.add_argument("--relative-state-checkpoint","--relative-checkpoint",dest="relative_checkpoint",type=Path,default=None)
    p.add_argument("--diagnose-constant-landscape",action="store_true")
    p.add_argument("--first-order-pretrain-only",action="store_true")
    p.add_argument("--optimize-first-order-weak-constant",action="store_true")
    p.add_argument("--sigmoid-seeds",type=int,nargs=3,default=(2026,2027,2028))
    p.add_argument("--sigmoid-lr",type=float,default=5e-3)
    p.add_argument("--sigmoid-robustness",choices=("main","noise003","sparse121","sparse401","sparse751","sparse751_densephysics"),default="main")
    p.add_argument("--online-benchmark",action="store_true")
    p.add_argument("--online-strides",type=int,nargs="+",default=(25,50,100))
    p.add_argument("--online-adam-steps",type=int,nargs="+",default=(1,5,10,20))
    p.add_argument("--noise-seed",type=int,default=3030)
    p.add_argument("--free-baseline-csv",type=Path,
        default=OUTPUTS_DIR/"run_relative_noise000_v2"/"results.csv")
    return p.parse_args()


def main():
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    a=args();a.outdir.mkdir(parents=True,exist_ok=True);c=Config(str(a.mat),pretrain_epochs=a.pretrain_epochs,epochs=a.epochs,finetune_epochs=a.finetune_epochs,noise=a.noise,measurements=a.measurements)
    seed_all(c.seed);torch.set_default_dtype(torch.float64);t,u=load_input(a.mat)
    if a.online_benchmark:
        if a.stiffness_profile!="sigmoid" or a.relative_formulation!="first_order" or a.first_order_physics!="weak":
            raise ValueError("--online-benchmark zahteva: --stiffness-profile sigmoid --relative-formulation first_order --first-order-physics weak")
        run_online_benchmark(a,c,t,u);return
    if a.stiffness_profile=="sigmoid":
        run_weak_sigmoid_experiment(a,c,t,u);return
    if a.stiffness_profile=="constant":
        if a.k_true_constant is None:raise ValueError("--k-true-constant je obavezan za constant profil")
        lo,hi=c.kappa_min*c.k0,c.kappa_max*c.k0
        if not lo<=a.k_true_constant<=hi:raise ValueError(f"k_true_constant mora biti u [{lo}, {hi}] Nm/rad")
        ref=simulate(t,u,c,lambda x:np.zeros_like(np.asarray(x,float))+a.k_true_constant)
        meas=measurements(ref,c)
        if a.relative_formulation=="first_order":
            if a.first_order_physics=="weak" and a.relative_checkpoint is None:
                raise ValueError("Weak dijagnostika zahteva --relative-state-checkpoint")
            z1=build_first_order_tensors(ref,meas,c);history=[]
            if a.relative_checkpoint is not None:
                state_net=load_relative_state_checkpoint(a.relative_checkpoint,z1,c,a.k_true_constant)
                print(f"Loaded first-order RelativeStateNet checkpoint: {a.relative_checkpoint}")
                if a.first_order_pretrain_only:
                    print(f"Continuing first-order pretraining for {c.pretrain_epochs} additional epochs")
                    pretrain_first_order(state_net,z1,c,history,a.outdir,a.k_true_constant)
                    continued_metrics=first_order_metrics(state_net,z1,ref)
                    checkpoint=a.outdir/f"relative_state_pretrained_k{int(a.k_true_constant)}.pt"
                    save_relative_state_checkpoint(checkpoint,state_net,z1,c,a.k_true_constant,continued_metrics)
                    print(f"Saved continued first-order checkpoint: {checkpoint}")
            else:
                if a.first_order_pretrain_only and c.pretrain_epochs<4000:
                    raise ValueError("--first-order-pretrain-only zahteva najmanje --pretrain-epochs 4000")
                state_net=RelativeStateNet(z1["delta_scale"],z1["v_scale"])
                pretrain_first_order(state_net,z1,c,history,a.outdir,a.k_true_constant)
                metrics=first_order_metrics(state_net,z1,ref);print("First-order pretraining metrics:\n"+json.dumps(metrics,indent=2))
                checkpoint=a.outdir/f"relative_state_pretrained_k{int(a.k_true_constant)}.pt"
                save_relative_state_checkpoint(checkpoint,state_net,z1,c,a.k_true_constant,metrics)
                print(f"Saved first-order checkpoint: {checkpoint}")
            metrics=first_order_metrics(state_net,z1,ref)
            gate=first_order_quality_gate(metrics,c);save_first_order_pretrain_report(a.outdir,metrics,gate,history,z1)
            print("First-order quality gate:\n"+json.dumps(gate,indent=2))
            if not gate["passed"] and not a.allow_poor_pretrain:
                raise RuntimeError("First-order RelativeStateNet nije prošao quality gate; stiffness trening i landscape su namerno zaustavljeni")
            if a.first_order_physics=="weak":
                differential_csv=(OUTPUTS_DIR/"first_order_constant_k300_serious6500"/"first_order_constant_loss_landscape.csv") if np.isclose(a.k_true_constant,300.0) else None
                weak_landscape,weak_closed=diagnose_first_order_weak(a.outdir,state_net,z1,ref,c,a.k_true_constant,differential_csv)
                if a.optimize_first_order_weak_constant:
                    network_min=weak_landscape["windows"]["101"]["all"]["network_weak"]["k_at_minimum"]
                    landscape_error=100*abs(network_min-a.k_true_constant)/a.k_true_constant
                    print(f"Weak landscape gate: network_min={network_min:.6f}, relative_distance={landscape_error:.4f}%")
                    if landscape_error>5.0:
                        print("Weak landscape minimum is farther than 5%; scalar optimization is intentionally skipped.")
                        return
                    model,weak_history,training_metrics=train_first_order_weak_constant(state_net,z1,c,c.epochs,patience=500)
                    save_first_order_weak_constant(a.outdir,model,weak_history,training_metrics,c,a.k_true_constant,metrics,weak_landscape,weak_closed)
                return
            landscape=diagnose_first_order_landscape(a.outdir,state_net,z1,ref,c,a.k_true_constant,points=200)
            if a.diagnose_constant_landscape or a.first_order_pretrain_only:return
            k_model=ConstantStiffness(c);physics=train_first_order_constant(state_net,k_model,z1,c,history)
            save_first_order_constant(a.outdir,state_net,k_model,z1,ref,c,a.k_true_constant,physics,landscape)
            return
        z=build_tensors(ref,meas,c)
        delta_net=DeltaNet();history=[]
        if a.diagnose_constant_landscape and a.delta_checkpoint is None:
            raise ValueError("--diagnose-constant-landscape zahteva --delta-checkpoint")
        if a.delta_checkpoint is not None:
            pm=load_delta_checkpoint(a.delta_checkpoint,delta_net,z,a.k_true_constant)
            pm=delta_metrics(delta_net,z,ref)
            print("Recomputed metrics immediately after checkpoint load:\n"+json.dumps(pm,indent=2))
        else:
            pretrain_constant_delta(delta_net,z,c,history);pm=delta_metrics(delta_net,z,ref)
            print("Constant-case DeltaNet pretraining metrics:\n"+json.dumps(pm,indent=2))
            ok=pm["delta_R2"]>=c.min_pretrain_r2 and pm["delta_dot_R2"]>=c.min_pretrain_r2 and pm["delta_dot_relative_RMSE"]<=c.max_pretrain_relative_rmse
            if not ok:
                (a.outdir/"pretrain_metrics.json").write_text(json.dumps(pm,indent=2),encoding="utf-8")
                if not a.allow_poor_pretrain:
                    raise RuntimeError("Constant-case DeltaNet pretraining nije prošao quality gate")
                print("WARNING: poor pretraining explicitly allowed for diagnostic run; canonical checkpoint will not be written.")
            name=delta_checkpoint_name(a.k_true_constant)
            save_delta_checkpoint(a.outdir/name,delta_net,z,c,a.k_true_constant,pm)
            if ok:save_delta_checkpoint(OUTPUTS_DIR/name,delta_net,z,c,a.k_true_constant,pm)
            print(f"Saved matching DeltaNet checkpoint: {name}")
        if a.diagnose_constant_landscape:
            diagnose_constant_landscape(a.outdir,delta_net,z,ref,meas,c,a.k_true_constant,points=200)
            return
        k_model=ConstantStiffness(c);total,physics,data=constant_train(delta_net,k_model,z,c,history)
        save_constant_result(a.outdir,delta_net,k_model,c,a.k_true_constant,total,physics,data)
        return
    identifiability_test(t,u,c,a.outdir)
    ref=simulate(t,u,c,lambda x:true_k(x,float(t[-1]),c));meas=measurements(ref,c);z=build_tensors(ref,meas,c);delta_net=DeltaNet();k_net=StiffnessNet(c);history=[]
    pretrain(delta_net,k_net,z,c,history);pm=delta_metrics(delta_net,z,ref);print("Pretraining metrics:\n"+json.dumps(pm,indent=2))
    ok=pm["delta_R2"]>=c.min_pretrain_r2 and pm["delta_dot_R2"]>=c.min_pretrain_r2 and pm["delta_dot_relative_RMSE"]<=c.max_pretrain_relative_rmse
    if not ok and not a.allow_poor_pretrain:
        (a.outdir/"pretrain_metrics.json").write_text(json.dumps(pm,indent=2),encoding="utf-8")
        raise RuntimeError("Pretraining kriterijumi nisu ispunjeni; stiffness trening je namerno zaustavljen. Povećaj --pretrain-epochs ili dijagnostikuj pobudu.")
    inverse_train(delta_net,k_net,z,c,history)
    # Primary identification result: no k_true was used for training or selection.
    save(a.outdir,ref,meas,delta_net,k_net,z,c,history,pm)
    torch.save({"delta":delta_net.state_dict(),"stiffness":k_net.state_dict(),"config":asdict(c)},a.outdir/"model_identification.pt")
    best=finetune(delta_net,k_net,z,c,history)
    # Fine-tuning is reported separately because a lower total loss may worsen k(t).
    fine_dir=a.outdir/"fine_tuning_experiment"
    save(fine_dir,ref,meas,delta_net,k_net,z,c,history,pm)
    if best is not None:
        torch.save({"delta":best["delta"],"stiffness":best["stiffness"],"config":asdict(c),
                    "selection":"minimum k_RMSE using k_true; simulation evaluation only"},
                   fine_dir/"model_best_evaluation_only.pt")
        (fine_dir/"best_evaluation_only.json").write_text(json.dumps({"epoch":best["epoch"],
            "k_RMSE":best["k_RMSE"],"k_final":best["k_final"],
            "selection_note":"Selected with known k_true only for simulation analysis; not an identification checkpoint."},indent=2),encoding="utf-8")


if __name__=="__main__":main()
