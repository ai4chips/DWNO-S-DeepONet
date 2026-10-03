import sys,os,time,re,importlib,numpy as np,jax,jax.numpy as jnp,equinox as eqx
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); ML = os.path.join(BASE, 'multilayer'); DATA = os.path.join(BASE, 'data', 'multilayer')
key=jax.random.PRNGKey(0);T_amb,dT,fs=298.15,100.0,0.03;z_layers=[1,4,8,12,16]

def load_cls(srcdir,modfile,clsname):
    fp=os.path.join(BASE,srcdir,modfile)
    spec=importlib.util.spec_from_file_location(f'{srcdir.replace("/","_")}_{clsname}',fp)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return getattr(mod,clsname)

DWB=load_cls('3d/DWNO_S_DeepONet','model_dwno_deeponet_jax_3d.py','DWNOSeparableDeepONet3d')
DWP=load_cls('3d/DWNO_DeepONet','model_dwno_deeponet_jax_3d.py','DWNOSeparableDeepONet3d')
DFB=load_cls('3d/DFNO_S_DeepONet','model_dfno_s_deeponet_jax_3d.py','DFNOSeparableDeepONet3d')
DFP=load_cls('3d/DFNO_DeepONet','model_dfno_deeponet_jax_3d.py','DFNODeepONet3d')
MLP=load_cls('3d/MLP_DeepONet','model_deeponet_mlp.py','MLPDeepONet3d')
MBP=load_cls('3d/MLP_S_DeepONet','model_mlp_bpss.py','MLPDeepONet3dBPSS')

models_info=[
('DWNO-BPSS',   DWB,dict(trunk_layers=[50]*7,R=50,nx=64,ny=64,nz=48,width=64,level=4,P=25,num_param_levels=1,num_wavelet_layers=10),'DWNO_S_DeepONet/results','dwno_bpss_z','.eqqx'),
('MLP-DeepONet',MLP,dict(branch_layers=[190]*4+[50],trunk_layers=[50]*7,r=30,nx=64,ny=64,nz=48),'MLP_DeepONet/results','mlp_deeponet_z','.eqqx'),
('DWNO-pooled', DWP,dict(trunk_layers=[50]*7,R=50,nx=64,ny=64,nz=48,width=64,level=4,P=25,num_param_levels=1,num_wavelet_layers=10),'DWNO_DeepONet/results','dwno_pooled_z','.eqqx'),
('D-FNO-BPSS',  DFB,dict(trunk_layers=[50]*7,R=50,nx=64,ny=64,nz=48,width=64,P=25,modes=7,num_fourier_layers=10),'DFNO_S_DeepONet/results','dfno_bpss_z','.eqqx'),
('D-FNO-pooled',DFP,dict(trunk_layers=[50]*7,R=50,nx=64,ny=64,nz=48,width=64,P=25,modes=7,num_fourier_layers=10),'DFNO_DeepONet/results','dfno_pooled_z','.eqqx'),
('MLP-BPSS',    MBP,dict(encoder_layers=[190,190,190,95],head_layers=[97,32,32,50],trunk_layers=[50]*7,r=30,nx=64,ny=64,nz=48),'MLP_S_DeepONet/results','mlp_bpss_z','.eqqx'),
]

f_test_multi=np.load(os.path.join(DATA,'f_test_multi.npy'))
u_test_multi=np.load(os.path.join(DATA,'u_test_multi.npy'))
ntest=100

print(f'Loaded {ntest} multi-layer test cases')
print(f'{"Model":<16s} {"Rel.L2":>8s} {"MAE(K)":>8s} {"MaxAE(K)":>9s} {"PeakErr":>9s} {"t(s)":>7s}')
print('-'*64)

for name,cls,kwargs,res_dir,prefix,ext in models_info:
    t0=time.perf_counter()
    ms={}
    for z in z_layers:
        epath=os.path.join(ML,res_dir,f'z{z}',f'{prefix}{z}{ext}')
        m=cls(key=key,**kwargs);m=eqx.tree_deserialise_leaves(epath,m);ms[z]=m

    jit_ms={z:eqx.filter_jit(ms[z]) for z in z_layers}

    rl2s=[];peak_errs=[];mae_sum=0.0;maxae=0.0
    for idx in range(ntest):
        u_total=np.zeros((64,64,48))
        for li,z in enumerate(z_layers):
            src=jnp.array(f_test_multi[idx:idx+1,li]/fs,dtype=jnp.float32)
            p=np.array(jit_ms[z](src))[0]*dT+T_amb
            u_total+=p
        u_total=u_total-4.0*T_amb
        u_true=u_test_multi[idx]
        diff=u_total-u_true
        rl2s.append(np.linalg.norm(diff.ravel())/np.linalg.norm(u_true.ravel()))
        mae_sum+=np.mean(np.abs(diff))
        maxae=max(maxae,np.max(np.abs(diff)))
        peak_errs.append(abs(u_total.max()-u_true.max())/(u_true.max()-T_amb))

    rl2_mean=np.mean(rl2s)*100;mae=mae_sum/ntest;peak_err=np.mean(peak_errs)*100
    dt=time.perf_counter()-t0
    print(f'{name:<16s} {rl2_mean:7.4f}% {mae:8.4f} {maxae:9.4f} {peak_err:8.4f}% {dt:6.1f}')
    del ms
