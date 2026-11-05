"""Audited, capped upstream-component adapters. No held-out test input in fit()."""
from .common import *
import pickle, copy
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0,str(ROOT/'external/tabsyn'))
from tabsyn.vae.model import Model_VAE, Encoder_model, Decoder_model
from tabsyn.model import MLPDiffusion, Model
from tabsyn.diffusion_utils import sample as diffusion_sample

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if DEVICE.type=='cuda' else []}

def set_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu())
    if DEVICE.type=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda']])

def checkpoint(path,obj):
    path=Path(path); tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)

def load(path):return torch.load(path,map_location=DEVICE,weights_only=False)

class TabSynAdapter:
    def __init__(self,b,free,cfg):
        self.b=b;self.free=free;self.cfg=cfg
        self.num_indices=b.free if free else list(range(len(b.cols)))
        self.cat_indices=b.generated_cats if free else list(range(len(b.cats)))
        self.categories=[b.codec.sizes[j] for j in self.cat_indices]
        self.vae=Model_VAE(2,len(self.num_indices),self.categories,4,n_head=1,factor=32,bias=True).to(DEVICE)
        self.encoder=Encoder_model(2,len(self.num_indices),self.categories,4,1,32).to(DEVICE)
        self.decoder=Decoder_model(2,len(self.num_indices),self.categories,4,1,32).to(DEVICE)
        self.width=cfg['diffusion_width'];self.d=None
    def encode(self,x,k):
        a=self.normalizer.transform(x[:,self.num_indices].detach().cpu().numpy()).astype(np.float32)
        return torch.tensor(a,device=DEVICE),k[:,self.cat_indices]
    @staticmethod
    def loss(model,x,k):
        recon,cat,mu,logvar=model(x,k)
        mse=(recon-x).square().mean()
        ce=torch.stack([F.cross_entropy(p,k[:,i]) for i,p in enumerate(cat)]).mean()
        kl=-.5*(1+logvar-mu.square()-logvar.exp()).mean()
        return mse,ce,kl
    def save_bundle(self,path):
        checkpoint(path,{'vae':self.vae.state_dict(),'diffusion':self.d.state_dict(),'mean':self.mean,'normalizer':self.normalizer,'free':self.free,'config':self.cfg})
    def load_bundle(self,path):
        obj=load(path);self.normalizer=obj['normalizer'];self.vae.load_state_dict(obj['vae'])
        self.encoder.load_weights(self.vae);self.decoder.load_weights(self.vae)
        dim=(len(self.num_indices)+len(self.cat_indices))*4
        self.d=Model(denoise_fn=MLPDiffusion(dim,self.width).to(DEVICE),hid_dim=dim).to(DEVICE)
        self.d.load_state_dict(obj['diffusion']);self.mean=obj['mean'];self.encoder.eval();self.decoder.eval();self.d.eval()
    @torch.no_grad()
    def sample(self,n,s):
        self.d.eval();self.decoder.eval()
        with isolated_rng(s):
            # Chunk sampling limits memory; same chunk policy for all variants.
            xs=[];ks=[]
            for start in range(0,n,1024):
                m=min(1024,n-start);dim=self.mean.shape[0]
                z=diffusion_sample(self.d.denoise_fn_D,m,dim,num_steps=self.cfg['sample_steps'],device=str(DEVICE))
                z=z*2+self.mean;num,cat=self.decoder(z.reshape(m,-1,4))
                raw=self.normalizer.inverse_transform(num.cpu().numpy()).astype(np.float32)
                x=torch.tensor(raw,device=DEVICE); k=torch.zeros(m,len(self.b.cats),device=DEVICE,dtype=torch.long)
                for j,index in enumerate(self.cat_indices):k[:,index]=cat[j].argmax(1)
                if self.free:k=fixk(self.b,k);x=decode(self.b,x,k)
                else:x=self.b.spec.clamp_continuous(x)
                xs.append(x);ks.append(k)
            return torch.cat(xs),torch.cat(ks)
    def fit(self,folder,s):
        b=self.b;assert b.reference=='val','fit must receive validation bundle, never test'
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        if (folder/'trained.json').exists():self.load_bundle(folder/'model.pt');return json.loads((folder/'trained.json').read_text())
        cfg=self.cfg;seed(s);sync();started=time.perf_counter()
        self.normalizer=QuantileTransformer(n_quantiles=min(len(b.train_x),max(min(len(b.train_x)//30,1000),10)),output_distribution='normal',subsample=1000000000,random_state=s)
        self.normalizer.fit(b.train_x[:,self.num_indices].cpu().numpy())
        x,k=self.encode(b.train_x,b.train_k);vx,vk=self.encode(b.ref_x,b.ref_k)
        opt=torch.optim.Adam(self.vae.parameters(),lr=cfg['lr']); sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.95,patience=10)
        best=float('inf');bad=0;beta=.01;history=[];start_epoch=0;best_epoch=0
        resume=folder/'vae_resume.pt'
        resumed_any=resume.exists()
        if resume.exists():
            obj=load(resume);self.vae.load_state_dict(obj['model']);opt.load_state_dict(obj['opt']);sched.load_state_dict(obj['sched'])
            start_epoch=obj['epoch'];best=obj['best'];bad=obj['bad'];beta=obj['beta'];history=obj['history'];best_epoch=obj['best_epoch'];set_rng(obj['rng'])
        for epoch in range(start_epoch,cfg['vae_max_epochs']):
            if bad>=cfg['vae_patience'] and epoch>=cfg['vae_min_epochs']:break
            self.vae.train();perm=torch.randperm(len(x),device=DEVICE);losses=[]
            for ix in perm.split(cfg['batch_size']):
                mse,ce,kl=self.loss(self.vae,x[ix],k[ix]); loss=mse+ce+beta*kl
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite TabSyn VAE loss')
                opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()))
            self.vae.eval()
            with torch.no_grad(),isolated_rng(s+30000):
                terms=[]
                for ix in torch.arange(len(vx),device=DEVICE).split(2048):
                    mse,ce,kl=self.loss(self.vae,vx[ix],vk[ix]);terms.append((len(ix),float(mse),float(ce)))
                val=sum(n*(m+c) for n,m,c in terms)/len(vx)
            sched.step(val)
            if val<best-1e-6:
                best=val;bad=0;best_epoch=epoch+1;checkpoint(folder/'vae_best.pt',self.vae.state_dict())
            else:
                bad+=1
                if bad%10==0:beta=max(1e-5,beta*.7)
            history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation_reconstruction':val,'beta':beta,'lr':opt.param_groups[0]['lr']})
            if (epoch+1)%cfg['check_every']==0 or epoch+1==cfg['vae_max_epochs'] or bad>=cfg['vae_patience']:
                checkpoint(resume,{'model':self.vae.state_dict(),'opt':opt.state_dict(),'sched':sched.state_dict(),'epoch':epoch+1,'best':best,'bad':bad,'beta':beta,'best_epoch':best_epoch,'history':history,'rng':rng_state()})
                pd.DataFrame(history).to_csv(folder/'vae_history.csv',index=False)
                event(b.out,stage='tabsyn_vae',dataset=b.name,split=b.split,seed=s,free=self.free,epoch=epoch+1,best=best,elapsed=round(time.perf_counter()-started,1))
        self.vae.load_state_dict(load(folder/'vae_best.pt'));self.encoder.load_weights(self.vae);self.decoder.load_weights(self.vae)
        self.encoder.eval();self.decoder.eval()
        with torch.no_grad():
            z=torch.cat([self.encoder(x[ix],k[ix])[:,1:,:].reshape(len(ix),-1) for ix in torch.arange(len(x),device=DEVICE).split(2048)])
        self.mean=z.mean(0);z=(z-self.mean)/2
        self.d=Model(denoise_fn=MLPDiffusion(z.shape[1],self.width).to(DEVICE),hid_dim=z.shape[1]).to(DEVICE)
        opt=torch.optim.Adam(self.d.parameters(),lr=cfg['lr']);sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.9,patience=20)
        best_sw=float('inf');bad_epochs=0;diffhist=[];start_epoch=0;selected_epoch=0
        resume=folder/'diffusion_resume.pt'
        resumed_any=resumed_any or resume.exists()
        if resume.exists():
            obj=load(resume);self.d.load_state_dict(obj['model']);opt.load_state_dict(obj['opt']);sched.load_state_dict(obj['sched'])
            start_epoch=obj['epoch'];best_sw=obj['best'];bad_epochs=obj['bad'];diffhist=obj['history'];selected_epoch=obj['selected_epoch'];set_rng(obj['rng'])
        for epoch in range(start_epoch,cfg['diffusion_max_epochs']):
            if bad_epochs>=cfg['diffusion_patience'] and epoch>=cfg['diffusion_min_epochs']:break
            self.d.train();losses=[]
            for ix in torch.randperm(len(z),device=DEVICE).split(cfg['batch_size']):
                loss=self.d(z[ix]).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite TabSyn diffusion loss')
                opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()))
            trainloss=float(np.mean(losses));sched.step(trainloss)
            row={'epoch':epoch+1,'train_loss':trainloss,'lr':opt.param_groups[0]['lr']}
            if (epoch+1)%cfg['check_every']==0 or epoch+1==cfg['diffusion_max_epochs']:
                xx,kk=self.sample(len(vx),s+40000);val=metrics(b,xx,kk)['sw'];row['validation_sw']=val
                if val<best_sw-1e-5:
                    best_sw=val;bad_epochs=0;selected_epoch=epoch+1;self.save_bundle(folder/'model.pt')
                else:bad_epochs+=cfg['check_every']
                diffhist.append(row)
                checkpoint(resume,{'model':self.d.state_dict(),'opt':opt.state_dict(),'sched':sched.state_dict(),'epoch':epoch+1,'best':best_sw,'bad':bad_epochs,'history':diffhist,'selected_epoch':selected_epoch,'rng':rng_state()})
                pd.DataFrame(diffhist).to_csv(folder/'diffusion_history.csv',index=False)
                event(b.out,stage='tabsyn_diffusion',dataset=b.name,split=b.split,seed=s,free=self.free,epoch=epoch+1,val_sw=val,best_sw=best_sw,elapsed=round(time.perf_counter()-started,1))
            else:diffhist.append(row)
        sync();info={'seed':s,'free':self.free,'vae_selected_epoch':best_epoch,'vae_epochs_run':len(history),'vae_hit_cap':len(history)>=cfg['vae_max_epochs'],
                   'diffusion_selected_epoch':selected_epoch,'diffusion_epochs_run':len(diffhist),'diffusion_hit_cap':len(diffhist)>=cfg['diffusion_max_epochs'],
                   'validation_sw':best_sw,'training_seconds_this_process':time.perf_counter()-started,'resumed':resumed_any,
                   'recipe':f'upstream components; quantile; MLP{self.width}; validation-selected capped adapter; see config'}
        self.load_bundle(folder/'model.pt');dump(folder/'trained.json',info);return info

def ddpm_modules():
    path=ROOT/'external/tab-ddpm';sys.path.insert(0,str(path));sys.path.insert(0,str(path/'scripts'))
    from scripts.train import train
    from scripts.sample import sample
    lc.patch_tabddpm_quantile_normalizer()
    return train,sample

def ddpm_setup(b,folder):
    assert b.reference=='val'
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if not (folder/'info.json').exists():
        for name,x,k in [('train',b.train_x,b.train_k),('val',b.ref_x,b.ref_k),('test',b.ref_x,b.ref_k)]:
            a=x.cpu().numpy();np.save(folder/f'X_num_{name}.npy',a[:,:-1]);np.save(folder/f'y_{name}.npy',a[:,-1]);np.save(folder/f'X_cat_{name}.npy',k.cpu().numpy())
        dump(folder/'info.json',{'name':b.name,'id':b.name,'task_type':'regression','n_classes':None,'train_size':len(b.train_x),'val_size':len(b.ref_x),'test_size':len(b.ref_x),'n_num_features':len(b.cols)-1,'n_cat_features':len(b.cats),'test_placeholder':'validation, never held-out test'})

def ddpm_fit(b,folder,data,s,steps,cfg):
    train,_=ddpm_modules();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    args=dict(parent_dir=str(folder),real_data_path=str(data),batch_size=b.cfg.batch_size,model_type='mlp',
        model_params={'num_classes':0,'is_y_cond':False,'rtdl_params':{'d_layers':[128,128],'dropout':0.}},
        num_timesteps=cfg['timesteps'],gaussian_loss_type='mse',scheduler='cosine',
        T_dict={'seed':s,'normalization':'quantile','num_nan_policy':None,'cat_nan_policy':None,'cat_min_frequency':None,'cat_encoding':None,'y_policy':'default'},
        num_numerical_features=len(b.cols)-1,device=DEVICE,seed=s)
    if not (folder/'trained.json').exists():
        sync();t=time.perf_counter();train(**copy.deepcopy(args),steps=steps,lr=.002,weight_decay=0.);sync()
        dump(folder/'trained.json',{'training_seconds':time.perf_counter()-t,'steps':steps,'seed':s})
    return args

def ddpm_sample(b,folder,args,n):
    _,sample=ddpm_modules();folder=Path(folder)
    sample(**copy.deepcopy(args),num_samples=n,model_path=str(folder/'model.pt'))
    a=np.load(folder/'X_num_train.npy');y=np.load(folder/'y_train.npy').reshape(-1,1)
    x=torch.tensor(np.concatenate([a,y],1),device=DEVICE,dtype=torch.float32)
    k=torch.tensor(np.load(folder/'X_cat_train.npy',allow_pickle=True).astype(np.int64),device=DEVICE)
    assert x.shape==(n,len(b.cols))
    return b.spec.clamp_continuous(x),k
