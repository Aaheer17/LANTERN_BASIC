# standard python libraries
import numpy as np
import torch
import torch.nn as nn
import os, time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from matplotlib.backends.backend_pdf import PdfPages
import os
import sys 
#from torch.cuda.amp import autocast, GradScaler
from eval_loss import run_comprehensive_eval_noangles
# Other functions of project
from Util.util import *
from datasets import *
from documenter import Documenter
from plotting_util import *
from transforms import *
from challenge_files import *
from challenge_files import evaluate # avoid NameError: 'evaluate' is not defined
import Models
from Models import *
import time
import os
import pwd


from Util.plot_csv import main as plot_csv_main


def get_username():
    return pwd.getpwuid(os.getuid())[0]


class GenerativeModel(nn.Module):
    """
    Base Class for Generative Models to inherit from.
    Children classes should overwrite the individual methods as needed.
    Every child class MUST overwrite the methods:

    def build_net(self): should register some NN architecture as self.net
    def batch_loss(self, x): takes a batch of samples as input and returns the loss
    def sample_n_parallel(self, n_samples): generates and returns n_samples new samples

    See tbd.py for an example of child class

    Structure:

    __init__(params)      : Read in parameters and register the important ones
    build_net()           : Create the NN and register it as self.net
                            HAS TO BE OVERWRITTEN IN CHILD CLASS
    prepare_training()    : Read in the appropriate parameters and prepare the model for training
                            Currently this is called from run_training(), so it should not be called on its own
    run_training()        : Run the actual training.
                            Necessary parameters are read in and the training is performed.
                            This calls on the methods train_one_epoch() and validate_one_epoch()
    train_one_epoch()     : Performs one epoch of model training.
                            This calls on the method batch_loss(x)
    validate_one_epoch()  : Performs one epoch of validation.
                            This calls on the method batch_loss(x)
    batch_loss(x)         : Takes one batch of samples as input and returns the loss.
                            HAS TO BE OVERWRITTEN IN CHILD CLASS
    sample_n(n_samples)   : Generates and returns n_samples new samples as a numpy array
                            HAS TO BE OVERWRITTEN IN CHILD CLASS
    sample_and_plot       : Generates n_samples and makes plots with them.
                            This is meant to be used during training if intermediate plots are wanted

    """
    def __init__(self, params, device, doc):
        """
        :param params: file with all relevant model parameters
        """
        super().__init__()
        self.doc = doc
        self.params = params
        #print("inside the modelbase: ",self.params)
        self.device = device
        self.shape = self.params['shape']#get(self.params,'shape')
        self.conditional = get(self.params,'conditional',False)
        self.single_energy = get(self.params, 'single_energy', None) # Train on a single energy
        self.eval_mode = get(self.params, 'eval_mode', 'all')

        self.batch_size = self.params["batch_size"]
        self.batch_size_sample = get(self.params, "batch_size_sample", 10_000)
        print("in modelbase")
        self.net = self.build_net()
        self.model=self.net  # added by farzana to debug
        param_count = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f'init model: Model has {param_count} parameters')
        self.params['parameter_count'] = param_count

        self.epoch = get(self.params, "total_epochs", 0)
        self.iterations = get(self.params,"iterations", 1)
        self.regular_loss = []
        self.kl_loss = []
        self.attn_epoch=[0,10,20,30,40,50,60,70,80,90,99]
        self.runs = get(self.params, "runs", 0)
        self.iterate_periodically = get(self.params, "iterate_periodically", False)
        self.validate_every = get(self.params, "validate_every", 50)
        
        # augment data
        self.aug_transforms = get(self.params, "augment_batch", False)

        # load autoencoder for latent modelling
        self.ae_dir = get(self.params, "autoencoder", None)
        if self.ae_dir is None:
            self.transforms = get_transformations(
                params.get('transforms', None), doc=self.doc
            )
            self.latent = False
        else:
            self.ae = self.load_other(self.ae_dir)# model_class='AE'
            self.transforms = self.ae.transforms
            self.latent = True
        
    def build_net(self):
        pass

    def prepare_training(self):
        
        print("train_model: Preparing model training")

        self.train_loader, self.val_loader, self.bounds = get_loaders(
            self.params.get('eval_hdf5_file'),  #was hdf5_file
            self.params.get('particle_type'),
            self.params.get('xml_filename'),
            self.params.get('val_frac'),
            self.params.get('batch_size'),
            self.transforms,
            self.params.get('eps', 1.e-10),
            device=self.device,
            shuffle=True,
            width_noise=self.params.get('width_noise', 1.e-6),
            single_energy=self.params.get('single_energy', None),
            aug_transforms=self.aug_transforms
        )

        self.use_scheduler = get(self.params, "use_scheduler", False)
        self.train_losses = np.array([])
        self.train_losses_epoch = np.array([])
        self.val_losses = np.array([])
        self.val_losses_epoch = np.array([])

        self.n_trainbatches = len(self.train_loader)
        self.n_traindata = self.n_trainbatches*self.batch_size
        self.set_optimizer(steps_per_epoch=self.n_trainbatches)
        #self.scalar= GradScaler()
        if hasattr(self, 'setup_ema'):
            print("Setting up EMA model...")
            self.setup_ema()

        self.sample_periodically = get(self.params, "sample_periodically", False)
        if self.sample_periodically:
            self.sample_every = get(self.params, "sample_every", 1)
            self.sample_every_n_samples = get(self.params, "sample_every_n_samples", 100000)
            print(f'train_model: sample_periodically set to True. Sampling {self.sample_every_n_samples} every'
                  f' {self.sample_every} epochs. This may significantly slow down training!')

        self.log = get(self.params, "log", True)
        if self.log:
            log_dir = self.doc.basedir
            self.logger = SummaryWriter(log_dir)
            print(f"train_model: Logging to log_dir {log_dir}")
        else:
            print("train_model: log set to False. No logs will be written")

    def set_optimizer(self, steps_per_epoch=1, params=None):
        """ Initialize optimizer and learning rate scheduling """
        if params is None:
            params = self.params

        self.optimizer = torch.optim.AdamW(
                self.net.parameters(),
                lr = params.get("lr", 0.0002),
                betas = params.get("betas", [0.9, 0.999]),
                eps = params.get("eps", 1e-6),
                weight_decay = params.get("weight_decay", 0.)
                )
        self.scheduler = set_scheduler(self.optimizer, params, steps_per_epoch, last_epoch=-1)

    def run_training(self):

        self.prepare_training()

        samples = []
        n_epochs = get(self.params, "n_epochs", 100)
        past_epochs = get(self.params, "total_epochs", 0)
        if past_epochs != 0:
            self.load(epoch=past_epochs)
            self.scheduler = set_scheduler(
                self.optimizer, self.params, self.n_trainbatches,
                last_epoch=self.params.get("total_epochs", -1) * self.n_trainbatches
            )
        print(f"train_model: Model has been trained for {past_epochs} epochs before.")
        print(f"train_model: Beginning training. n_epochs set to {n_epochs}")
        t_0 = time.time()

        # ensure attrs exist so finally can close them safely
        if not hasattr(self, "csv_f"):
            self.csv_f = None
        if not hasattr(self, "csv_writer"):
            self.csv_writer = None

        try:
            self.opt_step = 0
            for e in range(n_epochs):
                t0 = time.time()
                self.epoch = past_epochs + e

                self.net.train()
                one_s = time.time()
                self.train_one_epoch()
                print("One epoch time: ", time.time() - one_s)

                if (self.epoch + 1) % self.validate_every == 0:
                    self.eval()
                    self.validate_one_epoch()

                if self.sample_periodically and (self.epoch + 1) % self.sample_every == 0:
                    self.eval()
                    if get(self.params, "reconstruct", False):
                        samples, c = self.reconstruct_n()
                    else:
                        samples, c = self.sample_n()
                    self.plot_samples(
                        samples=samples, conditions=c, name=self.epoch,
                        energy=self.single_energy, mode=self.eval_mode
                    )

                if get(self.params, "save_periodically", False):
                    if (self.epoch + 1) % get(self.params, "save_every", 10) == 0 or self.epoch == 0:
                        self.save(epoch=f"{self.epoch}")

                print(f"Epoch {self.epoch}: LR = {self.scheduler.get_last_lr()[0]}")
                if e == 0:
                    t1 = time.time()
                    dtEst = (t1 - t0) * n_epochs
                    print(f"Training time estimate: {dtEst/60:.2f} min = {dtEst/60**2:.2f} h")
                sys.stdout.flush()

            t_1 = time.time()
            traintime = t_1 - t_0
            self.params['train_time'] = traintime
            print(
                f"train_model: Finished training {n_epochs} epochs after {traintime:.2f} s = "
                f"{traintime / 60:.2f} min = {traintime / 60 ** 2:.2f} h.", flush=True
            )

            print("train_model: Saving final model: ", flush=True)
            np.save(self.doc.get_file('train_losses_epoch.npy'), self.train_losses_epoch)
            self.save()

            if get(self.params, "sample", True):
                if get(self.params, "reconstruct", False):
                    samples, c = self.reconstruct_n()
                else:
                    samples, c = self.comprehensive_eval_noangles(
                        n=self.params['n_samples'],
                        sampling_type="ddpm",
                        postprocess=True,
                        skip_normalize_by_elayer=True,
                        return_numpy=False,
                    )
                self.plot_samples(samples=samples, conditions=c, energy=self.single_energy)

        finally:
            # always close the CSV if it was opened by train_one_epoch
            if getattr(self, "csv_f", None) is not None:
                try:
                    self.csv_f.flush()
                except Exception:
                    pass
                try:
                    self.csv_f.close()
                except Exception:
                    pass
                self.csv_f = None
                self.csv_writer = None
                
         # --- now that the file is closed, generate plots ---
        try:
            #import os, sys
            os.environ.setdefault("MPLBACKEND", "Agg")  # headless-safe

            # If you imported earlier:
            # from plot_training_csv import main as plot_csv_main
            csv_path = self.doc.get_file("training_log.csv")
            plot_csv_main(csv_path, smooth=200, r_clip=None, bins=100)

            # Or, if you prefer subprocess:
            # script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plot_training_csv.py")
            # subprocess.run([sys.executable, script, "--csv", csv_path, "--smooth", "200", "--bins", "100"], check=True)

            print(f"plot_training_csv: figures saved next to {csv_path}")
        except Exception as e:
            print("plot_training_csv: skipped due to error:", repr(e))

        
    def train_one_epoch(self):
    
        # one-time setup
        if not hasattr(self, "opt_step"):
            self.opt_step = 0
    
        # CSV writer
        if not hasattr(self, "csv_writer") or self.csv_writer is None:
            import os, csv
            csv_path = self.doc.get_file("training_log.csv")
            need_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
            self.csv_f = open(csv_path, "a", newline="")
            self.csv_fields = [
                "opt_step","epoch","batch_id",
                "loss_total","loss_param",
                "CFD_metric",
                "mse_noise","mse_x0","mse_v",
                "cfd_raw","w_t_mean","lambda_warm","lambda_cfd_eff","r_eff",
                "cfd_fro_pred","cfd_fro_shrunk","cfd_fro_smooth","cfd_ema_dist",
                "lr","mvn_loss", "mvn_mode", "mvn_trace_val","raw_mvn", "energy_loss"
            ]
            self.csv_writer = csv.DictWriter(self.csv_f, fieldnames=self.csv_fields)
            if need_header:
                self.csv_writer.writeheader()
                self.csv_f.flush()
    
        # accumulators for epoch-averages
        epoch_sums = {k: 0.0 for k in [
            "loss_total","loss_param","CFD_metric","mse_noise","mse_x0","mse_v",
            "cfd_raw","w_t_mean","lambda_warm","lambda_cfd_eff",
            "cfd_fro_pred","cfd_fro_shrunk","cfd_fro_smooth","cfd_ema_dist","mvn_loss", "mvn_trace_val", "raw_mvn", "energy_loss"
        ]}
        epoch_counts = 0
    
        def _csv_val(d, k):
            v = d.get(k, None)
            if v is None:
                return ""
            try:
                vv = float(v)
                if np.isnan(vv) or np.isinf(vv):
                    return ""
                return vv
            except Exception:
                return ""
    
        train_losses = []
    
        for batch_id, x in enumerate(self.train_loader):
            self.optimizer.zero_grad(set_to_none=True)
    
            loss, comps = self.batch_loss(x)
    
            # (optional) attention rollout snapshot
            if (self.epoch in getattr(self, "attn_epoch", [])) and batch_id == 0:
                if hasattr(self.net, 'plot_attention_rollout'):
                    self.net.plot_attention_rollout(
                        save_path=self.doc.get_file(f'attn_epoch{self.epoch}_batch{batch_id}.png')
                    )
    
            if np.isfinite(loss.item()):
                loss.backward()
                clip = self.params.get('clip_gradients_to', None)
                if clip:
                    nn.utils.clip_grad_norm_(self.net.parameters(), clip)
    
                self.optimizer.step()
    
                # EMA update
                if hasattr(self, 'update_ema'):
                    self.update_ema()
    
                train_losses.append(loss.item())
    
                # Scheduler + lr
                cur_lr = None
                if getattr(self, "use_scheduler", False):
                    self.scheduler.step()
                    try:
                        cur_lr = float(self.scheduler.get_last_lr()[0])
                    except Exception:
                        cur_lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
    
                # effective relative weight r_eff = (lambda_eff * cfd_raw) / mse_noise
                mse = comps.get("mse_noise", None)
                cfd = comps.get("cfd_raw", None)
                lam_eff = comps.get("lambda_cfd_eff", None)
                if (mse is not None) and (cfd is not None) and (lam_eff is not None):
                    try:
                        r_eff = (lam_eff * cfd) / mse if mse > 0 else ""
                    except Exception:
                        r_eff = ""
                else:
                    r_eff = ""
    
                self.opt_step += 1
    
                # CSV row
                row = {
                    "opt_step": int(self.opt_step),
                    "epoch": int(self.epoch),
                    "batch_id": int(batch_id),
                    "loss_total": float(loss.item()),
                    "loss_param": _csv_val(comps, "loss_param"),
                    "CFD_metric": _csv_val(comps, "CFD_metric"),
                    "mse_noise": _csv_val(comps, "mse_noise"),
                    "mse_x0":    _csv_val(comps, "mse_x0"),
                    "mse_v":     _csv_val(comps, "mse_v"),
                    "cfd_raw": _csv_val(comps, "cfd_raw"),
                    "w_t_mean": _csv_val(comps, "w_t_mean"),
                    "lambda_warm": _csv_val(comps, "lambda_warm"),
                    "lambda_cfd_eff": _csv_val(comps, "lambda_cfd_eff"),
                    "r_eff": r_eff if r_eff == "" else float(r_eff),
                    "cfd_fro_pred":   _csv_val(comps, "cfd_fro_pred"),
                    "cfd_fro_shrunk": _csv_val(comps, "cfd_fro_shrunk"),
                    "cfd_fro_smooth": _csv_val(comps, "cfd_fro_smooth"),
                    "cfd_ema_dist":   _csv_val(comps, "cfd_ema_dist"),
                    "lr": "" if cur_lr is None else cur_lr,
                    "mvn_loss": _csv_val(comps, "mvn_loss"),
                    "mvn_mode":comps.get("mvn_mode", ""),
                    "mvn_trace_val":  _csv_val(comps, "mvn_trace_val"),
                    "raw_mvn": _csv_val(comps, "raw_mvn"),
                    "energy_loss": _csv_val(comps, "energy_loss")
                    }
                self.csv_writer.writerow(row)
                self.csv_f.flush()
    
                # epoch sums
                for k in epoch_sums.keys():
                    v = _csv_val(comps, k)
                    if v != "":
                        epoch_sums[k] += float(v)
                epoch_counts += 1
    
            else:
                print(f"[warn] Unstable loss. Skipped backprop for epoch {self.epoch}, batch {batch_id}")
    
        # per-epoch arrays
        if train_losses:
            tl_mean = float(np.mean(train_losses))
            self.train_losses_epoch = np.append(self.train_losses_epoch, tl_mean)
            self.train_losses = np.concatenate([self.train_losses, np.array(train_losses)], axis=0)
    
            if getattr(self, "log", False):
                self.logger.add_scalar("train_losses_epoch", tl_mean, self.epoch)
                if getattr(self, "use_scheduler", False):
                    try:
                        self.logger.add_scalar("learning_rate_epoch", self.scheduler.get_last_lr()[0], self.epoch)
                    except Exception:
                        pass
    
        # (optional) write epoch-averages to a separate CSV
        if epoch_counts > 0:
            import os, csv
            epoch_csv = self.doc.get_file("training_log_epoch.csv")
            need_header = (not os.path.exists(epoch_csv)) or (os.path.getsize(epoch_csv) == 0)
            with open(epoch_csv, "a", newline="") as ef:
                ew = csv.DictWriter(ef, fieldnames=["epoch"] + list(epoch_sums.keys()))
                if need_header:
                    ew.writeheader()
                avg = {k: (epoch_sums[k] / epoch_counts) for k in epoch_sums}
                ew.writerow({"epoch": int(self.epoch), **avg})

    @torch.inference_mode()
    def validate_one_epoch(self):
        
        val_losses = np.array([])
        # iterate batch wise over input
        for batch_id, x in enumerate(self.val_loader):

            # calculate batch loss
            loss,_ = self.batch_loss(x)
            val_losses = np.append(val_losses, loss.item())
            # if self.log:
            #     self.logger.add_scalar("val_losses", val_losses[-1], self.epoch*self.n_trainbatches + batch_id)

        self.val_losses_epoch = np.append(self.val_losses_epoch, val_losses.mean())
        self.val_losses = np.concatenate([self.val_losses, val_losses], axis=0)
        if self.log:
            self.logger.add_scalar("val_losses_epoch", self.val_losses_epoch[-1], self.epoch)

    def batch_loss(self, x):
        pass

    def generate_Einc_ds1(self, energy=None, sample_multiplier=1000):
        """ generate the incident energy distribution of CaloChallenge ds1
            sample_multiplier controls how many samples are generated: 10* sample_multiplier for low energies,
            and 5, 3, 2, 1 times sample multiplier for the highest energies

        """
        ret = np.logspace(8, 18, 11, base=2)
        ret = np.tile(ret, 10)
        ret = np.array(
            [*ret, *np.tile(2. ** 19, 5), *np.tile(2. ** 20, 3), *np.tile(2. ** 21, 2), *np.tile(2. ** 22, 1)])
        ret = np.tile(ret, sample_multiplier)
        if energy is not None:
            ret = ret[ret == energy]
        np.random.shuffle(ret)
        return ret
    @torch.inference_mode()
    def sample_trained_model_energy(self,size=10**5,sampling_type='ddim'):
        #print("inside the sample_trained_model_energy! ",self.params)
        self.eval()
        energy_model = self.load_other(self.params['energy_model'])
        #print("after loading the energy model: ",self.params)
        t_0 = time.time()

        Einc = torch.tensor(
            10**np.random.uniform(3, 6, size=get(self.params, "n_samples", 10**5)) 
            if self.params['eval_dataset'] in ['2', '3'] else
            self.generate_Einc_ds1(energy=self.single_energy),
            dtype=torch.get_default_dtype(),
            device=self.device
        ).unsqueeze(1)
        
        # transform Einc to basis used in training
        dummy, transformed_cond = None, Einc
        print("before starting the loop Einc min and max: ", Einc.min(), Einc.max())
        for fn in self.transforms:
            if hasattr(fn, 'cond_transform'):
                dummy, transformed_cond = fn(dummy, transformed_cond)
                print("Einc min and max: ",transformed_cond.min(), transformed_cond.max())

        batch_size_sample = get(self.params, "batch_size_sample", 10000)
        transformed_cond_loader = DataLoader(
            dataset=transformed_cond, batch_size=batch_size_sample, shuffle=False
        )
        #print("param----s", self.params)
        sample = torch.vstack([energy_model.sample_batch(c,sampling_type=sampling_type).cpu()
                               for c in transformed_cond_loader])
        
        t_1 = time.time()
        sampling_time = t_1 - t_0
        self.params["sample_time"] = sampling_time
        print(
            f"generate_samples: Finished generating {len(sample)} samples "
            f"after {sampling_time} s.", flush=True
        )
        
        return sample, transformed_cond.cpu()
    # inside class ModelBase(...)
    @torch.no_grad()
    def comprehensive_eval_noangles(
        self,
        n: int = 15000,
        sampling_type: str = "ddpm",
        postprocess: bool = True,
        skip_normalize_by_elayer: bool = True,
        return_numpy: bool = False,
    ):
        """
        High-level wrapper:
          - builds CaloChallengeDataset (energy-only condition)
          - runs the no-angles evaluation pipeline
          - returns RAW (non-postprocessed) generated samples and RAW conditions

        Returns:
            (U_gen_raw, E_cond_raw)
            - if return_numpy=False: torch.FloatTensor on CPU
                * U_gen_raw: [N, 45]
                * E_cond_raw: [N, 1]
            - if return_numpy=True: NumPy arrays with same shapes
        """
        # --- required params ---
        params = getattr(self, "params", {})
        ref_file = params.get("eval_hdf5_file", None)
        xml_file = params.get("xml_filename", None)
        ptype    = params.get("particle_type", None)
        if not (ref_file and xml_file and ptype):
            raise ValueError(
                "Missing required params: eval_hdf5_file, eval_xml_file, particle_type"
            )

        # --- dataset ---
        ds = CaloChallengeDataset(
            hdf5_file=ref_file,
            particle_type=ptype,
            xml_filename=xml_file,
            val_frac=0.0,
            transform=getattr(self, "transforms", None),
            split=None,
            device=self.device,
            single_energy=None,
            aug_transforms=False,
        )

        # --- outputs dir & label ---
        basedir = getattr(self.doc, "basedir", os.getcwd())
        outdir  = os.path.join(basedir, "eval_results")
        label   = params.get("LBL", ptype or "noangles")

        # --- run evaluator ---
        U_gen_raw, E_cond_raw = run_comprehensive_eval_noangles(
            model=self,
            dataset=ds,
            n_samples=params.get("n_samples", n),
            batch_size=params.get("batch_size_sample", 1000),
            sampling_type=sampling_type,
            postprocess=postprocess,
            skip_normalize_by_elayer=False,
            outdir=outdir,
            label=label,
            return_numpy=return_numpy,
        )
        return U_gen_raw, E_cond_raw

    @torch.inference_mode()
    def sample_n(self, size=10**5):
        print("Inside sample_n")
        self.eval()
        

        # if self.net.bayesian:
        #     self.net.map = get(self.params, "fix_mu", False)
        #     for bay_layer in self.net.bayesian_layers:
        #         bay_layer.random = None
       
        # sample = []

        t_0 = time.time()

        Einc = torch.tensor(
            10**np.random.uniform(3, 6, size=get(self.params, "n_samples", 10**5)) 
            if self.params['eval_dataset'] in ['2', '3'] else
            self.generate_Einc_ds1(energy=self.single_energy),
            dtype=torch.get_default_dtype(),
            device=self.device
        ).unsqueeze(1)
        
        # transform Einc to basis used in training
        dummy, transformed_cond = None, Einc
        for fn in self.transforms:
            if hasattr(fn, 'cond_transform'):
                dummy, transformed_cond = fn(dummy, transformed_cond)

        batch_size_sample = get(self.params, "batch_size_sample", 1000)
        transformed_cond_loader = DataLoader(
            dataset=transformed_cond, batch_size=batch_size_sample, shuffle=False
        )
        
        # sample u_i's if self is a shape model
        if self.params['model_type'] == 'shape': 
            # load energy model
            energy_model = self.load_other(self.params['energy_model'])

            if self.params.get('sample_us', True):
                # sample us
                u_samples = torch.vstack([
                    energy_model.sample_batch(c) for c in transformed_cond_loader
                ])
                if self.latent:
                    # post-process u-samples according to energy config
                    # CAUTION: shape config pre-processing may then also be necessary!
                    dummy = torch.empty(1, 1)
                    for fn in energy_model.transforms[:0:-1]: # skip NormalizeByElayer
                        u_samples, dummy = fn(u_samples, dummy, rev=True)                
                transformed_cond = torch.cat([transformed_cond, u_samples], dim=1)
            else: # optionally use truth us
                transformed_cond = CaloChallengeDataset(
                self.params.get('eval_hdf5_file'),
                self.params.get('particle_type'),
                self.params.get('xml_filename'),
                transform=self.transforms,
                device=self.device,
                single_energy=self.single_energy
                ).energy
            
            # concatenate with Einc
            transformed_cond_loader = DataLoader(
                dataset=transformed_cond, batch_size=batch_size_sample, shuffle=False
            )
        print("Before calling the autoregressive_sample_batch: ", batch_size_sample)       
        sample = torch.vstack([self.sample_batch(c).cpu() for c in transformed_cond_loader])
        #sample = torch.vstack([self.autoregressive_sample_batch(c).cpu() for c in transformed_cond_loader])

        t_1 = time.time()
        sampling_time = t_1 - t_0
        self.params["sample_time"] = sampling_time
        print(
            f"generate_samples: Finished generating {len(sample)} samples "
            f"after {sampling_time} s.", flush=True
        )

        return sample, transformed_cond.cpu()
    
    def reconstruct_n(self,):
        print("inside reconstruct_n")
        if ~hasattr(self, 'train_loader'):
            self.train_loader, self.val_loader, self.bounds = get_loaders(
                self.params.get('hdf5_file'),
                self.params.get('particle_type'),
                self.params.get('xml_filename'),
                self.params.get('val_frac'),
                self.params.get('batch_size_sample'),
                self.transforms,
                self.params.get('eps', 1.e-10),
                device=self.device,
                shuffle=False,
                width_noise=self.params.get('width_noise', 1.e-6),
                single_energy=self.params.get('single_energy', None)
            )

        recos = []
        energies = []

        self.eval()
        for n, x in enumerate(self.train_loader):
            reco, cond = self.sample_batch(x)
            recos.append(reco)
            energies.append(cond)
        for n, x in enumerate(self.val_loader):
            reco, cond = self.sample_batch(x)
            recos.append(reco)
            energies.append(cond)

        recos = torch.vstack(recos)
        energies = torch.vstack(energies)
        return recos, energies

    def sample_batch(self, batch):
        pass
    def plot_samples_mimic(self, samples, conditions,name="",energy=None, doc=None):
        transforms = self.transforms
        if doc is None: doc = self.doc

        if self.params['model_type'] == 'energy':
            reference = CaloChallengeDataset(
                self.params.get('eval_hdf5_file'),
                self.params.get('particle_type'),
                self.params.get('xml_filename'),
                transform=transforms, # TODO: Or, apply NormalizeEByLayer popped from model transforms
                device=self.device,
                single_energy=self.single_energy
            )
            ref_layer=reference.layers
            ref_energy=reference.energy
            print("shape of ref_layer and ref_energy: ", ref_layer.shape, ref_energy.min(),ref_energy.max())
            print("----------------End of Geant4----------------")
            # postprocess
            print("before plotting shape of SAMPLES and Conditions: ",samples.shape,conditions.min(),conditions.max())
            for fn in transforms[::-1]:
                # if fn.__class__.__name__ == 'NormalizeByElayer':
                #     break # this might break plotting
                samples, conditions = fn(samples, conditions, rev=True)
                ref_layer, ref_energy = fn(ref_layer, ref_energy, rev=True)
                print("inside the loop : ", samples.shape, conditions.shape, ref_layer.shape, ref_energy.shape)
                
                #reference, conditions_r = fn(reference, conditions, rev=True)
            samples = samples.detach().cpu().numpy()
            conditions_s = conditions.detach().cpu().numpy()
#             samples = samples.detach().cpu().numpy()
#             conditions_s = conditions_s.detach().cpu().numpy()
            
#             reference=reference.detach().cpu().numpy()
#             conditions_r = conditions_r.detach().cpu().numpy()
                
            s_dict={'Layer' : samples,
                   'Incident Energy' : conditions_s }

            torch.save(s_dict, doc.get_file('s_dict.pt'))
        
           
            reference= ref_layer.detach().cpu().numpy()
            conditions_r = ref_energy.detach().cpu().numpy()
            
            r_dict={
                'Layer' : reference,
                'Incident Energy' : conditions_r
                
            }
            
        
            
            torch.save(r_dict, doc.get_file('r_dict.pt'))
            
    def plot_samples_old(self, sample_path, reference_path, doc):
        
        samples = torch.load(sample_path)
        reference = torch.load(reference_path)

        samples[:,1:] = torch.clip(samples[:,1:], min=0., max=1.)
        reference[:,1:] = torch.clip(reference[:,1:], min=0., max=1.)

        samples_np = samples.detach().cpu().numpy()
        reference_np = reference.detach().cpu().numpy()

        print("LBL....", self.params['LBL'])

        plot_ui_dists(samples_np, reference_np, documenter=doc, LBL=self.params['LBL'])
        evaluate.eval_ui_dists(samples_np, reference_np, documenter=doc, params=self.params)
        

        
    def plot_samples(self, samples, conditions, name="", energy=None, doc=None):
        #print(f"Samples requires_grad {samples.requires_grad}")
        transforms = self.transforms
        if doc is None: doc = self.doc

        if self.params['model_type'] == 'energy':
            reference = CaloChallengeDataset(
                self.params.get('eval_hdf5_file'),
                self.params.get('particle_type'),
                self.params.get('xml_filename'),
                transform=transforms, # TODO: Or, apply NormalizeEByLayer popped from model transforms
                device=self.device,
                single_energy=self.single_energy
            ).layers
            
            # postprocess
            #print("before plotting shape of SAMPLES: ",samples.shape)
            print("*******  Starting Reverse Transformation *******")
            for fn in transforms[::-1]:
                if fn.__class__.__name__ == 'NormalizeByElayer':
                    break # this might break plotting
                print("Sampling information")
                samples, _ = fn(samples, conditions, rev=True)
                print("Geant4 information")
                reference, _ = fn(reference, conditions, rev=True)
            torch.save(samples, doc.get_file("samples.pt"))
            torch.save(reference, doc.get_file("reference.pt"))
            # clip u_i's (except u_0) to [0,1] 
            samples[:,1:] = torch.clip(samples[:,1:], min=0., max=1.)
            reference[:,1:] = torch.clip(reference[:,1:], min=0., max=1.)
            
            print("LBL....",self.params['LBL'])
            plot_ui_dists(
                samples.detach().cpu().numpy(),
                reference.detach().cpu().numpy(),
                documenter=doc,
                LBL=self.params['LBL']
            )
            #plot_losses(self.train_losses,self.val_losses, documenter=doc)
            print("before passing to the evaluate: ",samples.requires_grad, reference.requires_grad)
            evaluate.eval_ui_dists(
                samples.detach().cpu().numpy(),
                reference.detach().cpu().numpy(),
                documenter=doc,
                params=self.params,
            )
        else:
            if self.latent:
                #save generated latent space
                self.save_sample(samples, conditions, name=name+'_latent', doc=doc) 
                    
            # postprocess
            for fn in transforms[::-1]:
                samples, conditions = fn(samples, conditions, rev=True)
            
            samples = samples.detach().cpu().numpy()
            conditions = conditions.detach().cpu().numpy()

            self.save_sample(samples, conditions, name=name, doc=doc)
            evaluate.run_from_py(samples, conditions, doc, self.params)

    def plot_saved_samples(self, name="", energy=None, doc=None):
        if doc is None: doc = self.doc
        mode = self.params.get('eval_mode', 'all')
        script_args = (
            f"-i {doc.basedir}/ "
            f"-r {self.params['eval_hdf5_file']} -m {mode} --cut {self.params['eval_cut']} "
            f"-d {self.params['eval_dataset']} --output_dir {doc.basedir}/final/ --save_mem"
        ) + (f" --energy {energy}" if energy is not None else '')
        evaluate.main(script_args.split())

    def save_sample(self, sample, energies, name="", doc=None):
        """Save sample in the correct format"""
        if doc is None: doc = self.doc
        save_file = h5py.File(doc.get_file(f'samples{name}.hdf5'), 'w')
        save_file.create_dataset('incident_energies', data=energies)
        save_file.create_dataset('showers', data=sample)
        save_file.close()            
 
    def save(self, epoch=""):
        """ Save the model, and more if needed"""
        # torch.save({"opt": self.optimizer.state_dict(),
        #             "net": self.net.state_dict(),
        #             "losses": self.train_losses_epoch,
        #             "epoch": self.epoch,
        #             "scheduler": self.scheduler.state_dict()}
        #             , self.doc.get_file(f"model{epoch}.pt"))
        
        """ Save the model, and more if needed"""
        save_dict = {
            "opt": self.optimizer.state_dict(),
            "net": self.net.state_dict(),
            "losses": self.train_losses_epoch,
            "epoch": self.epoch,
            "scheduler": self.scheduler.state_dict()
        }

        # Add EMA model if it exists
        if hasattr(self, 'ema_model') and self.ema_model is not None:
            save_dict["ema_net"] = self.ema_model.state_dict()

        torch.save(save_dict, self.doc.get_file(f"model{epoch}.pt"))

    def load(self, epoch=""):
        """ Load the model, and more if needed"""
        name = self.doc.get_file(f"model{epoch}.pt")
        state_dicts = torch.load(name, map_location=self.device, weights_only=False)
        self.net.load_state_dict(state_dicts["net"])
        
        if "losses" in state_dicts:
            self.train_losses_epoch = state_dicts.get("losses", {})
        if "epoch" in state_dicts:
            self.epoch = state_dicts.get("epoch", 0)
        #if "opt" in state_dicts:
        #    self.optimizer.load_state_dict(state_dicts["opt"])
        #if "scheduler" in state_dicts:
        #    self.scheduler.load_state_dict(state_dicts["scheduler"])
        self.net.to(self.device)
        
        # Load EMA model if it exists
        if "ema_net" in state_dicts and hasattr(self, 'ema_model') and self.ema_model is not None:
            self.ema_model.load_state_dict(state_dicts["ema_net"])

    def load_other(self, model_dir):
        """ Load a different model (e.g. to sample u_i's)"""
        
        with open(os.path.join(model_dir, 'params.yaml')) as f:
            params_old = yaml.load(f, Loader=yaml.FullLoader)

        model_class = params_old['model']
        # choose model
        if model_class == 'TBD':
            Model = self.__class__
        if model_class == 'TransfusionAR':
            from Models import TransfusionAR
            Model = TransfusionAR
        elif model_class == 'AE':
            from Models import AE
            Model = AE
        elif model_class =='TransfusionDDPM':
            from Models import TransfusionDDPM
            Model = TransfusionDDPM
        elif model_class=='TBD_DIFF':
            from Models import TBD_DIFF
            Model=TBD_DIFF
            

        # load model
        doc_trained = Documenter(None, existing_run=model_dir, read_only=True)
        other = Model(params_old, self.device, doc_trained)
        state_dicts = torch.load(
            os.path.join(model_dir, 'model.pt'), map_location=self.device, weights_only=False
        )
        other.net.load_state_dict(state_dicts["net"])
        
        # use eval mode and freeze weights
        other.eval()
        for p in other.parameters():
            p.requires_grad = False

        return other
