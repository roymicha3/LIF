import torch
from omegaconf import DictConfig

from tools.utils import SEQ_LEN
from network.kernel.kernel import Kernel
from data.spike.spike_sample import SpikeSample, digest_batch

from network.kernel.functional import cpu_based as cpu
from network.kernel.functional import gpu_based as gpu

from experiment_manager.environment import Environment
from experiment_manager.common.serializable import YAMLSerializable




@YAMLSerializable.register("SequentialDenKernel")
class SequentialDenKernel(Kernel, YAMLSerializable):
    
    def __init__(
        self,
        env : Environment,
        n,
        tau_s,
        tau_m,
        scale = False,
        learning = False
    ):
        super(SequentialDenKernel, self).__init__(n, (n, n), learning)
        super(YAMLSerializable, self).__init__()
        
        self.env = env
        self.n = n
        self.tau_s = tau_s
        self.tau_m = tau_m
        self.scale = scale
        
        self.dt = env.args.dt
        self.v_0 = env.args.v_0
        self.v_th = env.args.v_th
        self.device = env.device
        self.time_seq = SEQ_LEN(env.args.T, self.dt)
        
        self._beta = 1 - self.dt / self.tau_s
        self._alpha = 1 - self.dt / self.tau_m
        
        if self.device == "cuda":
            self.kernel = gpu.den
        else:
            self.kernel = cpu.den
        
        self.v_prev = None
        self.inner_state = None
        
    
    def reset(self):
        self.v_prev = None
        
    
    def silence(self, indices, output_index):
        # TODO: implement this
        return NotImplementedError("Silence function is not implemented yet.")
    

    def forward(self, input_):
        input_ = input_.to(self.device)
        
        if self.v_prev is None:
            self.v_prev = torch.zeros_like(input_)
            self.inner_state = torch.zeros_like(input_)
        
        inner_state, inner_spike, output, spike_output = \
            self.kernel(
                input_,
                self.inner_state,
                self.v_prev,
                self.v_th,
                self._alpha,
                self._beta,
                hard_reset=False)
        
        self.v_prev = output.clone()
        self.inner_state = inner_state.clone()
        
        if self.scale:
            return self.v_0 * output / self.dt
        
        return output
    
    def __call__(self, input_):
        """
        Call the forward function of the kernel
        """
        self.reset()
        generator = digest_batch(input_)
        
        for t in range(self.time_seq):
            data_t = next(generator)
            
            res = self.forward(data_t)
            
            self.v_prev = res.clone()
            
            if self.scale:
                res = self.v_0 * res / self.dt
            
            yield res
            

    
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment) -> "SequentialDenKernel":
        return cls(
            env,
            config.n,
            config.tau_s,
            config.tau_m,
            scale=config.scale,
            learning=config.learning,
        )
