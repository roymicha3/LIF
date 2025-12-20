import torch
from omegaconf import DictConfig

from tools.utils import SEQ_LEN
from network.kernel.kernel import Kernel
from data.spike import SpikeSample, digest_batch

from network.kernel.functional import cpu_based as cpu
from network.kernel.functional import gpu_based as gpu

from experiment_manager import Environment
from experiment_manager.common import YAMLSerializable




@YAMLSerializable.register("SequentialLeakyKernel")
class SequentialLeakyKernel(Kernel, YAMLSerializable):
    
    def __init__(
        self,
        env : Environment,
        n,
        tau,
        scale = False,
        learning = False
    ):
        super(SequentialLeakyKernel, self).__init__(n, (n, n), learning)
        super(YAMLSerializable, self).__init__()
        
        self.env = env
        self.n = n
        self.tau = tau
        self.scale = scale
        
        self.dt = env.args.dt
        self.v_0 = env.args.v_0
        self.v_th = env.args.v_th
        self.device = env.device
        self.time_seq = SEQ_LEN(env.args.T, self.dt)
        
        self._beta = 1 - self.dt / self.tau
        
        if self.device == "cuda":
            self.kernel = gpu.lif
        else:
            self.kernel = cpu.lif
        
        self.v_prev = None
        
    
    def reset(self):
        self.v_prev = None
        
    
    def silence(self, indices, output_index):
        # TODO: implement this
        return NotImplementedError("Silence function is not implemented yet.")
    

    def forward(self, input_):
        input_ = input_.to(self.device)
        
        if self.v_prev is None:
            self.v_prev = torch.zeros_like(input_)
        
        output, spike_output = self.kernel(
            input_,
            self.v_prev,
            self.v_th,
            self._beta,
            hard_reset=False)
        
        self.v_prev = output.clone()
        
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
    def from_config(cls, config: DictConfig, env: Environment) -> "SequentialLeakyKernel":
        return cls(
            env,
            config.n,
            config.tau,
            scale=config.scale,
            learning=config.learning,
        )
    
    @staticmethod    
    def assimulate_response(input_seq : SpikeSample, tau, dt):
        """
        this function assimulate how the output of the model should behave
        """
        seq_length = input_seq.seq_len 
        input_size = input_seq.size

        # Initialize the response tensor
        response = torch.zeros(seq_length, input_size)
        
        times = torch.arange(seq_length, dtype=torch.float32)
        
        for neuron in input_seq.get():
            spikes = neuron.get_spike_times()
            i = neuron.get_index()
            for t in spikes:
                # Compute the exponential decay terms
                exp = torch.exp(-(times[t:] - t) * dt / tau)
                response[t:, i] += (1 / tau) * exp

        return response
