import torch
import psutil
import GPUtil
import pytorch_lightning as L


class HardwareMonitorCallback(L.Callback):
    def __init__(self, log_every_n_steps=10):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.log_every_n_steps == 0 and trainer.is_global_zero:
            pl_module.log("system/CPU_usage_percent", psutil.cpu_percent(), rank_zero_only=True)
            pl_module.log("system/RAM_usage_percent", psutil.virtual_memory().percent, rank_zero_only=True)

            if torch.cuda.is_available():
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    pl_module.log(f"system/GPU_{i}_util_percent", gpu.load * 100, rank_zero_only=True)
                    pl_module.log(f"system/GPU_{i}_mem_percent", gpu.memoryUtil * 100, rank_zero_only=True)
                    pl_module.log(f"system/GPU_{i}_mem_used_mb", gpu.memoryUsed, rank_zero_only=True)
