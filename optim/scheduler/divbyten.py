#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from . import register_scheduler
from .base_scheduler import BaseLRScheduler
import argparse
import math


@register_scheduler("divbyten")
class DivByTen(BaseLRScheduler):
    """
    DivByTen learning rate scheduler: divides by 10 of max lr at specified epoch steps
    """
    def __init__(self, opts, **kwargs) -> None:
        is_iter_based = getattr(opts, "scheduler.is_iteration_based", True)
        super(DivByTen, self).__init__(opts=opts)

        max_iterations = getattr(opts, "scheduler.max_iterations", 150000)
        warmup_iterations = getattr(opts, "scheduler.warmup_iterations", 10000)

        self.min_lr = getattr(opts, "scheduler.divbyten.min_lr", 1e-5)
        self.max_lr = getattr(opts, "scheduler.divbyten.max_lr", 0.4)
        self.step1epoch = getattr(opts, "scheduler.divbyten.step1epoch", 40)
        self.step2epoch = getattr(opts, "scheduler.divbyten.step2epoch", 60)
        self.step3epoch = getattr(opts, "scheduler.divbyten.step3epoch", 80)

        self.warmup_iterations = max(warmup_iterations, 0)
        if self.warmup_iterations > 0:
            warmup_init_lr = getattr(opts, "scheduler.warmup_init_lr", 1e-7)
            self.warmup_init_lr = warmup_init_lr
            self.warmup_step = (self.max_lr - self.warmup_init_lr) / self.warmup_iterations

        self.period = max_iterations - self.warmup_iterations + 1 if is_iter_based \
            else getattr(opts, "scheduler.max_epochs", 350)

        self.is_iter_based = is_iter_based

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title="DivByTen LR arguments", description="DivByTen LR arguments")

        group.add_argument('--scheduler.divbyten.min-lr', type=float, default=1e-5,
                           help="Minimum LR in DivByTen LR scheduler")
        group.add_argument('--scheduler.divbyten.max-lr', type=float, default=0.1,
                           help="Maximum LR in DivByTen LR scheduler")
        group.add_argument('--scheduler.divbyten.step1epoch', type=float, default=40,
                           help="Step1 Epoch Value in DivByTen LR scheduler")
        group.add_argument('--scheduler.divbyten.step2epoch', type=float, default=60,
                           help="Step2 Epoch value in DivByTen LR scheduler")      
        group.add_argument('--scheduler.divbyten.step3epoch', type=float, default=80,
                           help="Step3 Epoch value in DivByTen LR scheduler")                    

        return parser

    def get_lr(self, epoch: int, curr_iter: int) -> float:
        if curr_iter < self.warmup_iterations:
            curr_lr = self.warmup_init_lr + curr_iter * self.warmup_step
        else:
            if self.is_iter_based:
                curr_iter = curr_iter - self.warmup_iterations
                curr_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * curr_iter / self.period))
            else:
                if epoch < self.step1epoch:
                    curr_lr = self.max_lr
                elif epoch >= self.step1epoch and epoch < self.step2epoch:
                    curr_lr = self.max_lr / 10
                elif epoch >= self.step2epoch and epoch < self.step3epoch:
                    curr_lr = self.max_lr / 100
                elif epoch >= self.step3epoch:
                    curr_lr = self.max_lr / 1000
                else:
                    curr_lr = self.max_lr
                    
                if curr_lr < self.min_lr:
                    curr_lr = self.min_lr
                
        return max(0.0, curr_lr)

    def __repr__(self) -> str:
        repr_str = '{}('.format(self.__class__.__name__)
        repr_str += '\n \t min_lr={}\n \t max_lr={}\n \t period={}'.format(self.min_lr, self.max_lr, self.period)
        if self.warmup_iterations > 0:
            repr_str += '\n \t warmup_init_lr={}\n \t warmup_iters={}'.format(self.warmup_init_lr, self.warmup_iterations)

        repr_str += '\n )'
        return repr_str
