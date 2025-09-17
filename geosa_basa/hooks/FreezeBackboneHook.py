from mmseg.registry import HOOKS
from mmengine.hooks import Hook
from typing import Dict, Optional, Sequence, Union

DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class FreezeBackboneHook(Hook):
    def __init__(self, freeze_iters=20000):
        self.freeze_iters = freeze_iters

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:
        if runner.iter < self.freeze_iters:
            # Freeze backbone at the start of training
            for param in runner.model.backbone.parameters():
                param.requires_grad = False

            if runner.iter==0:
                runner.logger.info('Backbone frozen.')
                # Log the requires_grad state of each parameter
                for name, param in runner.model.named_parameters():
                    runner.logger.info(f'Parameter: {name}, requires_grad: {param.requires_grad}')

        elif runner.iter >= self.freeze_iters:
            # Unfreeze backbone after freeze_iters
            for param in runner.model.backbone.parameters():
                param.requires_grad = True

            if runner.iter==self.freeze_iters:
                runner.logger.info('Backbone unfrozen.')
                # Log the requires_grad state of each parameter
                for name, param in runner.model.named_parameters():
                    runner.logger.info(f'Parameter: {name}, requires_grad: {param.requires_grad}')
