import time

import torch
import copy


# class _RequiredParameter(object):
#     """Singleton class representing a required parameter for an Optimizer."""
#
#     def __repr__(self):
#         return "<required parameter>"
#
#
# required = _RequiredParameter()
from time_meta import record_data


class EmaAmsGrad(torch.optim.Adam):
    def __init__(self, params, shadow_model, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, ema=0.999):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad=True)
        self.ema = ema
        self.shadow_model = shadow_model
        tmp_optim = torch.optim.Adam(shadow_model.parameters())
        self.shadow_param_groups = tmp_optim.param_groups
        self.deactivated = (ema < 0)

    def step(self, closure=None):
        # t0 = time.time()

        loss = super().step(closure)

        # t0 = record_data('AMS grad', t0)
        if not self.deactivated:
            for shadow, m_params in zip(self.shadow_param_groups, self.param_groups):
                for p_shadow, p_model in zip(shadow['params'], m_params['params']):
                    p_shadow.data.add_((1 - self.ema) * (p_model.data - p_shadow.data))

        # t0 = record_data('shadow update', t0)
        return loss


class MySGD(torch.optim.SGD):
    """
    my wrap of SGD for compatibility issues
    """
    def __init__(self, model, *args, **kwargs):
        self.shadow_model = model
        super(MySGD, self).__init__(model.parameters(), *args, **kwargs)

    def step(self, closure=None):
        return super(MySGD, self).step(closure)
