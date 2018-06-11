import torch
from torch.optim.optimizer import Optimizer, required
# from .optimizer import Optimizer, required


class DropoutSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params_tensors, connections, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.params_tensors = params_tensors
        self.connections = connections

        # self.param_group_dicts = list({key: {'params': (param)} for (key, param) in dic.items()} for dic in self.param_dicts)

        params = list({'params': (param_tensor)} for param_tensor in params_tensors)

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(DropoutSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DropoutSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        raise NotImplementedError()
        # loss = None
        # if closure is not None:
        #     loss = closure()
        #
        #
        # steps_dicts = []
        # for dic in self.param_group_dicts:
        #     step_dict = {}
        #     steps_dicts.append(step_dict)
        #     for key, group in dic.items():
        #
        #         # for group in self.param_groups:
        #         weight_decay = group['weight_decay']
        #         momentum = group['momentum']
        #         dampening = group['dampening']
        #         nesterov = group['nesterov']
        #
        #         for p in group['params']:
        #             if p.grad is None:
        #                 continue
        #             d_p = p.grad.data
        #             if weight_decay != 0:
        #                 d_p.add_(weight_decay, p.data)
        #             if momentum != 0:
        #                 param_state = self.state[p]
        #                 if 'momentum_buffer' not in param_state:
        #                     buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
        #                     buf.mul_(momentum).add_(d_p)
        #                 else:
        #                     buf = param_state['momentum_buffer']
        #                     buf.mul_(momentum).add_(1 - dampening, d_p)
        #                 if nesterov:
        #                     d_p = d_p.add(momentum, buf)
        #                 else:
        #                     d_p = buf
        #
        #             step_dict[key] = (-group['lr']*d_p).data.item()
        #             # p.data.add_(-group['lr'], d_p)
        #
        # return loss

    def step_grad(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()


        steps_dicts = []
        for group in self.param_groups:

            # for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                step_dict = {}
                steps_dicts.append(step_dict)
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                gradient = (-group['lr']*d_p).data.tolist()

                for i, key in enumerate(self.connections):
                    if gradient[i] != 0:
                        step_dict[key] = gradient[i]
                # p.data.add_(-group['lr'], d_p)

        return steps_dicts

    # def full_reset_grad(self):
    #     r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
    #     for dic in self.param_group_dicts:
    #         for key, group in dic.items():
    #             for p in group['params']:
    #                 p.grad = None

    def full_reset_grad(self):
        # self.zero_grad()
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad = None
