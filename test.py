import torch
import torch.nn as nn
import numpy as np
import taichi as ti

# init taichi
arch = ti.cuda
ti.init(arch=arch)


@ti.kernel
def test_grad(
        x: ti.types.ndarray(),
        y: ti.types.ndarray(),
):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y[i, j] = 3 * x[i, j]


class GradTest(nn.Module):
    def __init__(self):
        super(GradTest, self).__init__()

        self._test_grad = test_grad

        class _module_function(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                y = torch.zeros_like(
                    x,
                    requires_grad=True
                )

                self._test_grad(x, y)
                ctx.save_for_backward(x, y)

                return y

            @staticmethod
            def backward(ctx, dy):
                x, y = ctx.saved_tensors
                y.grad = dy

                self._test_grad.grad(x, y)

                return x.grad

        self._module_function = _module_function.apply

    def forward(self, x):
        return self._module_function(x.contiguous())


if __name__ == "__main__":
    x = torch.ones((20, 30), dtype=torch.float32, requires_grad=True).cuda()

    grad_test = GradTest()
    y = grad_test(x)

    L1_loss = nn.L1Loss()
    loss = L1_loss(x, y)
    print('loss:', loss)

    loss.backward()  # error: Exporting data to external array (such as numpy array) not supported in AutoDiff for now
    print('grad: ', x.grad[0])