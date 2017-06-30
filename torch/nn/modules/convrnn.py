import math
import torch
from .module import Module
from .utils import _pair


class ConvRNNBase(Module):

    def __init__(self, mode, in_channels, hidden_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, batch_first=False, dropout=0):
        super(ConvRNNBase, self).__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout

        self._all_weights = []
        if mode == 'LSTM':
            gate_size = 4*hidden_channels
        elif mode == 'GRU':
            gate_size = 3*hidden_channels
        else:
            gate_size = hidden_channels

        w_ih = Parameter(torch.Tensor(gate_size, in_channels, *kernel_size))
        w_hh = Parameter(torch.Tensor(gate_size, hidden_channels, *kernel_size))
        b_ih = Parameter(torch.Tensor(gate_size))
        b_hh = Parameter(torch.Tensor(gate_size))

        weights = ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']
        setattr(self, weights[0], w_ih)
        setattr(self, weights[1], w_hh)
        if bias:
            setattr(self, weights[2], b_ih)
            setattr(self, weights[3], b_hh)
            self._all_weights += [weights]
        else:
            self._all_weights += [weights[:2]]

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {hidden_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
        

class ConvRNNCellBase(Module):

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class ConvLSTM2dCell(ConvRNNCellBase):

        r"""A long short-term memory (LSTM) cell.

    .. math::

        \begin{array}{ll}
        i = \mathrm{sigmoid}(W_{ii} \star x + b_{ii} + W_{hi} \star h + b_{hi}) \\
        f = \mathrm{sigmoid}(W_{if} \star x + b_{if} + W_{hf} \star h + b_{hf}) \\
        g = \tanh(W_{ig} \star x + b_{ig} + W_{hc} h + b_{hg}) \\
        o = \mathrm{sigmoid}(W_{io} \star x + b_{io} + W_{ho} \star h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c_t) \\
        \end{array}

        where \star denotes the convolution operator

        Args:
        input_channels (int): Number of channels in the input  
        hidden_channels (int): Number of channels in the hidden state
        in_kernel_size (int or tuple): Size of the convolving kernel for the input, must be odd
        hid_kernel_size (int or tuple): Size of the convolving kernel for the hidden state, must be odd
        in_stride (int or tuple, optional): Stride of the input convolution, Default: 1
        hid_stride (int or tuple, optional): Stride of the hidden convolution, Default: 1
        in_dilation (int or tuple, optional): Spacing between input convolving kernel elements, Default: 1
        hid_dilation (int or tuple, optional): Spacing between hidden convolving kernal elements, Default: 1
        bias (bool, optional): If `False`, then the layer does not use bias weights `b_ih` and `b_hh`. Default: True

    Inputs: input, (h_0, c_0)
        - **input** (batch, in_channels, C_in, H_in): tensor containing input features
        - **h_0** (batch, hidden_channels, C_in, H_in): tensor containing the initial hidden state for each element in the batch.
        - **c_0** (batch, hidden_channels, C_in, H_in): tensor containing the initial cell state for each element in the batch.

    Outputs: h_1, c_1
        - **h_1** (batch, hidden_channels, C_in, H_in): tensor containing the next hidden state for each element in the batch
        - **c_1** (batch, hidden_channels, C_in, H_in): tensor containing the next cell state for each element in the batch

    Attributes:
        weight_ih (Tensor): the learnable input-hidden weights, of shape (hidden_channels, in_channels, kernel_size[0], kernel_size[1])
        weight_hh: the learnable hidden-hidden weights, of shape  (hidden_channels, in_channels, kernel_size[0], kernel_size[1])
        bias_ih: the learnable input-hidden bias, of shape `(hidden_channels)`
        bias_hh: the learnable hidden-hidden bias, of shape `(hidden_channels)`

    Examples::

        >>> rnn = nn.ConvLSTM2dCell(10, 20, 3)
        >>> input = Variable(torch.randn(6, 3, 10, 12, 12))
        >>> hx = Variable(torch.randn(3, 20, 12, 12))
        >>> cx = Variable(torch.randn(3, 20, 12, 12))
        >>> output = []
        >>> for i in range(6):
        ...     hx, cx = rnn(input[i], (hx, cx))
        ...     output.append(hx)
    """

    def __init__(self, in_channels, hidden_channels, in_kernel_size, hid_kernel_size, in_stride=1, hid_stride=1, in_dilation=1, hid_dilation=1, bias=True):
        super(ConvLSTM2dCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.in_kernel_size = _pair(in_kernel_size)
        
        if isinstance(in_kernel_size, int):
            self.in_padding = ((in_kernel_size-1)/2, (in_kernel_size-1)/2)
        else:
            self.in_padding = ((in_kernel_size[0]-1)/2, (in_kernel_size[1]-1)/2)
            
        if isinstance(hid_kernel_size, int):
            self.hid_padding = ((hid_kernel_size-1)/2, (hid_kernel_size-1)/2)
        else:
            self.hid_padding = ((hid_kernel_size[0]-1)/2, (hid_kernel_size[1]-1)/2)
            
        self.hid_kernel_size = _pair(hid_kernel_size)
        self.in_stride = _pair(in_stride)
        self.hid_stride = _pair(hid_stride)
        self.in_dilation = _pair(in_dilation)
        self.hid_dilation = _pair(hid_dilation)
        self.bias = bias

        self.weight_ih = Parameter(torch.Tensor(4 * hidden_channels, in_channels, *in_kernel_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_channels, hidden_channels, *hid_kernel_size))

        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_channels))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_channels))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. /math.sqrt(n)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        return self._backend.ConvLSTM2dCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
            self.in_stride, self.hid_stride,
            self.in_padding, self.hid_padding,
            self.in_dilation, self.hid_dilation
        )
        
        


