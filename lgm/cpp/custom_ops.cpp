/*
###############################################################################
# This file is part of LGM.
#
# Copyright 2018 Yuesong Shen
# Copyright 2018,2019 Technical University of Munich
#
# Developed by Yuesong Shen <yuesong dot shen at tum dot de>.
#
# If you use this file for your research, please cite the following paper:
#
# "Probabilistic Discriminative Learning with Layered Graphical Models" by
# Yuesong Shen, Tao Wu, Csaba Domokos and Daniel Cremers
#
# LGM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# LGM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with LGM. If not, see <http://www.gnu.org/licenses/>.
###############################################################################
*/
#include <torch/extension.h>
#include <vector>
#include <tuple>


double get_eps(at::Tensor t){
    if (t.dtype() == at::kFloat) {
        return 1.1754944e-38;
    }
    else if (t.dtype() == at::kDouble){
        return 2.2250738585072014e-308;
    }
    else if (t.dtype() == at::kHalf){
        return 6.1035e-05;
    }
    else{
        AT_ERROR("not half, float or double!");
    }
}


at::Tensor logep_forward(at::Tensor t){
    return at::log(t + t.type().scalarTensor(get_eps(t)));
}


at::Tensor logep_backward(at::Tensor grad, at::Tensor t){
    return grad / (t + t.type().scalarTensor(get_eps(t)));
}


at::Tensor unzip0_forward(at::Tensor t, int64_t dim){
    auto sz = t.sizes().vec();
    dim = dim < 0 ? sz.size()+dim : dim;
    sz[dim] = 1;
    return at::cat({t, at::zeros(sz, t.type())}, dim);
}


at::Tensor unzip0_backward(at::Tensor grad, int64_t dim){
    return at::slice(grad, dim, 0, at::size(grad, dim) - 1, 1);
}


at::Tensor softmax0fo_forward(at::Tensor t, int64_t dim){
    return at::softmax(unzip0_forward(t, dim), dim);
}


at::Tensor softmax0fo_backward(at::Tensor grad, at::Tensor result, at::Tensor t, int64_t dim){
    return unzip0_backward(at::_softmax_backward_data(grad, result, dim, t), dim);
}


at::Tensor logsoftmax0fo_forward(at::Tensor t, int64_t dim){
    return at::log_softmax(unzip0_forward(t, dim), dim);
}


at::Tensor logsoftmax0fo_backward(at::Tensor grad, at::Tensor result, at::Tensor t, int64_t dim){
    return unzip0_backward(at::_log_softmax_backward_data(grad, result, dim, t), dim);
}


at::Tensor logsumexp0_forward(at::Tensor t, int64_t dim, bool keepdim){
    auto tmax = at::clamp_min(at::max_values(t, dim, true), 0);
    auto result = at::where(tmax == INFINITY,
		            tmax,
		            tmax + at::log(at::sum(at::exp(t - tmax), dim, true) + at::exp(- tmax)));
    if (! keepdim)
        result.squeeze_(dim);
    return result;
}


at::Tensor logsumexp0_backward(at::Tensor grad, at::Tensor t, at::Tensor result, int64_t dim, bool keepdim){
    if (! keepdim) {
        grad = grad.unsqueeze(dim);
        result = result.unsqueeze(dim);
    }
    return grad * at::exp(t - result);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("logep_forward", &logep_forward, "log epsilon plus forward");
    m.def("logep_backward", &logep_backward, "log epsilon plus backward");
    m.def("unzip0_forward", &unzip0_forward, "unzip 0 forward");
    m.def("unzip0_backward", &unzip0_backward, "unzip 0 backward");
    m.def("softmax0fo_forward", &softmax0fo_forward, "softmax with 0 full output forward");
    m.def("softmax0fo_backward", &softmax0fo_backward, "softmax with 0 full output backward");
    m.def("logsoftmax0fo_forward", &logsoftmax0fo_forward, "logsoftmax with 0 full output forward");
    m.def("logsoftmax0fo_backward", &logsoftmax0fo_backward, "logsoftmax with 0 full output backward");
    m.def("logsumexp0_forward", &logsumexp0_forward, "logsumexp with 0 forward");
    m.def("logsumexp0_backward", &logsumexp0_backward, "logsumexp with 0 backward");
}
