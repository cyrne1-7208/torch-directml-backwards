#include "module.h"

#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <new>

at::Tensor torch_dml::CdistForward(const at::Tensor& x1,
                                  const at::Tensor& x2,
                                  double p,
                                  std::optional<int64_t> compute_mode)
{
    const auto batch1 = x1.sizes().slice(0, x1.dim() - 2);
    const auto batch2 = x2.sizes().slice(0, x2.dim() - 2);
    const auto batch = at::infer_size(batch1, batch2);

    auto outShape = batch;
    outShape.push_back(x1.size(-2));
    outShape.push_back(x2.size(-2));
    if (x1.numel() == 0 || x2.numel() == 0)
        return at::zeros(outShape, x1.options());

    auto shape1 = batch;
    shape1.push_back(x1.size(-2));
    shape1.push_back(x1.size(-1));
    auto shape2 = batch;
    shape2.push_back(x2.size(-2));
    shape2.push_back(x2.size(-1));

    // The legacy operator mutates its inputs via unsqueeze_; pass 3D copies to protect caller views.
    int64_t count = 1;
    for (auto dim : batch)
        count *= dim;
    auto a1 = x1.expand(shape1).reshape({count, x1.size(-2), x1.size(-1)}).clone();
    auto a2 = x2.expand(shape2).reshape({count, x2.size(-2), x2.size(-1)}).clone();

    using NativeFn = at::Tensor (*)(const at::Tensor&, const at::Tensor&, double, const std::optional<int64_t>&);
    auto nativeCdist = reinterpret_cast<NativeFn>(torch_dml::ModuleBase() + 0x911c0);
    return nativeCdist(a1, a2, p, compute_mode).reshape(outShape);
}

at::Tensor& torch_dml::CdistForwardOut(const at::Tensor& x1,
                                     const at::Tensor& x2,
                                     double p,
                                     std::optional<int64_t> compute_mode,
                                     at::Tensor& out)
{
    auto result = CdistForward(x1, x2, p, compute_mode);
    out.resize_(result.sizes());
    return out.copy_(result);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> torch_dml::Unique2(const at::Tensor& self,
                                                                  bool sorted,
                                                                  bool return_inverse,
                                                                  bool return_counts)
{
    if (return_inverse || return_counts || self.scalar_type() == at::kDouble || self.scalar_type() == at::kBool)
    {
        // DirectML 1.15.4 does not compute inverse/counts on GPU; fall back to CPU for unsupported paths.
        TORCH_WARN("DirectML unique fallback to CPU: return_inverse, return_counts or unsupported dtype");
        auto [out, inv, counts] = at::_unique2(self.cpu(), sorted, return_inverse, return_counts);
        return {out.to(self.device()), inv.to(self.device()), counts.to(self.device())};
    }
    if (self.numel() <= 1)
    {
        auto out = self.numel() == 0 ? at::empty({0}, self.options()) : self.reshape({1}).clone();
        auto empty = at::empty({0}, self.options().dtype(at::kLong));
        return {out, empty, empty};
    }
    // The legacy operator expects a contiguous 1D input tensor.
    auto flat = self.reshape({-1});
    using NativeFn = std::tuple<at::Tensor, at::Tensor, at::Tensor> (*)(const at::Tensor&, bool, bool, bool);
    return reinterpret_cast<NativeFn>(torch_dml::ModuleBase() + 0x2d860)(flat, sorted, false, false);
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> torch_dml::Unique2Out(const at::Tensor& self,
                                                                        bool sorted,
                                                                        bool return_inverse,
                                                                        bool return_counts,
                                                                        at::Tensor& out0,
                                                                        at::Tensor& out1,
                                                                        at::Tensor& out2)
{
    auto [o0, o1, o2] = Unique2(self, sorted, return_inverse, return_counts);
    out0.resize_(o0.sizes());
    out0.copy_(o0);
    out1.resize_(o1.sizes());
    out1.copy_(o1);
    out2.resize_(o2.sizes());
    out2.copy_(o2);
    return {out0, out1, out2};
}

at::Tensor torch_dml::PreluKernel(const at::Tensor& self, const at::Tensor& weight)
{
    auto w = weight;
    if (w.dim() == 1 && w.numel() > 1 && self.dim() > 1)
    {
        std::vector<int64_t> shape(self.dim(), 1);
        shape[1] = w.numel();
        w = w.reshape(shape);
    }
    auto pos = self.gt(0);
    return at::where(pos, self, self.mul(w));
}

std::tuple<at::Tensor, at::Tensor> torch_dml::PreluKernelBackward(const at::Tensor& grad_output,
                                                                  const at::Tensor& self,
                                                                  const at::Tensor& weight)
{
    auto w = weight;
    if (w.dim() == 1 && w.numel() > 1 && self.dim() > 1)
    {
        std::vector<int64_t> shape(self.dim(), 1);
        shape[1] = w.numel();
        w = w.reshape(shape);
    }
    auto pos = self.gt(0);
    auto gradWeight = at::where(pos, at::zeros_like(grad_output), grad_output.mul(self));
    return {at::where(pos, grad_output, grad_output.mul(w)), at::sum_to(gradWeight, weight.sizes())};
}

// Legacy NMS accessed dispatcher arrays directly; replace that access with the current typed API.
extern "C" __declspec(dllexport) at::Tensor* tdml_nms(at::Tensor* output,
                                                      const at::Tensor& boxes,
                                                      const at::Tensor& scores,
                                                      double threshold)
{
    auto& dispatcher = c10::Dispatcher::realSingleton();
    const auto operation = dispatcher.findSchemaOrThrow("torchvision::nms", "")
                               .typed<at::Tensor(const at::Tensor&, const at::Tensor&, double)>();
    auto cpuScores = scores.to(c10::Device(c10::DeviceType::CPU));
    auto cpuBoxes = boxes.to(c10::Device(c10::DeviceType::CPU));
    auto result = dispatcher.call<at::Tensor, const at::Tensor&, const at::Tensor&, double>(
        operation, cpuBoxes, cpuScores, threshold);
    return new (output) at::Tensor(result.to(boxes.device()));
}
