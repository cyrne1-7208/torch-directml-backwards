#include "module.h"

#include <ATen/autocast_mode.h>
#include <ATen/core/grad_mode.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

namespace torch_dml
{
    // Use the PyTorch DLL constructor and existing schemas to avoid adding schema-inference TLS to this extension.
    __declspec(dllimport) void
    ConstructCpp(torch::CppFunction*, c10::KernelFunction, std::optional<c10::impl::CppSignature>, std::unique_ptr<c10::FunctionSchema>) asm(
        "??0CppFunction@torch@@AEAA@VKernelFunction@c10@@V?$optional@VCppSignature@impl@c10@@@std@@V?$unique_ptr@UFunctionSchema@c10@@U?$default_delete@UFunctionSchema@c10@@@std@@@5@@Z");
} // namespace torch_dml

namespace
{
    class KernelRegistration
    {
    public:
        explicit KernelRegistration(torch::Library& library) : m_library(library)
        {
        }

        template <typename Name, typename Function>
        void impl(Name name, Function* function)
        {
            alignas(torch::CppFunction) std::byte storage[sizeof(torch::CppFunction)];
            auto* current = reinterpret_cast<torch::CppFunction*>(storage);
            torch_dml::ConstructCpp(current,
                                    c10::KernelFunction::makeFromUnboxedRuntimeFunction(function),
                                    c10::impl::CppSignature::make<Function>(),
                                    nullptr);
            std::unique_ptr<torch::CppFunction, void (*)(torch::CppFunction*)> cleanup(
                current, [](auto* value) { value->~CppFunction(); });
            m_library.impl(name, std::move(*current));
        }

    private:
        torch::Library& m_library;
    };

    void CheckScalar(const at::Tensor& value, at::ScalarType dtype, const char* name)
    {
        TORCH_CHECK(value.device().type() == at::kPrivateUse1 && value.scalar_type() == dtype && value.numel() == 1,
                    name,
                    " must be a one-element DirectML tensor of dtype ",
                    dtype);
    }

    at::Tensor Reciprocal(const at::Tensor& value)
    {
        if (value.scalar_type() != at::kDouble)
        {
            // Preserve the existing allocation and type promotion rules for other dtypes.
            return at::_ops::reciprocal::redispatch(c10::DispatchKeySet(), value);
        }
        // GradScaler computes the scale reciprocal in FP64, which DirectML 1.15.4 does not support for this operator.
        // TORCH_WARN_ONCE adds static-initialization TLS in this build, so use the ordinary warning.
        TORCH_WARN("DirectML reciprocal does not support float64; using CPU for this dtype. "
                   "GradScaler uses it for its one-element scale tensor.");
        return at::reciprocal(value.cpu()).to(value.device());
    }

    void Unscale(at::TensorList gradients, at::Tensor& foundInf, const at::Tensor& invScale)
    {
        CheckScalar(foundInf, at::kFloat, "found_inf");
        CheckScalar(invScale, at::kFloat, "inv_scale");
        TORCH_CHECK(foundInf.device() == invScale.device(), "AMP tensors must share a device");
        at::NoGradGuard noGrad;
        for (const auto& gradient : gradients)
        {
            TORCH_CHECK(gradient.device() == foundInf.device() && gradient.layout() == at::kStrided &&
                            (gradient.scalar_type() == at::kFloat || gradient.scalar_type() == at::kHalf),
                        "AMP gradients must be dense FP16/FP32 tensors on the scale's device");
            // Check for non-finite gradients and unscale with existing DirectML operators, without CPU readback.
            if (gradient.numel() == 0)
                continue;
            auto invalid = at::logical_not(at::isfinite(gradient)).any();
            foundInf.copy_(at::where(invalid, at::ones_like(foundInf), foundInf));
            gradient.mul_(invScale);
        }
    }

    at::Tensor& UpdateScale(at::Tensor& scale,
                            at::Tensor& growthTracker,
                            const at::Tensor& foundInf,
                            double growthFactor,
                            double backoffFactor,
                            int64_t growthInterval)
    {
        CheckScalar(scale, at::kFloat, "scale");
        CheckScalar(growthTracker, at::kInt, "growth_tracker");
        CheckScalar(foundInf, at::kFloat, "found_inf");
        TORCH_CHECK(scale.device() == growthTracker.device() && scale.device() == foundInf.device(),
                    "AMP tensors must share a device");
        at::NoGradGuard noGrad;
        auto invalid = foundInf.ne(0);
        auto nextGrowthTracker = growthTracker + 1;
        auto shouldGrow = nextGrowthTracker.eq(growthInterval);
        auto grownScale = scale * growthFactor;
        // Follow PyTorch's growth, backoff and reset rules without overflowing the scale itself.
        auto finiteGrownScale = at::where(at::isfinite(grownScale), grownScale, scale);
        scale.copy_(at::where(invalid, scale * backoffFactor, at::where(shouldGrow, finiteGrownScale, scale)));
        growthTracker.copy_(
            at::where(at::logical_or(invalid, shouldGrow), at::zeros_like(growthTracker), nextGrowthTracker));
        return scale;
    }

    at::Tensor BinaryCrossEntropyBanned(const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&, int64_t)
    {
        TORCH_CHECK(false,
                    "binary_cross_entropy is unsafe under FP16 autocast; "
                    "use binary_cross_entropy_with_logits instead");
    }

    struct AmpRegistration
    {
        torch::Library fallback{torch::Library::IMPL, "_", c10::DispatchKey::AutocastPrivateUse1, __FILE__, __LINE__};
        torch::Library kernels{torch::Library::IMPL, "aten", c10::DispatchKey::AutocastPrivateUse1, __FILE__, __LINE__};
        torch::Library gradients{torch::Library::IMPL, "aten", c10::DispatchKey::PrivateUse1, __FILE__, __LINE__};

        AmpRegistration()
        {
            fallback.fallback(torch::CppFunction::makeFallthrough());
            // PyTorch's KERNEL macro requires the registration variable to be named m.
            KernelRegistration m(kernels);
            // Use PyTorch's FP16 policy for matrix products and convolutions; leave out= and in-place overloads unchanged.
            KERNEL(at::kPrivateUse1, mm, lower_precision_fp)
            KERNEL(at::kPrivateUse1, matmul, lower_precision_fp)
            KERNEL(at::kPrivateUse1, bmm, lower_precision_fp)
            KERNEL(at::kPrivateUse1, addmm, lower_precision_fp)
            KERNEL(at::kPrivateUse1, baddbmm, lower_precision_fp)
            KERNEL(at::kPrivateUse1, linear, lower_precision_fp)
            KERNEL(at::kPrivateUse1, conv1d, lower_precision_fp)
            KERNEL(at::kPrivateUse1, conv2d, lower_precision_fp)
            KERNEL(at::kPrivateUse1, conv3d, lower_precision_fp)
            KERNEL(at::kPrivateUse1, conv_transpose1d, lower_precision_fp)
            KERNEL(at::kPrivateUse1, conv_transpose2d, input, lower_precision_fp)
            KERNEL(at::kPrivateUse1, conv_transpose3d, input, lower_precision_fp)
            KERNEL(at::kPrivateUse1, convolution, lower_precision_fp)

            // Use FP32 for sensitive operations and preserve explicit dtypes through PyTorch's policies.
            KERNEL(at::kPrivateUse1, exp, fp32)
            KERNEL(at::kPrivateUse1, log, fp32)
            KERNEL(at::kPrivateUse1, log2, fp32)
            KERNEL(at::kPrivateUse1, log10, fp32)
            KERNEL(at::kPrivateUse1, rsqrt, fp32)
            KERNEL(at::kPrivateUse1, reciprocal, fp32)
            KERNEL(at::kPrivateUse1, pow, Tensor_Scalar, fp32)
            KERNEL(at::kPrivateUse1, pow, Tensor_Tensor, fp32)
            KERNEL(at::kPrivateUse1, pow, Scalar, fp32)
            KERNEL(at::kPrivateUse1, layer_norm, fp32)
            KERNEL(at::kPrivateUse1, native_layer_norm, fp32)
            KERNEL(at::kPrivateUse1, group_norm, fp32)
            KERNEL(at::kPrivateUse1, native_group_norm, fp32)
            KERNEL(at::kPrivateUse1, mse_loss, fp32)
            KERNEL(at::kPrivateUse1, l1_loss, fp32)
            KERNEL(at::kPrivateUse1, smooth_l1_loss, fp32)
            KERNEL(at::kPrivateUse1, cross_entropy_loss, fp32)
            KERNEL(at::kPrivateUse1, binary_cross_entropy_with_logits, fp32)
            KERNEL(at::kPrivateUse1, softmax, int, fp32_set_opt_dtype)
            KERNEL(at::kPrivateUse1, log_softmax, int, fp32_set_opt_dtype)
            KERNEL(at::kPrivateUse1, sum, fp32_set_opt_dtype)
            KERNEL(at::kPrivateUse1, sum, dim_IntList, fp32_set_opt_dtype)
            KERNEL(at::kPrivateUse1, mean, fp32_set_opt_dtype)
            KERNEL(at::kPrivateUse1, mean, dim, fp32_set_opt_dtype)
            KERNEL(at::kPrivateUse1, linalg_vector_norm, fp32_set_opt_dtype)
            m.impl("binary_cross_entropy", &BinaryCrossEntropyBanned);

            KernelRegistration gradientKernels(gradients);
            gradientKernels.impl("reciprocal", &Reciprocal);
            gradientKernels.impl("_amp_foreach_non_finite_check_and_unscale_", &Unscale);
            gradientKernels.impl("_amp_update_scale_", &UpdateScale);
        }
    };
} // namespace

void torch_dml::RegisterAmp()
{
    // Called once by the original DeviceGuard registration; retain the registrations for the backend's lifetime.
    new AmpRegistration;
}
