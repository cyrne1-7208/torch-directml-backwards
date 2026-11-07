#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>

#include <bit>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

namespace torch_dml
{
    extern "C" IMAGE_DOS_HEADER __ImageBase;

    void RegisterAmp();
    void* GetInitializedDevice(c10::DeviceIndex device);
    bool QueryDevice(c10::DeviceIndex device);
    void SynchronizeDevice(c10::DeviceIndex device);

    at::Tensor CdistForward(const at::Tensor& x1,
                            const at::Tensor& x2,
                            double p,
                            std::optional<int64_t> compute_mode);
    at::Tensor& CdistForwardOut(const at::Tensor& x1,
                               const at::Tensor& x2,
                               double p,
                               std::optional<int64_t> compute_mode,
                               at::Tensor& out);

    std::tuple<at::Tensor, at::Tensor, at::Tensor> Unique2(const at::Tensor& self,
                                                          bool sorted,
                                                          bool return_inverse,
                                                          bool return_counts);
    std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> Unique2Out(const at::Tensor& self,
                                                                bool sorted,
                                                                bool return_inverse,
                                                                bool return_counts,
                                                                at::Tensor& out0,
                                                                at::Tensor& out1,
                                                                at::Tensor& out2);

    at::Tensor PreluKernel(const at::Tensor& self, const at::Tensor& weight);
    std::tuple<at::Tensor, at::Tensor> PreluKernelBackward(const at::Tensor& grad_output,
                                                          const at::Tensor& self,
                                                          const at::Tensor& weight);

    inline uintptr_t ModuleBase()
    {
        // The added code resides in the original extension, so no filename-based module lookup is needed.
        return reinterpret_cast<uintptr_t>(&__ImageBase);
    }

    struct __single_inheritance MethodOwner
    {
    };

    // Use a member-function pointer to preserve MSVC's member-function calling convention for structure returns.
    template <uint32_t Rva, typename Result, typename... Args>
    Result InvokeMethod(const void* self, Args... args)
    {
        using Method = Result (MethodOwner::*)(Args...) const;
        static_assert(sizeof(Method) == sizeof(uintptr_t));
        auto method = std::bit_cast<Method>(ModuleBase() + Rva);
        return (static_cast<const MethodOwner*>(self)->*method)(std::forward<Args>(args)...);
    }
} // namespace torch_dml
