#include "module.h"

#include <torch/library.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <string_view>

namespace torch_dml
{
    using Boxed = c10::BoxedKernel;
    using Kernel = c10::KernelFunction;
    using Signature = std::optional<c10::impl::CppSignature>;
    using Schema = std::unique_ptr<c10::FunctionSchema>;

    template <typename T>
    struct CurrentArgument
    {
        using type = T;
    };
    template <>
    struct CurrentArgument<c10::basic_string_view<char>>
    {
        using type = std::string_view;
    };
    template <>
    struct CurrentArgument<std::optional<c10::basic_string_view<char>>>
    {
        using type = std::optional<std::string_view>;
    };

    template <typename... Args>
    auto StringSignature()
    {
        using S = c10::impl::CppSignature;
        return std::pair{S::make<at::Tensor&(Args...)>(),
                         S::make<at::Tensor&(typename CurrentArgument<Args>::type...)>()};
    }

    Signature CurrentSignature(Signature signature)
    {
        // Convert only the five signatures in the original DLL; let PyTorch report other mismatches normally.
        using T = const at::Tensor&;
        using O = at::Tensor&;
        using V = c10::basic_string_view<char>;
        const std::array pairs{StringSignature<T, V, O>(),
                               StringSignature<T, T, V, O>(),
                               StringSignature<T, T, std::optional<V>, O>(),
                               StringSignature<T, int64_t, T, T, V, O>(),
                               StringSignature<T, int64_t, T, const at::Scalar&, V, O>()};
        if (signature)
        {
            for (const auto& [old, current] : pairs)
            {
                if (*signature == old)
                    return current;
            }
        }
        return signature;
    }

    // Both string_view types use pointer/length order under MSVC, preserving the by-value ABI.
    static_assert(sizeof(c10::basic_string_view<char>) == 16);
    static_assert(sizeof(std::optional<c10::basic_string_view<char>>) == sizeof(std::optional<std::string_view>));

    // The legacy allocation has no room for tokens_; do not construct a current KernelFunction in it.
    struct LegacyKernel
    {
        Boxed boxed;
        void* unboxed;
        void* symUnboxed;
    };

    struct LegacyCppFunction
    {
        std::optional<c10::DispatchKey> dispatch;
        LegacyKernel kernel;
        Signature signature;
        Schema schema;
        std::string debug;
    };

    static_assert(sizeof(Boxed) == 16 && sizeof(Kernel) == 40);
    static_assert(sizeof(LegacyKernel) == 32 && sizeof(LegacyCppFunction) == 96);
    static_assert(sizeof(torch::CppFunction) == 104 && sizeof(torch::Library) == 120);
    static_assert(std::is_trivially_copyable_v<std::optional<c10::DispatchKey>>);

    // Access specifiers do not affect the calling convention; use the constructors exported by the PyTorch DLLs.
    __declspec(dllimport) void
    ConstructBoxed(Boxed*, std::unique_ptr<c10::OperatorKernel>, Boxed::InternalBoxedKernelFunction*) asm(
        "??0BoxedKernel@c10@@AEAA@V?$unique_ptr@UOperatorKernel@c10@@U?$default_delete@UOperatorKernel@c10@@@std@@@std@@P6AXPEAUOperatorKernel@1@AEBVOperatorHandle@1@VDispatchKeySet@1@PEAV?$vector@UIValue@c10@@V?$allocator@UIValue@c10@@@std@@@3@@Z@Z");
    __declspec(dllimport) void ConstructKernel(Kernel*,
                                               Boxed,
                                               void*,
                                               void*) asm("??0KernelFunction@c10@@AEAA@VBoxedKernel@1@PEAX1@Z");
    __declspec(dllimport) void ConstructCpp(torch::CppFunction*, Kernel, Signature, Schema) asm(
        "??0CppFunction@torch@@AEAA@VKernelFunction@c10@@V?$optional@VCppSignature@impl@c10@@@std@@V?$unique_ptr@UFunctionSchema@c10@@U?$default_delete@UFunctionSchema@c10@@@std@@@5@@Z");
    __declspec(dllimport) torch::Library*
    RegisterImpl(torch::Library*, const char*, torch::CppFunction&&, torch::_RegisterOrVerify) asm(
        "?_impl@Library@torch@@AEGAAAEAV12@PEBD$$QEAVCppFunction@2@W4_RegisterOrVerify@2@@Z");
    __declspec(dllimport) torch::Library* RegisterFallback(torch::Library*, torch::CppFunction&&) asm(
        "?_fallback@Library@torch@@AEGAAAEAV12@$$QEAVCppFunction@2@@Z");

    // Destroy objects on exception only after their DLL constructor has completed.
    template <typename T>
    class NativeObject
    {
    public:
        template <typename Constructor, typename... Args>
        explicit NativeObject(Constructor constructor, Args&&... args)
        {
            constructor(reinterpret_cast<T*>(m_storage), std::forward<Args>(args)...);
        }
        NativeObject(const NativeObject&) = delete;
        NativeObject& operator=(const NativeObject&) = delete;
        ~NativeObject()
        {
            Get()->~T();
        }
        T* Get()
        {
            return std::launder(reinterpret_cast<T*>(m_storage));
        }

    private:
        alignas(T) std::byte m_storage[sizeof(T)];
    };

    Kernel CurrentKernel(LegacyKernel& old)
    {
        NativeObject<Kernel> kernel(ConstructKernel, std::move(old.boxed), old.unboxed, old.symUnboxed);
        return std::move(*kernel.Get());
    }

    class CurrentFunction
    {
    public:
        explicit CurrentFunction(LegacyCppFunction& old)
            : m_function(ConstructCpp,
                         CurrentKernel(old.kernel),
                         CurrentSignature(std::move(old.signature)),
                         std::move(old.schema))
        {
            // dispatch_ is the same leading optional<DispatchKey> in both versions; current APIs manage the remaining fields.
            std::memcpy(static_cast<void*>(Get()), &old.dispatch, sizeof(old.dispatch));
            std::move(*Get()).debug(std::move(old.debug));
        }
        torch::CppFunction* Get()
        {
            return m_function.Get();
        }

    private:
        NativeObject<torch::CppFunction> m_function;
    };

    template <typename Function>
    torch::Library* RegisterOp(torch::Library* self,
                               const char* name,
                               Function* function,
                               torch::_RegisterOrVerify mode)
    {
        NativeObject<torch::CppFunction> current(
            ConstructCpp,
            c10::KernelFunction::makeFromUnboxedRuntimeFunction(function),
            c10::impl::CppSignature::make<Function>(),
            nullptr);
        return RegisterImpl(self, name, std::move(*current.Get()), mode);
    }
} // namespace torch_dml

using namespace torch_dml;

extern "C" __declspec(dllexport) LegacyKernel* tdml_kernel_construct(LegacyKernel* self,
                                                                     std::unique_ptr<c10::OperatorKernel> functor,
                                                                     Boxed::InternalBoxedKernelFunction* boxed,
                                                                     void* unboxed,
                                                                     void* symUnboxed)
{
    NativeObject<Boxed> current(ConstructBoxed, std::move(functor), boxed);
    return new (self) LegacyKernel{std::move(*current.Get()), unboxed, symUnboxed};
}

extern "C" __declspec(dllexport) void tdml_kernel_destroy(LegacyKernel* self)
{
    self->~LegacyKernel();
}

extern "C" __declspec(dllexport) LegacyCppFunction* tdml_cpp_move(LegacyCppFunction* self, LegacyCppFunction&& other)
{
    return new (self) LegacyCppFunction(std::move(other));
}

extern "C" __declspec(dllexport) void tdml_cpp_destroy(LegacyCppFunction* self)
{
    self->~LegacyCppFunction();
}

// Construct the structure return value in the hidden output buffer passed by the MSVC calling convention.
extern "C" __declspec(dllexport) LegacyCppFunction* tdml_cpp_boxed(LegacyCppFunction* output, Boxed boxed)
{
    return new (output)
        LegacyCppFunction{std::nullopt, {std::move(boxed), nullptr, nullptr}, std::nullopt, nullptr, {}};
}

extern "C" __declspec(dllexport) torch::Library* tdml_library_impl(torch::Library* self,
                                                                   const char* name,
                                                                   LegacyCppFunction&& function,
                                                                   torch::_RegisterOrVerify mode)
{
    // Intercept legacy operators with structural defects or incomplete dimension handling.
    std::string_view opName(name);
    if (opName.starts_with("aten::"))
        opName.remove_prefix(6);
    if (opName == "_cdist_forward")
        return RegisterOp(self, name, &CdistForward, mode);
    if (opName == "_cdist_forward.out")
        return RegisterOp(self, name, &CdistForwardOut, mode);
    if (opName == "_unique2")
        return RegisterOp(self, name, &Unique2, mode);
    if (opName == "_unique2.out")
        return RegisterOp(self, name, &Unique2Out, mode);
    if (opName == "_prelu_kernel")
        return RegisterOp(self, name, &PreluKernel, mode);
    if (opName == "_prelu_kernel_backward")
        return RegisterOp(self, name, &PreluKernelBackward, mode);

    CurrentFunction current(function);
    return RegisterImpl(self, name, std::move(*current.Get()), mode);
}

extern "C" __declspec(dllexport) torch::Library* tdml_library_fallback(torch::Library* self,
                                                                       LegacyCppFunction&& function)
{
    CurrentFunction current(function);
    return RegisterFallback(self, std::move(*current.Get()));
}
