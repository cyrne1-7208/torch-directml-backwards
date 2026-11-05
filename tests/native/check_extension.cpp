#include <ATen/Functions.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/TensorImpl.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <new>
#include <string>
#include <utility>

using Nms = at::Tensor* (*)(at::Tensor*, const at::Tensor&, const at::Tensor&, double);

at::Tensor CallNms(Nms function, const at::Tensor& boxes, const at::Tensor& scores)
{
    alignas(at::Tensor) std::byte storage[sizeof(at::Tensor)];
    auto* output = function(reinterpret_cast<at::Tensor*>(storage), boxes, scores, 0.5);
    at::Tensor result = std::move(*output);
    output->~Tensor();
    return result;
}

extern "C" __declspec(dllexport) int tdml_check_native(void* module, unsigned nmsRva, unsigned tensorRva)
{
    try
    {
        auto* image = static_cast<char*>(module);
        TORCH_CHECK(image, "Native extension is not loaded");
        using Fallback = void (*)(const c10::OperatorHandle&, c10::Stack*, bool);
        auto fallback = reinterpret_cast<Fallback>(image + 0x158710);
        auto operation = c10::Dispatcher::realSingleton().findSchemaOrThrow("aten::add", "Tensor");
        auto left = at::arange(12, at::TensorOptions().dtype(at::kFloat)).reshape({3, 4});
        auto right = at::ones_like(left);
        c10::Stack stack{left, right, 1};
        fallback(operation, &stack, false);
        TORCH_CHECK(stack.size() == 1 && at::equal(stack[0].toTensor(), at::add(left, right)),
                    "In-module CPU fallback returned the wrong value");
        std::puts("PASS: actual in-module four-argument CPU fallback");

        using Construct =
            c10::TensorImpl* (*)(void*, c10::DispatchKeySet, caffe2::TypeMeta, std::optional<c10::Device>);
        auto construct = reinterpret_cast<Construct>(image + tensorRva);
        // Exercise the actual Tensor implementation and virtual destructor within the original 192-byte allocation.
        auto* tensor = construct(::operator new(192),
                                 c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
                                 caffe2::TypeMeta::Make<float>(),
                                 c10::Device(c10::DeviceType::PrivateUse1, 0));
        TORCH_CHECK(tensor->dtype() == caffe2::TypeMeta::Make<float>(), "Tensor dtype changed");
        auto detached = tensor->shallow_copy_and_detach(c10::VariableVersion(0), true);
        TORCH_CHECK(detached->dtype() == tensor->dtype() && detached->device() == tensor->device(),
                    "Detached Tensor metadata changed");
        detached.reset();
        delete tensor;
        std::puts("PASS: in-module Tensor construction, detach and destruction");

        auto nms = reinterpret_cast<Nms>(image + nmsRva);
        auto boxes = at::tensor({0.f, 0.f, 2.f, 2.f, 0.f, 0.f, 2.f, 2.f, 3.f, 3.f, 4.f, 4.f}).reshape({3, 4});
        auto scores = at::tensor({0.9f, 0.8f, 0.7f});
        TORCH_CHECK(at::equal(CallNms(nms, boxes, scores), at::tensor({int64_t(0), int64_t(2)})),
                    "In-module NMS returned the wrong indices");
        // Check that exceptions unwind temporary Tensors in the added code and reach the caller.
        bool rejected = false;
        try
        {
            CallNms(nms, at::ones({1, 3}), at::ones({1}));
        }
        catch (const std::exception& error)
        {
            rejected = std::string(error.what()).find("boxes should have 4 elements") != std::string::npos;
        }
        TORCH_CHECK(rejected, "Invalid NMS input did not propagate an exception");
        std::puts("PASS: in-module NMS and exception unwinding");

        return 0;
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "Native check failed: %s\n", error.what());
        return 1;
    }
}
