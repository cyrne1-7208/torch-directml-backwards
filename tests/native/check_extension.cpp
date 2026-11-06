#include <ATen/Functions.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <utility>

using Nms = at::Tensor* (*)(at::Tensor*, const at::Tensor&, const at::Tensor&, double);

at::Tensor CallNms(Nms function, const at::Tensor& boxes, const at::Tensor& scores, double threshold)
{
    alignas(at::Tensor) std::byte storage[sizeof(at::Tensor)];
    auto* output = function(reinterpret_cast<at::Tensor*>(storage), boxes, scores, threshold);
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
        // The first two boxes have IoU 1/3, so the result must change with the threshold.
        auto boxes = at::tensor({0.f, 0.f, 2.f, 2.f, 1.f, 0.f, 3.f, 2.f, 4.f, 4.f, 5.f, 5.f}).reshape({3, 4});
        auto scores = at::tensor({0.9f, 0.8f, 0.7f});
        TORCH_CHECK(at::equal(CallNms(nms, boxes, scores, 0.2), at::tensor({int64_t(0), int64_t(2)})),
                    "In-module NMS returned the wrong indices");
        TORCH_CHECK(at::equal(CallNms(nms, boxes, scores, 0.5), at::arange(3, at::kLong)),
                    "In-module NMS ignored its threshold");
        // Check that exceptions unwind temporary Tensors in the added code and reach the caller.
        bool rejected = false;
        try
        {
            CallNms(nms, at::ones({1, 3}), at::ones({1}), 0.5);
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

namespace
{
    // Read the pinned queue layout directly so the tested guard is not its own completion oracle.
    struct Queue
    {
        ID3D12CommandQueue* queue;
        D3D12_COMMAND_LIST_TYPE type;
        ID3D12Fence* fence;
        uint64_t submitted;
    };

    struct ExecutionState
    {
        void* device;
        Queue* queue;
        void* queueOwner;
        void* recorder;
    };

    struct ExecutionContext
    {
        std::mutex mutex;
        ExecutionState* state;
    };

    struct Submission
    {
        Microsoft::WRL::ComPtr<ID3D12Fence> fence;
        uint64_t value;
        bool recording;
    };

    enum class CompletionOperation
    {
        Query,
        Stream,
        Device,
    };

    static_assert(offsetof(Queue, fence) == 0x10 && offsetof(ExecutionContext, state) == 0x50);
}

extern "C" __declspec(dllexport) int tdml_check_synchronization(void* module, int index)
{
    try
    {
        using Microsoft::WRL::ComPtr;
        c10::Device device(c10::DeviceType::PrivateUse1, index);
        c10::impl::VirtualGuardImpl guard(device.type());
        auto stream = guard.getDefaultStream(device);
        auto value = at::ones({32, 32}, at::TensorOptions().device(device));
        auto output = at::empty_like(value);
        at::mm_out(output, value, value);
        output.cpu(); // Compile and allocate before blocking the queue.

        auto* image = static_cast<std::byte*>(module);
        auto* manager = **reinterpret_cast<std::byte***>(image + 0x1d3ce8);
        auto** devices = *reinterpret_cast<std::byte***>(manager + 0x18);
        auto* backend = devices[index];
        auto* context = *reinterpret_cast<ExecutionContext**>(backend + 0x38);
        auto* d3dDevice = *reinterpret_cast<ID3D12Device**>(backend + 8);

        auto snapshot = [&]
        {
            std::lock_guard lock(context->mutex);
            auto* state = context->state;
            return Submission{state->queue->fence, state->queue->submitted, state->recorder != nullptr};
        };
        for (auto operation : {CompletionOperation::Query, CompletionOperation::Stream, CompletionOperation::Device})
        {
            ComPtr<ID3D12Fence> gate;
            TORCH_CHECK(SUCCEEDED(d3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate))),
                        "Cannot create the independent queue gate");
            std::unique_ptr<void, decltype(&CloseHandle)> ready(CreateEventW(nullptr, FALSE, FALSE, nullptr), &CloseHandle);
            TORCH_CHECK(ready, "Cannot create the gate control event");
            std::atomic<HRESULT> releaseResult{E_PENDING};
            // Arm the short delay after enqueueing. The timeout also releases a queue blocked during setup or unwinding.
            std::jthread release([&]
            {
                if (WaitForSingleObject(ready.get(), 10000) == WAIT_OBJECT_0)
                    Sleep(2000);
                releaseResult = gate->Signal(1);
            });
            TORCH_CHECK(SUCCEEDED(context->state->queue->queue->Wait(gate.Get(), 1)), "Cannot gate the backend queue");
            at::mm_out(output, value, value);
            const auto before = snapshot();
            TORCH_CHECK(gate->GetCompletedValue() == 0, "Queue gate expired during setup");
            TORCH_CHECK(before.recording || before.fence->GetCompletedValue() < before.value,
                        "No recorded or submitted work was available for the check");
            TORCH_CHECK(SetEvent(ready.get()), "Cannot arm the queue gate release");
            bool reportedComplete = false;
            if (operation == CompletionOperation::Query)
                reportedComplete = guard.queryStream(stream);
            else if (operation == CompletionOperation::Stream)
                guard.synchronizeStream(stream);
            else
                guard.synchronizeDevice(index);
            // Observe the real queue before joining or reading back, without submitting through a test helper.
            const auto after = snapshot();
            const auto completed = after.fence->GetCompletedValue();
            TORCH_CHECK(!after.recording && (!before.recording || after.value > before.value),
                        "Completion operation did not submit recorded commands");
            if (operation == CompletionOperation::Query)
            {
                TORCH_CHECK(gate->GetCompletedValue() == 0 && !reportedComplete && completed < after.value,
                            "Query did not observe work blocked by the independent gate");
            }
            else
            {
                TORCH_CHECK(completed >= after.value && completed != UINT64_MAX,
                            "Synchronization returned while the actual backend fence was incomplete");
            }
            release.join();
            TORCH_CHECK(SUCCEEDED(releaseResult.load()), "Cannot release the independent queue gate");
            output.cpu(); // Finish the query-only case; the wait assertions above already observed the fence.
            TORCH_CHECK(guard.queryStream(stream), "Completed work was reported as pending");
        }
        std::puts("PASS: stream/device waits observed on the actual queue fence before readback");
        return 0;
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "Native synchronization check failed: %s\n", error.what());
        return 1;
    }
}
