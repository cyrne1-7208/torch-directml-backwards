#include "module.h"

#include <c10/util/Exception.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>

namespace
{
    using Microsoft::WRL::ComPtr;

    struct LegacyBackendManager
    {
        uintptr_t adapters[3];
        void** devicesBegin;
        void** devicesEnd;
        void** devicesCapacity;
        INIT_ONCE* initialized;
    };

    // Layout from the pinned upstream wheel; transfers and compute share this execution context and queue.
    struct LegacyQueue
    {
        ID3D12CommandQueue* queue;
        D3D12_COMMAND_LIST_TYPE type;
        ID3D12Fence* fence;
        uint64_t submitted;
    };

    static_assert(offsetof(LegacyQueue, fence) == 0x10 && offsetof(LegacyQueue, submitted) == 0x18);

    struct LegacyExecutionState
    {
        void* device;
        LegacyQueue* queue;
        void* queueOwner;
        void* recorder;
    };

    struct LegacyExecutionContext
    {
        std::mutex mutex;
        LegacyExecutionState* state;
    };

    static_assert(offsetof(LegacyExecutionContext, state) == 0x50);

    struct Submission
    {
        ComPtr<ID3D12Device> device;
        ComPtr<ID3D12Fence> fence;
        uint64_t fenceValue = 0;

        bool IsComplete() const
        {
            if (!fence)
                return true;
            const auto completedValue = fence->GetCompletedValue();
            // Device removal returns UINT64_MAX, which would otherwise satisfy the completion comparison below.
            TORCH_CHECK(completedValue != std::numeric_limits<uint64_t>::max(),
                        "DirectML device was removed during synchronization (HRESULT ",
                        static_cast<uint32_t>(device->GetDeviceRemovedReason()),
                        ")");
            return completedValue >= fenceValue;
        }
    };

    Submission GetLastSubmission(c10::DeviceIndex index)
    {
        auto* device = static_cast<std::byte*>(torch_dml::GetInitializedDevice(index));
        // A completion query must not initialize an unused device.
        if (!device)
            return {};
        auto* context = *reinterpret_cast<LegacyExecutionContext**>(device + 0x38);
        std::lock_guard lock(context->mutex);
        auto* state = context->state;
        if (state->recorder)
        {
            // Submit through the CPU-readback path so completion checks include commands still being recorded.
            torch_dml::InvokeMethod<0xad380, void>(state->recorder);
            state->recorder = nullptr;
        }
        auto* queue = state->queue;
        return {*reinterpret_cast<ID3D12Device**>(device + 8), queue->fence, queue->submitted};
    }
} // namespace

void* torch_dml::GetInitializedDevice(c10::DeviceIndex index)
{
    BOOL pending = FALSE;
    const auto image = ModuleBase();
    if (!InitOnceBeginInitialize(
            reinterpret_cast<INIT_ONCE*>(image + 0x1d3ce0), INIT_ONCE_CHECK_ONLY, &pending, nullptr))
        return nullptr;
    auto* manager = **reinterpret_cast<LegacyBackendManager***>(image + 0x1d3ce8);
    const auto count = manager->devicesEnd - manager->devicesBegin;
    if (index < 0 || index >= count)
        return nullptr;
    if (!InitOnceBeginInitialize(manager->initialized + index, INIT_ONCE_CHECK_ONLY, &pending, nullptr))
        return nullptr;
    return manager->devicesBegin[index];
}

bool torch_dml::QueryDevice(c10::DeviceIndex index)
{
    return GetLastSubmission(index).IsComplete();
}

void torch_dml::SynchronizeDevice(c10::DeviceIndex index)
{
    auto submission = GetLastSubmission(index);
    if (submission.IsComplete())
        return;
    std::unique_ptr<void, decltype(&CloseHandle)> event(CreateEventW(nullptr, FALSE, FALSE, nullptr), &CloseHandle);
    TORCH_CHECK(event, "Cannot create a DirectML completion event (Win32 ", GetLastError(), ")");
    auto result = submission.fence->SetEventOnCompletion(submission.fenceValue, event.get());
    TORCH_CHECK(
        SUCCEEDED(result), "Cannot register a DirectML completion event (HRESULT ", static_cast<uint32_t>(result), ")");
    const auto waitResult = WaitForSingleObject(event.get(), INFINITE);
    TORCH_CHECK(waitResult == WAIT_OBJECT_0, "DirectML completion wait failed (Win32 ", GetLastError(), ")");
    // Device removal also signals the event; notification alone does not establish successful completion.
    TORCH_CHECK(submission.IsComplete(), "DirectML completion event fired before the submitted work completed");
}
