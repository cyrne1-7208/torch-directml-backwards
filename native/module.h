#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <c10/core/Device.h>

#include <bit>
#include <cstdint>
#include <utility>

namespace torch_dml
{
    extern "C" IMAGE_DOS_HEADER __ImageBase;

    void RegisterAmp();
    void* GetInitializedDevice(c10::DeviceIndex device);
    bool QueryDevice(c10::DeviceIndex device);
    void SynchronizeDevice(c10::DeviceIndex device);

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
