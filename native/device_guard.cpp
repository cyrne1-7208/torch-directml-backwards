#include "module.h"

#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace
{
    using namespace c10;
    using torch_dml::InvokeMethod;

    class DmlGuardImpl final : public impl::DeviceGuardImplInterface
    {
    public:
        explicit DmlGuardImpl(const void* legacy) : m_legacy(legacy)
        {
        }

        DeviceType type() const override
        {
            return DeviceType::PrivateUse1;
        }
        Device exchangeDevice(Device device) const override
        {
            return InvokeMethod<0x69c0, Device, Device>(m_legacy, device);
        }
        Device getDevice() const override
        {
            return InvokeMethod<0x69c0, Device>(m_legacy);
        }
        // These upstream methods are no-ops; do not add independent device state.
        void setDevice(Device) const override
        {
        }
        void uncheckedSetDevice(Device) const noexcept override
        {
        }
        Stream getStream(Device device) const override
        {
            return InvokeMethod<0x69f0, Stream, Device>(m_legacy, device);
        }
        Stream getDefaultStream(Device device) const override
        {
            return InvokeMethod<0x69f0, Stream, Device>(m_legacy, device);
        }
        Stream exchangeStream(Stream stream) const override
        {
            return InvokeMethod<0x6a40, Stream, Stream>(m_legacy, stream);
        }
        bool queryStream(const Stream& stream) const override
        {
            TORCH_CHECK(stream.id() == 0, "DirectML only supports the default stream");
            TORCH_CHECK(stream.device_type() == type(), "Expected a DirectML stream");
            const auto index = stream.device_index();
            TORCH_CHECK(index >= 0 && index < deviceCount(), "Invalid DirectML device index: ", index);
            return torch_dml::QueryDevice(index);
        }
        void synchronizeStream(const Stream& stream) const override
        {
            TORCH_CHECK(stream.id() == 0, "DirectML only supports the default stream");
            TORCH_CHECK(stream.device_type() == type(), "Expected a DirectML stream");
            synchronizeDevice(stream.device_index());
        }
        void synchronizeDevice(DeviceIndex index) const override
        {
            TORCH_CHECK(index >= 0 && index < deviceCount(), "Invalid DirectML device index: ", index);
            torch_dml::SynchronizeDevice(index);
        }
        void destroyEvent(void*, DeviceIndex) const noexcept override
        {
        }
        void record(void** event, const Stream& stream, DeviceIndex device, EventFlag flag) const override
        {
            InvokeMethod<0x6b40, void, void**, const Stream&, DeviceIndex, EventFlag>(
                m_legacy, event, stream, device, flag);
        }
        void block(void* event, const Stream& stream) const override
        {
            InvokeMethod<0x6b80, void, void*, const Stream&>(m_legacy, event, stream);
        }
        bool queryEvent(void* event) const override
        {
            return InvokeMethod<0x6bc0, bool, void*>(m_legacy, event);
        }
        DeviceIndex deviceCount() const noexcept override
        {
            return InvokeMethod<0x6a90, DeviceIndex>(m_legacy);
        }

    private:
        const void* m_legacy;
    };
} // namespace

extern "C" __declspec(dllexport) c10::impl::DeviceGuardImplRegistrar* tdml_guard_register(
    c10::impl::DeviceGuardImplRegistrar* self,
    c10::DeviceType device,
    const c10::impl::DeviceGuardImplInterface* legacy)
{
    // The original DLL owns the legacy object; replace only the interface registered with PyTorch.
    // The registry does not take ownership, so retain the guard through shutdown as upstream does.
    const auto* guard = new DmlGuardImpl(legacy);
    auto* registrar = new (self) c10::impl::DeviceGuardImplRegistrar(device, guard);
    // Register the original backend name so current autograd can discover the accelerator.
    c10::register_privateuse1_backend("privateuseone");
    torch_dml::RegisterAmp();
    return registrar;
}
