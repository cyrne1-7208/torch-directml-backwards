#include "module.h"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

namespace
{
    class GeneratorHooks final : public at::PrivateUse1HooksInterface
    {
    public:
        explicit GeneratorHooks(const at::GeneratorFuncType& factory) : m_factory(factory)
        {
        }

        at::Generator getNewGenerator(c10::DeviceIndex device) const override
        {
            return m_factory(device);
        }
        const at::Generator& getDefaultGenerator(c10::DeviceIndex) const override
        {
            // The original backend generates random values on CPU; Python's manual_seed_all uses the same generator.
            return at::detail::getDefaultCPUGenerator();
        }

        bool hasPrimaryContext(c10::DeviceIndex device) const override
        {
            // Inspect the original initialization state without initializing a GPU for this query.
            return torch_dml::GetInitializedDevice(device) != nullptr;
        }

    private:
        at::GeneratorFuncType m_factory;
    };
} // namespace

extern "C" __declspec(dllexport) void* tdml_generator_register(void* self, const at::GeneratorFuncType& factory)
{
    // PyTorch retains a non-owning pointer; keep the hooks alive through global destruction.
    auto* hooks = new GeneratorHooks(factory);
    at::RegisterPrivateUse1HooksInterface(hooks);
    return self;
}
