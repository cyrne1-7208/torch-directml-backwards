#include "module.h"

#include <c10/core/TensorImpl.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

namespace
{
    using torch_dml::InvokeMethod;

    class DmlTensorImpl final : public c10::TensorImpl
    {
    public:
        DmlTensorImpl(c10::DispatchKeySet keys, caffe2::TypeMeta type, std::optional<c10::Device> device)
            : TensorImpl(keys, type, device)
        {
            // Keep opaque sizes and strides after detach; copy_tensor_metadata does not copy this policy.
            set_storage_access_should_throw();
            set_custom_sizes_strides(SizesStridesPolicy::CustomSizes);
        }

        ~DmlTensorImpl() override
        {
            ReplaceOpaque(nullptr);
        }

        void release_resources() override
        {
            TensorImpl::release_resources();
            ReplaceOpaque(nullptr);
        }

        c10::SymBool sym_is_contiguous_custom(c10::MemoryFormat format) const override
        {
            return c10::SymBool(InvokeMethod<0xf340, bool, c10::MemoryFormat>(this, format));
        }

        c10::IntArrayRef sizes_custom() const override
        {
            return InvokeMethod<0xf300, c10::IntArrayRef>(this);
        }
        c10::IntArrayRef strides_custom() const override
        {
            return InvokeMethod<0xf400, c10::IntArrayRef>(this);
        }
        int64_t numel_custom() const override
        {
            return InvokeMethod<0xf280, int64_t>(this);
        }
        int64_t storage_offset_custom() const override
        {
            return 0;
        }
        int64_t dim_custom() const override
        {
            return InvokeMethod<0xf290, int64_t>(this);
        }
        c10::Device device_custom() const override
        {
            return InvokeMethod<0xf2a0, c10::Device>(this);
        }
        c10::SymIntArrayRef sym_sizes_custom() const override
        {
            return InvokeMethod<0xf2c0, c10::SymIntArrayRef>(this);
        }
        c10::SymIntArrayRef sym_strides_custom() const override
        {
            return InvokeMethod<0xf3c0, c10::SymIntArrayRef>(this);
        }
        c10::SymInt sym_numel_custom() const override
        {
            return InvokeMethod<0xf250, c10::SymInt>(this);
        }
        c10::SymInt sym_storage_offset_custom() const override
        {
            return c10::SymInt(0);
        }

        void set_size(int64_t dim, int64_t value) override
        {
            InvokeMethod<0xf810, void, int64_t, int64_t>(this, dim, value);
        }
        void set_stride(int64_t dim, int64_t value) override
        {
            InvokeMethod<0xf7c0, void, int64_t, int64_t>(this, dim, value);
        }
        void set_storage_offset(int64_t offset) override
        {
            InvokeMethod<0xf770, void, int64_t>(this, offset);
        }

        c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(const c10::VariableVersion& version,
                                                                    bool allowChange) const override
        {
            return Detach(version, allowChange);
        }
        c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(c10::VariableVersion&& version,
                                                                    bool allowChange) const override
        {
            return Detach(std::move(version), allowChange);
        }
        void shallow_copy_from(const c10::intrusive_ptr<c10::TensorImpl>& source) override
        {
            const auto* tensor = dynamic_cast<const DmlTensorImpl*>(source.get());
            TORCH_CHECK(tensor, "Cannot shallow-copy a different TensorImpl into a DirectML tensor");
            copy_tensor_metadata(tensor, this, version_counter(), allow_tensor_metadata_change());
            ReplaceOpaque(tensor->m_opaque);
            refresh_numel();
        }

    private:
        // Retain the replacement handle (slot 1) before releasing the previous handle (slot 2).
        void ReplaceOpaque(void* value)
        {
            if (value == m_opaque)
                return;
            if (value)
            {
                auto table = *static_cast<void***>(value);
                reinterpret_cast<void (*)(void*)>(table[1])(value);
            }
            if (auto previous = std::exchange(m_opaque, value))
            {
                auto table = *static_cast<void***>(previous);
                reinterpret_cast<void (*)(void*)>(table[2])(previous);
            }
        }

        template <typename Version>
        c10::intrusive_ptr<c10::TensorImpl> Detach(Version&& version, bool allowChange) const
        {
            auto result = c10::make_intrusive<DmlTensorImpl>(key_set(), dtype(), device());
            copy_tensor_metadata(this, result.get(), std::forward<Version>(version), allowChange);
            result->ReplaceOpaque(m_opaque);
            result->refresh_numel();
            return result;
        }

        // The original extension reads m_opaque at this+0xb8; preserve this layout.
        std::byte m_legacyPadding[8]{};
        void* m_opaque = nullptr;
    };

    static_assert(sizeof(c10::TensorImpl) == 176);
    static_assert(sizeof(DmlTensorImpl) == 192);
} // namespace

extern "C" __declspec(dllexport) c10::TensorImpl* tdml_tensor_construct(void* self,
                                                                        c10::DispatchKeySet keys,
                                                                        caffe2::TypeMeta type,
                                                                        std::optional<c10::Device> device)
{
    return new (self) DmlTensorImpl(keys, type, device);
}
