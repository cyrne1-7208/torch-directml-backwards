#include "module.h"

extern "C"
{
    alignas(16) __declspec(dllexport) extern const float tdml_resample_input_offsets[4] = {0.0f, 0.0f, 0.5f, 0.5f};
    alignas(16) __declspec(dllexport) extern const float tdml_resample_output_offsets[4] = {0.0f, 0.0f, -0.5f, -0.5f};
}

// Called from the original array initialization; RBP is the caller's frame. Update the array and XMM0 without changing general-purpose registers.
// The caller next constructs a vector and does not read the comparison flags modified here.
// Mode 1 is LINEAR. Apply half-pixel H/W offsets only when align_corners=False; leave N/C offsets at zero.
#define TDML_RESAMPLE_OFFSETS(name, mode, align, destination, values)                                                  \
    extern "C" __declspec(dllexport) __attribute__((naked)) void name()                                                \
    {                                                                                                                  \
        __asm__("xorps %xmm0, %xmm0\n"                                                                                 \
                "cmpl $1, " #mode "(%rbp)\n"                                                                           \
                "jne 1f\n"                                                                                             \
                "cmpb $0, " #align "(%rbp)\n"                                                                          \
                "jne 1f\n"                                                                                             \
                "movaps " #values "(%rip), %xmm0\n"                                                                    \
                "1: movups %xmm0, " #destination "(%rbp)\n"                                                            \
                "ret\n");                                                                                              \
    }

// Frame offsets belong to the source wheel pinned in common.toml.
TDML_RESAMPLE_OFFSETS(tdml_upsample_input_offsets, 0x350, 0x360, 0x28, tdml_resample_input_offsets)
TDML_RESAMPLE_OFFSETS(tdml_upsample_output_offsets, 0x350, 0x360, 0x38, tdml_resample_output_offsets)
// ResampleGrad1 needs inverse offsets; forward offsets would shift gradient contributions away from pixel centers.
TDML_RESAMPLE_OFFSETS(tdml_upsample_grad_input_offsets, 0x368, 0x378, 0x40, tdml_resample_output_offsets)
TDML_RESAMPLE_OFFSETS(tdml_upsample_grad_output_offsets, 0x368, 0x378, 0x50, tdml_resample_input_offsets)

#undef TDML_RESAMPLE_OFFSETS
