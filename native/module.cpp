#include "module.h"

extern "C" void __cdecl __security_init_cookie();

extern "C" __declspec(dllexport) __declspec(safebuffers) BOOL WINAPI tdml_entry(HINSTANCE module,
                                                                                DWORD reason,
                                                                                LPVOID reserved)
{
    // Initialize this code's GS cookie before entering the original CRT startup sequence.
    if (reason == DLL_PROCESS_ATTACH)
        __security_init_cookie();
    using Entry = BOOL(WINAPI*)(HINSTANCE, DWORD, LPVOID);
    auto original = reinterpret_cast<Entry>(torch_dml::ModuleBase() + TDML_SOURCE_ENTRY_RVA);
    return original(module, reason, reserved);
}
