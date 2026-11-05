// Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.
// GIL handling is based on pybind11 2.12.0 (BSD-3-Clause); see NOTICE for the license.
#include "module.h"

#include <Python.h>

#include <cstddef>
#include <new>

namespace
{
    struct GILState
    {
        PyThreadState* thread;
        bool release;
        bool active;
    };

    // Original pybind11 internals ABI v5; this layout is separate from CPython's own structures.
    struct BindingInternals
    {
        std::byte prefix[0x1e8];
        Py_tss_t threadKey;
        Py_tss_t loaderKey;
        PyInterpreterState* interpreter;
    };

    static_assert(sizeof(GILState) == 16 && offsetof(GILState, release) == 8);
    static_assert(offsetof(BindingInternals, interpreter) == 0x1f8);

    BindingInternals& GetOriginalInternals()
    {
        using GetInternals = BindingInternals& (*)();
        return reinterpret_cast<GetInternals>(torch_dml::ModuleBase() + 0x111140)();
    }
} // namespace

extern "C" __declspec(dllexport) GILState* tdml_gil_acquire(GILState* self) noexcept(false)
{
    new (self) GILState{nullptr, true, true};
    auto& internals = GetOriginalInternals();
    self->thread = static_cast<PyThreadState*>(PyThread_tss_get(&internals.threadKey));
    if (!self->thread)
        self->thread = PyGILState_GetThisThreadState();
    if (!self->thread)
    {
        self->thread = PyThreadState_New(internals.interpreter);
        if (!self->thread)
            throw std::bad_alloc();
        self->thread->gilstate_counter = 0;
        PyThread_tss_set(&internals.threadKey, self->thread);
    }
    else
    {
        self->release = PyThreadState_GetUnchecked() != self->thread;
    }
    if (self->release)
        PyEval_AcquireThread(self->thread);
    // Use the target CPython headers for field offsets instead of reusing the old binary layout.
    ++self->thread->gilstate_counter;
    return self;
}

extern "C" __declspec(dllexport) void tdml_gil_dec_ref(GILState* self)
{
    if (--self->thread->gilstate_counter == 0)
    {
        PyThreadState_Clear(self->thread);
        if (self->active)
            PyThreadState_DeleteCurrent();
        PyThread_tss_set(&GetOriginalInternals().threadKey, nullptr);
        self->release = false;
    }
}

extern "C" __declspec(dllexport) int tdml_python_traverse(PyObject* self, visitproc visit, void* arg)
{
    int result = PyObject_VisitManagedDict(self, visit, arg);
    if (result)
        return result;
    Py_VISIT(Py_TYPE(self));
    return 0;
}

extern "C" __declspec(dllexport) int tdml_python_clear(PyObject* self)
{
    PyObject_ClearManagedDict(self);
    return 0;
}
