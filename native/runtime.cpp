#include <c10/util/logging_common.h>
#include <torch/version.h>

#include <memory>
#include <new>
#include <sstream>

extern "C" __declspec(dllexport) const char tdml_torch_version[] = TORCH_VERSION;

namespace
{
    struct LogLocation
    {
        const char* file;
        int line;
    };

    // The original inline stream() reads this+8; preserve the legacy stringstream layout.
    struct LegacyLogger
    {
        std::unique_ptr<LogLocation> location;
        std::stringstream stream;
        int severity;

        LegacyLogger(const char* file, int line, int level)
            : location(std::make_unique<LogLocation>(LogLocation{file, line})), severity(level)
        {
        }
    };

    static_assert(sizeof(LegacyLogger) == 264);
    static_assert(sizeof(c10::MessageLogger) == 280);
} // namespace

extern "C" __declspec(dllexport) LegacyLogger* tdml_log_construct(LegacyLogger* self,
                                                                  const char* file,
                                                                  int line,
                                                                  int severity)
{
    return new (self) LegacyLogger(file, line, severity);
}

extern "C" __declspec(dllexport) std::stringstream* tdml_log_stream(LegacyLogger* self)
{
    return &self->stream;
}

extern "C" __declspec(dllexport) void tdml_log_destroy(LegacyLogger* self)
{
    std::unique_ptr<LegacyLogger, void (*)(LegacyLogger*)> cleanup(
        self, [](LegacyLogger* logger) { logger->~LegacyLogger(); });
    c10::MessageLogger current(
        c10::SourceLocation::current(self->location->file, "", self->location->line), self->severity, true);
    current.stream() << self->stream.str();
}
