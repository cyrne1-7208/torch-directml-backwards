#include <ATen/core/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <new>

// Legacy NMS accessed dispatcher arrays directly; replace that access with the current typed API.
extern "C" __declspec(dllexport) at::Tensor* tdml_nms(at::Tensor* output,
                                                      const at::Tensor& boxes,
                                                      const at::Tensor& scores,
                                                      double threshold)
{
    auto& dispatcher = c10::Dispatcher::realSingleton();
    const auto operation = dispatcher.findSchemaOrThrow("torchvision::nms", "")
                               .typed<at::Tensor(const at::Tensor&, const at::Tensor&, double)>();
    auto cpuScores = scores.to(c10::Device(c10::DeviceType::CPU));
    auto cpuBoxes = boxes.to(c10::Device(c10::DeviceType::CPU));
    auto result = dispatcher.call<at::Tensor, const at::Tensor&, const at::Tensor&, double>(
        operation, cpuBoxes, cpuScores, threshold);
    return new (output) at::Tensor(result.to(boxes.device()));
}
