#include <torch/torch.h>
#include <pybind11/pybind11.h>

#include "device.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &init);
   // m.def('destroy', &destroy);
   // m.def("get_world_size", &get_world_size);
   // m.def("get_rank", &get_rank);
    m.def("get_handle", &get_handle);
    m.def("open_handles", &open_handles);
    m.def("allreduce", &allreduce);
}
