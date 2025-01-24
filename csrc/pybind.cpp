#include <cstdlib>
#include <ops.h>
#include <pybind11/pybind11.h>

#include "migration_manager.cuh"
#include "rnccl.h"

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // vLLM custom ops
  pybind11::module ops = m.def_submodule("ops", "migration operators");

  ops.def("init_migration_manager", &init_migration_manager,
          "init migration manager");
  ops.def("migrate", &migrate, "perform migrate from p instance to d instance");

  py::class_<KVCacheHandlerConfig>(m, "KVCacheHandlerConfig")
      .def(py::init<int, int, int, int,
                    std::vector<std::vector<std::vector<std::string>>> &,
                    std::vector<std::vector<std::vector<int64_t>>> &>())
      .def(py::init<int, int, int, int>())
      .def(py::init<>());
  m.def("get_nccl_unique_id", &get_nccl_unique_id,
        "Get a NCCL unique id for initializing RNCCLComm");

  py::class_<RNCCLComm>(m, "RNCCLComm")
      .def(py::init<ncclUniqueIdVec, int, int>())
      .def("nccl_group_start", &RNCCLComm::nccl_group_start)
      .def("nccl_group_end", &RNCCLComm::nccl_group_end)
      .def("nccl_send", &RNCCLComm::nccl_send)
      .def("nccl_recv", &RNCCLComm::nccl_recv)
      .def("wait_for_default_stream", &RNCCLComm::wait_for_default_stream)
      .def("let_default_stream_wait", &RNCCLComm::let_default_stream_wait);
}
