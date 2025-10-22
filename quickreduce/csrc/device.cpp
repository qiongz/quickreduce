#include "device.h"
#include <utility>   
#include <thread>   
#include <exception> 


quickreduce::fptr_t init(int world_size, int rank, std::optional<int64_t> qr_max) {
  if (world_size > 8)  throw std::invalid_argument("world size > 8 is not supported");
  if (world_size == 6) throw std::invalid_argument("world size == 6 is not supported");
  if (world_size % 2 != 0) throw std::invalid_argument("Odd num gpus is not supported for now");
  if (rank < 0 || rank >= world_size) throw std::invalid_argument("invalid rank passed in");
  auto* fptr = new quickreduce::DeviceComms();
  fptr->init(world_size, rank, qr_max);
  return reinterpret_cast<quickreduce::fptr_t>(fptr);
}

void destroy(quickreduce::fptr_t _fa) {
  if (_fa) {
    auto* fa = reinterpret_cast<quickreduce::DeviceComms*>(_fa);
    fa->destroy();
    delete fa;
  }
}

torch::Tensor get_handle(quickreduce::fptr_t _fa) {
  auto* fa = reinterpret_cast<quickreduce::DeviceComms*>(_fa);
  hipIpcMemHandle_t handle = fa->get_handle();
  auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  auto data = torch::empty({static_cast<int64_t>(sizeof(hipIpcMemHandle_t))}, options);
  std::memcpy(data.data_ptr(), &handle, sizeof(hipIpcMemHandle_t));
  return data;
}

void open_handles(quickreduce::fptr_t _fa, const std::vector<torch::Tensor>& handles) {
  auto* fa = reinterpret_cast<quickreduce::DeviceComms*>(_fa);
  std::vector<hipIpcMemHandle_t> ipc_handles;
  ipc_handles.reserve(handles.size());
  for (auto& h : handles) {
    hipIpcMemHandle_t ipc{};
    std::memcpy(&ipc, h.data_ptr(), sizeof(hipIpcMemHandle_t));
    ipc_handles.push_back(ipc);
  }
  fa->open_ipc_handles(ipc_handles);
}


void allreduce(quickreduce::fptr_t _fa,
               at::Tensor& inp,
               int64_t quant_level,
               bool cast_bf2half) {
  auto* fa = reinterpret_cast<quickreduce::DeviceComms*>(_fa);
  at::cuda::OptionalCUDAGuard guard(inp.device());
  auto stream = at::cuda::getCurrentCUDAStream(); 
  TORCH_CHECK_LE(inp.numel(), fa->kMaxProblemSize);
  if (inp.scalar_type() == at::ScalarType::Half) {
    fa->allreduce(reinterpret_cast<half*>(inp.data_ptr()),
                  inp.numel(), quant_level, stream, false);
  } else {
    throw std::runtime_error("quick allreduce only supports float16 and bfloat16");
  }
}


c10::intrusive_ptr<c10::ivalue::Future>
allreduce_async(quickreduce::fptr_t fa_addr,
                at::Tensor& tensor,         
                int64_t quant_level,
                bool cast_bf2half) {

  TORCH_CHECK(tensor.is_cuda(), "quick_allreduce expects CUDA/HIP tensor");
  auto in_dtype = tensor.scalar_type();
  TORCH_CHECK(in_dtype == at::kFloat || in_dtype == at::kBFloat16 || in_dtype == at::kHalf,
              "quick allreduce supports float32/bfloat16/float16 as input");

  at::cuda::OptionalCUDAGuard guard(tensor.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  at::cuda::CUDAStreamGuard sg(stream);  
  auto* fa = reinterpret_cast<quickreduce::DeviceComms*>(fa_addr);
  TORCH_CHECK_LE(tensor.numel(), fa->kMaxProblemSize);

  at::Tensor t_fp16;
  if (in_dtype == at::kHalf) {
    t_fp16 = tensor;                      
  } else {
    t_fp16 = tensor.to(at::kHalf);        
    c10::cuda::CUDACachingAllocator::recordStream(t_fp16.storage().data_ptr(), stream);
  }
  // call Comms->allreduce
  fa->allreduce(reinterpret_cast<half*>(t_fp16.data_ptr()),
                t_fp16.numel(), quant_level, stream, false);

  if (in_dtype != at::kHalf) {
    
    tensor.copy_(t_fp16, true);
  }
  at::cuda::CUDAEvent done_event;
  done_event.record(stream);
  auto fut = c10::make_intrusive<c10::ivalue::Future>(c10::TensorType::get());
  std::thread([evt = std::move(done_event), fut, out = tensor, hold = std::move(t_fp16)]() mutable {
    try {
      evt.synchronize();  
      fut->markCompleted(out); 
    } catch (...) {
      fut->setError(std::current_exception());
    }
  }).detach();
  c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
  return fut;
}


int64_t qr_max_size() {
  // The default is 2GB (2,147,483,648 bytes)
  return static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
}