#include "device.h"

#include <utility>   // for std::move
#include <thread>    // std::thread
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

void qr_destroy(quickreduce::fptr_t _fa) {
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

// 在“当前流”上入队 quickreduce 的 allreduce
/*
void allreduce(quickreduce::fptr_t _fa,
               torch::Tensor& inp,
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
*/

/*
c10::intrusive_ptr<c10::ivalue::Future>
allreduce_async(quickreduce::fptr_t fa_addr, 
  at::Tensor & tensor, int64_t quant_level, bool cast_bf2half) {
  TORCH_CHECK(tensor.is_cuda(), "quick_allreduce expects CUDA/HIP tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kHalf || tensor.scalar_type() == at::kBFloat16,
              "quick allreduce only supports float16/bfloat16");

  at::cuda::OptionalCUDAGuard guard(tensor.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* fa = reinterpret_cast<quickreduce::DeviceComms*>(fa_addr);
  TORCH_CHECK_LE(tensor.numel(), fa->kMaxProblemSize);
  if (tensor.scalar_type() == at::ScalarType::Half) {
    fa->allreduce(reinterpret_cast<half*>(tensor.data_ptr()),
                  tensor.numel(), quant_level, stream, false);
  } else {
    throw std::runtime_error("quick allreduce only supports float16 and bfloat16");
  }
  c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
  at::cuda::CUDAEvent done_event;
  done_event.record(stream);

  auto fut = c10::make_intrusive<c10::ivalue::Future>(c10::TensorType::get());
  std::thread([evt = std::move(done_event), fut, tensor]() mutable {
  try {
        evt.synchronize();
        fut->markCompleted(tensor);
      } catch (...) {
        fut->setError(std::current_exception());  
    }).detach();
  return fut;
}
*/

c10::intrusive_ptr<c10::ivalue::Future>
allreduce_async(quickreduce::fptr_t fa_addr,
                at::Tensor& tensor,          // ★ 传入的就是 DDP bucket buffer
                int64_t quant_level,
                bool cast_bf2half) {

  TORCH_CHECK(tensor.is_cuda(), "quick_allreduce expects CUDA/HIP tensor");
  auto in_dtype = tensor.scalar_type();
  TORCH_CHECK(in_dtype == at::kFloat || in_dtype == at::kBFloat16 || in_dtype == at::kHalf,
              "quick allreduce supports float32/bfloat16/float16 as input");

  // 统一设备 + 当前通信流（ROCm 下复用 at::cuda 语义）
  at::cuda::OptionalCUDAGuard guard(tensor.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  at::cuda::CUDAStreamGuard sg(stream);   // ★ 之后的 op（to/copy_/allreduce）都排队到同一流

  auto* fa = reinterpret_cast<quickreduce::DeviceComms*>(fa_addr);
  TORCH_CHECK_LE(tensor.numel(), fa->kMaxProblemSize);

  // 1) 输入 → FP16（在同一流里转换）
  at::Tensor t_fp16;
  if (in_dtype == at::kHalf) {
    t_fp16 = tensor;                      // ★ 已经是 FP16，直接复用输入（真正 in-place 路径）
  } else {
    t_fp16 = tensor.to(at::kHalf);        // ★ 异步 cast 入队到同一流
    c10::cuda::CUDACachingAllocator::recordStream(t_fp16.storage().data_ptr(), stream);
  }

  // 2) allreduce in FP16（假设后端 in-place 写 t_fp16）
  fa->allreduce(reinterpret_cast<half*>(t_fp16.data_ptr()),
                t_fp16.numel(), quant_level, stream, false);

  // 3) 把结果“就地写回”输入 tensor（保持原 dtype 不变）
  //    - 如果输入是 FP16，什么都不用做（已经 in-place 完成）
  //    - 如果输入是 BF16/FP32，使用 copy_ 进行 dtype 转换并回写
  if (in_dtype != at::kHalf) {
    // 这里会排队到同一流，且允许 dtype 转换（FP16 -> BF16/FP32）
    tensor.copy_(t_fp16, /*non_blocking=*/true);
  }

  // 4) 事件记录在“最后一步回写之后”，保证 Future 完成时数据已在输入 tensor 内
  at::cuda::CUDAEvent done_event;
  done_event.record(stream);

  // 5) 返回 Future[Tensor]（值就是“输入 tensor 本身”）
  auto fut = c10::make_intrusive<c10::ivalue::Future>(c10::TensorType::get());

  // ★ 把 t_fp16 捕获进线程以延长其生命周期，直到 GPU 完成为止
  std::thread([evt = std::move(done_event), fut, out = tensor, hold = std::move(t_fp16)]() mutable {
    try {
      evt.synchronize();          // 等待：to(FP16) + allreduce + copy_ 回写 全部完成
      fut->markCompleted(out);    // ★ 就地语义：返回“输入 tensor”
    } catch (...) {
      fut->setError(std::current_exception());
    }
  }).detach();

  // （可选）把“输入 tensor 的存储”登记到该流，避免异步回写期间被复用/回收
  c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
  return fut;
}


int64_t qr_max_size() {
  // The default is 2GB (2,147,483,648 bytes)
  return static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
}