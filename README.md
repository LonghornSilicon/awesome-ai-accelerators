<div align="center">

# Awesome AI Accelerators [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of papers, tools, and resources for **AI accelerators** — spanning the entire stack from silicon to serving systems.

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](LICENSE)

</div>

AI accelerators are the engines behind modern machine learning — the GPUs, TPUs, NPUs, and custom ASICs that make training and inference practical at scale. Building and programming them is a deeply interdisciplinary craft that spans deep learning, computer architecture, compilers, GPU programming, RTL design, and physical implementation.

This list is a **curated, opinionated map** of that landscape. It favors canonical papers and widely adopted open-source projects over exhaustive coverage. Whether you are an undergraduate taking your first architecture course, a PhD student researching dataflow architectures, or an engineer optimizing kernels at a frontier lab, the goal is to give you the best resource for each topic — and a clear path through the rest.

---

## Contents

- [AI Foundations](#ai-foundations)
  - [AI Fundamentals](#ai-fundamentals)
  - [Deep Learning](#deep-learning)
  - [Transformers & LLMs](#transformers--llms)
  - [Diffusion Models](#diffusion-models)
  - [Vision Models](#vision-models)
  - [Reinforcement Learning](#reinforcement-learning)
- [AI Inference](#ai-inference)
  - [Quantization](#quantization)
  - [KV Cache](#kv-cache)
  - [Speculative Decoding](#speculative-decoding)
  - [Prefix Caching](#prefix-caching)
  - [Continuous Batching](#continuous-batching)
  - [Mixture of Experts](#mixture-of-experts)
  - [Sparsity & Pruning](#sparsity--pruning)
  - [Scheduling](#scheduling)
  - [Memory Optimization](#memory-optimization)
  - [Serving Systems](#serving-systems)
- [GPU Programming](#gpu-programming)
  - [CUDA](#cuda)
  - [Triton](#triton)
  - [ROCm](#rocm)
  - [Metal](#metal)
- [AI Kernels](#ai-kernels)
  - [GEMM](#gemm)
  - [FlashAttention](#flashattention)
  - [PagedAttention](#pagedattention)
  - [Normalization & Softmax](#normalization--softmax)
  - [RoPE & Positional Encodings](#rope--positional-encodings)
  - [Fused Operators](#fused-operators)
  - [Kernel Optimization Guides](#kernel-optimization-guides)
- [Runtime Systems](#runtime-systems)
- [Compilers](#compilers)
- [Computer Architecture](#computer-architecture)
  - [Foundations](#foundations)
  - [Memory Systems & Cache](#memory-systems--cache)
  - [Interconnect & NoC](#interconnect--noc)
  - [Dataflow & Systolic Arrays](#dataflow--systolic-arrays)
  - [SIMD, SIMT & Tensor Cores](#simd-simt--tensor-cores)
  - [Processing-in-Memory](#processing-in-memory)
- [AI Accelerator Architecture](#ai-accelerator-architecture)
  - [GPUs](#gpus)
  - [TPUs & Systolic ASICs](#tpus--systolic-asics)
  - [NPUs & Edge AI](#npus--edge-ai)
  - [FPGA & Reconfigurable](#fpga--reconfigurable)
  - [RISC-V for AI](#risc-v-for-ai)
- [Simulators & Modeling](#simulators--modeling)
- [Chip Design](#chip-design)
  - [Blogs & Newsletters](#blogs--newsletters)
  - [YouTube Channels](#youtube-channels)
  - [Interactive & Visual](#interactive--visual)
- [RTL Design](#rtl-design)
- [Design Verification](#design-verification)
- [Physical Design](#physical-design)
- [University Courses](#university-courses)

---

## AI Foundations

Understanding the workloads is the prerequisite for accelerating them. This section covers the models that drive accelerator design.

### AI Fundamentals

- [Deep Learning](https://www.deeplearningbook.org/) — Goodfellow, Bengio, and Courville's foundational textbook, freely available online.
- [Dive into Deep Learning (D2L)](https://d2l.ai/) — Interactive book with runnable code in PyTorch, JAX, and TensorFlow.
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael Nielsen's intuitive introduction to the core ideas.
- [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/) — Parr and Howard's practical primer on the math behind backprop.

### Deep Learning

- [ImageNet Classification with Deep CNNs (AlexNet)](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) — The 2012 paper that ignited the deep learning era.
- [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385) — Residual connections that enabled very deep networks.
- [Batch Normalization](https://arxiv.org/abs/1502.03167) — Normalizing activations to accelerate and stabilize training.
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — The default optimizer for most deep learning workloads.
- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) — Rich Sutton's essay on why general methods that scale with compute win.

### Transformers & LLMs

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — The Transformer architecture; the single most important paper for modern AI accelerators.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar's visual walkthrough of the architecture.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP's line-by-line implementation in PyTorch.
- [BERT](https://arxiv.org/abs/1810.04805) — Bidirectional pretraining that reshaped NLP.
- [GPT-2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Scaling generative pretraining toward zero-shot task transfer.
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — Demonstrated emergent in-context learning at scale.
- [T5: Exploring the Limits of Transfer Learning](https://arxiv.org/abs/1910.10683) — The unified text-to-text framing of NLP tasks.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Kaplan et al.'s power-law relationships between compute, data, and loss.
- [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) — The compute-optimal scaling result that reset training recipes.
- [LLaMA](https://arxiv.org/abs/2302.13971) — Efficient open foundation models that catalyzed the open LLM ecosystem.
- [Mistral 7B](https://arxiv.org/abs/2310.06825) — Grouped-query and sliding-window attention for efficient inference.
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) — A 671B fine-grained MoE model trained efficiently with FP8 and MLA.
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — Eliciting strong reasoning through large-scale reinforcement learning.
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy's minimal, hackable GPT training and inference codebase.
- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — Hugging Face's practical guide to training LLMs on GPU clusters (data, tensor, pipeline, and expert parallelism).
- [The Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) — Hugging Face's end-to-end field guide to building world-class LLMs.

### Diffusion Models

- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239) — The paper that made diffusion models practical.
- [High-Resolution Image Synthesis with Latent Diffusion (Stable Diffusion)](https://arxiv.org/abs/2112.10752) — Diffusion in latent space for efficient, high-quality generation.
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — The guidance technique underlying modern text-to-image models.
- [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — Replacing the U-Net backbone with transformers.
- [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — Lilian Weng's comprehensive tutorial.

### Vision Models

- [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929) — Applying pure transformers to image classification.
- [Segment Anything (SAM)](https://arxiv.org/abs/2304.02643) — A promptable foundation model for image segmentation.
- [CLIP: Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020) — Contrastive language-image pretraining that bridges vision and text.
- [YOLO](https://arxiv.org/abs/1506.02640) — Real-time object detection as a single regression problem.

### Reinforcement Learning

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) — Sutton and Barto's canonical textbook, freely available.
- [Playing Atari with Deep RL (DQN)](https://arxiv.org/abs/1312.5602) — Deep Q-networks that learned to play from pixels.
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) — The workhorse policy-gradient algorithm, now central to RLHF.
- [Spinning Up in Deep RL](https://spinningup.openai.com/) — OpenAI's educational resource with clean implementations.

---

## AI Inference

Inference — not training — dominates the lifetime compute of a deployed model. This section covers the algorithms and systems that make LLM serving fast and cheap.

### Quantization

- [TurboQuant](https://arxiv.org/abs/2504.19874) — Online vector quantization with near-optimal distortion, delivering large KV cache memory savings and attention speedups.
- [LLM.int8()](https://arxiv.org/abs/2208.07339) — 8-bit matrix multiplication for transformers with outlier handling.
- [GPTQ](https://arxiv.org/abs/2210.17323) — Accurate post-training quantization to 3–4 bits.
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) — Protecting salient weights for low-bit LLM inference.
- [SmoothQuant](https://arxiv.org/abs/2211.10438) — Migrating quantization difficulty from activations to weights.
- [QLoRA](https://arxiv.org/abs/2305.14314) — 4-bit NormalFloat quantization enabling fine-tuning on a single GPU.
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) — The NVIDIA/Arm/Intel proposal behind Hopper's FP8 support.
- [A Survey of Quantization Methods](https://arxiv.org/abs/2103.13630) — Comprehensive survey of efficient neural network inference.

### KV Cache

- [Efficient Memory Management for LLM Serving (PagedAttention)](https://arxiv.org/abs/2309.06180) — The vLLM paper; treats KV cache like OS virtual memory.
- [Multi-Query Attention](https://arxiv.org/abs/1911.02150) — Sharing keys and values across heads to shrink the KV cache.
- [GQA: Grouped-Query Attention](https://arxiv.org/abs/2305.13245) — Interpolating between multi-head and multi-query attention.
- [H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048) — KV cache eviction based on attention scores.
- [KVQuant](https://arxiv.org/abs/2401.18079) — Ultra-low-bit KV cache quantization for long context.
- [DeepSeek-V2 (Multi-head Latent Attention)](https://arxiv.org/abs/2405.04434) — Low-rank KV compression that dramatically cuts cache size.

### Speculative Decoding

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — The original speculative decoding paper from Google.
- [Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — DeepMind's concurrent formulation.
- [Medusa](https://arxiv.org/abs/2401.10774) — Multiple decoding heads for parallel speculation without a draft model.
- [EAGLE](https://arxiv.org/abs/2401.15077) — Speculative sampling at the feature level for higher acceptance rates.
- [Lookahead Decoding](https://arxiv.org/abs/2402.02057) — Breaking the sequential dependency without a draft model.

### Prefix Caching

- [SGLang: Efficient Execution with RadixAttention](https://arxiv.org/abs/2312.07104) — Automatic KV reuse across requests via a radix tree.
- [Prompt Cache](https://arxiv.org/abs/2311.04934) — Modular attention reuse for recurring text segments.
- [ChunkAttention](https://arxiv.org/abs/2402.15220) — Prefix-aware KV cache with a shared prefix trie.

### Continuous Batching

- [Orca: A Distributed Serving System for Transformers](https://www.usenix.org/conference/osdi22/presentation/yu) — Introduced iteration-level (continuous) batching at OSDI '22.
- [How Continuous Batching Enables 23x Throughput](https://www.anyscale.com/blog/continuous-batching-llm-inference) — Anyscale's widely cited explainer.
- [DistServe](https://arxiv.org/abs/2401.09670) — Disaggregating prefill and decode for goodput-optimal serving.
- [Sarathi-Serve](https://arxiv.org/abs/2403.02310) — Chunked prefills that balance throughput and latency.

### Mixture of Experts

- [Outrageously Large Neural Networks (Sparsely-Gated MoE)](https://arxiv.org/abs/1701.06538) — Shazeer et al.'s foundational MoE layer.
- [Switch Transformers](https://arxiv.org/abs/2101.03961) — Simplified routing scaling to trillion-parameter models.
- [GShard](https://arxiv.org/abs/2006.16668) — Scaling MoE with automatic sharding.
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — A strong open sparse MoE model.
- [MegaBlocks](https://arxiv.org/abs/2211.15841) — Block-sparse GPU kernels for dropless MoE training.
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) — Fine-grained MoE with auxiliary-loss-free load balancing at scale.

### Sparsity & Pruning

- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) — Sparse subnetworks that train to full accuracy.
- [Learning both Weights and Connections (Deep Compression)](https://arxiv.org/abs/1506.02626) — Han et al.'s pruning + quantization + Huffman coding pipeline.
- [SparseGPT](https://arxiv.org/abs/2301.00774) — One-shot pruning of massive language models.
- [Wanda](https://arxiv.org/abs/2306.11695) — Pruning by weights and activations, no retraining required.
- [Accelerating Sparse DNNs with 2:4 Sparsity](https://arxiv.org/abs/2104.08378) — NVIDIA's structured sparsity for Tensor Cores.

### Scheduling

- [FastServe](https://arxiv.org/abs/2305.05920) — Preemptive scheduling to minimize job completion time.
- [Llumnix](https://arxiv.org/abs/2406.03243) — Dynamic request rescheduling across serving instances.
- [Splitwise](https://arxiv.org/abs/2311.18677) — Phase-splitting prefill and decode across heterogeneous hardware.

### Memory Optimization

- [ZeRO: Memory Optimizations Toward Training Trillion-Parameter Models](https://arxiv.org/abs/1910.02054) — Partitioning optimizer state, gradients, and parameters.
- [FlexGen](https://arxiv.org/abs/2303.06865) — High-throughput generative inference with offloading on a single GPU.
- [Reducing Activation Recomputation (Selective Checkpointing)](https://arxiv.org/abs/2205.05198) — Trading compute for memory in large transformers.
- [InfiniGen](https://arxiv.org/abs/2406.19707) — Dynamic KV cache management with speculative prefetching.

### Serving Systems

- [DeepSpeed-Inference](https://arxiv.org/abs/2207.00032) — Multi-GPU inference optimizations for transformer models.
- [AlpaServe](https://arxiv.org/abs/2302.11665) — Statistical multiplexing with model parallelism for serving.
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) — Google's partitioning analysis for TPU inference at scale.
- [Mooncake](https://arxiv.org/abs/2407.00079) — KVCache-centric disaggregated architecture behind Kimi.

---

## GPU Programming

The lingua franca of accelerator programming. This section covers the major parallel programming models.

### CUDA

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — The authoritative reference from NVIDIA.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — Performance optimization guidance straight from the source.
- [Programming Massively Parallel Processors](https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) — Hwu, Kirk, and El Hajj's definitive CUDA textbook (PMPP).
- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) — Simon Boehm's step-by-step walk to near-cuBLAS performance.
- [GPU MODE Lectures](https://github.com/gpu-mode/lectures) — Community lecture series and code on writing fast GPU kernels.
- [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA's CUDA templates for high-performance GEMM and beyond.

### Triton

- [Triton: An Intermediate Language and Compiler](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) — The original paper by Philippe Tillet.
- [Triton](https://github.com/triton-lang/triton) — OpenAI's Python DSL for writing efficient GPU kernels.
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) — Official tutorials from vector-add to fused attention.

### ROCm

- [ROCm Documentation](https://rocm.docs.amd.com/) — AMD's open compute platform for GPU acceleration.
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/) — CUDA-portable C++ runtime API for AMD and NVIDIA GPUs.
- [Composable Kernel](https://github.com/ROCm/composable_kernel) — AMD's performance library for GEMM and fused operators.

### Metal

- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders) — Apple's optimized GPU compute primitives.
- [MLX](https://github.com/ml-explore/mlx) — Apple's NumPy-like array framework for Apple silicon.
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) — The reference for writing Metal compute kernels.

---

## AI Kernels

The hand-tuned building blocks where most accelerator FLOPs are spent.

### GEMM

- [Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf) — Goto and van de Geijn's classic on blocking for the memory hierarchy.
- [CUTLASS](https://github.com/NVIDIA/cutlass) — Composable GEMM templates that expose Tensor Core performance.
- [How To Optimize GEMM](https://github.com/flame/how-to-optimize-gemm) — A pedagogical, step-by-step optimization of matrix multiply.
- [CuTe Layout Algebra](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md) — CUTLASS 3.x's abstraction for tensor layouts and tiling.

### FlashAttention

- [FlashAttention](https://arxiv.org/abs/2205.14135) — IO-aware exact attention that avoids materializing the score matrix.
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) — Better parallelism and work partitioning for near-peak throughput.
- [FlashAttention-3](https://arxiv.org/abs/2407.08608) — Exploiting Hopper asynchrony and FP8 for the H100.
- [flash-attention](https://github.com/Dao-AILab/flash-attention) — The reference implementation from Tri Dao's lab.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — A kernel library for LLM serving with attention variants.

### PagedAttention

- [PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) — Non-contiguous KV cache blocks to eliminate memory fragmentation.
- [vLLM Kernels](https://github.com/vllm-project/vllm/tree/main/csrc) — Production PagedAttention and quantized kernel implementations.

### Normalization & Softmax

- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm, the normalization used in most modern LLMs.
- [Layer Normalization](https://arxiv.org/abs/1607.06450) — Ba, Kiros, and Hinton's original formulation.
- [Online Normalizer Calculation for Softmax](https://arxiv.org/abs/1805.02867) — The single-pass softmax trick underlying FlashAttention.

### RoPE & Positional Encodings

- [RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE, the positional encoding used across modern LLMs.
- [YaRN](https://arxiv.org/abs/2309.00071) — Efficient context-window extension for RoPE models.
- [ALiBi](https://arxiv.org/abs/2108.12409) — Attention with linear biases for length extrapolation.

### Fused Operators

- [Automatic Kernel Fusion (TVM)](https://arxiv.org/abs/1802.04799) — Operator fusion as part of an end-to-end DL compiler.
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) — Fused Triton kernels for efficient LLM training.
- [NVIDIA Apex](https://github.com/NVIDIA/apex) — Fused Adam, LayerNorm, and other mixed-precision training kernels.

### Kernel Optimization Guides

- [GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html) — NVIDIA's model for reasoning about arithmetic intensity and roofline.
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) — Horace He on compute, memory, and overhead bound regimes.
- [What Every Programmer Should Know About Memory](https://www.akkadia.org/drepper/cpumemory.pdf) — Ulrich Drepper's essential treatment of the memory hierarchy.

---

## Runtime Systems

Frameworks that take a trained model and run it efficiently on real hardware.

- [vLLM](https://github.com/vllm-project/vllm) — High-throughput LLM serving engine built around PagedAttention.
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — NVIDIA's optimized LLM inference library with in-flight batching.
- [SGLang](https://github.com/sgl-project/sglang) — Fast serving with RadixAttention and a co-designed frontend language.
- [MLC-LLM](https://github.com/mlc-ai/mlc-llm) — Universal LLM deployment across GPUs, phones, and browsers via TVM.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — Portable C/C++ inference with aggressive quantization (GGUF).
- [Ollama](https://github.com/ollama/ollama) — Simple local model running built on llama.cpp.
- [TensorRT](https://github.com/NVIDIA/TensorRT) — NVIDIA's high-performance deep learning inference SDK.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — Cross-platform inference and training accelerator.
- [Apache TVM](https://github.com/apache/tvm) — End-to-end deep learning compiler stack (see [Compilers](#compilers)).
- [tinygrad](https://github.com/tinygrad/tinygrad) — A minimalist framework with a lazy, fused execution model.
- [ExecuTorch](https://github.com/pytorch/executorch) — PyTorch's on-device inference runtime for mobile and edge.
- [OpenXLA / XLA](https://github.com/openxla/xla) — The domain-specific compiler behind JAX and TensorFlow.
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) — Microsoft's training and inference optimization library.
- [LMDeploy](https://github.com/InternLM/lmdeploy) — Toolkit for compressing and serving LLMs.
- [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) — Hugging Face's production serving toolkit.

---

## Compilers

The bridge between ML frameworks and silicon.

- [MLIR](https://mlir.llvm.org/) — Multi-Level IR; the shared compiler infrastructure behind most modern AI stacks.
- [MLIR: Scaling Compiler Infrastructure (paper)](https://arxiv.org/abs/2002.11054) — Lattner et al.'s introduction to the framework.
- [LLVM](https://llvm.org/) — The compiler infrastructure that underpins essentially everything below.
- [TVM: An Automated End-to-End Optimizing Compiler](https://www.usenix.org/conference/osdi18/presentation/chen) — The Apache TVM paper.
- [IREE](https://github.com/iree-org/iree) — An MLIR-based end-to-end compiler and runtime for ML models.
- [OpenXLA](https://github.com/openxla/xla) — XLA and StableHLO, the compiler ecosystem for JAX/TF/PyTorch.
- [Triton Compiler](https://github.com/triton-lang/triton) — Lowers a Python tile DSL to fast GPU code through MLIR.
- [torch.compile / TorchInductor](https://pytorch.org/get-started/pytorch-2.0/) — PyTorch 2.x's graph capture and Triton-based codegen.
- [Glow](https://github.com/pytorch/glow) — A graph-lowering compiler for neural network hardware accelerators.
- [Mojo](https://www.modular.com/mojo) — Modular's Python-superset language for AI systems programming.
- [Halide](https://halide-lang.org/) — Decoupling algorithm from schedule for high-performance tensor and imaging code.
- [The Deep Learning Compiler: A Comprehensive Survey](https://arxiv.org/abs/2002.03794) — A map of the DL compiler landscape.

---

## Computer Architecture

The foundation every accelerator is built on.

### Foundations

- [Computer Architecture: A Quantitative Approach](https://www.elsevier.com/books/computer-architecture/hennessy/978-0-12-811905-1) — Hennessy and Patterson's definitive graduate text.
- [Computer Organization and Design](https://www.elsevier.com/books/computer-organization-and-design-risc-v-edition/patterson/978-0-12-820331-6) — Patterson and Hennessy's undergraduate RISC-V edition.
- [Processor Microarchitecture: An Implementation Perspective](https://dl.icdst.org/pdfs/files/15b09def448c317556dc0fc412aee571.pdf) — González, Latorre, and Magklis's concise tour of how out-of-order cores are actually built.
- [A New Golden Age for Computer Architecture](https://cacm.acm.org/magazines/2019/2/234352-a-new-golden-age-for-computer-architecture/fulltext) — Hennessy and Patterson's Turing Award lecture on domain-specific architectures.
- [Onur Mutlu's Computer Architecture Lectures](https://safari.ethz.ch/architecture/) — ETH Zürich's comprehensive, freely available course.

### Memory Systems & Cache

- [What Every Programmer Should Know About Memory](https://www.akkadia.org/drepper/cpumemory.pdf) — The reference on caches, DRAM, and memory-aware programming.
- [High Bandwidth Memory (HBM) Standard](https://www.jedec.org/standards-documents/docs/jesd235a) — The JEDEC standard powering modern accelerator memory.
- [DRAMSim3](https://github.com/umd-memsys/DRAMsim3) — A cycle-accurate DRAM simulator for architecture research.
- [Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2) — A fast, extensible DRAM simulator for modern memory standards.

### Interconnect & NoC

- [Principles and Practices of Interconnection Networks](https://www.elsevier.com/books/principles-and-practices-of-interconnection-networks/dally/978-0-12-200751-4) — Dally and Towles's definitive text.
- [NVLink and NVSwitch](https://www.nvidia.com/en-us/data-center/nvlink/) — NVIDIA's high-bandwidth GPU-to-GPU interconnect.
- [BookSim2](https://github.com/booksim/booksim2) — A cycle-accurate interconnection network simulator.
- [Ultra Ethernet Consortium](https://ultraethernet.org/) — An emerging open standard for scale-out AI fabrics.

### Dataflow & Systolic Arrays

- [Why Systolic Architectures?](https://www.eecs.harvard.edu/~htk/publication/1982-kung-why-systolic-architecture.pdf) — H. T. Kung and Charles Leiserson's foundational 1982 work.
- [Eyeriss: Energy-Efficient Reconfigurable Accelerator](https://ieeexplore.ieee.org/document/7738524) — Chen et al.'s influential dataflow accelerator and taxonomy.
- [Efficient Processing of Deep Neural Networks](https://link.springer.com/book/10.1007/978-3-031-01766-7) — Sze, Chen, Yang, and Emer's essential survey book.
- [Timeloop / Accelergy](https://github.com/NVlabs/timeloop) — Infrastructure for evaluating and mapping DNN accelerator dataflows.
- [MAESTRO](https://github.com/maestro-project/maestro) — An analytical cost model for DNN dataflows.

### SIMD, SIMT & Tensor Cores

- [Volta Architecture Whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf) — The introduction of Tensor Cores.
- [Dissecting the NVIDIA Volta GPU via Microbenchmarking](https://arxiv.org/abs/1804.06826) — A detailed reverse-engineering of Tensor Core behavior.
- [Dissecting Tensor Cores via Microbenchmarks](https://arxiv.org/abs/2206.02874) — Programmability, performance, and precision across generations.
- [Modal GPU Glossary](https://modal.com/gpu-glossary) — A clear reference for SIMT execution and GPU terminology.

### Processing-in-Memory

- [A Modern Primer on Processing in Memory](https://arxiv.org/abs/2012.03112) — Mutlu et al.'s comprehensive introduction to PIM.
- [PRIME: A Novel PIM Architecture for Neural Network Computation](https://ieeexplore.ieee.org/document/7551380) — ReRAM-based neural network acceleration in memory.
- [Samsung HBM-PIM (Aquabolt-XL)](https://ieeexplore.ieee.org/document/9731711) — A commercialized processing-in-memory design.

---

## AI Accelerator Architecture

The chips themselves, organized by class.

### GPUs

- [NVIDIA Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core) — H100: FP8, the Transformer Engine, and TMA.
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) — The GB200/B200 generation for trillion-parameter models.
- [AMD CDNA Architecture](https://www.amd.com/en/technologies/cdna.html) — The compute architecture behind the MI300 series.
- [Dissecting the Ampere GPU Architecture](https://arxiv.org/abs/2402.13499) — Microbenchmark-driven analysis of modern GPU internals.

### TPUs & Systolic ASICs

- [In-Datacenter Performance Analysis of a TPU (TPUv1)](https://arxiv.org/abs/1704.04760) — Jouppi et al.'s landmark ISCA '17 paper.
- [A Domain-Specific Supercomputer for Training DNNs (TPUv3/v4)](https://cacm.acm.org/magazines/2020/7/245702-a-domain-specific-supercomputer-for-training-deep-neural-networks/fulltext) — Google's system-level TPU design.
- [TPU v4: An Optically Reconfigurable Supercomputer](https://arxiv.org/abs/2304.01433) — Optical circuit switching for large-scale training.
- [Ten Lessons From Three Generations of TPUs](https://ieeexplore.ieee.org/document/9138922) — Jouppi and Patterson on hardware-software co-design.

### NPUs & Edge AI

- [Apple Machine Learning Research](https://machinelearning.apple.com/) — Publications on the Apple Neural Engine and on-device ML.
- [Coral / Edge TPU](https://coral.ai/) — Google's edge inference accelerator.
- [MLPerf Tiny / TinyML](https://github.com/mlcommons/tiny) — Benchmarks and models for microcontroller-class inference.
- [Efficient Deep Learning (MIT 6.5940 / TinyML)](https://efficientml.ai/) — Song Han's course on model compression and edge deployment.
- [A Survey of Accelerator Architectures for DNNs](https://ieeexplore.ieee.org/document/9060877) — Broad coverage of the accelerator design space.

### FPGA & Reconfigurable

- [FINN](https://github.com/Xilinx/finn) — AMD/Xilinx's framework for quantized neural network inference on FPGAs.
- [hls4ml](https://github.com/fastmachinelearning/hls4ml) — Translating ML models to FPGA firmware via high-level synthesis.
- [Vitis AI](https://github.com/Xilinx/Vitis-AI) — AMD/Xilinx's development stack for FPGA/ACAP inference.
- [Brainwave: A Configurable Cloud-Scale DNN Processor](https://www.microsoft.com/en-us/research/publication/serving-dnns-real-time-datacenter-scale-project-brainwave/) — Microsoft's real-time FPGA AI serving.

### RISC-V for AI

- [RISC-V Vector Extension (RVV)](https://github.com/riscv/riscv-v-spec) — The vector ISA extension for data-parallel workloads.
- [RISC-V Specifications](https://riscv.org/technical/specifications/) — The open ISA underpinning much accelerator research.
- [Ara](https://github.com/pulp-platform/ara) — A RISC-V vector processor from ETH Zürich's PULP platform.

---

## Simulators & Modeling

Tools for exploring the design space before committing to silicon.

- [gem5](https://www.gem5.org/) — The de facto community architecture simulator.
- [Accel-Sim](https://accel-sim.github.io/) — A framework for simulating and validating GPU architectures.
- [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) — Cycle-level simulation of NVIDIA GPUs running CUDA/OpenCL.
- [SCALE-Sim](https://github.com/ARM-software/SCALE-Sim) — ARM's systolic-array CNN accelerator simulator.
- [Timeloop / Accelergy](https://github.com/NVlabs/timeloop) — Mapping and energy/area modeling for DNN accelerators.
- [Sparseloop](https://github.com/Accelergy-Project/sparseloop) — Analytical modeling for sparse tensor accelerators.
- [ASTRA-sim](https://github.com/astra-sim/astra-sim) — Distributed ML training platform and network co-design simulator.
- [LLMCompass](https://github.com/PrincetonUniversity/LLMCompass) — Hardware evaluation for LLM inference.
- [Roofline: An Insightful Visual Performance Model](https://dl.acm.org/doi/10.1145/1498765.1498785) — Williams, Waterman, and Patterson's foundational model.

---

## Chip Design

Following the people and publications that explain how chips are actually designed, built, and brought to market — the industry context that connects everything above.

### Blogs & Newsletters

- [SemiAnalysis](https://semianalysis.com/) — Dylan Patel's deep dives on the semiconductor industry, AI hardware, and the supply chain.
- [The EDA Primer: From RTL to Silicon](https://newsletter.semianalysis.com/p/the-eda-primer-from-rtl-to-silicon) — SemiAnalysis's end-to-end walkthrough of the chip design flow; the single best starting point for the big picture.

### YouTube Channels

- [Asianometry](https://www.youtube.com/@Asianometry) — Accessible, well-researched videos on semiconductor history, manufacturing, and the industry.
- [CalebWritesCode](https://www.youtube.com/@CalebWritesCode) — Hands-on GPU programming and low-level systems tutorials.
- [Dwarkesh Patel](https://www.youtube.com/@DwarkeshPatel) — Long-form interviews with leading figures in AI and hardware.
- [3Blue1Brown](https://www.youtube.com/@3blue1brown) — Grant Sanderson's visual explanations of the math and neural networks behind the workloads.

### Interactive & Visual

- [brrrviz](https://brrrviz.com/) — An interactive visualization of GPU architecture and how modern GPUs go brrr.

---

## RTL Design

Register-transfer-level design in Verilog and SystemVerilog — the entry point to any accelerator implementation. Start with the textbook, follow the lectures, then build fluency on the practice sites.

- **Textbook** — [Digital Design and Computer Architecture](https://www.elsevier.com/books/digital-design-and-computer-architecture-risc-v-edition/harris/978-0-12-820064-3), Harris & Harris. The standard undergraduate text taking you from logic gates to a working RISC-V processor.
- **Lectures** — [Onur Mutlu — Digital Design & Computer Architecture (ETH Zürich)](https://safari.ethz.ch/digitaltechnik/). A complete, freely recorded lecture series widely regarded as the best online course on the subject.
- **Practice** — [HDLBits](https://hdlbits.01xz.net/). Interactive Verilog problems with instant feedback; the single most effective way to build RTL coding fluency.
- **Reference** — [ASIC World](https://www.asic-world.com/verilog/veritut.html). A free, comprehensive Verilog/SystemVerilog tutorial and syntax reference to keep open while you code.

---

## Design Verification

Confirming the RTL does what it should — typically the majority of the effort on a real chip. Learn the methodology from the textbook, then practice UVM on the training sites.

- **Textbook** — [SystemVerilog for Verification](https://link.springer.com/book/10.1007/978-1-4614-0715-7), Spear & Tumbush. The definitive introduction to the SystemVerilog features and testbench techniques used in industry.
- **Lectures & Resources** — [Verification Academy](https://verificationacademy.com/). Siemens EDA's free hub of courses, the canonical UVM Cookbook, and methodology guides — the go-to reference for DV.
- **Training** — [ChipVerify](https://www.chipverify.com/). Free, approachable SystemVerilog and UVM tutorials with worked examples, ideal for getting members productive quickly.
- **Hands-on** — [cocotb](https://www.cocotb.org/). Write testbenches in Python instead of SystemVerilog; the most accessible on-ramp to writing real verification.

---

## Physical Design

Turning a synthesized netlist into a manufacturable, timing-clean layout (synthesis → floorplan → placement → CTS → routing → timing/signoff). Ground yourself in the textbooks, then run a full flow hands-on.

- **Textbook** — [VLSI Physical Design: From Graph Partitioning to Timing Closure](https://link.springer.com/book/10.1007/978-90-481-9591-6), Kahng, Lienig, Markov & Hu. The standard text covering every stage of the physical design flow.
- **Textbook (Circuits)** — [CMOS VLSI Design: A Circuits and Systems Perspective](https://pages.hmc.edu/harris/cmosvlsi/4e/index.html), Weste & Harris. The classic bridge from transistors and layout to full VLSI systems.
- **Textbook (Timing)** — [Static Timing Analysis for Nanometer Designs](https://link.springer.com/book/10.1007/978-0-387-93820-2), Bhasker & Chadha. The definitive reference for understanding timing closure.
- **Lectures** — [NPTEL — VLSI Physical Design Automation](https://nptel.ac.in/courses/106105161). A complete, freely available lecture course on the algorithms behind each PD stage.
- **Training** — [VLSI System Design (VSD)](https://www.vlsisystemdesign.com/). Popular hands-on courses and workshops that walk you through a real RTL-to-GDSII flow.
- **Hands-on** — [OpenLane](https://github.com/The-OpenROAD-Project/OpenLane). An open, automated RTL-to-GDSII flow on the Sky130 PDK — the best way to actually run physical design end-to-end for free.

---

## University Courses

The best openly available courses across the stack.

- [Stanford CS217 — Hardware Accelerators for ML](https://cs217.stanford.edu/) — Systems and hardware for machine learning.
- [Stanford CS149 — Parallel Computing](https://gfxcourses.stanford.edu/cs149) — Kayvon Fatahalian's parallel programming course.
- [MIT 6.5940 — TinyML & Efficient Deep Learning](https://efficientml.ai/) — Song Han's course on compression, quantization, and acceleration.
- [MIT 6.172 — Performance Engineering](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/) — Squeezing performance out of modern hardware.
- [CMU 15-418 — Parallel Computer Architecture](https://www.cs.cmu.edu/~418/) — The classic parallelism course.
- [CMU 10-414/714 — Deep Learning Systems](https://dlsyscourse.org/) — Build a deep learning framework from scratch.
- [Berkeley CS152/252 — Computer Architecture](https://inst.eecs.berkeley.edu/~cs152/) — Krste Asanović's architecture sequence.
- [ETH Zürich — Digital Design & Computer Architecture](https://safari.ethz.ch/digitaltechnik/) — Onur Mutlu's freely recorded lectures.
- [UW CSE 599W — Systems for ML](https://dlsys.cs.washington.edu/) — Tianqi Chen's ML systems course.

