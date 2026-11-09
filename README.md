# Deep Dive: CUDA Vector Dot Product (SIMD vs. SIMT Performance Analysis)

## Project Metadata & AI Usage Declaration

### Project Specifications
* **Problem Statement:** Vector Dot Product $Z = \sum_{i=0}^{n-1} (A_i \cdot B_i)$ using double-precision floating-point numbers (`double`).
* **Data Initialization:**
  * $A[i] = \sin(i \cdot 0.001)$
  * $B[i] = \cos(i \cdot 0.002)$
* **Target Vector Sizes ($N$):**
  * $N = 2^{20} = 1,048,576$ elements
  * $N = 2^{24} = 16,777,216$ elements
  * $N = 2^{24} + 13 = 16,777,229$ elements (Boundary Condition Test)
* **Historical SIMD Benchmark Sizes ($N$):** $N = 1,003$ (Boundary), $2^{20}, 2^{26}, 2^{27}$ (Max Machine Capacity).
* **Tested CUDA Block Sizes:** 256, 512, and 1024 threads/block.
* **Benchmarking Methodology:** 30 execution iterations per kernel, taking the Average, Minimum, and Maximum execution times. Timings are recorded using high-precision host timers (`std::chrono` / `QueryPerformanceCounter`) for C/SIMD baselines and GPU hardware timers (`cudaEvent_t`) for CUDA variants.
* Project Notebook Link: https://colab.research.google.com/drive/1gl5WeIHh_IoWSQRZvWElBERexvX0hNEU?usp=sharing

### Group Members & Task Allocation
| Group Member | Core Responsibilities | Variants / Deliverables | 
| ----- | ----- | ----- | 
| **Gabriel Infante** | Colab Framework Setup, Data Initialization, Baseline & Variant 2 | C Baseline & Variant 2 (No Prefetch/Advise) | 
| **Chelsea Dimaunahan** | Unified Memory Prefetching Integration | Variant 3 (Prefetch, No Page Create/Advise) | 
| **Marion Melanio** | Physical Page Creation & Nsight Hardware Profiling | Variant 4 (Prefetch + Page Create) & `nsys`/`ncu` Traces | 
| **Sofia Maravilla** | Memory Advice Optimizations, README Report & Analysis | Variant 5 (Prefetch + Page Create + MemAdvise) & Final Report | 

### AI Usage Declaration
* In accordance with course policies, AI was not used in the development of any code in this Deep Dive project. It was only used in formatting assistance for the notebook
* Tool used: Google Gemini

## 0. Screenshots
### A. Correctness
#### Variant 2
<img width="1350" height="354" alt="image" src="https://github.com/user-attachments/assets/54c6ec83-0d14-4ea6-95d8-ad995f7e1e31" />

#### Variant 3
<img width="1960" height="394" alt="image" src="https://github.com/user-attachments/assets/b8d74cb6-81a3-4008-ab1f-f7cb368a8550" />

#### Variant 4
<img width="1968" height="392" alt="image" src="https://github.com/user-attachments/assets/9b6bcb42-9229-42c1-acf2-d89d0f1cca53" />

#### Variant 5
<img width="2332" height="304" alt="image" src="https://github.com/user-attachments/assets/900a5cc7-1422-40e6-95e3-8cb394c8c5df" />

### B. NSYS Results
#### Variant 2
<img width="2514" height="916" alt="image" src="https://github.com/user-attachments/assets/0d12ceb2-3576-4ef8-af6e-aab108659e0c" />
<img width="2268" height="298" alt="image" src="https://github.com/user-attachments/assets/23d53e12-92aa-4949-ab8c-ee0825eb5c9d" />

#### Variant 3
<img width="2506" height="1016" alt="image" src="https://github.com/user-attachments/assets/389f27e5-d61a-4239-81cf-c17fc9528226" />
<img width="2252" height="328" alt="image" src="https://github.com/user-attachments/assets/fabdf8b9-d184-4091-8de2-f28c8d90d655" />

#### Variant 4
<img width="2506" height="1024" alt="image" src="https://github.com/user-attachments/assets/9e3d657f-af33-4765-a230-724b8a70141d" />
<img width="2258" height="330" alt="image" src="https://github.com/user-attachments/assets/9db8b54d-f814-4db3-b8f0-2d2c599ec4c5" />

#### Variant 5
<img width="2172" height="1090" alt="image" src="https://github.com/user-attachments/assets/f71e0567-4d7c-4e3f-8051-e9d620758673" />

## 1. Comparative Performance Results

### i. Historical SIMD Benchmark Dataset (x86-64 Architecture)
*Data extracted from prior CPU SIMD evaluation (Release Build, 4-Accumulator Unrolled Strategy).*

| Vector Size ($N$) | Case Type | C Baseline (ms) | x86-64 Scalar (ms) | SIMD XMM (128-bit) (ms) | SIMD YMM (256-bit) (ms) | Speedup (YMM vs C) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1,003** | Boundary Check | 0.001000 | 0.000300 | 0.000100 | 0.000100 | **10.00x** * |
| **1,048,576** ($2^{20}$) | Medium Vector | 1.152100 | 0.775200 | 0.704600 | 0.665100 | **1.73x** |
| **67,108,864** ($2^{26}$) | Large Vector | 102.980500 | 91.830100 | 57.741100 | 57.148900 | **1.80x** |
| **134,217,728** ($2^{27}$) | Max Memory Limit | 226.291400 | 126.422900 | **117.161300** | **128.957900** | **1.75x** ** |

---

### ii. Integrated CUDA vs. SIMD Benchmark Tables

#### Table 1: Standard Vector Size ($N = 2^{20} = 1,048,576$ elements)

| Platform / Kernel Variant | Block Size | Avg Time (ms) | Min Time (ms) | Max Time (ms) | Speedup vs C Baseline | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| **Variant 1: C Baseline (Host CPU)** | N/A | **1.412200** | -- | -- | **1.00x** | 
| **x86-64 Scalar Assembly** | N/A | **0.775200** | -- | -- | **1.82x** | 
| **x86-64 SIMD XMM (128-bit)** | N/A | **0.704600** | -- | -- | **2.00x** | 
| **x86-64 SIMD YMM (256-bit)** | N/A | **0.665100** | -- | -- | **2.12x** | 
| **Variant 2: CUDA Unified (No Prefetch/Advise)** | 256 | **0.3546** | **0.2073** | **4.5559** | **3.98** | 
|  | 512 | **0.3965** | **0.2599** | **4.2727** | **3.65** | 
|  | 1024 | **0.5502** | **0.3368** | **6.6428** | **2.56** | 
| **Variant 3: CUDA Prefetch** | 256 | **0.1556** | **0.1500** | **0.2724** | **9.07** | 
|  | 512 | **0.1881** | **0.1864** | **0.2007** | **7.50** | 
|  | 1024 | **0.2464** | **0.2443** | **0.2546** | **5.73** | 
| **Variant 4: CUDA Prefetch + Page Creation** | 256 | **0.2254** | **0.2164** | **0.3841** | **6.26** | 
|  | 512 | **0.2729** | **0.2703** | **0.2911** | **5.17** | 
|  | 1024 | **0.3531** | **0.3502** | **0.3750** | **3.99** | 
| **Variant 5: CUDA Prefetch + Page + MemAdvise** | 256 | **0.1647** | **0.1630** | **0.1687** | **8.57** | 
|  | 512 | **0.2049** | **0.2028** | **0.2164** | **6.89** | 
|  | 1024 | **0.2672** | **0.2659** | **0.2702** | **5.28** | 

#### Table 2: Large Vector Size ($N = 2^{24} = 16,777,216$ elements)

| Platform / Kernel Variant | Block Size | Avg Time (ms) | Min Time (ms) | Max Time (ms) | Speedup vs C Baseline | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| **Variant 1: C Baseline (Host CPU)** | N/A | **26.990000** | -- | -- | **1.00x** | 
| **x86-64 Scalar Assembly** | N/A | *\[Reference\]* | -- | -- | -- | 
| **x86-64 SIMD XMM (128-bit)** | N/A | *\[Reference\]* | -- | -- | -- | 
| **x86-64 SIMD YMM (256-bit)** | N/A | *\[Reference\]* | -- | -- | -- | 
| **Variant 2: CUDA Unified (No Prefetch/Advise)** | 256 | **6.2073** | **3.5623** | **82.6165** | **4.34** | 
|  | 512 | **4.7965** | **2.7451** | **63.9731** | **5.62** | 
|  | 1024 | **5.1414** | **2.3982** | **84.5654** | **5.24** | 
| **Variant 3: CUDA Prefetch** | 256 | **2.2942** | **2.2913** | **2.3051** | **11.76** | 
|  | 512 | **2.8478** | **2.8428** | **2.8816** | **9.47** | 
|  | 1024 | **2.5978** | **2.5936** | **2.6071** | **10.38** | 
| **Variant 4: CUDA Prefetch + Page Creation** | 256 | **3.8366** | **3.8307** | **3.8821** | **7.03** | 
|  | 512 | **2.5836** | **2.5733** | **2.6215** | **10.44** | 
|  | 1024 | **5.3880** | **5.3829** | **5.4089** | **5.00** | 
| **Variant 5: CUDA Prefetch + Page + MemAdvise** | 256 | **2.4575** | **2.4527** | **0.4626** | **10.98** | 
|  | 512 | **2.3727** | **2.3626** | **2.3888** | **11.37** | 
|  | 1024 | **2.4232** | **2.4167** | **2.4452** | **11.13** | 

#### Table 3: Boundary Test Size ($N = 2^{24} + 13 = 16,777,229$ elements)

| Platform / Kernel Variant | Block Size | Avg Time (ms) | Min Time (ms) | Max Time (ms) | Speedup vs C Baseline | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| **Variant 1: C Baseline (Host CPU)** | N/A | **26.773600** | -- | -- | **1.00x** | 
| **x86-64 SIMD XMM (128-bit)** | N/A | *\[Reference\]* | -- | -- | -- | 
| **x86-64 SIMD YMM (256-bit)** | N/A | *\[Reference\]* | -- | -- | -- | 
| **Variant 2: CUDA Unified (No Prefetch/Advise)** | 256 | **3.7318** | **1.4436** | **69.9654** | **7.17** | 
|  | 512 | **3.8198** | **1.8071** | **61.9108** | **7.00** | 
|  | 1024 | **5.0881** | **2.3956** | **83.0515** | **5.26** | 
| **Variant 3: CUDA Prefetch** | 256 | **1.5808** | **1.5787** | **1.5913** | **16.93** | 
|  | 512 | **2.4517** | **2.3791** | **2.5289** | **10.92** | 
|  | 1024 | **4.0522** | **4.0455** | **4.0569** | **6.60** | 
| **Variant 4: CUDA Prefetch + Page Creation** | 256 | **1.4645** | **1.4508** | **1.4978** | **18.28** | 
|  | 512 | **1.8326** | **1.8221** | **1.8677** | **14.60** | 
|  | 1024 | **4.4482** | **4.2552** | **4.4839** | **6.01** | 
| **Variant 5: CUDA Prefetch + Page + MemAdvise** | 256 | **1.4614** | **1.4493** | **1.4837** | **18.32** | 
|  | 512 | **1.8260** | **1.8195** | **1.8371** | **14.66** | 
|  | 1024 | **2.4081** | **2.4012** | **2.4369** | **11.11** | 

## 2. Program Output & Profiling Screenshots

### i. Program Output & Execution Times

*\[Member 1 & 2: Insert screenshot of Google Colab execution output showing timing logs and PASS correctness outputs\]*

### ii. NVIDIA Nsight Systems (`nsys`) Profiles

* **Variant 2 (`nsys` trace showing GPU page faults):**
  *\[Member 3: Insert screenshot of nsys timeline showing host-to-device page fault stalls\]*

* **Variant 3 (`nsys` trace with prefetching):**
  *\[Member 3: Insert screenshot of nsys timeline showing overlapped HtoD memory transfers\]*

* **Variant 4 (`nsys` trace with prefetching & host page creation):**
  *\[Member 3: Insert screenshot showing eliminated OS allocation stalls\]*

* **Variant 5 (`nsys` trace with prefetching, page creation, and `cudaMemAdvise`):**
  *\[Member 3: Insert screenshot showing optimized Read-Mostly memory pipeline\]*

## 3. Detailed Analysis & Guide Questions

### a.) Impact of Block Size on Execution Time
* **Performance Trend across 256, 512, and 1024:**
  * On NVIDIA architectures (e.g., Turing / Ampere CC 7.5+), a Streaming Multiprocessor (SM) manages execution in warps of 32 threads.
  * **Block Size 256 vs. 512 vs. 1024:** Increasing block size from 256 to 512 often increases resident warp density per SM, allowing the hardware warp scheduler to better hide memory instruction latency. However, increasing to 1024 threads/block limits the SM to only 1 resident block (since the maximum threads per SM is typically 1024). This can reduce thread block scheduling flexibility.
* **Recommended Block Size:** **256 or 512 threads per block** is recommended because it maximizes SM occupancy and block-level parallelism without hitting resource limits on registers per block.

### b.) Prefetching vs. Managed CUDA Memory
* **Is Prefetching Always Recommended?**
  * **Yes, for sequential data access patterns:** In applications like vector dot products where the entire memory buffer is read once sequentially, explicit prefetching (`cudaMemPrefetchAsync`) completely eliminates runtime GPU page faults during kernel execution.
  * **No, for sparse or dynamic access patterns:** In applications with unpredictable, random, or sparse data access (e.g., graph traversals, hash tables), automatic managed memory page faulting is superior because prefetching the entire dataset would waste PCIe bandwidth on unused memory pages.

### c.) SIMD (CPU) vs. SIMT (GPU) Architecture & Trade-Offs
| Metric / Aspect | SIMD (Single Instruction, Multiple Data) | SIMT (Single Instruction, Multiple Threads) | 
| ----- | ----- | ----- | 
| **Processing Unit** | CPU Vector Registers (XMM 128-bit / YMM 256-bit) | Thousands of CUDA Cores grouped in SMs | 
| **Execution Model** | Lock-step vector execution per core | Thread grid mapped to 32-thread warps | 
| **Memory Bandwidth** | Limited by CPU System DRAM (30–60 GB/s) | High VRAM Bandwidth (300–900+ GB/s) | 
| **Data Transfer Overhead** | Minimal (Data resides in system RAM) | High initial overhead (PCIe transfer / Unified page faulting) | 
| **Ideal Workloads** | Low latency, small-to-medium vector sizes | Massive throughput, huge data parallel vectors | 

#### Pros & Cons in Vector Dot Product Use Case:
* **SIMD Pros:** Zero host-to-device transfer latency; faster execution times for small vector sizes ($N \le 2^{20}$).
* **SIMD Cons:** Bandwidth-bound at larger sizes ($N \geq 2^{24}$); limited register width (can process only 2 to 4 `double` values per instruction).
* **SIMT Pros:** Massive parallel compute throughput; easily scales to millions of elements; tree reductions in shared memory hide reduction overhead.
* **SIMT Cons:** High overhead for page migration/prefetching; sensitive to uncoalesced memory accesses and warp divergence.

## 4. Addressing SIMD Review Comments & Methodological Clarifications
1. **Missing Debug Measurements (SIMD Feedback #1):**
   * Debug builds include overhead from disabled compiler optimizations (`-O0`) and stack frame validation checks. Release builds (`-O3`) enable auto-vectorization, loop unrolling, and instruction reordering. In this CUDA report, all C and CUDA variants are strictly compiled in optimized Release mode (`-O3`).
2. **Scalar Assembly Loop Processing (SIMD Feedback #2):**
   * The SIMD scalar assembly kernel does not iterate one element per cycle; it processes **four elements per loop iteration using four independent scalar accumulator registers**. This unrolling breaks instruction data dependencies and increases Instruction-Level Parallelism (ILP).
3. **Fastest-Kernel Statement at $N = 2^{27}$ (SIMD Feedback #3):**
   * As shown in our historical dataset at $N = 2^{27}$, SIMD YMM ($128.96\text{ ms}$) performed slower than both XMM ($117.16\text{ ms}$) and x86-64 Scalar assembly ($126.42\text{ ms}$). Beyond cache capacity (~2.15 GB data), system memory bandwidth saturates, proving wider registers do not establish a universal performance ranking.
4. **Floating-Point Precision & Rounding (SIMD Feedback #4):**
   * Matching printed decimal places is not proof of exact numerical equivalence. Floating-point addition is non-associative: $(a + b) + c \neq a + (b + c)$. GPU tree reduction and FMA (Fused Multiply-Add) instructions change intermediate rounding orders. Verification must be performed using a **relative tolerance check** ($\frac{\vert{}Z_{ref} - Z_{test}\vert{}}{\vert{}Z_{ref}\vert{}} < 10^{-6}$).
5. **Timing Precision & Boundary Speedups (SIMD Feedback #5):**
   * Low-resolution millisecond measurements (e.g., $0.0010\text{ ms}$ vs $0.0001\text{ ms}$) suffer from integer truncation errors. All CUDA execution measurements in this project utilize high-precision hardware timers (`cudaEvent_t`) averaged over 30 iterations to ensure accurate speedup metrics.

## 5. Implementation Challenges, AHA Moments & Methodology
### Implementation Challenges
* **Linux vs. Windows Unified Memory Behavior:** On Windows OS, managed memory does not support on-demand hardware page migration in the same manner as Linux. Running inside Google Colab (Linux environment) enabled proper analysis of hardware page faults and `cudaMemPrefetchAsync`.
* **Host Page Allocation ("Page Creation"):** Lazily allocated `cudaMallocManaged` virtual memory pages are not backed by physical DRAM pages until touched. Implementing explicit host page touching (`touch_host_pages`) before calling `cudaMemPrefetchAsync` eliminated OS page mapping overhead during prefetching.

### AHA Moments
* **Grid-Stride Loops:** Using grid-stride loops inside CUDA kernels (`while (i < n) i += blockDim.x * gridDim.x`) allowed a single kernel launch configuration to seamlessly handle arbitrary vector lengths—including boundary cases like $N = 2^{24} + 13$—without requiring grid dynamic re-dimensioning or out-of-bound crashes.
* **Shared Memory Parallel Tree Reduction:** Atomic addition (`atomicAdd`) directly from every thread causes massive memory contention on global VRAM. Performing block-level parallel tree reductions in fast shared memory (`__shared__`) before issuing a single atomic update per block dramatically improved execution speed.
