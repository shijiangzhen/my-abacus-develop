#include "fft_cpu.h"
#include "fftw3.h"
namespace ModuleBase
{

template <typename FPTYPE>
void FFT_CPU<FPTYPE>::initfft(int nx_in, 
                              int ny_in, 
                              int nz_in, 
                              int lixy_in, 
                              int rixy_in, 
                              int ns_in, 
                              int nplane_in, 
				              int nproc_in, 
                              bool gamma_only_in, 
                              bool xprime_in)
{
    this->gamma_only = gamma_only_in;
    this->xprime = xprime_in;
    this->fftnx = this->nx = nx_in;
    this->fftny = this->ny = ny_in;
    if (this->gamma_only)
    {
        if (xprime) {
            this->fftnx = int(this->nx / 2) + 1;
        } else {
            this->fftny = int(this->ny / 2) + 1;
        }
    }
    this->nz = nz_in;
    this->ns = ns_in;
    this->lixy = lixy_in;
    this->rixy = rixy_in;
    this->nplane = nplane_in;
    this->nproc = nproc_in;
    this->nxy = this->nx * this->ny;
    this->fftnxy = this->fftnx * this->fftny;
    const int nrxx = this->nxy * this->nplane;
    const int nsz = this->nz * this->ns;
    this->maxgrids = (nsz > nrxx) ? nsz : nrxx;
}
template <>
void FFT_CPU<double>::setupFFT()
{
    
    unsigned int flag = FFTW_ESTIMATE;
    switch (this->fft_mode)
    {
    case 0:
        flag = FFTW_ESTIMATE;
        break;
    case 1:
        flag = FFTW_MEASURE;
        break;
    case 2:
        flag = FFTW_PATIENT;
        break;
    case 3:
        flag = FFTW_EXHAUSTIVE;
        break;
    default:
        break;
    }
    z_auxg = (std::complex<double>*)fftw_malloc(sizeof(fftw_complex) * this->maxgrids);
    z_auxr = (std::complex<double>*)fftw_malloc(sizeof(fftw_complex) * this->maxgrids);
    d_rspace = (double*)z_auxg;
    this->planzfor = fftw_plan_many_dft(1, 
                                        &this->nz, 
                                        this->ns, 
                                        (fftw_complex*)z_auxg, 
                                        &this->nz, 
                                        1, 
                                        this->nz,
                                        (fftw_complex*)z_auxg, 
                                        &this->nz, 
                                        1, 
                                        this->nz, 
                                        FFTW_FORWARD, 
                                        flag);
    
/**
 * fftw_plan_many_dft用 FFTW 库创建了一个批量 1D 逆FFT 计划，用于对所有 stick（柱）上的 z 方向数据同时做逆FFT。
 * planzbac参数说明:
 *   - 1: 维度数 (rank)，这里为1，表示进行1D FFT。
 *   - &this->nz: 每个维度的大小数组，指向nz，表示每个FFT的长度为nz。
 *   - this->ns: 执行多少个独立的FFT（batch数），即ns个1D FFT。
 *   - (fftw_complex*)z_auxg: 输入数据的指针。
 *   - &this->nz: 输入embed数组，指定输入数据在每个维度的物理存储跨度。这里为nz，表示数据是连续存储的。
 *   - 1: 输入stride，输入数组中同一FFT不同元素之间的步长。为1表示数据是连续的。
 *   - this->nz: 输入dist，输入数组中不同FFT之间的距离（元素数）。为nz表示每个FFT的数据块大小。
 *   - (fftw_complex*)z_auxg: 输出数据的指针（与输入相同，表示in-place变换）。
 *   - &this->nz: 输出embed数组，指定输出数据在每个维度的物理存储跨度。这里为nz，表示数据是连续存储的。
 *   - 1: 输出stride，输出数组中同一FFT不同元素之间的步长。为1表示数据是连续的。
 *   - this->nz: 输出dist，输出数组中不同FFT之间的距离（元素数）。为nz表示每个FFT的数据块大小。
 *   - FFTW_BACKWARD: 变换方向，表示执行逆FFT。
 *   - flag: FFTW计划标志，决定FFTW如何优化（如FFTW_ESTIMATE、FFTW_MEASURE等）。
 */
    this->planzbac = fftw_plan_many_dft(1, 
                                        &this->nz, 
                                        this->ns, 
                                        (fftw_complex*)z_auxg, 
                                        &this->nz, 
                                        1, 
                                        this->nz,
                                        (fftw_complex*)z_auxg, 
                                        &this->nz, 
                                        1, 
                                        this->nz, 
                                        FFTW_BACKWARD, 
                                        flag);

    //---------------------------------------------------------
    //                              2 D - XY
    //---------------------------------------------------------
    // 1D+1D is much faster than 2D FFT!
    // in-place fft is better for c2c and out-of-place fft is better for c2r
    int* embed = nullptr;
    int npy = this->nplane * this->ny;
    if (this->xprime)
    {
        this->planyfor = fftw_plan_many_dft(1, 
                                            &this->ny, 
                                            this->nplane, 
                                            (fftw_complex*)z_auxr, 
                                            embed,
                                            this->nplane, 
                                            1,
                                            (fftw_complex*)z_auxr, 
                                            embed,
                                            this->nplane, 
                                            1, 
                                            FFTW_FORWARD, 
                                            flag);
        this->planybac = fftw_plan_many_dft(1, 
                                            &this->ny, 
                                            this->nplane, 
                                            (fftw_complex*)z_auxr, 
                                            embed,
                                            this->nplane, 
                                            1,
                                            (fftw_complex*)z_auxr, 
                                            embed,
                                            this->nplane, 
                                            1, 
                                            FFTW_BACKWARD, 
                                            flag);
        if (this->gamma_only)
        {
            this->planxr2c = fftw_plan_many_dft_r2c(1,  
                                                    &this->nx, 
                                                    npy, 
                                                    d_rspace, 
                                                    embed, 
                                                    npy, 
                                                    1, 
                                                    (fftw_complex*)z_auxr,
                                                    embed, 
                                                    npy, 
                                                    1, 
                                                    flag);
            this->planxc2r = fftw_plan_many_dft_c2r(1, 
                                                    &this->nx, 
                                                    npy, 
                                                    (fftw_complex*)z_auxr, 
                                                    embed, 
                                                    npy, 
                                                    1, 
                                                    d_rspace,
                                                    embed, 
                                                    npy, 
                                                    1, 
                                                    flag);
        }
        else
        {
            this->planxfor1 = fftw_plan_many_dft(1, 
                                                 &this->nx, 
                                                 npy, 
                                                 (fftw_complex*)z_auxr, 
                                                 embed, 
                                                 npy, 
                                                 1,
                                                 (fftw_complex*)z_auxr, 
                                                 embed, 
                                                 npy, 
                                                 1, 
                                                 FFTW_FORWARD, 
                                                 flag);
            this->planxbac1 = fftw_plan_many_dft(1, 
                                                 &this->nx, 
                                                 npy, 
                                                 (fftw_complex*)z_auxr, 
                                                 embed, 
                                                 npy, 
                                                 1,
                                                (fftw_complex*)z_auxr, 
                                                embed, 
                                                npy, 
                                                1, 
                                                FFTW_BACKWARD, 
                                                flag);
        }
    }
    else
    {
        this->planxfor1 = fftw_plan_many_dft(1, 
                                             &this->nx, 
                                             this->nplane * (this->lixy + 1), 
                                             (fftw_complex*)z_auxr, 
                                             embed, 
                                             npy,
                                             1, 
                                             (fftw_complex*)z_auxr, 
                                             embed, 
                                             npy, 
                                             1, 
                                             FFTW_FORWARD, 
                                             flag);
        this->planxbac1 = fftw_plan_many_dft(1, 
                                            &this->nx, 
                                            this->nplane * (this->lixy + 1), 
                                            (fftw_complex*)z_auxr, 
                                            embed, 
                                            npy,
                                            1, 
                                            (fftw_complex*)z_auxr, 
                                            embed, 
                                            npy, 
                                            1, 
                                            FFTW_BACKWARD, 
                                            flag);
        if (this->gamma_only)
        {
            this->planyr2c = fftw_plan_many_dft_r2c(1, 
                                                    &this->ny, 
                                                    this->nplane,
                                                    d_rspace, 
                                                    embed, 
                                                    this->nplane, 
                                                    1,
                                                    (fftw_complex*)z_auxr, 
                                                    embed, 
                                                    this->nplane, 
                                                    1, 
                                                    flag);
            this->planyc2r = fftw_plan_many_dft_c2r(1, 
                                                    &this->ny, 
                                                    this->nplane, 
                                                    (fftw_complex*)z_auxr, 
                                                    embed,
                                                    this->nplane, 
                                                    1, 
                                                    d_rspace, 
                                                    embed, 
                                                    this->nplane, 
                                                    1, 
                                                    flag);
        }
        else
        {
            this->planxfor2 = fftw_plan_many_dft(1, 
                                                &this->nx, 
                                                this->nplane * (this->ny - this->rixy), 
                                                (fftw_complex*)z_auxr, 
                                                embed,
                                                npy, 
                                                1, (fftw_complex*)z_auxr, 
                                                embed, 
                                                npy, 
                                                1, 
                                                FFTW_FORWARD, 
                                                flag);
            this->planxbac2 = fftw_plan_many_dft(1, 
                                                 &this->nx, 
                                                 this->nplane * (this->ny - this->rixy), 
                                                 (fftw_complex*)z_auxr, 
                                                 embed,
                                                 npy, 
                                                 1, 
                                                 (fftw_complex*)z_auxr, 
                                                 embed, 
                                                 npy, 
                                                 1, 
                                                 FFTW_BACKWARD, 
                                                 flag);
            this->planyfor = fftw_plan_many_dft(1, 
                                                &this->ny, 
                                                this->nplane, 
                                                (fftw_complex*)z_auxr, 
                                                embed, 
                                                this->nplane,
                                                1, 
                                                (fftw_complex*)z_auxr, 
                                                embed, 
                                                this->nplane, 
                                                1, 
                                                FFTW_FORWARD, 
                                                flag);
            this->planybac = fftw_plan_many_dft(1, 
                                                &this->ny, 
                                                this->nplane, 
                                                (fftw_complex*)z_auxr, 
                                                embed, 
                                                this->nplane,
                                                1, 
                                                (fftw_complex*)z_auxr, 
                                                embed, 
                                                this->nplane, 
                                                1, 
                                                FFTW_BACKWARD, 
                                                flag);
        }
    }
    return;
}

template <>
void FFT_CPU<double>::clearfft(fftw_plan& plan)
{
    if (plan)
    {
        fftw_destroy_plan(plan);
        plan = nullptr;
    }
}

template <>
void FFT_CPU<double>::cleanFFT()
{
    clearfft(planzfor);
    clearfft(planzbac);
    clearfft(planxfor1);
    clearfft(planxbac1);
    clearfft(planxfor2);
    clearfft(planxbac2);
    clearfft(planyfor);
    clearfft(planybac);
    clearfft(planxr2c);
    clearfft(planxc2r);
    clearfft(planyr2c);
    clearfft(planyc2r);
}

template <>
void FFT_CPU<double>::clear()
{
    this->cleanFFT();
    if (z_auxg != nullptr)
    {
        fftw_free(z_auxg);
        z_auxg = nullptr;
    }
    if (z_auxr != nullptr)
    {
        fftw_free(z_auxr);
        z_auxr = nullptr;
    }
    d_rspace = nullptr;
}

template <>
void FFT_CPU<double>::fftxyfor(std::complex<double>* in, std::complex<double>* out) const
{
    int npy = this->nplane * this->ny;
    if (this->xprime)
    {
        
        fftw_execute_dft(this->planxfor1, (fftw_complex*)in, (fftw_complex*)out);
        #pragma omp parallel for
        for (int i = 0; i < this->lixy + 1; ++i)
        {
            fftw_execute_dft(this->planyfor, (fftw_complex*)&in[i * npy], (fftw_complex*)&out[i * npy]);
        }
        #pragma omp parallel for
        for (int i = rixy; i < this->nx; ++i)
        {
            fftw_execute_dft(this->planyfor, (fftw_complex*)&in[i * npy], (fftw_complex*)&out[i * npy]);
        }
    }
    else
    {
        #pragma omp parallel for
        for (int i = 0; i < this->nx; ++i)
        {
            fftw_execute_dft(this->planyfor, (fftw_complex*)&in[i * npy], (fftw_complex*)&out[i * npy]);
        }
        fftw_execute_dft(this->planxfor1, (fftw_complex*)in, (fftw_complex*)out);
        fftw_execute_dft(this->planxfor2, (fftw_complex*)&in[rixy * nplane], (fftw_complex*)&out[rixy * nplane]);
    }
}

template <>
void FFT_CPU<double>::fftxybac(std::complex<double>* in,std::complex<double>* out) const
{
    // 计算了二维 FFT 操作中每一行的实际长度，即 npy。
    // 其中，nplane 表示当前进程负责的 z 层数（或平面数），ny 是 y 方向的网格点数。
    // 这样，npy 就代表了在二维平面（xy）FFT时，每个 x 方向数据块的长度（即每个 x 上有多少个 y*z 数据点）。
    int npy = this->nplane * this->ny;
    /*
    在 Gamma-only 模式下，倒空间数据具有厄米对称性（Hermite Symmetry）。为了节省内存和计算量，只需要存储约一半的数据。
    xprime 决定了是在 x 轴 还是 y 轴 上利用这种对称性进行压缩。
    网格尺寸缩减：
    根据 initfft 中的代码逻辑：
    如果 xprime = true：x 轴方向的网格点数被缩减为 fftnx = nx/2 + 1，而 y 轴保持 ny。
    如果 xprime = false：y 轴方向的网格点数被缩减为 fftny = ny/2 + 1，而 x 轴保持 nx。

    在非 Gamma-only 模式下，xprime 不再负责“网格尺寸减半（nx/2+1）”的任务（因为此时是复数到复数的全变换），
    但它依然控制着以下两个核心逻辑：

    1. 并行循环的顺序与结构
    xprime 决定了 2D FFT 在 xy 平面上执行 1D+1D 变换的先后顺序和分段策略。

    如果 xprime = true：
    代码会优先处理 y 方向的 batch（批量）变换，或者以特定的 x 索引范围（如 lixy 和 rixy）作为外层循环。
    在 fftxybac 中，你可以看到它先处理两个并行的 y 方向循环，最后再执行一个整体的 x 方向计划。
    如果 xprime = false：
    代码逻辑会反过来，可能先处理 x 方向的计划（分成了 planxbac1 和 planxbac2 两段），再并行处理 y 方向。

    2. 多进程并行索引（lixy 与 rixy）的挂钩
    在 ABACUS 的并行 FFT 实现中，xy 平面的数据往往被切分到不同进程上。

    lixy (left index xy) 和 rixy (right index xy) 定义了当前进程在 xy 平面上负责的边界。
    xprime 的取值决定了这些边界值如何作为 for 循环的起止点，从而确保每个进程只计算自己那一块网格点的 1D FFT。
    */

    if (this->xprime)
    {
    /*
    1) in/out 与 npy 代表什么布局？
    在 fftxybac 里先计算：

    npy = nplane * ny;
    这意味着：把 3D 数据（至少是本进程负责的 nplane 个 z 层）按 x 为“大行” 排布：

    对固定的 x=i，这一“行”的长度是 ny * nplane（即 npy）
    因此 &in[i*npy] / &out[i*npy] 指向 第 i 个 x 行 的起点

    2) 第一步：对每个 x 行做 “沿 y 的 1D 逆 FFT”（并行）
    这两段 OpenMP 循环都在执行同一个计划 planybac：

    planybac 实际在做什么？
    在 setupFFT() 里（当 xprime==true）创建：

    rank=1，变换长度 ny
    howmany = nplane
    stride = nplane，dist = 1
    这组参数的含义是：对每个固定的 z_local，沿 y 做一次长度为 ny 的 1D FFT，
    并且对所有 z_local=0..nplane-1 批量做（howmany=nplane）。

    也就是说，对某个固定的 x=i：它会做 nplane 次 1D FFT

    为什么分成两个 i 区间：[0, lixy] 和 [rixy, nx-1]？
    这通常用于 Gamma-only 的半谱/对称性 或某些 分布/裁剪后的有效区间：

    中间区间 (lixy+1 ... rixy-1) 可能是“无效/无需计算”的 x 行（例如 Gamma-only 下利用共轭对称，某些行不需要显式变换）
    也可能是并行分区后当前进程只需要处理两端的行
    当 lixy+1 == rixy 时，这两个循环等价于覆盖完整范围（中间没有空洞）。

    要点：这一步是 线程并行（OpenMP），每个线程处理不同的 x=i 行。
    */
        #pragma omp parallel for
        for (int i = 0; i < this->lixy + 1; ++i)
        {
            fftw_execute_dft(this->planybac, (fftw_complex*)&in[i * npy], (fftw_complex*)&out[i * npy]);
        }
        #pragma omp parallel for
        for (int i = rixy; i < this->nx; ++i)
        {
            fftw_execute_dft(this->planybac, (fftw_complex*)&in[i * npy], (fftw_complex*)&out[i * npy]);
        }
    /*
    3) 第二步：做 “沿 x 的 1D 逆 FFT”（一次批量执行）
    planxbac1 在 setupFFT()（xprime==true && !gamma_only）中创建为：

    rank=1，变换长度 nx
    howmany = npy（也就是 ny*nplane 个序列）
    stride = npy，dist = 1
    含义是：对每个固定的 (y, z_local) 组合，沿 x 做一次长度为 nx 的 1D FFT；
    这样的序列总共有 ny*nplane 条，所以 howmany=npy。

    这样就完成了：先 y 后 x 的二维逆 FFT（对每个 z_local 平面都是一套 2D 逆 FFT）。

    4) 关于 in/out：为什么 y 步骤写到 out，x 步骤又从 in 读？
    因为外部调用时是 fftxybac(auxr, auxr)），即原地变换：in == out 指向同一块缓冲区。
    所有前面的 y 变换写到 out，等价于写回 in，后面的 x 变换读 in 成立。

    5) 这段代码在三维逆 FFT 里的位置
    结合 PW/FFT 流水线，这段 fftxybac 通常是：

    z 方向做批量逆 FFT（stick 上连续）
    通信/重排把数据放到 xy 网格（每进程拿到本地 nplane 层）
    这里：对每个本地 z 层做 2D（x,y）逆 FFT，得到实空间数据（或进入下一步）
    */
        fftw_execute_dft(this->planxbac1, (fftw_complex*)in, (fftw_complex*)out);
    }
    /*
    else这段代码是 FFT_CPU<double>::fftxybac() 里 xprime == false 分支（也就是 不走“先 y 后 x” 的那条路），
    它实现的是 二维逆 FFT 的另一种执行顺序：先做 x 方向，再做 y 方向，
    并且在 x 方向上把要做 FFT 的序列分成了两段（用 planxbac1 + planxbac2）。

    1) 为什么要先执行 planxbac1 和 planxbac2（先做 x 方向）
    在 setupFFT() 的 xprime == false && !gamma_only 分支里，x 方向逆 FFT 的 plan 是这样建的（关键是 howmany）：

    planxbac1: howmany = nplane * (lixy + 1)
    planxbac2: howmany = nplane * (ny - rixy)
    它们的含义是：沿 x 做很多条 1D 逆 FFT（长度为 nx），每条对应一个固定的 (y, z_local)。

    但这里并不对所有 y 都做 x-FFT，而是只对两段 y 范围做：

    第一段：y = 0 .. lixy
    第二段：y = rixy .. ny-1
    中间段：y = lixy+1 .. rixy-1（被跳过）
    这通常是为了配合并行切分/有效区域（例如某些分布下中间区域在本进程不需要/不拥有，或 Gamma 压缩时只有两侧有效）。
    关键点：lixy/rixy 在 xprime==false 时，起作用的是 y 方向的边界（而在 xprime==true 时更像是 x 边界用法）。

    为什么要拆成两个 plan？

    因为要做 FFT 的 y 索引不是一个连续区间，而是“两头一段 + 一段”，拆成两个连续块可以让 FFTW 的 batch（howmany）保持 连续内存描述，避免复杂的“跳跃式批处理”。
    对应到代码：

    planxbac1 从 in[0] 开始覆盖 y=0..lixy（以及所有 z）
    planxbac2 从 in[rixy*nplane] 开始覆盖 y=rixy..ny-1（以及所有 z）

    3) 再对每个 x 行做 planybac（后做 y 方向）
    内部的for循环含义是：

    对每个 x=i 的“大行”（长度 npy = ny*nplane）做 y 方向的批量 1D 逆 FFT
    planybac（在 setupFFT() 里建的）相当于：对固定的 z_local，沿 y 做长度 ny 的 1D 逆 FFT，
    并对 nplane 个 z_local 批量执行。
    这里用 OpenMP 并行，是把不同 x 行分给不同线程。

    4) 总结：这个 else 分支整体在做什么
    当 xprime == false 时，二维逆 FFT 的执行顺序是：

    x 方向逆 FFT（批量）：只对两段有效 y 区间做（用 planxbac1 + planxbac2）
    y 方向逆 FFT（批量）：对每个 x 行做（OpenMP 并行）
    这与 xprime == true 分支（先 y 后 x，并且可能在 x 上按 lixy/rixy 分段）形成对照。
    */
    else
    {
        fftw_execute_dft(this->planxbac1, (fftw_complex*)in, (fftw_complex*)out);
        fftw_execute_dft(this->planxbac2, (fftw_complex*)&in[rixy * nplane], (fftw_complex*)&out[rixy * nplane]);
        #pragma omp parallel for
        for (int i = 0; i < this->nx; ++i)
        {
            fftw_execute_dft(this->planybac, (fftw_complex*)&in[i * npy], (fftw_complex*)&out[i * npy]);
        }
    }
}

template <>
void FFT_CPU<double>::fftzfor(std::complex<double>* in, std::complex<double>* out) const
{
    fftw_execute_dft(this->planzfor, (fftw_complex*)in, (fftw_complex*)out);
}

template <>
void FFT_CPU<double>::fftzbac(std::complex<double>* in, std::complex<double>* out) const
{
    // 这个函数调用了 FFTW 库的 fftw_execute_dft，使用事先创建好的 planzbac 计划（plan）。
    // 对所有 stick 的 z 方向数据批量执行 1D 逆FFT，是三维逆傅里叶变换的第一步。
    // fftw_execute_dft 是 FFTW 库的执行接口，this->planzbac 是提前用fftw_plan_many_dft创建好的批量1D逆FFT计划（plan）。
    // in 和 out 是输入和输出的复数数组指针，通常为同一个数组，实现原地变换。
    // 该操作会对所有 stick（柱）上的 z 方向数据批量执行逆FFT，为后续的 xy 平面变换做准备。
    fftw_execute_dft(this->planzbac, (fftw_complex*)in, (fftw_complex*)out);
}

template <>
void FFT_CPU<double>::fftxyr2c(double* in, std::complex<double>* out) const
{
    int npy = this->nplane * this->ny;
    if (this->xprime)
    {
        fftw_execute_dft_r2c(this->planxr2c, in, (fftw_complex*)out);
        #pragma omp parallel for
        for (int i = 0; i < this->lixy + 1; ++i)
        {
            fftw_execute_dft(this->planyfor, (fftw_complex*)&out[i * npy], (fftw_complex*)&out[i * npy]);
        }
    }
    else
    {
        #pragma omp parallel for
        for (int i = 0; i < this->nx; ++i)
        {
            fftw_execute_dft_r2c(this->planyr2c, &in[i * npy], (fftw_complex*)&out[i * npy]);
        }
        fftw_execute_dft(this->planxfor1, (fftw_complex*)out, (fftw_complex*)out);
    }
}

template <>
void FFT_CPU<double>::fftxyc2r(std::complex<double> *in,double *out) const
{
    int npy = this->nplane * this->ny;
    if (this->xprime)
    {
        #pragma omp parallel for
        for (int i = 0; i < this->lixy + 1; ++i)
        {
            fftw_execute_dft(this->planybac, (fftw_complex*)&in[i * npy], (fftw_complex*)&in[i * npy]);
        }
        fftw_execute_dft_c2r(this->planxc2r, (fftw_complex*)in, out);
    }
    else
    {
        fftw_execute_dft(this->planxbac1, (fftw_complex*)in, (fftw_complex*)in);
        #pragma omp parallel for
        for (int i = 0; i < this->nx; ++i)
        {
            fftw_execute_dft_c2r(this->planyc2r, (fftw_complex*)&in[i * npy], &out[i * npy]);
        }
    }
}

template <> double* 
FFT_CPU<double>::get_rspace_data() const {return d_rspace;}
template <> std::complex<double>* 
FFT_CPU<double>::get_auxr_data()   const {return z_auxr;}
template <> std::complex<double>* 
// 在this->planzbac = fftw_plan_many_dft()里创建的FFT计划中，使用了z_auxg作为输入和输出数组指针。
FFT_CPU<double>::get_auxg_data()   const {return z_auxg;}

template FFT_CPU<float>::FFT_CPU();
template FFT_CPU<float>::~FFT_CPU();
template FFT_CPU<double>::FFT_CPU();
template FFT_CPU<double>::~FFT_CPU();
}