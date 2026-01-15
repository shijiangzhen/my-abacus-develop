#include "source_base/timer.h"
#include "source_basis/module_pw/kernels/pw_op.h"
#include "pw_basis_k.h"
#include "pw_gatherscatter.h"

#include <cassert>
#include <complex>

namespace ModulePW
{

/**
 * @brief transform real space to reciprocal space
 * @details real wave function f(k,r):
 *          f(k,r)=1/V*\sum_{g} c(k,g)*exp(i(g+k)*r) \equiv exp(ikr)f'(k.r)
 *          c(k,g)=\int dr*f(k,r)*exp(-i(g+k)*r)
 *          However, we use f'(k,r)!!! :
 *          f'(k,r)=1/V*\sum_{g} c(k,g)*exp(ig*r)
 *          c(k,g)=\int dr*f'(k,r)*exp(-ig*r)
 *
 *          This function tranform f'(r) to c(k,g).
 * @param in: (nplane,ny,nx), std::complex<double> data
 * @param out: (nz, ns),  std::complex<double> data
 */
template <typename FPTYPE>
void PW_Basis_K::real2recip(const std::complex<FPTYPE>* in,
                            std::complex<FPTYPE>* out,
                            const int ik,
                            const bool add,
                            const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "real2recip");

    assert(this->gamma_only == false);
    auto* auxr = this->fft_bundle.get_auxr_data<FPTYPE>();
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
    for (int ir = 0; ir < this->nrxx; ++ir)
    {
        auxr[ir] = in[ir];
    }
    this->fft_bundle.fftxyfor(fft_bundle.get_auxr_data<FPTYPE>(), fft_bundle.get_auxr_data<FPTYPE>());

    this->gatherp_scatters(this->fft_bundle.get_auxr_data<FPTYPE>(), this->fft_bundle.get_auxg_data<FPTYPE>());

    this->fft_bundle.fftzfor(fft_bundle.get_auxg_data<FPTYPE>(), fft_bundle.get_auxg_data<FPTYPE>());

    const int startig = ik * this->npwk_max;
    const int npwk = this->npwk[ik];
    auto* auxg = this->fft_bundle.get_auxg_data<FPTYPE>();
    if (add)
    {
        FPTYPE tmpfac = factor / FPTYPE(this->nxyz);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
        for (int igl = 0; igl < npwk; ++igl)
        {
            out[igl] += tmpfac * auxg[this->igl2isz_k[igl + startig]];
        }
    }
    else
    {
        FPTYPE tmpfac = 1.0 / FPTYPE(this->nxyz);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
        for (int igl = 0; igl < npwk; ++igl)
        {
            out[igl] = tmpfac * auxg[this->igl2isz_k[igl + startig]];
        }
    }
    ModuleBase::timer::tick(this->classname, "real2recip");
}

/**
 * @brief transform real space to reciprocal space
 * @details real wave function f(k,r):
 *          f(k,r)=1/V*\sum_{g} c(k,g)*exp(i(g+k)*r) \equiv exp(ikr)f'(k.r)
 *          c(k,g)=\int dr*f(k,r)*exp(-i(g+k)*r)
 *          However, we use f'(k,r)!!! :
 *          f'(k,r)=1/V*\sum_{g} c(k,g)*exp(ig*r)
 *          c(k,g)=\int dr*f'(k,r)*exp(-ig*r)
 *
 *          This function tranform f'(r) to c(k,g).
 * @param in: (nplane,ny,nx), double data
 * @param out: (nz, ns),  std::complex<double> data
 */
template <typename FPTYPE>
void PW_Basis_K::real2recip(const FPTYPE* in,
                            std::complex<FPTYPE>* out,
                            const int ik,
                            const bool add,
                            const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "real2recip");
    assert(this->gamma_only == true);
    // for(int ir = 0 ; ir < this->nrxx ; ++ir)
    // {
    //     this->fft_bundle.get_rspace_data<FPTYPE>()[ir] = in[ir];
    // }
    // r2c in place
    const int npy = this->ny * this->nplane;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 4096 / sizeof(FPTYPE))
#endif
    for (int ix = 0; ix < this->nx; ++ix)
    {
        for (int ipy = 0; ipy < npy; ++ipy)
        {
            this->fft_bundle.get_rspace_data<FPTYPE>()[ix * npy + ipy] = in[ix * npy + ipy];
        }
    }

    this->fft_bundle.fftxyr2c(fft_bundle.get_rspace_data<FPTYPE>(), fft_bundle.get_auxr_data<FPTYPE>());

    this->gatherp_scatters(this->fft_bundle.get_auxr_data<FPTYPE>(), this->fft_bundle.get_auxg_data<FPTYPE>());

    this->fft_bundle.fftzfor(fft_bundle.get_auxg_data<FPTYPE>(), fft_bundle.get_auxg_data<FPTYPE>());

    const int startig = ik * this->npwk_max;
    const int npwk = this->npwk[ik];
    auto* auxg = this->fft_bundle.get_auxg_data<FPTYPE>();
    if (add)
    {
        FPTYPE tmpfac = factor / FPTYPE(this->nxyz);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
        for (int igl = 0; igl < npwk; ++igl)
        {
            out[igl] += tmpfac * auxg[this->igl2isz_k[igl + startig]];
        }
    }
    else
    {
        FPTYPE tmpfac = 1.0 / FPTYPE(this->nxyz);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
        for (int igl = 0; igl < npwk; ++igl)
        {
            out[igl] = tmpfac * auxg[this->igl2isz_k[igl + startig]];
        }
    }
    ModuleBase::timer::tick(this->classname, "real2recip");
    return;
}

/**
 * @brief transform reciprocal space to real space
 * @details real wave function f(k,r):
 *          f(k,r)=1/V*\sum_{g} c(k,g)*exp(i(g+k)*r) \equiv exp(ikr)f'(k.r)
 *          c(k,g)=\int dr*f(k,r)*exp(-i(g+k)*r)
 *          However, we use f'(k,r)!!! :
 *          f'(k,r)=1/V*\sum_{g} c(k,g)*exp(ig*r)
 *          c(k,g)=\int dr*f'(k,r)*exp(-ig*r)
 *
 *          This function tranform c(k,g) to f'(r).
 * @param in: (nz,ns), std::complex<double>
 * @param out: (nplane, ny, nx), std::complex<double>
 */
template <typename FPTYPE>
void PW_Basis_K::recip2real(const std::complex<FPTYPE>* in,
                            std::complex<FPTYPE>* out,
                            const int ik,
                            const bool add,
                            const FPTYPE factor) const
{
    // tick是一个计时器函数，用于记录当前类执行 "recip2real" 这段代码的耗时
    ModuleBase::timer::tick(this->classname, "recip2real");
    // assert是一个断言，确保当前对象的 gamma_only 成员变量为 false。
    // 作用是保证此处执行的是非 Gamma-only的逆傅里叶变换（即处理的是一般的复数 FFT，而不是只处理实数 Gamma 点的特殊情况）。
    // 如果条件不满足，程序会在此处终止并报错
    assert(this->gamma_only == false);
    // fft_bundle.get_auxg_data<FPTYPE>() 返回一个指向 G 空间辅助数据的指针（类型为 FPTYPE，如 float 或 double）。
    // this->nst * this->nz 是需要清零的元素总数。
    // ModuleBase::GlobalFunc::ZEROS(...) 是一个工具函数，用于将指定数组的所有元素设置为 0。
    // 用途：在进行傅里叶变换前，先把辅助数组清零，避免残留数据影响后续计算
    ModuleBase::GlobalFunc::ZEROS(fft_bundle.get_auxg_data<FPTYPE>(), this->nst * this->nz);

    // npwk_max 是每个 k 点分配的最大平面波数，startig 用于定位当前 k 点的平面波数据在一维数组中的起始位置。
    // 作用是计算当前 k 点（ik）在全局平面波数组中的起始索引。
    const int startig = ik * this->npwk_max;
    // 获取当前 k 点实际的平面波数（即 G 向量的数量）。
    const int npwk = this->npwk[ik];
    // 获取一个指向 G 空间辅助数组的指针，类型为 FPTYPE（如 float 或 double）。
    auto* auxg = this->fft_bundle.get_auxg_data<FPTYPE>();
#ifdef _OPENMP
// 4096 / sizeof(FPTYPE) 计算出每个线程一次处理多少个元素，
// 如果是 float，每块 4096/4 = 1024 个元素。如果是 double，每块 4096/8 = 512 个元素。
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
    /*
    举例说明下面的for循环的重排过程：
    1. 设定场景
    FFT 网格 (auxg)：逻辑大小为 8（即 N_stems×N_z=8）。
    注意，假设的auxg的这个大小也许并不合适，因为FFT网格至少要包围整个截断球。
    K 点信息：共 2 个 k 点，每个 k 点最大分配空间 npwk_max = 3。
    当前处理：第二个 k 点（ik = 1），该 k 点实际只有 2 个平面波（npwk = 2）。

    2. 内存状态
    in (输入指针)：
    由于 ABACUS 每次只传入当前 k 点的指针，in 指向的是长度为 2 的数组：[B1, B2]。
    startig (地图偏移)：
    ik * npwk_max = 1 * 3 = 3。
    igl2isz_k (全局座位表)：
    这是一个大数组，存着所有 k 点的映射。
    索引: [ 0, 1, 2, | 3, 4, 5 ]
    内容: [ ?, ?, ?, | 2, 5, ? ] （前 3 个是 k=0 的，后 3 个是 k=1 的）。
    在 igl2isz_k 数组中，每个 k 点从 npwk 到 npwk_max 之间的位置存储的是 无效数据或无意义的填充值。
    为了能够通过 ik * npwk_max 这种简单的乘法快速定位不同 k 点的映射表，为每个k点分配了等长的空间（即 npwk_max）。
    auxg (FFT 网格)：
    初始化全为 0：[0, 0, 0, 0, 0, 0, 0, 0]。

    3. 执行映射循环
    代码逻辑：auxg[this->igl2isz_k[igl + startig]] = in[igl];

    第 1 次迭代 (igl = 0)：
    取行李：in[0] 是 B1。
    看地图：igl + startig = 3。查询 igl2isz_k[3] 得到网格位置 2。
    放行李：auxg[2] = B1。
    第 2 次迭代 (igl = 1)：
    取行李：in[1] 是 B2。
    看地图：igl + startig = 4。查询 igl2isz_k[4] 得到网格位置 5。
    放行李：auxg[5] = B2。
    */
    /*
    下面的for循环实现了G 空间数据的重排。
    
    原始数据 in 是一个 1D 数组，长度为 npwk。它只按顺序存储了那些处于能量截断球内的平面波系数,
    这些系数在内存中是连续排列的，没有反映它们在 3D 倒空间网格中的几何位置；
    而FFT 要求数据必须放在逻辑 3D 网格的对应坐标点上，所以要通过映射表 igl2isz_k，
    将紧凑 1D 数组中的值 in[igl] 搬运到 3D 网格辅助数组 auxg 的对应位置上，
    那些不在截断球内的网格点（即没有对应的 igl）会被预先清零（ModuleBase::GlobalFunc::ZEROS）。
    
    in[igl]= i * (k_α + G_α) * ρ_G，即当前k点下第igl个平面波分量在某一笛卡尔方向上的倒空间梯度系数，
    也就是说，外部是将三个方向上（x,y,z）的倒空间梯度系数依次传入进来并映射到逻辑3D网格中,
    传入的是x方向上时，就映射到逻辑3D网格的x方向上，y和z同理。
    
    in (一维紧凑数组)：长度为 npwk。它只存储了在能量截断半径内的平面波系数，没有任何“空位”。
    
    auxg ：存储的是 当前k点处于 3D 逻辑网格位置上的平面波系数，也是一维数组（连续内存指针），长度为 nst * nz。
    在逻辑布局上是二维数组（stems × z），而这个逻辑二维数组又是从 三维倒空间网格中提取出的有效数据部分。
    nst：逻辑上的“柱”（stems）的数量（代表 x,y 平面上的位置）。nz：z 方向的网格点数（每个柱上的点数）。
    其逻辑排列是ABACUS 特有的 Stem-Z 布局，主要为了并行 FFT 优化：
    1.第一级（Stem）：将 3D 网格的 x 和 y 坐标合并成一个索引（即“柱”）。
    2.第二级（Z-length）：每个“柱”对应一根 z 方向的线。
    3.线性索引公式：数据存储在 auxg[isz]，其中 isz = is * nz + iz（is 是柱索引，iz 是 z 坐标）
   
    igl2isz_k[]：是一个映射表（索引字典），npwk_max*nks 大小的一维数组。
    它存储了第 igl 个平面波在逻辑 3D 网格（打平后的一维 auxg）中对应的绝对位置（一维线性下标）。

    逻辑过程：
        1.先用 ZEROS 把 auxg 全部清零。
        2.遍历紧凑的 in 数组。
        3.根据映射表，把 in 中的系数像“填坑”一样，放到 auxg 对应的网格点下标处。
    */
    for (int igl = 0; igl < npwk; ++igl)
    {
        auxg[this->igl2isz_k[igl + startig]] = in[igl];
    }

    /*
    在 ABACUS 及类似的平面波 DFT 程序中，Stick（柱） 思想是处理倒空间（G 空间）数据和优化 FFT 变换的核心。

    1. 什么是 Stick（原理）

    在倒空间中，平面波基组由满足能量截断条件 |G+k|^2 ≤ E_{cut} 的格点组成。几何上，这些点分布在一个球体内。

    几何定义：将这个球体放入 3D FFT 网格 (N_x, N_y, N_z) 后，我们观察z方向。
            对于每一对固定的(G_x, G_y) 坐标，在 z 方向上对应的一串连续格点被称为一个Stick（或 Stem）。
    有效性：只有穿过截断球的 Stick 才包含有效数据。不穿过球体的 Stick 全部为 0，程序在处理时会跳过它们以节省内存。
    内存布局：在 auxg数组中，数据按照isz = is*nz + iz排布。这里is是Stick的编号，iz是z方向的索引。这就是所谓的Stem-Z布局。

    2. Stick 的作用

    A. 提高 FFT 效率（维度分解）    
    3D FFT 可以分解为：1D FFT (z方向) + 2D FFT (xy平面)。
    由于数据是按 Stick 存储的，z方向的 1D FFT 可以在连续的内存块上执行，这极大地提高了 CPU 缓存命中率。

    B. 并行化优化（数据分布）
    在多核并行计算中，Stick 是任务分配的最小单位：
    1. 分配方式：程序将不同的 Stick 分配给不同的进程（MPI Process）。
    2. 无通信计算：在进行z方向的 FFT 时，由于每个进程拥有完整的 Stick（即 z 方向的所有点），进程之间不需要进行任何数据交换。
        只有在 z 变换完成、准备进行 xy 变换时，才需要通过 gather/scatter操作进行一次全局通信（转置）。

    C. 压缩存储
    减少空值：如果使用完整的 3D 阵列存储，截断球外的点（约占 50% 以上的空间）都是 0。
    按需存储：只存储有数据的 Stick（即代码中的 nst 个柱子），显著降低了内存需求。

    3. 在代码中的体现
    在你提供的 recip2real代码中：
    （1）映射：igl2isz_k 将数据直接放入对应 Stick 的对应 z位置。
    （2）Z变换：this->fft_bundle.fftzbac 沿着每一个 Stick 进行分步 1D 逆 FFT。
    （3）重排与通信：this->gathers_scatterp将处理好的 Stick 数据重新分布，为接下来的 xy 平面变换做准备。

    总结
    Stick 思想将 3D 球形数据转化为一簇 1D “柱子”，实现了内存压缩、高性能缓存访问和高效率的并行任务划分。
    */

    // 调用了 fft_bundle 对象的 fftzbac 方法（本质上应该调用的是FFTW库的函数fftw_execute_dft），
    // 对 auxg的所有 Stick（柱）上的 z 方向数据做逆FFT，是三维逆傅里叶变换的第一步。
    // get_auxg_data既是输入也是输出，表示原地（in-place）变换，即直接在原数组上进行操作。
    this->fft_bundle.fftzbac(fft_bundle.get_auxg_data<FPTYPE>(), fft_bundle.get_auxg_data<FPTYPE>());

    // 调用了 gathers_scatterp 方法，用于数据重排和分布。
    // 输入是 G 空间辅助数组 auxg，输出是实空间辅助数组 auxr。
    // 作用是将经过 z 方向逆FFT后的 Stick（柱）数据，按照 FFT 库的要求，重新分布到 xy 平面网格上，为后续的 xy 方向逆FFT做准备。
    // 这个过程通常涉及内存转置和进程间通信（在并行计算时），确保每个进程拥有完整的 xy 平面数据块。
    // 该代码实现了 Stick 数据到 xy 网格的数据重排，是三维逆傅里叶变换的第二步，为后续的 xy 方向逆FFT做准备。
    this->gathers_scatterp(this->fft_bundle.get_auxg_data<FPTYPE>(), this->fft_bundle.get_auxr_data<FPTYPE>());

    // 调用了 fftxybac 方法，对实空间辅助数组 auxr 执行二维逆FFT（xy方向），实现三维逆傅里叶变换的最后一步。
    // 输入/输出：auxr，即 xy 平面网格上的数据，原地变换（in-place）。
    // 作用：将经过 z 方向逆FFT和数据重排后的数据，在每个 z 层上做一次二维逆FFT，最终得到实空间的波函数分布。
    // 3D FFT流程：Stick（柱）数据 → z方向逆FFT → 数据重排到xy网格 → xy方向逆FFT（本行代码） → 得到实空间分布。
    this->fft_bundle.fftxybac(fft_bundle.get_auxr_data<FPTYPE>(), fft_bundle.get_auxr_data<FPTYPE>());
    auto* auxr = this->fft_bundle.get_auxr_data<FPTYPE>();
   
    // 默认情况下，add=false，factor=1.0
    // add为true时，将逆傅里叶变换（FFT）得到的实空间数据 auxr 累加到输出数组 out 上（如多波函数累加或外部叠加）。
    if (add)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
        for (int ir = 0; ir < this->nrxx; ++ir)
        {
            // 对实空间每个网格点，将 factor * auxr[ir] 加到 out[ir] 上。
            out[ir] += factor * auxr[ir];
        }
    }
    // 当 add == false 时，直接将 逆FFT得到的实空间数据 auxr 的内容赋值给 out，即覆盖原有的输出数据。
    else
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
        for (int ir = 0; ir < this->nrxx; ++ir)
        {
            out[ir] = auxr[ir];
        }
    }
    // 结束计时，记录 "recip2real" 代码段的执行时间。
    // 在进入和离开 recip2real 函数时各调用一次timer::tick，配合内部计时机制，
    // 可以统计本次逆傅里叶变换所消耗的时间。
    ModuleBase::timer::tick(this->classname, "recip2real");
}

/**
 * @brief transform reciprocal space to real space
 * @details real wave function f(k,r):
 *          f(k,r)=1/V*\sum_{g} c(k,g)*exp(i(g+k)*r) \equiv exp(ikr)f'(k.r)
 *          c(k,g)=\int dr*f(k,r)*exp(-i(g+k)*r)
 *          However, we use f'(k,r)!!! :
 *          f'(k,r)=1/V*\sum_{g} c(k,g)*exp(ig*r)
 *          c(k,g)=\int dr*f'(k,r)*exp(-ig*r)
 *
 *          This function tranform c(k,g) to f'(r).
 * @param in: (nz,ns), std::complex<double>
 * @param out: (nplane, ny, nx), double
 */
template <typename FPTYPE>
void PW_Basis_K::recip2real(const std::complex<FPTYPE>* in,
                            FPTYPE* out,
                            const int ik,
                            const bool add,
                            const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "recip2real");
    assert(this->gamma_only == true);
    ModuleBase::GlobalFunc::ZEROS(fft_bundle.get_auxg_data<FPTYPE>(), this->nst * this->nz);

    const int startig = ik * this->npwk_max;
    const int npwk = this->npwk[ik];
    auto* auxg = this->fft_bundle.get_auxg_data<FPTYPE>();
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
    for (int igl = 0; igl < npwk; ++igl)
    {
        auxg[this->igl2isz_k[igl + startig]] = in[igl];
    }
    this->fft_bundle.fftzbac(fft_bundle.get_auxg_data<FPTYPE>(), fft_bundle.get_auxg_data<FPTYPE>());

    this->gathers_scatterp(this->fft_bundle.get_auxg_data<FPTYPE>(), this->fft_bundle.get_auxr_data<FPTYPE>());

    this->fft_bundle.fftxyc2r(fft_bundle.get_auxr_data<FPTYPE>(), fft_bundle.get_rspace_data<FPTYPE>());

    // for(int ir = 0 ; ir < this->nrxx ; ++ir)
    // {
    //     out[ir] = this->fft_bundle.get_rspace_data<FPTYPE>()[ir] / this->nxyz;
    // }

    // r2c in place
    const int npy = this->ny * this->nplane;
    auto* rspace = this->fft_bundle.get_rspace_data<FPTYPE>();
    if (add)
    {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 4096 / sizeof(FPTYPE))
#endif
        for (int ix = 0; ix < this->nx; ++ix)
        {
            for (int ipy = 0; ipy < npy; ++ipy)
            {
                out[ix * npy + ipy] += factor * rspace[ix * npy + ipy];
            }
        }
    }
    else
    {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 4096 / sizeof(FPTYPE))
#endif
        for (int ix = 0; ix < this->nx; ++ix)
        {
            for (int ipy = 0; ipy < npy; ++ipy)
            {
                out[ix * npy + ipy] = rspace[ix * npy + ipy];
            }
        }
    }
    ModuleBase::timer::tick(this->classname, "recip2real");
}

template <>
void PW_Basis_K::real_to_recip(const base_device::DEVICE_CPU* /*dev*/,
                               const std::complex<float>* in,
                               std::complex<float>* out,
                               const int ik,
                               const bool add,
                               const float factor) const
{
    this->real2recip(in, out, ik, add, factor);
}
template <>
void PW_Basis_K::real_to_recip(const base_device::DEVICE_CPU* /*dev*/,
                               const std::complex<double>* in,
                               std::complex<double>* out,
                               const int ik,
                               const bool add,
                               const double factor) const
{
    #if defined(__DSP)
        this->real2recip_dsp(in,out,ik,add,factor);
    #else
        this->real2recip(in, out, ik, add, factor);
    #endif
}

// PW_Basis_K::recip_to_real 针对 CPU 设备和 float 类型的特化实现。
template <>
void PW_Basis_K::recip_to_real(const base_device::DEVICE_CPU* /*dev*/,
                               const std::complex<float>* in,
                               std::complex<float>* out,
                               const int ik,
                               const bool add,
                               const float factor) const
{
    this->recip2real(in, out, ik, add, factor);
}
// PW_Basis_K::recip_to_real 针对 CPU 设备和 double 类型的特化实现。
template <>
void PW_Basis_K::recip_to_real(const base_device::DEVICE_CPU* /*dev*/,
                               const std::complex<double>* in,
                               std::complex<double>* out,
                               const int ik,
                               const bool add,
                               const double factor) const
{
    // 如果定义了 __DSP，调用 recip2real_dsp（适配 DSP 平台的实现）
    // __DSP 是一个条件编译宏，用于区分是否在特定的 DSP（数字信号处理器，Digital Signal Processor）平台上编译和运行。
    #if defined(__DSP)
        this->recip2real_dsp(in,out,ik,add,factor);
    #else
        this->recip2real(in, out, ik, add, factor);
    #endif
}

#if (defined(__CUDA) || defined(__ROCM))
template <>
void PW_Basis_K::real_to_recip(const base_device::DEVICE_GPU* ctx,
                               const std::complex<float>* in,
                               std::complex<float>* out,
                               const int ik,
                               const bool add,
                               const float factor) const
{
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
    assert(this->gamma_only == false);
    assert(this->poolnproc == 1);

    base_device::memory::synchronize_memory_op<std::complex<float>, base_device::DEVICE_GPU, base_device::DEVICE_GPU>()(
        this->fft_bundle.get_auxr_3d_data<float>(),
        in,
        this->nrxx);

    this->fft_bundle.fft3D_forward(this->fft_bundle.get_auxr_3d_data<float>(), this->fft_bundle.get_auxr_3d_data<float>());

    const int startig = ik * this->npwk_max;
    const int npw_k = this->npwk[ik];
    set_real_to_recip_output_op<float, base_device::DEVICE_GPU>()(npw_k,
                                                                  this->nxyz,
                                                                  add,
                                                                  factor,
                                                                  this->ig2ixyz_k + startig,
                                                                  this->fft_bundle.get_auxr_3d_data<float>(),
                                                                  out);
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
}
template <>
void PW_Basis_K::real_to_recip(const base_device::DEVICE_GPU* ctx,
                               const std::complex<double>* in,
                               std::complex<double>* out,
                               const int ik,
                               const bool add,
                               const double factor) const
{
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
    assert(this->gamma_only == false);
    assert(this->poolnproc == 1);

    base_device::memory::synchronize_memory_op<std::complex<double>,
                                               base_device::DEVICE_GPU,
                                               base_device::DEVICE_GPU>()(this->fft_bundle.get_auxr_3d_data<double>(),
                                                                          in,
                                                                          this->nrxx);

    this->fft_bundle.fft3D_forward(this->fft_bundle.get_auxr_3d_data<double>(), this->fft_bundle.get_auxr_3d_data<double>());

    const int startig = ik * this->npwk_max;
    const int npw_k = this->npwk[ik];
    set_real_to_recip_output_op<double, base_device::DEVICE_GPU>()(npw_k,
                                                                   this->nxyz,
                                                                   add,
                                                                   factor,
                                                                   this->ig2ixyz_k + startig,
                                                                   this->fft_bundle.get_auxr_3d_data<double>(),
                                                                   out);
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
}

template <>
void PW_Basis_K::recip_to_real(const base_device::DEVICE_GPU* ctx,
                               const std::complex<float>* in,
                               std::complex<float>* out,
                               const int ik,
                               const bool add,
                               const float factor) const
{
    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
    assert(this->gamma_only == false);
    assert(this->poolnproc == 1);
    // ModuleBase::GlobalFunc::ZEROS(fft_bundle.get_auxr_3d_data<float>(), this->nxyz);
    base_device::memory::set_memory_op<std::complex<float>, base_device::DEVICE_GPU>()(
        this->fft_bundle.get_auxr_3d_data<float>(),
        0,
        this->nxyz);

    const int startig = ik * this->npwk_max;
    const int npw_k = this->npwk[ik];

    set_3d_fft_box_op<float, base_device::DEVICE_GPU>()(npw_k,
                                                        this->ig2ixyz_k + startig,
                                                        in,
                                                        this->fft_bundle.get_auxr_3d_data<float>());
    this->fft_bundle.fft3D_backward(this->fft_bundle.get_auxr_3d_data<float>(), this->fft_bundle.get_auxr_3d_data<float>());

    set_recip_to_real_output_op<float, base_device::DEVICE_GPU>()(this->nrxx,
                                                                  add,
                                                                  factor,
                                                                  this->fft_bundle.get_auxr_3d_data<float>(),
                                                                  out);

    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
}
template <>
void PW_Basis_K::recip_to_real(const base_device::DEVICE_GPU* ctx,
                               const std::complex<double>* in,
                               std::complex<double>* out,
                               const int ik,
                               const bool add,
                               const double factor) const
{
    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
    assert(this->gamma_only == false);
    assert(this->poolnproc == 1);
    // ModuleBase::GlobalFunc::ZEROS(fft_bundle.get_auxr_3d_data<double>(), this->nxyz);
    base_device::memory::set_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
        this->fft_bundle.get_auxr_3d_data<double>(),
        0,
        this->nxyz);

    const int startig = ik * this->npwk_max;
    const int npw_k = this->npwk[ik];

    set_3d_fft_box_op<double, base_device::DEVICE_GPU>()(npw_k,
                                                         this->ig2ixyz_k + startig,
                                                         in,
                                                         this->fft_bundle.get_auxr_3d_data<double>());
    this->fft_bundle.fft3D_backward(this->fft_bundle.get_auxr_3d_data<double>(), this->fft_bundle.get_auxr_3d_data<double>());

    set_recip_to_real_output_op<double, base_device::DEVICE_GPU>()(this->nrxx,
                                                                   add,
                                                                   factor,
                                                                   this->fft_bundle.get_auxr_3d_data<double>(),
                                                                   out);

    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
}

template <typename FPTYPE>
void PW_Basis_K::real2recip_gpu(const std::complex<FPTYPE>* in,
                               std::complex<FPTYPE>* out,
                               const int ik,
                               const bool add,
                               const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
    assert(this->gamma_only == false);
    assert(this->poolnproc == 1);

    base_device::memory::synchronize_memory_op<std::complex<FPTYPE>,
                                               base_device::DEVICE_GPU,
                                               base_device::DEVICE_GPU>()(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                          in,
                                                                          this->nrxx);

    this->fft_bundle.fft3D_forward(this->fft_bundle.get_auxr_3d_data<FPTYPE>(), this->fft_bundle.get_auxr_3d_data<FPTYPE>());

    const int startig = ik * this->npwk_max;
    const int npw_k = this->npwk[ik];
    set_real_to_recip_output_op<FPTYPE, base_device::DEVICE_GPU>()(npw_k,
                                                                   this->nxyz,
                                                                   add,
                                                                   factor,
                                                                   this->ig2ixyz_k + startig,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                   out);
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
}
template <typename FPTYPE>
void PW_Basis_K::recip2real_gpu(const std::complex<FPTYPE>* in,
                               std::complex<FPTYPE>* out,
                               const int ik,
                               const bool add,
                               const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
    assert(this->gamma_only == false);
    assert(this->poolnproc == 1);
    // ModuleBase::GlobalFunc::ZEROS(fft_bundle.get_auxr_3d_data<FPTYPE>(), this->nxyz);
    base_device::memory::set_memory_op<std::complex<FPTYPE>, base_device::DEVICE_GPU>()(
        this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
        0,
        this->nxyz);

    const int startig = ik * this->npwk_max;
    const int npw_k = this->npwk[ik];

    set_3d_fft_box_op<FPTYPE, base_device::DEVICE_GPU>()(npw_k,
                                                         this->ig2ixyz_k + startig,
                                                         in,
                                                         this->fft_bundle.get_auxr_3d_data<FPTYPE>());
    this->fft_bundle.fft3D_backward(this->fft_bundle.get_auxr_3d_data<FPTYPE>(), this->fft_bundle.get_auxr_3d_data<FPTYPE>());

    set_recip_to_real_output_op<FPTYPE, base_device::DEVICE_GPU>()(this->nrxx,
                                                                   add,
                                                                   factor,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                   out);

    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
}

template void PW_Basis_K::real2recip_gpu<float>(const std::complex<float>*,
                                                std::complex<float>*,
                                                const int,
                                                const bool,
                                                const float) const;

template void PW_Basis_K::real2recip_gpu<double>(const std::complex<double>*,
                                                 std::complex<double>*,
                                                 const int,
                                                 const bool,
                                                 const double) const;

template void PW_Basis_K::recip2real_gpu<float>(const std::complex<float>*,
                                                std::complex<float>*,
                                                const int,
                                                const bool,
                                                const float) const;

template void PW_Basis_K::recip2real_gpu<double>(const std::complex<double>*,
                                                 std::complex<double>*,
                                                 const int,
                                                 const bool,
                                                 const double) const;

#endif

template void PW_Basis_K::real2recip<float>(const float* in,
                                            std::complex<float>* out,
                                            const int ik,
                                            const bool add,
                                            const float factor) const; // in:(nplane,nx*ny)  ; out(nz, ns)
template void PW_Basis_K::real2recip<float>(const std::complex<float>* in,
                                            std::complex<float>* out,
                                            const int ik,
                                            const bool add,
                                            const float factor) const; // in:(nplane,nx*ny)  ; out(nz, ns)
template void PW_Basis_K::recip2real<float>(const std::complex<float>* in,
                                            float* out,
                                            const int ik,
                                            const bool add,
                                            const float factor) const; // in:(nz, ns)  ; out(nplane,nx*ny)
template void PW_Basis_K::recip2real<float>(const std::complex<float>* in,
                                            std::complex<float>* out,
                                            const int ik,
                                            const bool add,
                                            const float factor) const; // in:(nz, ns)  ; out(nplane,nx*ny)

template void PW_Basis_K::real2recip<double>(const double* in,
                                             std::complex<double>* out,
                                             const int ik,
                                             const bool add,
                                             const double factor) const; // in:(nplane,nx*ny)  ; out(nz, ns)
template void PW_Basis_K::real2recip<double>(const std::complex<double>* in,
                                             std::complex<double>* out,
                                             const int ik,
                                             const bool add,
                                             const double factor) const; // in:(nplane,nx*ny)  ; out(nz, ns)
template void PW_Basis_K::recip2real<double>(const std::complex<double>* in,
                                             double* out,
                                             const int ik,
                                             const bool add,
                                             const double factor) const; // in:(nz, ns)  ; out(nplane,nx*ny)
template void PW_Basis_K::recip2real<double>(const std::complex<double>* in,
                                             std::complex<double>* out,
                                             const int ik,
                                             const bool add,
                                             const double factor) const; // in:(nz, ns)  ; out(nplane,nx*ny)
} // namespace ModulePW
