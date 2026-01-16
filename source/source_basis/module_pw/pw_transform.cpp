#include "source_base/global_function.h"
#include "source_base/timer.h"
#include "source_basis/module_pw/kernels/pw_op.h"
#include "source_base/module_fft/fft_bundle.h"
#include "pw_basis.h"
#include "pw_gatherscatter.h"

#include <cassert>
#include <complex>

namespace ModulePW
{
//     const base_device::DEVICE_CPU* PW_Basis::get_default_device_ctx() {
//         static const base_device::DEVICE_CPU* default_device_cpu;
//     return default_device_cpu;
// }
/**
 * @brief transform real space to reciprocal space
 * @details c(g)=\int dr*f(r)*exp(-ig*r)
 *          Here we calculate c(g)
 * @param in: (nplane,ny,nx), std::complex<double> data
 * @param out: (nz, ns),  std::complex<double> data
 */
template <typename FPTYPE>
void PW_Basis::real2recip(const std::complex<FPTYPE>* in,
                          std::complex<FPTYPE>* out,
                          const bool add,
                          const FPTYPE factor) const
{
    // 开始计时
    ModuleBase::timer::tick(this->classname, "real2recip");
    
    // 这是一个断言，程序运行到这里时，gamma_only必须为false，否则程序会中断并报错，
    // 也就是说该函数只适用于非Gamma_only的情况
    assert(this->gamma_only == false);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
    // 遍历所有实空间点，把输入的复数数据in[ir]逐个复制到FFT辅助缓冲区 auxr_data 中
    for (int ir = 0; ir < this->nrxx; ++ir)
    {
        this->fft_bundle.get_auxr_data<FPTYPE>()[ir] = in[ir];
    }
    // 对auxr_data进行二维FFT（x和y方向），结果仍然保存在auxr_data中。
    this->fft_bundle.fftxyfor(fft_bundle.get_auxr_data<FPTYPE>(), fft_bundle.get_auxr_data<FPTYPE>());

    // 把FFT变换后的按平面（Z 切片上的 XY 网格）存储/分布的数据，转换成按 stick（每根柱的 Z 连续数据）存储/分布的数据，
    // 目的是为后续的 Z 方向 FFT 做准备，结果存放在 auxg_data 中。
    this->gatherp_scatters(this->fft_bundle.get_auxr_data<FPTYPE>(), this->fft_bundle.get_auxg_data<FPTYPE>());

    // 对 auxg_data 缓冲区进行 z 方向的一维正向快速傅里叶变换（FFT）。
    // 输入和输出都是 auxg_data，即原地（in-place）变换。
    // 这一步是在完成了 x、y 方向的二维 FFT 和数据重排（gatherp_scatters）之后进行的。
    // 经过这一步，auxg_data 中的数据就从实空间完全变换到了倒空间，每根 stick 的 z 方向数据都被做了 FFT。
    // 其实是直接用 FFTW 库的 fftw_execute_dft 函数来执行预先创建好的 z 方向 FFT 计划。
    this->fft_bundle.fftzfor(fft_bundle.get_auxg_data<FPTYPE>(), fft_bundle.get_auxg_data<FPTYPE>());

    // 累加 (if (add))：
    // 该分支处理 add 为 true 的情况。意味着变换后的结果不会覆盖 out 数组，而是累加到 out 原有的值上。
    // 这在处理多个k点或波函数的线性组合时非常有用。
    /*
    归一化与缩放 (tmpfac)：
    FPTYPE tmpfac = factor / FPTYPE(this->nxyz);
    nxyz 是实空间总网格点数 (N_x × N_y × N_z)。
    在离散傅里叶变换中，从实空间到倒空间通常需要除以总格点数来保持物理量纲或满足特定的归一化约定。
    factor 是用户传入的额外缩放系数.

    this->ig2isz[ig]：这是一个索引映射表。由于 FFT 变换得到的是一个完整的稠密网格数据，
    而程序通常只关心能量截断半径内的平面波点（即 npw 个点），ig2isz 负责将平面波索引 ig 映射到 FFT 网格中的正确物理位置.

    循环操作：
    通过循环遍历所有的平面波 ig，从 FFT 结果缓冲区中取出对应的值，乘以归一化因子 tmpfac，最后累加到输出数组 out[ig] 中。

    一句话总结：
    这段代码的作用是从 FFT 完成后的稠密倒空间网格中，根据映射表提取出所需的平面波系数，经过归一化处理后累加到输出数组中。
    */ 
    // 默认add为false，factor为1.0
    if (add)
    {
        FPTYPE tmpfac = factor / FPTYPE(this->nxyz);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
        for (int ig = 0; ig < this->npw; ++ig)
        {
            out[ig] += tmpfac * this->fft_bundle.get_auxg_data<FPTYPE>()[this->ig2isz[ig]];
        }
    }
    // 在 FFT 变换完成后，将结果直接赋值（覆盖）到输出数组 out 中，而不是进行累加。
    else
    {
        FPTYPE tmpfac = 1.0 / FPTYPE(this->nxyz);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
        for (int ig = 0; ig < this->npw; ++ig)
        {
            out[ig] = tmpfac * this->fft_bundle.get_auxg_data<FPTYPE>()[this->ig2isz[ig]];
        }
    }
    ModuleBase::timer::tick(this->classname, "real2recip");
}

/**
 * @brief transform real space to reciprocal space
 * @details c(g)=\int dr*f(r)*exp(-ig*r)
 *          Here we calculate c(g)
 * @param in: (nplane,ny,nx), double data
 * @param out: (nz, ns),  std::complex<double> data
 */
template <typename FPTYPE>
void PW_Basis::real2recip(const FPTYPE* in, std::complex<FPTYPE>* out, const bool add, const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "real2recip");
    if (this->gamma_only)
    {
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
    }
    else
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
        for (int ir = 0; ir < this->nrxx; ++ir)
        {
            this->fft_bundle.get_auxr_data<FPTYPE>()[ir] = std::complex<FPTYPE>(in[ir], 0);
        }
        this->fft_bundle.fftxyfor(fft_bundle.get_auxr_data<FPTYPE>(), fft_bundle.get_auxr_data<FPTYPE>());
    }
    this->gatherp_scatters(this->fft_bundle.get_auxr_data<FPTYPE>(), this->fft_bundle.get_auxg_data<FPTYPE>());

    this->fft_bundle.fftzfor(fft_bundle.get_auxg_data<FPTYPE>(), fft_bundle.get_auxg_data<FPTYPE>());

    if (add)
    {
        FPTYPE tmpfac = factor / FPTYPE(this->nxyz);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
        for (int ig = 0; ig < this->npw; ++ig)
        {
            out[ig] += tmpfac * this->fft_bundle.get_auxg_data<FPTYPE>()[this->ig2isz[ig]];
        }
    }
    else
    {
        FPTYPE tmpfac = 1.0 / FPTYPE(this->nxyz);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
        for (int ig = 0; ig < this->npw; ++ig)
        {
            out[ig] = tmpfac * this->fft_bundle.get_auxg_data<FPTYPE>()[this->ig2isz[ig]];
        }
    }
    ModuleBase::timer::tick(this->classname, "real2recip");
}

/**
 * @brief transform reciprocal space to real space
 * @details f(r)=1/V * \sum_{g} c(g)*exp(ig*r)
 *          Here we calculate f(r)
 * @param in: (nz,ns), std::complex<double>
 * @param out: (nplane, ny, nx), std::complex<double>
 */
template <typename FPTYPE>
void PW_Basis::recip2real(const std::complex<FPTYPE>* in,
                          std::complex<FPTYPE>* out,
                          const bool add,
                          const FPTYPE factor) const
{
    // 开始计时
    ModuleBase::timer::tick(this->classname, "recip2real");
    // 这是一个断言，程序运行到这里时，gamma_only必须为false，否则程序会中断并报错，
    // 也就是说该函数只适用于非Gamma_only的情况
    assert(this->gamma_only == false);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
    //该循环的作用是将 FFT 的倒空间辅助缓冲区 auxg_data 全部初始化为复数零(0,0) 。
    // this->nst：当前进程所拥有的本地柱（Sticks）的数量。每一根柱对应 xy 网格中的一个点。
    // this->nz：z 方向的网格点数。
    // this->nst * this->nz：代表了倒空间中所有有效网格点的总数。
    for (int i = 0; i < this->nst * this->nz; ++i)
    {
        fft_bundle.get_auxg_data<FPTYPE>()[i] = std::complex<FPTYPE>(0, 0);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
/*
1. 核心作用：数据填充/撒点（Scatter）
该循环的作用是将输入的稀疏平面波系数 in[ig] 放置到 FFT 辅助缓冲区 auxg_data 的对应稠密网格位置上。

2. 关键变量含义：
this->npw：处于能量截断半径内的平面波（G 向量）总数。
in[ig]：输入的倒空间系数数组，仅存储了有效的平面波点。
this->ig2isz[ig]：映射表。它记录了第 ig 个平面波在完整 FFT 倒空间网格中的线性索引位置。
fft_bundle.get_auxg_data<FPTYPE>()：准备进行逆 FFT 变换的稠密网格缓冲区。

为什么这样做？
从压缩到展开：在 ABACUS 中，为了节省内存，波函数等物理量在倒空间只存储能量截断以内的部分（npw 个点）。
符合 FFT 要求：快速傅里叶变换（FFT）要求输入必须是一个完整的、覆盖整个布里渊区的规则网格数据。
定位数据：通过 ig2isz 映射表，将原本连续存储的平面波系数“散布”到网格中正确的物理位置上。

总结
这一步紧接在“缓冲区清零”之后，其目的是构建逆 FFT 变换所需的稠密倒空间网格数据。
完成此步后，缓冲区中只有对应的 G 点处有值，其余位置均为零，随后即可调用 fftzbac 等函数执行逆变换。
*/
    for (int ig = 0; ig < this->npw; ++ig)
    {
        this->fft_bundle.get_auxg_data<FPTYPE>()[this->ig2isz[ig]] = in[ig];
    }
    // 对 auxg_data 缓冲区执行 z 方向的一维逆向快速傅里叶变换 (Backward FFT)。
    // 这是将数据从倒空间转回实空间（逆变换）的第一步，负责在本地对每个柱（Stick）执行 z 方向的一维逆 FFT。
    this->fft_bundle.fftzbac(fft_bundle.get_auxg_data<FPTYPE>(), fft_bundle.get_auxg_data<FPTYPE>());

    // 将数据从 “柱状布局（Stick Layout）” 重新分布回 “平面布局（Plane Layout）”，
    // 将数据从适合 z 变换的状态调整为适合 xy 变换的状态。
    // 输入：auxg_data。此时该缓冲区存放的是刚刚完成 z 方向一维逆 FFT（fftzbac）后的数据。
    // 数据的组织方式是按柱（Sticks）排列的（每根柱在内存中 z 方向连续）。
    // 输出：auxr_data。重分布后的数据将存放在此。数据的组织方式将变为按 XY 平面（Planes）切分。
    this->gathers_scatterp(this->fft_bundle.get_auxg_data<FPTYPE>(), this->fft_bundle.get_auxr_data<FPTYPE>());

    // 对 auxr_data 缓冲区执行xy 方向的二维逆向快速傅里叶变换，
    // 执行完毕后，auxr_data 缓冲区中存储的就是实空间网格上的复数数值。
    this->fft_bundle.fftxybac(fft_bundle.get_auxr_data<FPTYPE>(), fft_bundle.get_auxr_data<FPTYPE>());

    // 累加逻辑 (if (add))：处理 add 为 true 的情况。变换后的实空间网格数据不会直接覆盖 out 数组，
    // 而是通过 += 运算累加到 out 原有的值上。
    // 默认add为false，factor为1.0。
    // this->nrxx：代表当前进程所负责的实空间网格点总数。
    // 循环遍历所有本地点，将 auxr_data（稠密网格结果）搬运到最终的 out 存储区域。
    if (add)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
        for (int ir = 0; ir < this->nrxx; ++ir)
        {
            out[ir] += factor * this->fft_bundle.get_auxr_data<FPTYPE>()[ir];
        }
    }
    // 当调用者设置 add = false 时进入此分支。这表示变换后的实空间数据将直接覆盖输出数组 out，而不是累加。
    // 将缓冲区中的每一个网格点数值直接赋值给目标输出数组 out。
    // nrxx 代表当前进程所负责的本地实空间网格点总数。由于采用了并行分解，每个进程只处理并拷贝自己负责的那一部分物理空间数据。
    else
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
        for (int ir = 0; ir < this->nrxx; ++ir)
        {
            out[ir] = this->fft_bundle.get_auxr_data<FPTYPE>()[ir];
        }
    }
    ModuleBase::timer::tick(this->classname, "recip2real");
}

/**
 * @brief transform reciprocal space to real space
 * @details f(r)=1/V * \sum_{g} c(g)*exp(ig*r)
 *          Here we calculate f(r)
 * @param in: (nz,ns), std::complex<double>
 * @param out: (nplane, ny, nx), double
 */
template <typename FPTYPE>
void PW_Basis::recip2real(const std::complex<FPTYPE>* in, FPTYPE* out, const bool add, const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "recip2real");
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
    for (int i = 0; i < this->nst * this->nz; ++i)
    {
        fft_bundle.get_auxg_data<FPTYPE>()[i] = std::complex<FPTYPE>(0, 0);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
    for (int ig = 0; ig < this->npw; ++ig)
    {
        this->fft_bundle.get_auxg_data<FPTYPE>()[this->ig2isz[ig]] = in[ig];
    }
    this->fft_bundle.fftzbac(fft_bundle.get_auxg_data<FPTYPE>(), fft_bundle.get_auxg_data<FPTYPE>());

    this->gathers_scatterp(this->fft_bundle.get_auxg_data<FPTYPE>(), this->fft_bundle.get_auxr_data<FPTYPE>());

    if (this->gamma_only)
    {
        this->fft_bundle.fftxyc2r(fft_bundle.get_auxr_data<FPTYPE>(), fft_bundle.get_rspace_data<FPTYPE>());

        // r2c in place
        const int npy = this->ny * this->nplane;

        if (add)
        {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 4096 / sizeof(FPTYPE))
#endif
            for (int ix = 0; ix < this->nx; ++ix)
            {
                for (int ipy = 0; ipy < npy; ++ipy)
                {
                    out[ix * npy + ipy] += factor * this->fft_bundle.get_rspace_data<FPTYPE>()[ix * npy + ipy];
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
                    out[ix * npy + ipy] = this->fft_bundle.get_rspace_data<FPTYPE>()[ix * npy + ipy];
                }
            }
        }
    }
    else
    {
        this->fft_bundle.fftxybac(fft_bundle.get_auxr_data<FPTYPE>(), fft_bundle.get_auxr_data<FPTYPE>());
        if (add)
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
            for (int ir = 0; ir < this->nrxx; ++ir)
            {
                out[ir] += factor * this->fft_bundle.get_auxr_data<FPTYPE>()[ir].real();
            }
        }
        else
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
            for (int ir = 0; ir < this->nrxx; ++ir)
            {
                out[ir] = this->fft_bundle.get_auxr_data<FPTYPE>()[ir].real();
            }
        }
    }
    ModuleBase::timer::tick(this->classname, "recip2real");
}
template void PW_Basis::real2recip<float>(const float* in,
                                          std::complex<float>* out,
                                          const bool add,
                                          const float factor) const; // in:(nplane,nx*ny)  ; out(nz, ns)
template void PW_Basis::real2recip<float>(const std::complex<float>* in,
                                          std::complex<float>* out,
                                          const bool add,
                                          const float factor) const; // in:(nplane,nx*ny)  ; out(nz, ns)
template void PW_Basis::recip2real<float>(const std::complex<float>* in,
                                          float* out,
                                          const bool add,
                                          const float factor) const; // in:(nz, ns)  ; out(nplane,nx*ny)
template void PW_Basis::recip2real<float>(const std::complex<float>* in,
                                          std::complex<float>* out,
                                          const bool add,
                                          const float factor) const;

template void PW_Basis::real2recip<double>(const double* in,
                                           std::complex<double>* out,
                                           const bool add,
                                           const double factor) const; // in:(nplane,nx*ny)  ; out(nz, ns)
template void PW_Basis::real2recip<double>(const std::complex<double>* in,
                                           std::complex<double>* out,
                                           const bool add,
                                           const double factor) const; // in:(nplane,nx*ny)  ; out(nz, ns)
template void PW_Basis::recip2real<double>(const std::complex<double>* in,
                                           double* out,
                                           const bool add,
                                           const double factor) const; // in:(nz, ns)  ; out(nplane,nx*ny)
template void PW_Basis::recip2real<double>(const std::complex<double>* in,
                                           std::complex<double>* out,
                                           const bool add,
                                           const double factor) const;
} // namespace ModulePW