// This file contains interface to xc_functional class:
// 1. v_xc : which takes rho as input, and v_xc as output
// 2. v_xc_libxc : which does the same thing as v_xc, but calling libxc
// NOTE : it is only used for nspin = 1 and 2, the nspin = 4 case is treated in v_xc
// 3. v_xc_meta : which takes rho and tau as input, and v_xc as output

#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_io/module_parameter/parameter.h"
#include "xc_functional.h"

#ifdef USE_LIBXC
#include "xc_functional_libxc.h"
#endif

// [etxc, vtxc, v] = XC_Functional::v_xc(...)
std::tuple<double, double, ModuleBase::matrix> XC_Functional::v_xc(const int& nrxx, // number of real-space grid
                                                                   const Charge* const chr,
                                                                   const UnitCell* ucell) // core charge density
{
    ModuleBase::TITLE("XC_Functional", "v_xc");
    // 标记 XC_Functional 模块下 v_xc 函数的开始时间点
    ModuleBase::timer::tick("XC_Functional", "v_xc");

    if (use_libxc)
    {
#ifdef USE_LIBXC
        ModuleBase::timer::tick("XC_Functional", "v_xc");
        return XC_Functional_Libxc::v_xc_libxc(XC_Functional::get_func_id(),
                                               nrxx,
                                               ucell->omega,
                                               ucell->tpiba,
                                               chr,
                                               &(scaling_factor_xc));
#else
        ModuleBase::WARNING_QUIT("v_xc", "compile with LIBXC");
#endif
    }

    //Exchange-Correlation potential Vxc(r) from n(r)
    double etxc = 0.0;
    double vtxc = 0.0;
    ModuleBase::matrix v(PARAM.inp.nspin, nrxx);

    // the square of the e charge
    // in Rydeberg unit, so * 2.0.
    double e2 = 2.0;

    double vanishing_charge = 1.0e-10;

    if (PARAM.inp.nspin == 1 || ( PARAM.inp.nspin ==4 && !PARAM.globalv.domag && !PARAM.globalv.domag_z))
    {
        // 自旋非极化，只考虑一个自旋分量
        // spin-unpolarized case 
#ifdef _OPENMP
// reduction(+:etxc) 和 reduction(+:vtxc) 表示在并行过程中，
// 每个线程分别维护自己的 etxc 和 vtxc 局部变量，循环结束后自动把所有线程的结果累加到全局变量上，
// etxc 和 vtxc 是在循环中不断累加的变量（分别代表交换-相关能量和势的积分）。
// 如果直接在多线程下累加，容易出现数据竞争和错误。
// 用 reduction，OpenMP会自动处理每个线程的累加，保证并行安全和结果正确
#pragma omp parallel for reduction(+:etxc) reduction(+:vtxc)
#endif
        for (int ir = 0;ir < nrxx;ir++)
        {
            // total electron charge density
            // rho[自旋分量][网格点]：第一维表示自旋分量（如自旋上、下，或非共线的多个分量），第二维表示实空间网格点。
            // 这样可以同时存储每个网格点上的不同自旋分量的电子密度。
            // rho_core[网格点]：只表示每个网格点上的核芯电子密度，不区分自旋分量。
            // 因为在赝势方法中，核芯电子通常被认为是自旋非极化的（即对所有自旋分量贡献相同），所以只需一维数组。
            // 计算实空间网格点 ir 处的总电子密度， 
            // 总密度 = 价电子密度（通常由波函数计算得到）加上核芯电子密度（由赝势提供）。
            double rhox = chr->rho[0][ir] + chr->rho_core[ir];

            double arhox = std::abs(rhox);
            if (arhox > vanishing_charge)
            {
                double exc = 0.0;
                double vxc = 0.0;
                // 计算交换-关联能量密度（exc）和交换-关联势（vxc）
                XC_Functional::xc(arhox, exc, vxc);
                // 乘以Rydeberg单位换算因子，赋值给势矩阵 v 的第0个自旋分量、当前网格点 ir。
                // 这样就得到了实空间上每个网格点的交换-关联势。
                v(0,ir) = e2 * vxc;
                // consider the total charge density
                // 把当前网格点的交换-关联能量密度乘以电子密度和单位因子，累加到总的交换-相关能量 etxc 上
                etxc += e2 * exc * rhox;
                // only consider chr->rho
                // 把当前网格点的交换-关联势乘以价电子密度，累加到总的 vtxc 上
                vtxc += v(0, ir) * chr->rho[0][ir];
            } // endif
        } //enddo
    }
    else if(PARAM.inp.nspin ==2)
    {
        // 自旋极化
        // spin-polarized case
#ifdef _OPENMP
#pragma omp parallel for reduction(+:etxc) reduction(+:vtxc)
#endif
        for (int ir = 0;ir < nrxx;ir++)
        {
            // 把自旋上、下的价电子密度和核芯电子密度相加，得到每个网格点的总电子密度
            double rhox = chr->rho[0][ir] + chr->rho[1][ir] + chr->rho_core[ir]; //HLX(05-29-06): bug fixed
            double arhox = std::abs(rhox);

            if (arhox > vanishing_charge)
            {
                // 计算自旋极化参数 zeta，定义为自旋密度差与总密度的比值，反映了自旋极化程度，
                // 这个参数是自旋极化交换-相关泛函（如LSDA、GGA等）的核心输入之一。
                // 分子 chr->rho[0][ir] - chr->rho[1][ir] 是价电子的自旋向上和向下的密度差。
                // 由于芯电子密度 rho_core 被认为是无极化的（即 ρcore,↑=ρcore,↓），
                // 所以总密度的差值等同于价电子密度的差值。分母 arhox 是自旋上和下的总密度之和。
                // 分子分母都是总的自旋上和总的自旋下的密度，只是因为芯电子密度本身不带自旋，即自旋向上和向下的贡献相等，
                // 所以分子的差中就没有了 rho_core 项。
                double zeta = (chr->rho[0][ir] - chr->rho[1][ir]) / arhox; //HLX(05-29-06): bug fixed

                // zeta = 1表示全自旋向上，zeta = -1表示全自旋向下，zeta = 0 表示无自旋极化（自旋上、下电子数相等）。
                // 物理上自旋极化参数 (zeta) 的取值范围应在 ([-1, 1]) 之间。
                // 由于数值误差（如在密度极小的区域），zeta 可能略超出这个范围，
                // 所以这里做了截断，确保其数值在 ([-1, 1]) 区间内，保证其物理合理性
                if (std::abs(zeta)  > 1.0)
                {
                    zeta = (zeta > 0.0) ? 1.0 : (-1.0);
                }
                
                // 虽然芯电子密度 rho_core 本身不带自旋（即自旋向上和向下的贡献相等），
                // 但在计算交换-相关泛函时，必须考虑总密度对泛函的影响。
                // 在自旋极化计算中，我们将芯电子密度平均分配给两个自旋通道：
                // 总自旋向上密度：ρtotal,↑ = ρvalence,↑ + ρcore / 2
                // 总自旋向下密度：ρtotal,↓ = ρvalence,↓ + ρcore / 2
                // rhoup = arhox * (1.0+zeta) / 2.0 = (arhox + rho[0] - rho[1]) / 2
                //       = (rho[0] + rho[1] + rho_core + rho[0] - rho[1]) / 2 = rho[0] + 0.5 * rho_core
                // 同理：rhodw = rho[1] + 0.5 * rho_core
                // 代码通过这种方式，实际上是将不具自旋的 rho_core 对半平分到了 rhoup 和 rhodw 中。
                // 目的：这样得到的 rhoup 和 rhodw 包含了完整的电子密度信息。
                double rhoup = arhox * (1.0+zeta) / 2.0;
                double rhodw = arhox * (1.0-zeta) / 2.0;
                double exc = 0.0;
                double vxc[2];
                XC_Functional::xc_spin(arhox, zeta, exc, vxc[0], vxc[1]);

                // 对每个自旋分量（is=0,1）分别赋值交换-关联势。
                for (int is = 0;is < PARAM.inp.nspin;is++)
                {
                    // v(is, ir) 表示第 is 个自旋分量在第 ir 个实空间网格点的交换-相关势
                    v(is, ir) = e2 * vxc[is];
                }
                // 把当前网格点的交换-关联能量密度乘以电子密度和单位因子，累加到总的交换-相关能量 etxc 上
                etxc += e2 * exc * rhox;
                // 把当前网格点的自旋上和自旋下的交换-相关势乘以对应的电子密度，累加到总的 vtxc 上
                vtxc += v(0, ir) * chr->rho[0][ir] + v(1, ir) * chr->rho[1][ir];
            }
        }
    }
    else if(PARAM.inp.nspin == 4)//noncollinear case added by zhengdy
    {
#ifdef _OPENMP
#pragma omp parallel for reduction(+:etxc) reduction(+:vtxc)
#endif
        for(int ir = 0;ir<nrxx; ir++)
        {
            // chr->rho[0][ir]：总电子密度（标量密度），即该网格点上所有电子的数目，与自旋方向无关，与其他三个分量无代数关系。
            // chr->rho[1][ir], chr->rho[2][ir], chr->rho[3][ir]：自旋密度矢量的三个分量，
            // 分别对应自旋在 x、y、z 方向上的分布情况。
            // amag为非共线自旋体系下的磁矩模长，反映了该点自旋极化的强度，只与自旋的空间分量（x、y、z）有关
            double amag = sqrt( pow(chr->rho[1][ir],2) + pow(chr->rho[2][ir],2) + pow(chr->rho[3][ir],2) );

            // 计算该网格点的总电子密度，与其他三个自旋分量无关。
            double rhox = chr->rho[0][ir] + chr->rho_core[ir];

            double arhox = std::abs( rhox );

            if ( arhox > vanishing_charge )
            {
                // 计算自旋极化参数 zeta，定义为磁矩模长与总密度的比值，反映了自旋极化程度
                double zeta = amag / arhox; 
                double exc = 0.0;
                double vxc[2];
                
                // zeta取值范围应在[−1,1] 之间
                if ( std::abs( zeta ) > 1.0 )
                {
                    zeta = (zeta > 0.0) ? 1.0 : (-1.0);
                }//end if

                if(use_libxc)
                {
#ifdef USE_LIBXC
                    double rhoup = arhox * (1.0+zeta) / 2.0;
                    double rhodw = arhox * (1.0-zeta) / 2.0;
                    XC_Functional_Libxc::xc_spin_libxc(XC_Functional::get_func_id(), rhoup, rhodw, exc, vxc[0], vxc[1]);
#else
                    ModuleBase::WARNING_QUIT("v_xc", "compile with LIBXC");
#endif                    
                }
                else
                {
                    double rhoup = arhox * (1.0+zeta) / 2.0;
                    double rhodw = arhox * (1.0-zeta) / 2.0;
                    XC_Functional::xc_spin(arhox, zeta, exc, vxc[0], vxc[1]);
                }

                etxc += e2 * exc * rhox;

                // vxc[0] 和 vxc[1] 分别是用自旋极化泛函（如LSDA、GGA）计算得到的自旋向上和自旋向下的交换-相关势。
                // 由于非共线自旋体系的主分量（v(0, ir)）对应于总电子密度，所以这里取自旋上、下势的平均值，
                // 以让主分量势正确代表所有电子的交换-相关作用，而不是偏向某一自旋方向
                v(0, ir) = e2*( 0.5 * ( vxc[0] + vxc[1]) );
                // 将主分量的交换-相关势与主分量的电子密度相乘，并累加到总的 vtxc 上。
                vtxc += v(0,ir) * chr->rho[0][ir];

                // vs为计算自旋极化泛函（如LSDA、GGA）得到的自旋上、下交换-关联势的差的一半，
                // 代表了交换-关联作用对自旋极化的响应强度。
                double vs = 0.5 * ( vxc[0] - vxc[1] );
                // 只有当磁矩模长大于阈值（即该点有明显自旋极化）时，才计算自旋分量的交换-关联势。
                if ( amag > vanishing_charge )
                {
                    // 循环处理自旋密度矢量的三个分量（x、y、z）
                    for(int ipol = 1;ipol< 4;ipol++)
                    {
                        // chr->rho[ipol][ir] / amag 计算的是局部磁矩矢量在 ipol 方向（x, y, 或 z）上的单位方向分量（方向余弦）。
                        // 矢量化投影：将标量强度 vs 乘以该方向单位分量，从而将势能分配到笛卡尔坐标系的三个轴上
                        // 公式表示为 v_s * (m_ipol / |m|)
                        // 这种做法是物理上假设在每一个空间网格点上，交换-关联势产生的“有效磁场”方向与该点处的局部磁矩方向m完全平行，
                        // 所以能用vs直接乘自旋密度的方向余弦来得到vs在各个方向上的分量。
                        // 该代码通过方向投影，将标量自旋势转化为矢量势分量，是实现非共线自旋计算的核心数学手段。
                        v(ipol, ir) = e2 * vs * chr->rho[ipol][ir] / amag; 
                        // 将每个自旋分量的交换-关联势与对应自旋密度的乘积累加到总的 vtxc 上。
                        vtxc += v(ipol,ir) * chr->rho[ipol][ir];
                    }//end do
                }//end if
            }//end if
        }//end do
    }//end if
    // energy terms, local-density contributions

    // add gradient corrections (if any)
    // mohan modify 2009-12-15

    // the dummy variable dum contains gradient correction to stress
    // which is not used here
    std::vector<double> dum;
    gradcorr(etxc, vtxc, v, chr, chr->rhopw, ucell, dum);

    // parallel code : collect vtxc,etxc
    // mohan add 2008-06-01
#ifdef __MPI
// reduce_pool是一个 MPI 封装函数（通常底层调用 MPI_Allreduce 且操作符为 MPI_SUM）。
// 它会将所有参与计算的进程中的 etxc 和 vtxc 变量值相加，并将最终的总和重新分发给每一个进程。
// 虽然本文件里前面并没有MPI，但实空间网格可能在前面其他文件里就被切分到了不同的 MPI 进程中,
// 所以本文件的v_xc()函数参数中的 nrxx 并不是整个体系的总网格点数，而是当前 MPI 进程所分配到的局部网格点数
// 当前进程MPI在前面又使用了OPENMP进一步把网格切分为多线程计算
// 而物理上的总能量和总积分必须是全空间的积分，
// 因此，必须通过 MPI 通信，把所有进程手中的局部能量 etxc 和势积分 vtxc 加在一起，才能得到体系完整的物理量。
    Parallel_Reduce::reduce_pool(etxc);
    Parallel_Reduce::reduce_pool(vtxc);
#endif
    // ucell->omega：晶胞体积（即整个体系的体积）。chr->rhopw->nxyz：实空间网格的总点数。
    // 晶胞体积除以网格点数，就是每个网格点对应的体积（ΔV）。
    // 之前的循环中，etxc 和 vtxc 是对每个网格点的能量密度和势密度进行了求和，
    // 在数学上，连续积分在离散网格上的近似公式为：∫ f(r) dr ≈ Σ f(ri) * ΔV
    // 因为每个网格点的体积相等，所以可以先对格点处的能量或势密度求和，
    // 最后再乘以每个网格点的体积，得到近似的积分值。
    // 这一步是将离散求和转换为连续积分的关键步骤。
    etxc *= ucell->omega / chr->rhopw->nxyz;
    vtxc *= ucell->omega / chr->rhopw->nxyz;

    // 用于性能计时，标记 XC_Functional 模块下 v_xc 函数的结束时间点
    // 与函数开头的 tick 配对使用，可以统计整个 v_xc 函数的运行耗时
    ModuleBase::timer::tick("XC_Functional", "v_xc");
   
    // std::move(v)：作用是将矩阵 v 以右值引用的方式传递出去，
    // 避免在返回时发生一次不必要的深拷贝，提高效率，
    // 这样可以直接把局部变量 v 的内存“转交”给返回值，尤其适合大矩阵或大对象的返回。
    // make_tuple(): 用于创建并返回一个包含 etxc、vtxc 和 v 三个元素的元组,
    // 将多个不同类型的返回值打包成一个 tuple，这样函数就可以一次性返回多个结果。
    return std::make_tuple(etxc, vtxc, std::move(v));
}
