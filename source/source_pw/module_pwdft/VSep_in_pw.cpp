#include "VSep_in_pw.h"

#include "source_base/constants.h"
#include "source_base/libm/libm.h"
#include "source_base/math_integral.h"
#include "source_base/timer.h"
#include "source_base/tool_title.h"
#include "source_cell/sep.h"
#include "source_cell/sep_cell.h"

#include <cassert>
#include <cmath>
#include <complex>
#include <vector>

// namespace GlobalC
// {
// VSep vsep_cell;
// }
namespace
{
double sphere_cut(double r, double r_out, double r_power)
{
    if (r <= 0 || r >= r_out)
    {
        return 0.0;
    }
    // DFT-1/2球形截断函数
    return std::pow(1 - std::pow(r / r_out, r_power), 3);
}

double shell_cut(double r, double r_in, double r_out, double r_power)
{
    if (r_in <= 0)
    {
        return sphere_cut(r, r_out, r_power);
    }
    if (r_in >= r_out)
    {
        return 0.0;
    }
    if (r <= r_in || r >= r_out)
    {
        return 0.0;
    }
    //shell DFT-1/2球壳形截断函数
    return std::pow(1 - std::pow(2 * (r - r_in) / (r_out - r_in) - 1.0, r_power), 3);
}
} // namespace

VSep::VSep() = default;

VSep::~VSep() = default;

// 读取并处理外部输入的自能势数据，生成G空间的自能势分量，供哈密顿量使用。
void VSep::init_vsep(const ModulePW::PW_Basis& rho_basis, const Sep_Cell& sep_cell)
{
    ModuleBase::TITLE("VSep", "init_vsep");
    ModuleBase::timer::tick("VSep", "init_vsep");

    int ntype = sep_cell.get_ntype();

    this->vsep_form.create(ntype, rho_basis.ngg, true);

    const double d_fpi_omega = ModuleBase::FOUR_PI / sep_cell.get_omega();
    int igl0 = 0;
    for (int it = 0; it < ntype; ++it)
    {
        // 对每种原子类型，若启用DFT-1/2修正（get_sep_enable()），则读取其自能势参数
        if (!sep_cell.get_sep_enable()[it])
        {
            continue;
        }
        const SepPot* sep_pot = &sep_cell.get_seps()[it];
        // Simpson integral requires that the grid points be odd, if it is even, subtract one.
        int mesh = sep_pot->mesh;
        if ((mesh & 1) == 0)
        {
            mesh--;
        }

        double* r = sep_pot->r;
        double* rv = sep_pot->rv;
        std::vector<double> shell_rv(sep_pot->mesh);
        std::vector<double> rab(sep_pot->mesh);
        std::vector<double> aux(sep_pot->mesh);

        // calculate a and b of r [i] = a * (exp(b*i) - 1), i = 1,..., mesh. Note: no 0
        // for rab[i] = (r[i] + a) * b
        double b_val = log(r[1] / r[0] - 1);
        double a_val = r[0] / (exp(b_val) - 1);

        for (int ir = 0; ir < sep_pot->mesh; ++ir)
        {
            shell_rv[ir]
                = shell_cut(r[ir], sep_pot->r_in, sep_pot->r_out, sep_pot->r_power) * rv[ir] * sep_pot->enhence_a;
            rab[ir] = (r[ir] + a_val) * b_val;
        }

        igl0 = 0;
        // start from |G|=0 or not.
        if (rho_basis.gg_uniq[0] < 1.0e-8)
        {
            for (int ir = 0; ir < sep_pot->mesh; ++ir)
            {
                aux[ir] = r[ir] * shell_rv[ir];
            }
            ModuleBase::Integral::Simpson_Integral(mesh, aux.data(), rab.data(), this->vsep_form(it, 0));
            this->vsep_form(it, 0) *= d_fpi_omega;
            igl0 = 1;
        }

        for (int ig = igl0; ig < rho_basis.ngg; ++ig)
        {
            double gx2 = rho_basis.gg_uniq[ig] * sep_cell.get_tpiba2();
            double gx = std::sqrt(gx2);
            for (int ir = 0; ir < sep_pot->mesh; ++ir)
            {
                aux[ir] = shell_rv[ir] * ModuleBase::libm::sin(gx * r[ir]) / gx;
            }
            ModuleBase::Integral::Simpson_Integral(mesh, aux.data(), rab.data(), this->vsep_form(it, ig));
            this->vsep_form(it, ig) *= d_fpi_omega;
        }
    }

    ModuleBase::timer::tick("VSep", "init_vsep");
}

// 将G空间的自能势转换到实空间网格，用于后续的实空间操作或分析
void VSep::generate_vsep_r(const ModulePW::PW_Basis& rho_basis,
                           const ModuleBase::ComplexMatrix& sf_in,
                           const Sep_Cell& sep_cell)
{
    ModuleBase::TITLE("VSep", "generate_vsep_r");
    ModuleBase::timer::tick("VSep", "generate_vsep_r");

    //初始化实空间网格点数 nrxx，并分配 vsep_r 存储空间
    this->nrxx = rho_basis.nrxx;
    this->vsep_r.assign(rho_basis.nrxx, 0.0);

    std::unique_ptr<std::complex<double>[]> vg(new std::complex<double>[rho_basis.npw]);
    ModuleBase::GlobalFunc::ZEROS(vg.get(), rho_basis.npw);

    //为每个原子类型和每个G点，累加自能势的傅里叶分量（vsep_form(it, ig)）和结构因子（sf_in(it, ig)）的乘积
    for (int it = 0; it < sep_cell.get_ntype(); it++)
    {
        if (!sep_cell.get_sep_enable()[it])
        {
            continue;
        }

        for (int ig = 0; ig < rho_basis.npw; ++ig)
        {
            vg[ig] += this->vsep_form(it, rho_basis.ig2igg[ig]) * sf_in(it, ig);
        }
    }

    //调用 recip2real 方法，将G空间的自能势 vg 傅里叶变换到实空间，得到 vsep_r
    rho_basis.recip2real(vg.get(), this->vsep_r.data());

    ModuleBase::timer::tick("VSep", "generate_vsep_r");
}
