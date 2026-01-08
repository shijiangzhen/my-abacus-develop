// This file contains wrapper for the GGA functionals
// it includes 4 subroutines:
// 1. gcxc, which is the wrapper for gradient correction part
// 2. gcx_spin, spin polarized, exchange only
// 3. gcc_spin, spin polarized, correlation only
// 4. gcxc_libxc, the entire GGA functional, LIBXC, for nspin=1 case
// 5. gcxc_spin_libxc, the entire GGA functional, LIBXC, for nspin=2 case

#include "xc_functional.h"
#include <stdexcept>
#include "source_pw/module_pwdft/global.h"
#include "source_base/global_function.h"

#ifdef USE_LIBXC
#include "xc_functional_libxc.h"
#endif

void XC_Functional::gcxc(const double &rho, const double &grho, double &sxc,
          double &v1xc, double &v2xc)
{
    //-----------------------------------------------------------------------
    //     gradient corrections for exchange and correlation - Hartree a.u.
    //     exchange  :  Becke88
    //                  GGA (Generalized Gradient Approximation), PW91
    //                  PBE
    //                  revPBE
    //     correlation: Perdew86
    //                  GGA (PW91)
    //                  Lee-Yang-Parr
    //                  PBE
    //
    //     input:  rho, grho=|\nabla rho|^2
    //     definition:  E_x = \int E_x(rho,grho) dr
    //     output: sx = E_x(rho,grho)
    //             v1x= D(E_x)/D(rho)
    //             v2x= D(E_x)/D( D rho/D r_alpha ) / |\nabla rho|
    //             sc, v1c, v2c as above for correlation
    //
    // use funct
    // USE kinds
    // implicit none
    // real rho, grho, sx, sc, v1x, v2x, v1c, v2c;
    
    // rho：电子密度的绝对值
    // grho：rho的梯度的平方
    // sx：单位体积的交换能（rho和grho的函数）
    // v1：单位体积的交换或关联能对密度的导数
    // v2：单位体积的交换能或关联能对密度的梯度的导数再除以梯度
    const double small = 1.e-6;
    const double smallg = 1.e-10;
    double s,v1,v2;
    sxc = v1xc = v2xc = 0.0;

    // 如果密度或其梯度过小，则不进行计算，直接返回零值，
    // 避免在物理上无意义或数值不稳定的区域（如真空区或极小密度区）进行复杂的泛函计算
    if (rho <= small || grho < smallg)
    {
        return;
    }

    for(int id : func_id)
    {
        switch( id )
        {
            case XC_GGA_X_B88: //B88
                XC_Functional::becke88(rho, grho, s, v1, v2);break;
            case XC_GGA_X_PW91: //PW91_X
                XC_Functional::ggax(rho, grho, s, v1, v2);break;
            case XC_GGA_X_PBE: //PBX
                XC_Functional::pbex(rho, grho, 0, s, v1, v2);break;
            case XC_GGA_X_PBE_R: //revised PBX
                XC_Functional::pbex(rho, grho, 1, s, v1, v2);break;
            case XC_GGA_X_HCTH_A: //HCTH_X
                XC_Functional::hcth(rho, grho, s, v1, v2);break; //XC together
            case XC_GGA_C_HCTH_A: //HCTH_C
                s = 0.0; v1 = 0.0; v2 = 0.0;break;
            case XC_GGA_X_OPTX: //OPTX
                XC_Functional::optx(rho, grho, s, v1, v2);break;
            case XC_GGA_X_PBE_SOL: //PBXsol
                XC_Functional::pbex(rho, grho, 2, s, v1, v2);break;
            case XC_GGA_X_WC: //Wu-Cohen
                XC_Functional::wcx (rho, grho, s, v1, v2);break;
            case XC_GGA_C_P86: //P86
                XC_Functional::perdew86(rho, grho, s, v1, v2);break;
            case XC_GGA_C_PW91: //PW91_C
                XC_Functional::ggac(rho, grho, s, v1, v2);break;
            case XC_GGA_C_PBE: //PBC
                XC_Functional::pbec(rho, grho, 0, s, v1, v2);break;
            case XC_GGA_C_PBE_SOL: //PBCsol
                XC_Functional::pbec(rho, grho, 1, s, v1, v2);break;
            case XC_GGA_C_LYP: //BLYP
                XC_Functional::glyp(rho, grho, s, v1, v2); break;
            case XC_HYB_GGA_XC_PBEH: //PBE0
                double sx, v1x, v2x, sc, v1c, v2c;
                XC_Functional::pbex(rho, grho, 0, sx, v1x, v2x);
                // 计算杂化泛函(PBE0)的GGA交换部分的能量密度和势（单位体积），按混合比例缩放
                sx *= (1.0 - XC_Functional::hybrid_alpha); 
                v1x *= (1.0 - XC_Functional::hybrid_alpha); 
                v2x *= (1.0 - XC_Functional::hybrid_alpha);
                // 关联部分全部使用 pbe 泛函
                XC_Functional::pbec(rho, grho, 0, sc, v1c, v2c);
                // 将交换和关联部分的能量密度、势分别相加，得到杂化泛函GGA部分的总能量密度和势的两部分（单位体积）。
                s = sx + sc;
                v1 = v1x + v1c;
                v2 = v2x + v2c;
                break;
            default: //SCAN_X,SCAN_C,HSE, and so on
                throw std::domain_error("functional unfinished in "+std::string(__FILE__)+" line "+std::to_string(__LINE__));
        }
        // 将GGA泛函的交换部分和关联部分的贡献累加得到总的GGA能量密度和势的两部分（单位体积）。
        sxc += s;
        v1xc += v1;
        v2xc += v2;
    }

    return;
}

//-----------------------------------------------------------------------
void XC_Functional::gcx_spin(double rhoup, double rhodw, double grhoup2, double grhodw2,
              double &sx, double &v1xup, double &v1xdw, double &v2xup, double &v2xdw)
{
    //--------------------------------------------------------------------
    //     gradient corrections for exchange - Hartree a.u.
    //     Implemented:  Becke88, GGA (PW91), PBE, revPBE
    //
    // use funct
    // USE kinds
    // implicit none

    //     dummy arguments

    // real rhoup, rhodw, grhoup2, grhodw2, sx, v1xup, v1xdw,
    //	v2xup, v2xdw;
    // up and down charge
    // up and down gradient of the charge
    // exchange and correlation energies
    // derivatives of exchange wr. rho
    // derivatives of exchange wr. grho

    // parameter :
    double small = 1.e-10;
    double sxup, sxdw;
    int iflag;

    // exchange
    double rho = rhoup + rhodw;

    sx = 0.00;
    v1xup = 0.00; v2xup = 0.00;
    v1xdw = 0.00; v2xdw = 0.00;
    sxup  = 0.00; sxdw  = 0.00;

    if (rho <= small)
    {
        return;
    }

    // 这种做法并不正确，后续会修改，应该像其他的一样将交换和关联部分放在一起处理
    // 可以看到下面的代码中将func_id[0]（交换部分）和func_id[1]（关联部分）分开计算的
    // not the correct way to do things, will change later
    // should put exchange and correlation together
    // like the others
    //for(int id : func_id)
    //{
        int id = func_id[0];
        switch( id )
        {
            case XC_GGA_X_B88: //B88
                if (rhoup > small && sqrt(fabs(grhoup2)) > small)
                {
                    XC_Functional::becke88_spin(rhoup, grhoup2, sxup, v1xup, v2xup);
                }
                if (rhodw > small && sqrt(fabs(grhodw2)) > small)
                {
                    XC_Functional::becke88_spin(rhodw, grhodw2, sxdw, v1xdw, v2xdw);
                }
                break;
            case XC_GGA_X_PBE: //PBX
                if (rhoup > small && sqrt(fabs(grhoup2)) > small)
                {
                    // 根据自旋标度关系，有：
                    // E_x(ρ_up, ρ_down) = 0.5 * [ E_x(2ρ_up) + E_x(2ρ_down) ]
                    // 所以这里直接调用非自旋的函数来计算上自旋部分的交换能和势，但要输入2倍的密度
                    // 而且梯度的平方也要乘以4（因为grho = |∇ρ|^2，ρ变为2ρ时，|∇(2ρ)|^2 = 4|∇ρ|^2）
                    XC_Functional::pbex(2.0 * rhoup, 4.0 * grhoup2, 0, sxup, v1xup, v2xup);
                }
                if (rhodw > small && sqrt(fabs(grhodw2)) > small)
                {
                    XC_Functional::pbex(2.0 * rhodw, 4.0 * grhodw2, 0, sxdw, v1xdw, v2xdw);
                }
                break;
            case XC_GGA_X_PBE_R: //revised PBX
                if (rhoup > small && sqrt(fabs(grhoup2)) > small)
                {
                    XC_Functional::pbex(2.0 * rhoup, 4.0 * grhoup2, 1, sxup, v1xup, v2xup);
                }
                if (rhodw > small && sqrt(fabs(grhodw2)) > small)
                {
                    XC_Functional::pbex(2.0 * rhodw, 4.0 * grhodw2, 1, sxdw, v1xdw, v2xdw);
                }
                break;
            case XC_HYB_GGA_XC_PBEH: //PBE0
                if (rhoup > small && sqrt(fabs(grhoup2)) > small)
                {
                    XC_Functional::pbex(2.0 * rhoup, 4.0 * grhoup2, 0, sxup, v1xup, v2xup);
                    sxup *= (1.0 - XC_Functional::hybrid_alpha); 
                    v1xup *= (1.0 - XC_Functional::hybrid_alpha); 
                    v2xup *= (1.0 - XC_Functional::hybrid_alpha);
                }
                if (rhodw > small && sqrt(fabs(grhodw2)) > small)
                {
                    XC_Functional::pbex(2.0 * rhodw, 4.0 * grhodw2, 0, sxdw, v1xdw, v2xdw);
        	    	sxdw *= (1.0 - XC_Functional::hybrid_alpha); 
                    v1xdw *= (1.0 - XC_Functional::hybrid_alpha); 
                    v2xdw *= (1.0 - XC_Functional::hybrid_alpha);            
                }
                break;
            case XC_GGA_X_PBE_SOL: //PBXsol
                if (rhoup > small && sqrt(fabs(grhoup2)) > small)
                {
                    XC_Functional::pbex(2.0 * rhoup, 4.0 * grhoup2, 2, sxup, v1xup, v2xup);
                }
                if (rhodw > small && sqrt(fabs(grhodw2)) > small)
                {
                    XC_Functional::pbex(2.0 * rhodw, 4.0 * grhodw2, 2, sxdw, v1xdw, v2xdw);
                }
                break;
            default:
                sxup = 0.0; sxdw = 0.0;
                v1xup = 0.0; v2xup = 0.0;
                v1xdw = 0.0; v2xdw = 0.0;
        }
        /* 自旋极化的交换能密度和势按自旋标度关系组合：
         E_x(ρ_up, ρ_down) = 0.5 * [ E_x(2ρ_up) + E_x(2ρ_down) ]
                           = 0.5 * [ ∫ (2ρ_up) ε_x(2ρ_up) dr + ∫ (2ρ_down) ε_x(2ρ_down) dr ]
                           = 0.5 * [ ∫sxup dr + ∫sxdw dr ]
                           = ∫ 0.5 * (sxup + sxdw) dr
                           = ∫ sx dr
        所以 sx = 0.5 * (sxup + sxdw)
        */
        sx = 0.50 * (sxup + sxdw);
        /* pbex()函数输出的v1xup是d(sxup)/d(2*ρ_up)，记为v1xup00,其中sxup是基于2ρ_up计算的单位体积的交换能，
          而最终的v1xup应该是上旋部分的单位体积交换能（0.5*sxup）对上旋密度（ρ_up）的导数
        即 v1xup = d(0.5*sxup)/d(ρ_up)
        = 0.5 * d(sxup)/d(2*ρ_up) * d(2*ρ_up)/d(ρ_up)
        = 0.5 * v1xup00 * 2
        = v1xup00
        所以 最终的v1xup就是pbex()函数输出的v1xup值，不需要额外处理，
        同理 v1xdw 也是一样的道理。
       
        然而，对于v2xup，pbex()函数输出的是 d(sxup)/d(|∇(2ρ_up)|) / |∇(2ρ_up)|,记为v2xup00,
        而我们需要的v2xup是上旋部分的单位体积交换能（0.5*sxup）对上旋密度梯度的导数再除以梯度
        即 v2xup = d(0.5*sxup)/d(|∇ρ_up|) / |∇ρ_up|
        = 0.5 * d(sxup)/d(|∇(2ρ_up)|) * d(|∇(2ρ_up)|)/d(|∇ρ_up|) / |∇ρ_up|
        = 0.5 * d(sxup)/d(|∇(2ρ_up)|) * 2 / |∇ρ_up|
        = d(sxup)/d(|∇(2ρ_up)|) / |∇ρ_up|
        = 2 * v2xup00
        因此 需要将pbex()函数输出的v2xup乘以2，才能得到正确的v2xup值。
        同理 v2xdw 也一样。
        */
        v2xup = 2.0 * v2xup;
        v2xdw = 2.0 * v2xdw;       
    //}

    return;
} //end subroutine gcx_spin

//
//-----------------------------------------------------------------------
void XC_Functional::gcc_spin(double rho, double &zeta, double grho, double &sc,
              double &v1cup, double &v1cdw, double &v2c)
{
    //-------------------------------------------------------------------
    //     gradient corrections for correlations - Hartree a.u.
    //     Implemented:  Perdew86, GGA (PW91), PBE

    // use funct
    // USE kinds
    // implicit none

    //     dummy arguments

    // real(kind=DP) :: rho, zeta, grho, sc, v1cup, v1cdw, v2c
    // the total charge
    // the magnetization
    // the gradient of the charge squared
    // exchange and correlation energies
    // derivatives of correlation wr. rho
    // derivatives of correlation wr. grho

    // parameter :
    double small = 1.0e-10;
	double epsr = 1.0e-6;

    double x;

    sc = 0.00;
    v1cup = 0.00; v1cdw = 0.00;
    v2c = 0.00;
    // 如果自旋极化参数 zeta 的绝对值大于 1（允许一个很小的数值误差 small），说明自旋极化参数超出了物理允许范围，直接返回
    if (std::abs(zeta) - 1.0 > small || rho <= small || sqrt(std::abs(grho)) <= small)
    {
        return;
    }
    else
    {
        // 对自旋极化参数 zeta 进行截断，确保其数值不会超过物理允许的范围 [−1,1]，
        // 并且避免数值上恰好等于1 或 −1（这样会导致后续计算出现奇异或不稳定）
        // ... ( - 1.0 + epsr )  <  zeta  <  ( 1.0 - epsr )
        // zeta = SIGN( MIN( ABS( zeta ), ( 1.D0 - epsr ) ) , zeta )
        // Fortran中，SIGN(a, b)= |a| × sign(b)，即返回一个数值，其绝对值等于 |a|，符号与 b 相同
        // 如果 zeta 的绝对值超过1−epsr，就把它截断到 1−epsr，保持 zeta 的符号不变
        x = std::min(std::abs(zeta), (1.0 - epsr));
		if(zeta>0)
		{
			zeta = x;
		}
		else
		{
			zeta = -x;
		}
    } //endif

    if(func_id[0]==XC_HYB_GGA_XC_PBEH)
    {
        XC_Functional::pbec_spin(rho, zeta, grho, 1, sc, v1cup, v1cdw, v2c);
        return;
    }

    //for(int id : func_id)
    //{
        int id = func_id[1];
        switch( id )
        {          
            case XC_GGA_C_P86: //P86
                XC_Functional::perdew86_spin(rho, zeta, grho, sc, v1cup, v1cdw, v2c);break;
            case XC_GGA_C_PW91: //PW91_C
                ModuleBase::WARNING_QUIT("xc_wrapper_gcxc","there seems to be something wrong with ggac_spin, better use libxc version instead");break;
                //XC_Functional::ggac_spin(rho, zeta, grho, sc, v1cup, v1cdw, v2c);
            case XC_GGA_C_PBE: //PBC
                XC_Functional::pbec_spin(rho, zeta, grho, 1, sc, v1cup, v1cdw, v2c);break;
            case XC_GGA_C_PBE_SOL: //PBCsol
                XC_Functional::pbec_spin(rho, zeta, grho, 2, sc, v1cup, v1cdw, v2c);break;
        }

    //}
    return;
} //end subroutine gcc_spin
