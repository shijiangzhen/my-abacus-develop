// LDA泛函的封装（包括GGA的LDA部分），提供LDA及其自旋极化版本的计算接口，便于在其他模块中调用LDA相关计算
// This file contains wrapper for the LDA functionals
// it includes 3 subroutines:
// 1. xc, which is the wrapper of LDA part
// (i.e. LDA functional and LDA part of GGA functional)
// 2. xc_spin, which is the spin polarized counterpart of xc
// 3. xc_spin_libxc, which is the wrapper for LDA functional, spin polarized

#ifdef USE_LIBXC
#include <xc_funcs.h>
#endif	// ifdef USE_LIBXC

#include "xc_functional.h"
#include <stdexcept>

void XC_Functional::xc(const double &rho, double &exc, double &vxc)
{

	double third = 1.0 / 3.0;
	double pi34 = 0.6203504908994e0 ; // pi34=(3/4pi)^(1/3)
	double rs;
    double e,v;
    
    exc = vxc = 0.00;
	
    // Wigner_Seitz半径计算，公式为 rs = (3/(4πρ))^(1/3)
	rs = pi34 / std::pow(rho, third);

    // 交换和关联部分一般是分开计算的，而数组func_id里就存储了需要计算的泛函的交换和关联部分的各自编号，
    // 所以需要id来遍历func_id数组，分别找到对应的交换和关联部分进行计算
    for(int id : func_id)
    {
        switch( id )
        {
            // Exchange functionals containing slater exchange
            case XC_LDA_X: case XC_GGA_X_PBE: case XC_GGA_X_PBE_R: case XC_GGA_X_PBE_SOL:
            case XC_GGA_X_WC: case XC_GGA_X_B88: case XC_GGA_X_PW91:
            //  SLA,PBX,rPBX,PBXsol,WC,B88,PW91_X
            // 调用 slater 方法计算交换能和势。
                XC_Functional::slater(rs, e, v);break;

            // Exchange functionals containing attenuated slater exchange
            case XC_HYB_GGA_XC_PBEH:
            //  PBE0
                double ex, vx, ec, vc;
                XC_Functional::slater(rs, ex, vx);
                // 计算杂化泛函(PBE0)的LDA交换部分的能量密度和势
                // 按混合比例缩放，hybrid_alpha 是杂化比例（如PBE0中为0.25）
                ex *= (1 - XC_Functional::hybrid_alpha); 
                vx *= (1 - XC_Functional::hybrid_alpha);
                // 关联部分全部使用 pw 泛函
                XC_Functional::pw(rs, 0, ec, vc);
                // 将交换和相关部分的能量密度、势分别相加，得到杂化泛函LDA部分的总能量密度和势。
                e = ex + ec;
                v = vx + vc;
                break;

            // Correlation functionals containing PW correlation
            case XC_GGA_C_PBE: case XC_GGA_C_PW91: case XC_LDA_C_PW: case XC_GGA_C_PBE_SOL:
            //   PBC,PW91,PWLDA
                XC_Functional::pw(rs, 0, e, v);break;

            // Correlation functionals containing PZ correlation
            case XC_LDA_C_PZ: case XC_GGA_C_P86:
            //  PZ,P86
                XC_Functional::pz(rs, 0, e, v);break;

            // Correlation functionals containing LYP correlation
            case XC_GGA_C_LYP:
            //  BLYP
                XC_Functional::lyp(rs, e, v);break;
                            
            default:
                e = v = 0.0;
        }
        // 将每个泛函部分（交换和关联）的能量密度和势加起来得到总的交换-关联能量密度和势
        exc += e;
        vxc += v;
    }
	return;
}

// 自旋极化版本的LDA泛函计算接口
void XC_Functional::xc_spin(const double &rho, const double &zeta,
		double &exc, double &vxcup, double &vxcdw)
{
	static const double small = 1.e-10;
    double e, vup, vdw;
    exc = vxcup = vxcdw = 0.0;

	static const double third = 1.0 / 3.0;
	static const double pi34 = 0.62035049089940; 
	const double rs = pi34 / pow(rho, third);//wigner_sitz_radius;

    for(int id : func_id)
    {
        switch( id )
        {
            // Exchange functionals containing slater exchange
            case XC_LDA_X: case XC_GGA_X_PBE: case XC_GGA_X_PBE_R: case XC_GGA_X_PBE_SOL: 
            case XC_GGA_X_WC: case XC_GGA_X_B88: case XC_GGA_X_PW91:
            //  SLA,PBX,rPBX,PBXsol,WC,B88,PW91_X
            // 调用 slater_spin 方法计算自旋极化交换能和势。
                XC_Functional::slater_spin(rho, zeta, e, vup, vdw);	break;
            
            // Exchange functionals containing attenuated slater exchange
            case XC_HYB_GGA_XC_PBEH:
            //  PBE0
                double ex, vupx, vdwx, ec, vupc, vdwc;
                // 计算自旋极化的杂化泛函(PBE0)的LDA交换部分的能量密度和势
                XC_Functional::slater_spin(rho, zeta, ex, vupx, vdwx);
                ex *= (1.0 - XC_Functional::hybrid_alpha); 
                vupx *= (1.0 - XC_Functional::hybrid_alpha); 
                vdwx *= (1.0 - XC_Functional::hybrid_alpha);
                XC_Functional::pw_spin(rs, zeta, ec, vupc, vdwc);
                e = ex + ec;
                vup = vupx + vupc;
                vdw = vdwx + vdwc;
                break;

            // Correlation functionals containing PZ correlation
            case XC_LDA_C_PZ: case XC_GGA_C_P86:
            //  PZ,P86
                XC_Functional::pz_spin(rs, zeta, e, vup, vdw); break;

            // Correlation functionals containing PW correlationtests/integrate/101_PW_OU_pseudopot
            case XC_GGA_C_PBE: case XC_GGA_C_PBE_SOL: case XC_LDA_C_PW:
            //   PBC,PBCsol
                XC_Functional::pw_spin(rs, zeta, e, vup, vdw); break;

            // Cases that are only realized in LIBXC
            default:
                throw std::domain_error("functional unfinished in "+std::string(__FILE__)+" line "+std::to_string(__LINE__));	break;

        }
        exc += e;
        vxcup += vup;
        vxcdw += vdw;
	}
	return;
}