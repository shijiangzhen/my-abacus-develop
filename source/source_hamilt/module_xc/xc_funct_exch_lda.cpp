// This file contains realization of LDA exchange functionals
// Spin unpolarized ones:
//  1. slater: ordinary Slater exchange with alpha=2/3
//  2. slater1: Slater exchange with alpha=1
//  3. slater_rxc : Slater exchange with alpha=2/3 and Relativistic exchange
// And their spin polarized counterparts:
//  1. slater_spin
//  2. slater1_spin
//  3. slater_rxc_spin

#include "xc_functional.h"

//Slater exchange with alpha=2/3
void XC_Functional::slater(const double &rs, double &ex, double &vx)
{
	// f = -9/8*(3/2pi)^(2/3)
	const double f = -0.687247939924714e0;
	const double alpha = 2.00 / 3.00;
	ex = f * alpha / rs;
    /*
    ex推导过程如下：

    ex 是 LDA（局域密度近似）交换能密度。对于均匀电子气，交换能密度为：
        ex = - (3/4) * (3/π)^{1/3} * (ρ)^{1/3}

    但在 Slater Xα 方法中，通常写作：
        ex = f * α / rs

    其中：
    - rs 是 Wigner-Seitz 半径，定义为： (4/3)π rs^3 = 1/ρ  →  rs = [3/(4πρ)]^{1/3}
    - α 是经验参数，Slater 取 2/3，Slater1 取 1
    - f = -9/8 * (3/2π)^{2/3}，这是能量系数

    推导：
    1. 均匀电子气的交换能密度（每电子）：
       ε_x = - (3/4) * (3/π)^{1/3} * (ρ)^{1/3}

    2. 用 rs 表示 ρ：
       ρ = 3/(4π rs^3)  →  ρ^{1/3} = [3/(4π)]^{1/3} / rs

    3. 代入 ε_x：
       ε_x = - (3/4) * (3/π)^{1/3} * [3/(4π)]^{1/3} / rs
           = - (3/4) * [3/(2π)]^{2/3} / rs
           = 2/3 * [ - (9/8) * [3/(2π)]^{2/3} ] / rs
           = α * f / rs

    代码实现即为上述公式。
    */
	
    vx = 4.0 / 3.0 * f * alpha / rs;
    /*
    vx 是单位体积的交换能（ρ*ex）对密度的变分导数（交换势），即：
        vx = d(ρ * ex) / dρ

    已知 ex = f * α / rs
    其中 rs = [3/(4πρ)]^{1/3}

    首先，写出 ex 关于 ρ 的表达式：
        rs = [3/(4πρ)]^{1/3}
        → rs = C * ρ^{-1/3}，其中 C = [3/(4π)]^{1/3}
        → 1/rs = C^{-1} * ρ^{1/3}

    所以 ex = f * α * C^{-1} * ρ^{1/3}

    现在计算 vx:
        vx = d(ρ * ex) / dρ
           = d(ρ * f * α * C^{-1} * ρ^{1/3}) / dρ
           = f * α * C^{-1} * d(ρ^{4/3}) / dρ
           = f * α * C^{-1} * (4/3) * ρ^{1/3}
           = (4/3) * ex

    因为 ex = f * α / rs 
    所以 vx = (4/3) * ex = (4/3) * f * α / rs

    与代码一致。
    */
	return;
}

//Slater exchange with alpha=1, corresponding to -1.374/r_s Ry
//used to recover old results
// α = 1 是历史上Slater提出的一个简化近似，它来自于将总交换能“平均地”分摊给每个电子，
// 是一种物理上的直观近似，但并非从能量泛函变分得到的严格势。
void XC_Functional::slater1(const double &rs, double &ex, double &vx)
{
	const double f = -0.687247939924714e0;
	const double alpha = 1.0;
	ex = f * alpha / rs;
	vx = 4.0 / 3.0 * f * alpha / rs;
	return;
}

// Slater exchange with alpha=2/3 and Relativistic exchange
void XC_Functional::slater_rxc(const double &rs, double &ex, double &vx)
{
    const double trd = 1.0 / 3.0;
    //const double ftrd = 4.0 / 3.0;
    //const double tftm = pow(2.0, ftrd) - 2.0;
    const double a0 = pow((4.0 / (9.0 * ModuleBase::PI)), trd);
    // X-alpha parameter:
    const double alp = 2 * trd;

    double vxp = -3 * alp / (2 * ModuleBase::PI * a0 * rs);
    double exp = 3 * vxp / 4;
    const double beta = 0.014 / rs;
    const double sb = sqrt(1 + beta * beta);
    const double alb = log(beta + sb);
    vxp = vxp * (-0.5 + 1.5 * alb / (beta * sb));
    double x = (beta * sb - alb) / (beta * beta);
    exp = exp * (1.0 - 1.5 * x * x);
    vx = vxp;
    ex = exp;
    return;
}

// Slater exchange with alpha=2/3, spin-polarized case
void XC_Functional::slater_spin( const double &rho, const double &zeta, 
		double &ex, double &vxup, double &vxdw)
{
    const double f = - 1.107838149573033610;
    const double alpha = 2.00 / 3.00;
    // f = -9/8*(3/pi)^(1/3)
    const double third = 1.0 / 3.0;
    const double p43 = 4.0 / 3.0;

    double rho13 = pow(((1.0 + zeta) * rho) , third);
    double exup = f * alpha * rho13;
    /*
    exup 是自旋向上电子的交换能密度。推导如下：

    1. 均匀电子气的交换能密度（每电子）：
        ε_x = - (3/4) * (3/π)^{1/3} * ρ^{1/3}

    2. 对于自旋极化体系，总密度 ρ = ρ_up + ρ_down，自旋极化度 zeta = (ρ_up - ρ_down)/ρ
        则 ρ_up = (1 + zeta) * ρ / 2
            ρ_down = (1 - zeta) * ρ / 2
        不知道代码的pow()函数里面为什么没有除以2，系数f里也没有2^(-1/3)，是bug吗？

    3. 对于每一自旋分量，交换能密度为：
        ε_x^σ = - (3/4) * (3/π)^{1/3} * (ρ_σ)^{1/3}
              = 2/3 * [ - (9/8) * (3/π)^{1/3} ] * (ρ_σ)^{1/3}
              = f * α * (ρ_σ)^{1/3}
        
        其中 f = -9/8 * (3/π)^{1/3} ，α = 2/3
        所以，exup = f * α * (ρ_up)^{1/3}
            exdw = f * α * (ρ_down)^{1/3}

    */
    // 计算自旋向上电子的交换势
    // 前面已经推导出vx=4/3*ex，这里只是要对自旋向上和自旋向下分量分别计算
    vxup = p43 * f * alpha * rho13;
    rho13 = pow(((1.0 - zeta) * rho) , third);
    double exdw = f * alpha * rho13;
    vxdw = p43 * f * alpha * rho13;
    // 计算总交换能密度，按自旋分量加权平均，自旋向上占比 (1+zeta)/2，自旋向下占比 (1-zeta)/2
    // 这样可以保证在 zeta=0 时，两分量贡献相等，恢复到非自旋极化的交换能密度表达式，
    // 在 zeta=1 时，只有自旋向上分量贡献交换能密度；在 zeta=-1 时，只有自旋向下分量贡献交换能密度。
    ex = 0.50 * ((1.0 + zeta) * exup + (1.0 - zeta) * exdw);

    return;
}

// alpha = 1.00
// Slater exchange with alpha=2/3, spin-polarized case
void XC_Functional::slater1_spin( const double &rho, const double &zeta, double &ex, double &vxup, double &vxdw)
{
    const double f = - 1.107838149573033610;
	const double alpha = 1.00;
	const double third = 1.0 / 3.0;
	const double p43 = 4.0 / 3.0;
    // f = -9/8*(3/pi)^(1/3)

    double rho13 = pow(((1.0 + zeta) * rho) , third);
    double exup = f * alpha * rho13;
    vxup = p43 * f * alpha * rho13;
    rho13 = pow(((1.0 - zeta) * rho) , third);
    double exdw = f * alpha * rho13;
    vxdw = p43 * f * alpha * rho13;
    ex = 0.50 * ((1.0 + zeta) * exup + (1.0 - zeta) * exdw);

    return;
} // end subroutine slater1_spin

// Slater exchange with alpha=2/3, relativistic exchange case
void XC_Functional::slater_rxc_spin( const double &rho, const double &z, 
		double &ex, double &vxup, double &vxdw)
{
    if (rho <= 0.0)
    {
        ex = vxup = vxdw = 0.0;
        return;
    }

    const double trd = 1.0 / 3.0;
	const double ftrd = 4.0 / 3.0;
	double tftm = pow(2.0, ftrd) - 2;
    double a0 = pow((4 / (9 * ModuleBase::PI)), trd);

    double alp = 2 * trd;

	double fz = (pow((1 + z), ftrd) + pow((1 - z), ftrd) - 2) / tftm;
	double fzp = ftrd * (pow((1 + z), trd) - pow((1 - z), trd)) / tftm;

    double rs = pow((3 / (4 * ModuleBase::PI * rho)), trd);
    double vxp = -3 * alp / (2 * ModuleBase::PI * a0 * rs);
    double exp = 3 * vxp / 4;

    double beta = 0.014 / rs;
    double sb = sqrt(1 + beta * beta);
    double alb = log(beta + sb);
    vxp = vxp * (-0.5 + 1.5 * alb / (beta * sb));
    exp = exp * (1.0- 1.5*( (beta*sb-alb) / (beta*beta) )
			        * 1.5*( (beta*sb-alb) / (beta*beta) ));

    double x = (beta * sb - alb) / (beta * beta);

    exp = exp * (1.0 - 1.5 * x * x);

    double vxf = pow(2.0, trd) * vxp;

    double exf = pow(2.0, trd) * exp;

    vxup  = vxp + fz * (vxf - vxp) + (1 - z) * fzp * (exf - exp);

    vxdw  = vxp + fz * (vxf - vxp) - (1 + z) * fzp * (exf - exp);

    ex    = exp + fz * (exf - exp);

    return;
}
