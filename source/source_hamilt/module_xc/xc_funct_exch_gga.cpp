// This file contains realizations of gradient correction to exchange part
// Spin unpolarized ones:
//  1. becke88 : Becke88 exchange
//  2. ggax : PW91 exchange
//  3. pbex : PBE exchange (and revPBE)
//  4. optx : OPTX, Handy et al.
//  5. wcx : Wu-Cohen exchange
// And some of their spin polarized counterparts:
//  1. becke88_spin

#include "xc_functional.h"

void XC_Functional::becke88(const double &rho, const double &grho, double &sx, double &v1x, double &v2x)
{
    //-----------------------------------------------------------------------
    // Becke exchange: A.D. Becke, PRA 38, 3098 (1988)
    // only gradient-corrected part, no Slater term included
    //
    double beta, third, two13;
    beta = 0.00420;
    third = 1.0 / 3.0;
    two13 = 1.2599210498948730;
    double rho13, rho43, xs, xs2, sa2b8, shm1, dd, dd2, ee;

    rho13 = pow(rho, third);
    rho43 = pow(rho13, 4);
    xs = two13 * sqrt(grho) / rho43;
    xs2 = xs * xs;
    sa2b8 = sqrt(1.00 + xs2);
    shm1 = log(xs + sa2b8);
    dd = 1.00 + 6.00 * beta * xs * shm1;
    dd2 = dd * dd;
    ee = 6.00 * beta * xs2 / sa2b8 - 1.0;
    sx = two13 * grho / rho43 * (- beta / dd);
    v1x = - (4.0 / 3.0) / two13 * xs2 * beta * rho13 * ee / dd2;
    v2x = two13 * beta * (ee - dd) / (rho43 * dd2);

    return;
} // end subroutine becke88

void XC_Functional::ggax(const double &rho, const double &grho, double &sx, double &v1x, double &v2x)
{
    //-----------------------------------------------------------------------
    // Perdew-Wang GGA (PW91), exchange part:
    // J.P. Perdew et al.,PRB 46, 6671 (1992)
    const double f1 = 0.196450;
    const double f2 = 7.79560;
    const double f3 = 0.27430;
    const double f4 = 0.15080;
    const double f5 = 0.0040;
    const double fp1 = -0.0192920212964260;
    const double fp2 = 0.1616204596739950;
    // fp1 = -3/(16 pi)*(3 pi^2)^(-1/3)
    double rhom43, s, s2, s3, s4, exps, as, sa2b8, shm1, bs, das,
    dbs, dls;

    rhom43 = pow(rho, (- 4.0 / 3.0));
    s = fp2 * sqrt(grho) * rhom43;
    s2 = s * s;
    s3 = s2 * s;
    s4 = s2 * s2;
    exps = f4 * exp(- 100.0 * s2);
    as = f3 - exps - f5 * s2;
    sa2b8 = sqrt(1.00 + f2 * f2 * s2);
    shm1 = log(f2 * s + sa2b8);
    bs = 1.0 + f1 * s * shm1 + f5 * s4;
    das = (200.0 * exps - 2.0 * f5) * s;
    dbs = f1 * (shm1 + f2 * s / sa2b8) + 4.0 * f5 * s3;
    dls = (das / as - dbs / bs);
    
	sx = fp1 * grho * rhom43 * as / bs;
	v1x = - 4.0 / 3.0 * sx / rho * (1.0 + s * dls);
    v2x = fp1 * rhom43 * as / bs * (2.0 + s * dls);

    return;
} //end subroutine ggax

void XC_Functional::pbex(const double &rho, const double &grho, const int &iflag, 
double &sx, double &v1x, double &v2x)
{
    // PBE exchange (without Slater exchange):
    // iflag=0  J.P.Perdew, K.Burke, M.Ernzerhof, PRL 77, 3865 (1996)
    // iflag=1  "revised' PBE: Y. Zhang et al., PRL 80, 890 (1998)

    // input: charge and squared gradient
    // output: energy
    // output: potential
    // (3*pi2*|rho|)^(1/3)
    // |grho|
    // |grho|/(2*kf*|rho|)
    // s^2
    // n*ds/dn
    // n*ds/d(gn)
    // exchange energy LDA part
    // exchange energy gradient part
    
    // grho:电子密度梯度的模的平方∣∇ρ∣^2
    

	// numerical coefficients (NB: c2=(3 pi^2)^(1/3) )
    const double third = 1.0 / 3.0;
    const double c1 = 0.750 / ModuleBase::PI; 
    const double c2 = 3.0936677262801360;
    const double c5 = 4.0 * third; 
    // parameters of the functional
    // iflag=0: PBE, iflag=1: revised PBE
    // 对于PBE，k=0.804, mu=0.2195149727645171
    double k[3] = { 0.8040, 1.24500, 0.8040 };
    const double mu[3] = {0.2195149727645171, 0.2195149727645171, 0.12345679012345679} ;//modified by zhengdy, to ensure the same parameters with another dft code.

    const double agrho = sqrt(grho); // 密度梯度的模, ∣∇ρ∣
    const double kf = c2 * pow(rho, third); // 费米波矢, kF=(3π²ρ)^(1/3)
    const double dsg = 0.50 / kf; // 1/(2*kF), 用于计算无量纲梯度s
    const double s1 = agrho * dsg / rho; // 无量纲梯度s = ∣∇ρ∣/(2*kF*ρ)
    const double s2 = s1 * s1; // s的平方
    const double ds = - c5 * s1; // n*ds/dn，用于后续势的导数，其中 n=ρ（电子密度），见下方推导
    /*
    ds = n * ds/dn
    s = |∇ρ| / (2 * kF * ρ)
    其中 kF = c2 * ρ^(1/3)
    s1 = |∇ρ| / (2 * kF * ρ)
    ds/dn = 对 s 关于 n (即 ρ) 求导

    推导如下：
    s = |∇ρ| / (2 * kF * ρ)
        = |∇ρ| / (2 * c2 * ρ^(1/3) * ρ)
        = |∇ρ| / (2 * c2 * ρ^(4/3))

    令 A = |∇ρ| / (2 * c2)
    s = A * ρ^(-4/3)

    ds/dρ = A * (-4/3) * ρ^(-7/3)

    n * ds/dn = ρ * ds/dρ = ρ * A * (-4/3) * ρ^(-7/3)
                        = A * (-4/3) * ρ^(-4/3)
                        = - (4/3) * s

    所以 ds = - (4.0 / 3.0) * s1;
    */


    // Energy
    const double f1 = s2 * mu[iflag] / k [iflag]; // s^2 * mu / k
    const double f2 = 1.0 + f1;
    const double f3 = k [iflag] / f2; // k / (1 + s^2 * mu / k)
    
    // PBE中交换能密度的表达式为： ε_x^PBE = ε_x^LDA * F_x(s)
    // 其中 F_x(s) = 1 + k - k / (1 + s^2 * mu / k)
    // 这里代码的fx = k - k / (1 + s^2 * mu / k), 
    // 即交换能的梯度修正部分,比PBE的增强因子少了1（1就是LDA部分），因为LDA部分在后面单独计算
    const double fx = k [iflag] - f3; 
    const double exunif = - c1 * kf; // 交换能的LDA部分，推导如下
    /*
    exunif 是均匀电子气（LDA）交换能密度的表达式。推导如下：

    1. 均匀电子气的交换能密度（每单位体积）为：
        ε_x = - (3/4) * (3/π)^{1/3} * ρ^{1/3} * ρ
           = - (3/4) * (3/π)^{1/3} * ρ^{4/3}

    2. 但在实际计算中，常用的表达式为：
        ε_x = - (3/4) * (3/π)^{1/3} * ρ^{4/3}
             = - c1 * kf * ρ
        其中：
          c1 = 0.75 / π
          kf = (3π^2 ρ)^{1/3}  （费米波矢）

    3. 进一步推导：
        kf = (3π^2 ρ)^{1/3}
        kf * ρ = (3π^2)^{1/3} * ρ^{4/3}
        所以：
        exunif = - c1 * kf
        但此处 exunif 实际上是每电子的交换能密度（即除以 ρ），
        所以最终：
        exunif = - (3/4) * (3/π)^{1/3} * ρ^{1/3}
                 = - c1 * kf

    4. 代码中：
        c1 = 0.750 / ModuleBase::PI
        kf = c2 * pow(rho, 1/3)
        c2 = (3π^2)^{1/3}

        所以 exunif = - c1 * kf
    */
    sx = exunif * fx; // 交换能的梯度修正部分

    // Potential
    const double dxunif = exunif * third; // LDA部分对密度的导数再乘以密度ρ
    /*
    dxunif 是 exunif 关于 rho 的导数乘以 rho，即
        dxunif = rho * d(exunif)/d(rho)

    exunif = -c1 * kf
    其中 kf = c2 * rho^{1/3}
    所以 exunif = -c1 * c2 * rho^{1/3}

    对 rho 求导：
    d(exunif)/d(rho) = -c1 * c2 * d(rho^{1/3})/d(rho)
                     = -c1 * c2 * (1/3) * rho^{-2/3}

    所以
    dxunif = rho * d(exunif)/d(rho)
           = rho * [ -c1 * c2 * (1/3) * rho^{-2/3} ]
           = -c1 * c2 * (1/3) * rho^{1 - 2/3}
           = -c1 * c2 * (1/3) * rho^{1/3}

    而 exunif = -c1 * c2 * rho^{1/3}
    所以 dxunif = exunif * (1/3)
    即
        dxunif = exunif * third;
    */

    const double dfx1 = f2 * f2;// (1 + s^2 * mu / k)^2
    const double dfx = 2.0 * mu[iflag] * s1 / dfx1; 
    /*
    dfx 是 fx=F_x(s) 关于 s 的导数，用于交换势的梯度修正部分,链式法则用于后续求导。

    1. F_x(s) 的表达式：
        F_x(s) = k - k / (1 + s^2 * mu / k)
                 = k - k / f2
        其中 f2 = 1 + s^2 * mu / k

    2. 对 s 求导：
        令 f2 = 1 + s^2 * mu / k
        则 F_x(s) = k - k / f2

        dF_x/ds = -k * d(1/f2)/ds
                  = -k * (-1) * (1/f2^2) * d(f2)/ds
                  = k * (1/f2^2) * d(f2)/ds

        d(f2)/ds = d(1 + s^2 * mu / k)/ds = 2 * s * mu / k

        所以：
        dF_x/ds = k * (1/f2^2) * (2 * s * mu / k)
                  = (2 * s * mu) / f2^2

    3. 所以 dfx = dF_x/ds 
        dfx = (2 * mu[iflag] * s1 / dfx1)
        其中 dfx1 = f2^2

    最终：
        dfx = 2.0 * mu[iflag] * s1 / dfx1;
    */
    
    
	v1x = sx + dxunif * fx + exunif * dfx * ds;
    /*
    v1x 是交换能密度关于电子密度 rho 的导数d(sx * rho)/d(rho)，用于交换势的计算。
    推导如下：

    1. sx = exunif * fx
        其中 exunif = -c1 * kf
              kf = c2 * rho^{1/3}
              fx = k - k / (1 + s^2 * mu / k)
              s = |∇ρ| / (2 * kF * ρ)
       又有dxunif = rho * d(exunif)/d(rho)
       dfx = d(fx)/d(s)
       ds = n * ds/dn
              
    2. 详细链式法则推导：
        v1x  = d(sx * rho)/d(rho)
             = d(exunif * fx * rho)/d(rho)
             = exunif * fx + rho * d(exunif)/d(rho) * fx + rho * exunif * d(fx)/d(rho)
             = exunif * fx + rho * fx * dxunif/rho  + rho * exunif * d(fx)/ds * ds/d(rho)
             = exunif * fx + dxunif * fx + rho * exunif * dfx * ds/rho
             = sx + dxunif * fx + exunif * dfx * ds
  
    */

    v2x = exunif * dfx * dsg / agrho;
    /*
    v2x 是单位体积的交换能对电子密度梯度的导数再除以梯度，用于计算势的梯度修正部分。
    通常 GGA 势的梯度部分形式为 ∇⋅(∂(ε_x*ρ)/∂(∇ρ))。
    令 f = ε_x * ρ = exunif * fx * ρ
    
    需要计算 ∂f/∂(∇ρ)。
    
    根据链式法则：
    ∂f/∂(∇ρ) = ∂f/∂s * ∂s/∂(∇ρ)
    
    1. ∂f/∂s
       f = exunif * fx * ρ
       exunif 和 ρ 不依赖于 s (s 依赖于 ∇ρ)
       ∂f/∂s = exunif * ρ * ∂(fx)/∂s
             = exunif * ρ * dfx
             
    2. ∂s/∂(∇ρ)
       s = |∇ρ| / (2 * kF * ρ)
       dsg=1 / (2 * kF)
       
       ∂s/∂(∇ρ) = [1 / (2 * kF * ρ)] 
                = dsg / ρ
                
    3. 组合
       ∂f/∂(∇ρ) = (exunif * ρ * dfx) * (dsg / ρ) 
                = exunif * dfx * dsg 
                
    
    所以：
    v2x = ∂f/∂(∇ρ) / |∇ρ|
        = exunif * dfx * dsg / |∇ρ|
        = exunif * dfx * dsg / agrho
    */
    
    // 右边的sx是ϵx，即每个电子的交换能密度，乘以ρ后，sx变成了ρϵ_x，即该空间点上的总交换能密ρ*ϵx
    sx = sx * rho; // 得到的是每单位体积的交换能，用于后续积分得到总交换能

	return;
}

void XC_Functional::optx(const double rho, const double grho, double &sx, double &v1x, double &v2x)
{
    //     OPTX, Handy et al. JCP 116, p. 5411 (2002) and refs. therein
    //     Present release: Mauro Boero, Tsukuba, 10/9/2002
    //--------------------------------------------------------------------------
    //     rhoa = rhob = 0.5 * rho in LDA implementation
    //     grho is the SQUARE of the gradient of rho// --> gr=sqrt(grho)
    //     sx  : total exchange correlation energy at point r
    //     v1x : d(sx)/drho
    //     v2x : 1/gr*d(sx)/d(gr)
    //--------------------------------------------------------------------------
    // use kinds, only: DP
    // implicit none
    // real(kind=DP) :: rho, grho, sx, v1x, v2x

    // parameter :
    double small = 1.e-30;
    double smal2 = 1.e-10;
    //.......coefficients and exponents....................
    // parameter :
    double o43 = 4.00 / 3.00,
                 two13 = 1.2599210498948730,
                         two53 = 3.1748021039363990,
                                 gam = 0.0060,
                                       a1cx = 0.97845711702844210,
                                              a2 = 1.431690;
    double gr, rho43, xa, gamx2, uden, uu;
    //.......OPTX in compact form..........................

    if (rho <= small)
    {
        sx = 0.00;
        v1x = 0.00;
        v2x = 0.00;
    }
    else
    {
        gr = (grho > smal2) ? grho : smal2;	//max()
        rho43 = pow(rho, o43);
        xa = two13 * sqrt(gr) / rho43;
        gamx2 = gam * xa * xa;
        uden = 1.e+00 / (1.e+00 + gamx2);
        uu = a2 * gamx2 * gamx2 * uden * uden;
        uden = rho43 * uu * uden;
        sx = -rho43 * (a1cx + uu) / two13;
        v1x = o43 * (sx + two53 * uden) / rho;
        v2x = -two53 * uden / gr;
    } //endif

    return;
} // end subroutine optx

void XC_Functional::wcx(const double &rho,const double &grho, double &sx, double &v1x, double &v2x)
{
  double kf, agrho, s1, s2, es2, ds, dsg, exunif, fx;
  // (3*pi2*|rho|)^(1/3)
  // |grho|
  // |grho|/(2*kf*|rho|)
  // s^2
  // n*ds/dn
  // n*ds/d(gn)
  // exchange energy LDA part
  // exchange energy gradient part
  double dxunif, dfx, f1, f2, f3, dfx1, x1, x2, x3, dxds1, dxds2, dxds3;
  // numerical coefficients (NB: c2=(3 pi^2)^(1/3) )
  double third, c1, c2, c5, teneightyone; // c6

  third = 1.0/3.0;
  c1 = 0.75 / ModuleBase::PI;
  c2 = 3.093667726280136;
  c5 = 4.0 * third;
  teneightyone = 0.123456790123;
  // parameters of the functional
  double k, mu, cwc;
  k = 0.804;
  mu = 0.2195149727645171;
  cwc = 0.00793746933516;
  //
  agrho = sqrt (grho);
  kf = c2 * pow(rho,third);
  dsg = 0.5 / kf;
  s1 = agrho * dsg / rho;
  s2 = s1 * s1;
  es2 = exp(-s2);
  ds = - c5 * s1;
  //
  //   Energy
  //
  // x = 10/81 s^2 + (mu - 10/81) s^2 e^-s^2 + ln (1 + c s^4)
  x1 = teneightyone * s2;
  x2 = (mu - teneightyone) * s2 * es2;
  x3 = log(1.0 + cwc * s2 * s2);
  f1 = (x1 + x2 + x3) / k;
  f2 = 1.0 + f1;
  f3 = k / f2;
  fx = k - f3;
  exunif = - c1 * kf;
  sx = exunif * fx;
  //
  //   Potential
  //
  dxunif = exunif * third;
  dfx1 = f2 * f2;
  dxds1 = teneightyone;
  dxds2 = (mu - teneightyone) * es2 * (1.0 - s2);
  dxds3 = 2.0 * cwc * s2 / (1.0 + cwc * s2 *s2);
  dfx = 2.0 * s1 * (dxds1 + dxds2 + dxds3) / dfx1;
  v1x = sx + dxunif * fx + exunif * dfx * ds;
  v2x = exunif * dfx * dsg / agrho;

  sx = sx * rho;
  return;
}

void XC_Functional::becke88_spin(double rho, double grho, double &sx, double &v1x, double &v2x)
{
    //-------------------------------------------------------------------
    // Becke exchange: A.D. Becke, PRA 38, 3098 (1988) - Spin polarized case

    // USE kinds
    // implicit none
    // real(kind=DP) :: rho, grho, sx, v1x, v2x
    // input: charge
    // input: gradient
    // output: the up and down energies
    // output: first part of the potential
    // output: the second part of the potential

    double beta, third;
    // parameter :
    beta = 0.00420;
    third = 1.0 / 3.0;
    double rho13, rho43, xs, xs2, sa2b8, shm1, dd, dd2, ee;

    rho13 = pow(rho, third);
    rho43 = pow(rho13, 4);
    xs = sqrt(grho) / rho43;
    xs2 = xs * xs;
    sa2b8 = sqrt(1.00 + xs2);
    shm1 = log(xs + sa2b8);
    dd = 1.00 + 6.00 * beta * xs * shm1;
    dd2 = dd * dd;
    ee = 6.00 * beta * xs2 / sa2b8 - 1.0;
    sx = grho / rho43 * (- beta / dd);
    v1x = - (4.0 / 3.0) * xs2 * beta * rho13 * ee / dd2;
    v2x = beta * (ee - dd) / (rho43 * dd2);

    return;
} //end subroutine becke88_spin
