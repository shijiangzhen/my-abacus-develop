#include <source_hamilt/module_xc/kernels/xc_functional_op.h>

namespace hamilt {

template <typename T, typename Device>
void xc_functional_grad_wfc_op<T, Device>::operator()(
    const int& ik,
    const int& pol,
    const int& npw,
    const int& npwx,
	const Real& tpiba,
    const Real * gcar,
    const Real * kvec_c,
    const T * rhog,
    T* porter)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for(int ig = 0; ig < npw; ig++) {
		// the formula is : rho(r)^prime = \int iG * rho(G)e^{iGr} dG
		// double kplusg = wfc_basis->getgpluskcar(ik,ig)[ipol] * tpiba;
        /*
        推导过程如下：

        1. 在平面波展开中，波函数的梯度涉及到动量算符作用在波函数上。
        2. 在倒空间，梯度算符对应于乘以 i(G + k)，其中 G 是倒格矢，k 是布里渊区采样点。
        3. 由于单位制的不同，实际计算时需要乘以 tpiba（2π/a，a为晶格常数），将无量纲的倒格矢转换为实际单位。
        4. gcar[(ik * npwx + ig) * 3 + pol] 取出第 ik 个 k 点，第 ig 个 G 矢量，第 pol 分量的 G 分量。
        5. kvec_c[ik * 3 + pol] 取出第 ik 个 k 点，第 pol 分量的 k 分量。
        6. 两者相加得到 G + k 的第 pol 分量，再乘以 tpiba 得到实际的动量分量。

        最终公式：
        kplusg = (gcar[(ik * npwx + ig) * 3 + pol] + kvec_c[ik * 3 + pol]) * tpiba;
        */
        Real kplusg = (gcar[(ik * npwx + ig) * 3 + pol] +
                       kvec_c[ik * 3 + pol]) * tpiba;
                       
		// calculate the charge density gradient in reciprocal space.
		porter[ig] = T(0.0, kplusg) * rhog[ig];
	}
}

template <typename T, typename Device>
void xc_functional_grad_wfc_op<T, Device>::operator()(
    const int& ipol,
    const int& nrxx,
    const T * porter,
    T* grad)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for (int ir = 0; ir < nrxx; ++ir) {
        grad[ipol * nrxx + ir] = porter[ir];
	}
}

template struct xc_functional_grad_wfc_op<std::complex<float>, base_device::DEVICE_CPU>;
template struct xc_functional_grad_wfc_op<std::complex<double>, base_device::DEVICE_CPU>;

} // namespace hamilt   