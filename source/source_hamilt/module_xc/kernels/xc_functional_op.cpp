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

 /**
 *   1. 周期函数的平面波展开：
 *
 *      ρ(r) = Σ_G ρ_G e^{i (k + G) · r}
 *
 *      - G        : 倒格矢（plane-wave index 对应的倒空间向量）
 *      - k        : 当前 k 点的倒空间向量
 *      - ρ_G      : 在 (k + G) 处的傅里叶系数（本函数中的 rhog[ig]）
 *      - r        : 实空间坐标
 *
 *   2. 对 ρ(r) 求梯度：
 *
 *      ∇ρ(r) = Σ_G i (k + G) ρ_G e^{i (k + G) · r}
 *
 *      对第 α 个笛卡尔分量 (α = x, y, z)：
 *
 *      ∂ρ(r) / ∂r_α = Σ_G i (k_α + G_α) ρ_G e^{i (k + G) · r}
 *
 *   因此，在倒空间上，对每一个平面波分量 G，其在第 α 个方向上的
 *   “梯度系数”就是 i (k_α + G_α) 乘以 ρ_G。
 *
 * -----------------------------------------------------------------------------
 * 关键表达式中各项的物理/数学含义（标量 kplusg 的组成）：
 *
 *   kplusg = (gcar[(ik * npwx + ig) * 3 + pol]
 *             + kvec_c[ik * 3 + pol]) * tpiba;
 *
 *   1. gcar[(ik * npwx + ig) * 3 + pol]
 *
 *      - gcar      : 存储 G 向量在笛卡尔坐标中的分量（通常是以晶格常数
 *                    为单位的“无量纲倒空间坐标”，例如 2π/alat 尚未乘上）。
 *      - ik        : 当前 k 点索引。
 *      - npwx      : 每个 k 点分配的最大平面波数（用于在数组中做平铺）。
 *      - ig        : 当前平面波 (G) 的索引。
 *      - 3         : 每个向量有 3 个分量 (x, y, z)。
 *      - pol       : 当前计算的方向索引 (0,1,2 分别对应 x,y,z)。
 *
 *      索引展开过程：
 *
 *      - 对于第 ik 个 k 点，其平面波 G 从 0 到 npwx-1 顺序存储。
 *      - 对于每个 G，有 3 个分量 (G_x, G_y, G_z) 排在一起。
 *      - 所以：
 *            每个k点都用npwx个平面波存储，ik*npwx 定位到第 ik 个 k 点的起始位置，
 *            加上 ig 定位到该 k 点下的第 ig 个平面波。但每个平面波有 3 个分量，
 *            因此需要乘以 3 才能定位到该平面波的三个分量的起始位置，
 *            最后加上 pol 选中具体的分量（0 对应 x，1 对应 y，2 对应 z）。
 *            对于每个k点，其平面波数据是连续存储的，格式如下：
 *              | G0_x | G0_y | G0_z | G1_x | G1_y | G1_z | ... | G(npwx-1)_x | G(npwx-1)_y | G(npwx-1)_z |
 *            base_index   = ik * npwx + ig        // 定位到该 k 点下的第 ig 个 G
 *            vector_index = base_index * 3        // 该 G 的三个分量起始位置
 *            component    = vector_index + pol    // 选中第 pol 个分量
 *
 *      因此：
 *
 *            gcar[(ik * npwx + ig) * 3 + pol]  ≡  G_α(ik, ig)
 *
 *      表示第 ik 个 k 点下，第 ig 个平面波的倒格矢 G 在第 α=pol 个
 *      笛卡尔方向上的分量（尚未乘上 2π/alat 等因子）。
 *
 *   2. kvec_c[ik * 3 + pol]
 *
 *      - kvec_c    : 存储各 k 点的倒空间向量 k 在笛卡尔坐标中的 3 个分量。
 *      - ik * 3    : 第 ik 个 k 点在一维数组中的起始位置。
 *      - + pol     : 选取该 k 点在第 pol 个方向上的分量。
 *
 *      所以：
 *
 *            kvec_c[ik * 3 + pol]  ≡  k_α(ik)
 *
 *      表示第 ik 个 k 点的 k 向量在第 α=pol 个笛卡尔方向上的分量。
 *
 *   3. gcar[...] + kvec_c[...]
 *
 *      - 将同一方向上的 G_α 和 k_α 相加，得到 (k + G) 在该方向的分量：
 *
 *            (k + G)_α = k_α(ik) + G_α(ik, ig)
 *
 *      这正是梯度表达式 ∂ρ/∂r_α 中 i (k_α + G_α) 的实部系数。
 *
 *   4. 乘以 tpiba
 *
 *      - tpiba 一般在平面波 DFT 程序中表示一个将“晶格单位”倒空间矢量
 *        转换为物理单位 (Å^-1 或 Bohr^-1) 的比例因子：
 *
 *            tpiba ≈ 2π / a   （其中 a 是某个参考晶格长度，例如 alat）
 *
 *      - 由于 gcar 和 kvec_c 中的分量一般是以“相对于倒基矢”的无量纲形式
 *        存储（例如 b1, b2, b3 的线性组合中的系数），要得到真正的倒空间
 *        矢量分量，需要乘上 2π/alat 之类的因子。
 *
 *      因此：
 *
 *            kplusg = (k_α(ik) + G_α(ik, ig)) * tpiba
 *
 *      就是 (k + G) 向量在第 α 个方向上的物理量纲分量（单位为倒长度）。
 *
 * -----------------------------------------------------------------------------
 * 与梯度公式的对应关系：
 *
 *   - 在平面波展开下，第 α 个方向的梯度为：
 *
 *        ∂ρ(r) / ∂r_α = Σ_G i (k_α + G_α) ρ_G e^{i (k + G) · r}
 *
 *   - 数值实现中，对每个平面波分量 G（即循环中的 ig）：
 *
 *        • kplusg 对应 (k_α + G_α) 在物理单位下的数值；
 *        • rhog[ig] 对应 ρ_G；
 *        • porter[ig] 对应 i (k_α + G_α) ρ_G（即在倒空间的梯度系数）。
 *
 *   最终，通过对 porter[ig] 做逆傅里叶变换（以及对三个分量 pol=0,1,2
 *   分别计算），就可以在实空间上得到 ρ(r) 的梯度：
 *
 *        porter[ig] ← i (k_α + G_α) ρ_G
 *        ⇒ 逆 FFT ⇒ ∂ρ(r) / ∂r_α
 *
 * 综上，kplusg 的表达式精确地实现了从“数组索引 + 无量纲 G/k 分量”
 * 到“物理单位下的 (k + G) 在指定笛卡尔方向上的分量”的映射，从而为
 * 利用平面波展开计算电荷密度梯度提供了正确的系数。
 */
    for(int ig = 0; ig < npw; ig++) {
		// the formula is : rho(r)^prime = \int iG * rho(G)e^{iGr} dG
		// double kplusg = wfc_basis->getgpluskcar(ik,ig)[ipol] * tpiba;
        
        
        Real kplusg = (gcar[(ik * npwx + ig) * 3 + pol] +
                       kvec_c[ik * 3 + pol]) * tpiba;
                       
		// calculate the charge density gradient in reciprocal space.
        // T 被实例化为 std::complex<float/double>（文件底部明确实例化），
        // 所以 T(0.0, kplusg) 等价于 complex(0, kplusg) = i * kplusg（实部为0，虚部为kplusg）
        // porter[ig] = i * (k_α + G_α) * ρ_G，即第ik个k点下第ig个平面波分量在pol方向上的倒空间梯度系数
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

// 这两行是对模板类 xc_functional_grad_wfc_op 的显式实例化。
// 保证模板类在链接时有对应的实例代码（尤其在模板定义和使用分离时）。
// 这里明确指定 T 为 std::complex<float> 或 std::complex<double>，即复数类型，Device 为 CPU。
template struct xc_functional_grad_wfc_op<std::complex<float>, base_device::DEVICE_CPU>;
template struct xc_functional_grad_wfc_op<std::complex<double>, base_device::DEVICE_CPU>;

} // namespace hamilt   