// This file contains subroutines realted to gradient calculations
// it contains 5 subroutines:
// 1. gradcorr, which calculates gradient correction
// 2. grad_wfc, which calculates gradient of wavefunction
//		it is used in stress_func_mgga.cpp
// 3. grad_rho, which calculates gradient of density
// 4. grad_dot, which calculates divergence of something
// 5. noncolin_rho, which diagonalizes the spin density matrix
//  and gives the spin up and spin down components of the charge.

// 该文件包含与梯度计算相关的子程序
// 它包含5个子程序：
// 1. gradcorr，计算梯度校正
// 2. grad_wfc，计算波函数的梯度，它用于stress_func_mgga.cpp中
// 3. grad_rho，计算密度的梯度
// 4. grad_dot，计算某物的散度
// 5. noncolin_rho，对自旋密度矩阵进行对角化，并给出自旋向上和自旋向下的电荷分量。

#include "xc_functional.h"
#include "source_base/timer.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_io/module_parameter/parameter.h"
#include <ATen/core/tensor.h>
#include <ATen/core/tensor_map.h>
#include <ATen/core/tensor_types.h>
#include <source_hamilt/module_xc/kernels/xc_functional_op.h>

#ifdef USE_LIBXC
#include "xc_functional_libxc.h"
#endif

// from gradcorr.f90
void XC_Functional::gradcorr(double &etxc, double &vtxc, ModuleBase::matrix &v,
	const Charge* const chr, ModulePW::PW_Basis* rhopw, const UnitCell *ucell,
	std::vector<double> &stress_gga, const bool is_stress)
{
	ModuleBase::TITLE("XC_Functional","gradcorr");
	
	// 如果是 LDA 或无泛函，直接返回
	if(func_type == 0 || func_type == 1) { return; } // none or LDA functional

	bool igcc_is_lyp = false;
	// 看GGA关联部分是否使用LYP泛函
	if( func_id[1] == XC_GGA_C_LYP) { igcc_is_lyp = true; }

	// nspin0 先等于输入的自旋通道数
	// PARAM.inp.nspin可能是1,2,4
	// 1：无自旋极化（非磁性）；2：自旋极化（磁性，collinear）；4：非共线自旋
	int nspin0 = PARAM.inp.nspin;
	// 如果 nspin==4，默认 nspin0=1，即当非共线自旋但不考虑磁性时，按非极化处理。
	// domag 或 domag_z 指示是否考虑磁性或z方向磁性
	// 如果 nspin==4 且 domag 或 domag_z 为真，说明是非共线磁性体系，nspin0=2，即按自旋极化处理。
	if(PARAM.inp.nspin==4) { nspin0 =1; }
	if(PARAM.inp.nspin==4&&(PARAM.globalv.domag||PARAM.globalv.domag_z)) { nspin0 = 2; }

	// 这是一个断言（assertion），确保变量 nspin0 的值大于0，否则程序会在这里终止并报错
	// 根据前面代码， nspin0 应该是1或2
	assert(nspin0>0);
	const double fac = 1.0/ nspin0;

	// 判断是否需要计算应力
	if(is_stress)
	{
		// stress_gga：这是一个 std::vector<double> 类型的变量，
		// 用于存储梯度校正部分（GGA）对应力张量的贡献。应力张量在三维空间中是一个3×3的对称矩阵，共有9个分量。
		// resize() 的作用是改变 vector 的大小，
		// 这里就是将 stress_gga 的大小设置为9，确保有9个元素用于存储应力张量的各个分量。
		stress_gga.resize(9);
		for(int i=0;i<9;i++)
		{
			// 将所有分量初始化为0.0
			stress_gga[i] = 0.0;
		}
	}

	// doing FFT to get rho in G space: rhog1 
	// 将自旋通道0的实空间密度 chr->rho[0] 通过快速傅里叶变换（FFT）变换到倒空间，结果存储在 chr->rhog[0]
    rhopw->real2recip(chr->rho[0], chr->rhog[0]);
	// 如果是自旋极化体系（nspin=2），还需要处理自旋通道1
	if(PARAM.inp.nspin==2)//mohan fix bug 2012-05-28
	{
		// 把自旋通道1的实空间密度变换到倒空间。
		rhopw->real2recip(chr->rho[1], chr->rhog[1]);
	}
	// 把核区电荷密度从实空间变换到倒空间。
    rhopw->real2recip(chr->rho_core, chr->rhog_core);
		
	// sum up (rho_core+rho) for each spin in real space
	// and reciprocal space.
	double* rhotmp1 = nullptr;
	double* rhotmp2 = nullptr;
	std::complex<double>* rhogsum1 = nullptr;
	std::complex<double>* rhogsum2 = nullptr;
	ModuleBase::Vector3<double>* gdr1 = nullptr;
	ModuleBase::Vector3<double>* gdr2 = nullptr;
	ModuleBase::Vector3<double>* h1 = nullptr;
	ModuleBase::Vector3<double>* h2 = nullptr;
	double* neg = nullptr;
	double** vsave = nullptr;
	double** vgg = nullptr;
	
	// for spin unpolarized case, 
	// calculate the gradient of (rho_core+rho) in reciprocal space.
	// 自旋非极化体系（即只有一个自旋通道）
	// 但这部分并没有if判断 nspin0==1，因为即使 nspin0==2（自旋极化），
	// 也需要计算自旋通道0的部分（即自旋向上）
	// rhotmp1：实空间的总密度数组（长度为网格点数 nrxx），用于存储每个网格点上的 rho + rho_core
	rhotmp1 = new double[rhopw->nrxx];
	// rhogsum1：倒空间（G空间）的总密度数组（长度为平面波数 npw）
	rhogsum1 = new std::complex<double>[rhopw->npw];
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
	for(int ir=0; ir<rhopw->nrxx; ir++) 
	{
		// chr->rho[0][ir]：自旋通道0在第 ir 个实空间网格点上的密度值。
		// 计算实空间每个网格点上的总密度：rho + rho_core
		rhotmp1[ir] = chr->rho[0][ir] + fac * chr->rho_core[ir];
	}
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
	for(int ig=0; ig<rhopw->npw; ig++)
	{
		// chr->rhog[0][ig]：自旋通道0在第 ig 个 G 点上的倒空间密度。
		// 对倒空间的每个 G 点，将自旋密度和核区密度相加，得到总密度，存入 rhogsum1
		rhogsum1[ig] = chr->rhog[0][ig] + fac * chr->rhog_core[ig];
	}

	// 为每个实空间网格点分配一个三维向量（Vector3<double>），用于存储密度的梯度
	gdr1 = new ModuleBase::Vector3<double>[rhopw->nrxx];
	// 如果不计算应力，则为每个网格点分配一个三维向量 h1，用于后续存储梯度校正相关的中间量。
	if(!is_stress) { h1 = new ModuleBase::Vector3<double>[rhopw->nrxx]; }
	
	// 调用 grad_rho 函数，根据倒空间每个 G 点的总密度 rhogsum1 计算实空间每个网格点的密度梯度，结果存储在 gdr1 中。
	// ucell->tpiba=2π/a，a 是晶格常数，将计算结果从“内部约化单位”还原为真实的“物理单位”。
	XC_Functional::grad_rho( rhogsum1 , gdr1, rhopw, ucell->tpiba);

	// 自旋极化体系
	// for spin polarized case;
	// calculate the gradient of (rho_core+rho) in reciprocal space.
	if(PARAM.inp.nspin==2)
	{
		// 实空间的总密度数组（长度为网格点数 nrxx）
		rhotmp2 = new double[rhopw->nrxx];
		// 倒空间（G空间）的总密度数组（长度为平面波数 npw）
		rhogsum2 = new std::complex<double>[rhopw->npw];
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
		for(int ir=0; ir<rhopw->nrxx; ir++)
		{
			// rhotmp1（自旋向上的部分，即自旋通道0）已经在上面计算过了，
			// 这里计算自旋向下部分的总密度：rho + rho_core
			// 但但核区密度本身是无自旋的，为了在自旋通道的计算中保持总电子数守恒，需要将核区密度平均分配到两个自旋通道。
			// 这样，两个通道各自获得一半的核区密度，所以这里乘以 fac（1/nspin0，nspin0=2）
			rhotmp2[ir] = chr->rho[1][ir] + fac * chr->rho_core[ir];
		}
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
		for(int ig=0; ig<rhopw->npw; ig++)
		{
			// 同理，计算倒空间的每个 G 点，将自旋密度和核区密度相加，得到总密度，存入 rhogsum2
			rhogsum2[ig] = chr->rhog[1][ig] + fac * chr->rhog_core[ig];
		}

		// 为每个实空间网格点分配一个三维向量（Vector3<double>），用于存储自旋向下的密度的梯度
		gdr2 = new ModuleBase::Vector3<double>[rhopw->nrxx];
		// 如果不计算应力，则为每个网格点分配一个三维向量 h2，用于后续存储梯度校正相关的中间量。
		if(!is_stress) { h2 = new ModuleBase::Vector3<double>[rhopw->nrxx]; }
		
		// 调用 grad_rho 函数，根据倒空间每个 G 点的自旋向下的总密度 rhogsum2 计算实空间每个网格点的密度梯度，结果存储在 gdr2 中。
		XC_Functional::grad_rho( rhogsum2 , gdr2, rhopw, ucell->tpiba);
	}

	// 非共线自旋磁性体系，即 nspin==4 且 domag 或 domag_z 为真。
	// 此时体系的自旋密度是一个三维矢量（不仅有z分量，还可能有x、y分量）
	if(PARAM.inp.nspin == 4&&(PARAM.globalv.domag||PARAM.globalv.domag_z))
	{
		rhotmp2 = new double[rhopw->nrxx];
		rhogsum2 = new std::complex<double>[rhopw->npw];
		// 辅助数组，长度为网格点数 nrxx，用于后续判断自旋分量方向
 		neg = new double [rhopw->nrxx];
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
		for(int ir=0; ir<rhopw->nrxx; ir++)
		{
			rhotmp1[ir] = 0.0;
			rhotmp2[ir] = 0.0;
			neg[ir] = 0.0;
		}
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
		for(int ig=0; ig<rhopw->npw; ig++)
		{
			rhogsum1[ig] = 0.0;
			rhogsum2[ig] = 0.0;
		}
		if(!is_stress)
		{
			// 不计算应力时，为每个自旋通道分配一块用于暂存势能的二维数组
			vsave = new double* [PARAM.inp.nspin];
			// vsave：类型为 double**，用于保存每个自旋通道在所有实空间网格点上的势能（v）
			for(int is = 0;is<PARAM.inp.nspin;is++) {
				vsave[is]= new double [rhopw->nrxx];
			}
#ifdef _OPENMP
// collapse(2)用于将紧邻的两个 for 循环合并为一个大的循环，
// 也就是说把下面的内外两个循环“展开”为一个总共 PARAM.inp.nspin × rhopw->nrxx 次的单层循环，
// 然后把这些迭代任务平均分配给多个线程
// 如果不加 collapse(2)，OpenMP 只会对最外层的 for 循环进行并行，而内层循环不会被并行化
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
			// 将每个自旋通道、每个实空间网格点上的势能 v 暂存到 vsave 中，
			// 并将 v 清零，为后续计算做准备
			for(int is = 0;is<PARAM.inp.nspin;is++) {
				for(int ir =0;ir<rhopw->nrxx;ir++){
					vsave[is][ir] = v(is,ir);
					v(is,ir) = 0;
				}
			}
			// vgg：类型为 double**，后续存储每个自旋通道、每个网格点上的梯度校正势能（或中间结果）
			vgg = new double* [nspin0];
			for(int is = 0;is<nspin0;is++) {vgg[is] = new double[rhopw->nrxx];
}
		}
		// 对非共线自旋体系的自旋密度矩阵进行对角化，得到每个网格点上的“自旋向上”和“自旋向下”分量。
		// rhotmp1, rhotmp2：输出数组，分别存储对角化后的自旋向上和自旋向下密度。
		// neg：辅助数组，判断自旋分量方向（与量子化轴有关）。
		// chr->rho：输入的自旋密度矩阵（4个分量，分别为总密度和三个自旋分量）。
		// rhopw->nrxx：网格点总数。
		// ucell->magnet.ux_：量子化轴方向。
		// ucell->magnet.lsign_：是否采用固定量子化轴。
		noncolin_rho(rhotmp1,rhotmp2,neg,chr->rho,rhopw->nrxx,ucell->magnet.ux_,ucell->magnet.lsign_);
		// 将实空间的自旋密度（rhotmp1、rhotmp2）通过快速傅里叶变换转换到倒空间（G空间），分别存储到rhogsum1和rhogsum2
		rhopw->real2recip(rhotmp1, rhogsum1);
		rhopw->real2recip(rhotmp2, rhogsum2);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
		for(int ir=0; ir<rhopw->nrxx; ir++)
		{
			// 在非共线自旋体系中，经过对角化后得到的“自旋向上/下”分量只包含价电子部分,
			// 所以要将核区密度均匀加到自旋向上和自旋向下两个分量上（fac = 1/nspin0，nspin0=2）
			rhotmp2[ir] += fac * chr->rho_core[ir];
			rhotmp1[ir] += fac * chr->rho_core[ir];
		}
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
		for(int ig=0; ig<rhopw->npw; ig++)
		{
			// 将核区密度的倒空间分量均匀分配到自旋向上和自旋向下两个分量的倒空间密度（rhogsum1 和 rhogsum2）中
			rhogsum2[ig] += fac * chr->rhog_core[ig];
			rhogsum1[ig] += fac * chr->rhog_core[ig];
		}

		// gdr2：类型为 ModuleBase::Vector3<double>*，
		// 为每个实空间网格点分配一个三维向量（Vector3<double>），用于存储自旋向下的密度的梯度
		gdr2 = new ModuleBase::Vector3<double>[rhopw->nrxx];
		// 如果不计算应力，则为每个网格点分配一个三维向量 h2，用于后续存储梯度校正相关的中间量。
		if(!is_stress) h2 = new ModuleBase::Vector3<double>[rhopw->nrxx];

		// 分别计算自旋向上（或分量1）和自旋向下（或分量2）密度的梯度，结果分别存储在 gdr1 和 gdr2 中。
		// rhogsum1 / rhogsum2：倒空间（G空间）中自旋向上/向下的总密度（包含价电子和核区密度）。
		XC_Functional::grad_rho( rhogsum1 , gdr1, rhopw, ucell->tpiba);
		XC_Functional::grad_rho( rhogsum2 , gdr2, rhopw, ucell->tpiba);

	}
	
	const double epsr = 1.0e-6;
	const double epsg = 1.0e-10;

	double vtxcgc = 0.0;
	double etxcgc = 0.0;

#ifdef _OPENMP
#pragma omp parallel
{
	// #pragma omp parallel 的作用是开启一个多线程并行区域，
	// 在这个区域内的大括号 { ... } 里的代码会被每个线程各自执行一遍。
	// 所以如果启用了 OpenMP 并行编译（多线程），则每个线程都分配自己的局部变量：
	// local_stress_gga：每个线程自己的应力张量累加器（9个分量）。
	// local_vtxcgc、local_etxcgc：每个线程自己的梯度校正势和能量累加器。
	// if(is_stress)：如果需要计算应力，则初始化 local_stress_gga 为9个0。
	// 这样设计可以保证在多线程并行计算时，每个线程独立累加自己的结果，
	// 最后再统一归约（reduction）到全局变量，避免多线程写同一内存导致的错误。
	std::vector<double> local_stress_gga;
	double local_vtxcgc = 0.0;
	double local_etxcgc = 0.0;

	if(is_stress)
	{
		local_stress_gga.resize(9);
		for(int i=0;i<9;i++)
		{
			local_stress_gga[i] = 0.0;
		}
	}
#else
	// 如果没有启用 OpenMP 并行编译（单线程），
	// 则直接使用全局变量 stress_gga、vtxcgc、etxcgc 进行累加，无需线程私有变量。
	std::vector<double> &local_stress_gga = stress_gga;
	double &local_vtxcgc = vtxcgc;
	double &local_etxcgc = etxcgc;
#endif

	double grho2a = 0.0;
	double grho2b = 0.0;
	double sxc = 0.0;
	double v1xc = 0.0;
	double v2xc = 0.0;

	if(nspin0==1)
	{
		double segno;
#ifdef _OPENMP
// 这里#pragma omp for 前面没有加 parallel，是因为这段代码已经处于前面开启的 #pragma omp parallel 并行区域内部。
#pragma omp for
#endif
		for(int ir=0; ir<rhopw->nrxx; ir++)
		{
			// 取当前网格点上密度的绝对值
			const double arho = std::abs( rhotmp1[ir] );
			if(!is_stress) { h1[ir].x = h1[ir].y = h1[ir].z = 0.0;
}

			if(arho > epsr)
			{
				// 计算第 ir 个网格点上自旋通道1（或自旋向上）的密度梯度的模平方（x^2 + y^2 + z^2）
				grho2a = gdr1[ir].norm2();
				
				//normally values in rhotmp can either be >= 0 or < 0.
				// 判断当前网格点上的密度 rhotmp1[ir] 是正还是负。
				// 如果密度为正，则 segno 设为 1.0；如果为负，则设为 -1.0，用于后续能量累加时，保证能量符号与密度一致
				if( rhotmp1[ir] >= 0.0 ) { segno = 1.0;
				} else { segno = -1.0;
}
				if (use_libxc && is_stress)
				{
#ifdef USE_LIBXC
					if(func_type == 3 || func_type == 5) //the gradcorr part to stress of mGGA
					{
						double v3xc;
						double atau = chr->kin_r[0][ir]/2.0;
						XC_Functional_Libxc::tau_xc( func_id, arho, grho2a, atau, sxc, v1xc, v2xc, v3xc);
					}
					else
					{
						XC_Functional_Libxc::gcxc_libxc( func_id, arho, grho2a, sxc, v1xc, v2xc);
					}
#endif 
				} // end use_libxc
				else
				{
					// gcxc函数根据给定的电子密度（arho）和其梯度的平方（grho2a），
					// 计算对应的交换-关联能量密度（sxc）以及相关势（v1xc, v2xc）
					XC_Functional::gcxc( arho, grho2a, sxc, v1xc, v2xc);
				}
				/* GGA对应力张量的贡献（即对晶格应变的导数）可以表示为：
				   σ_ij = (1/V) ∫ [ (∂(ρ*ε_xc)/∂|∇ρ|) * (∇ρ_i) * (∇ρ_j) / |∇ρ| ] d^3r
				   其中 i,j = x,y,z 分量索引，V 是体系体积，ε_xc 是交换-关联能量密度。
				   ∇ρ_i 和 ∇ρ_j 分别是电子密度梯度在 i 和 j 方向的分量。
				   v2xc = ∂(ρ*ε_xc)/∂|∇ρ|/|∇ρ|
				   σ_ij反映了电子密度梯度对体系应力的贡献。	
				*/
				if(is_stress)
				{
					double tt[3];
					// 将密度梯度的三个分量（∇ρ_x, ∇ρ_y, ∇ρ_z）存入临时数组 tt 中，便于后续计算
					// 注意这里的ir，即所以 tt 存储的是当前网格点 ir 上的梯度分量，所以别忘了外层遍历所有网格点的循环
					tt[0] = gdr1[ir].x;
					tt[1] = gdr1[ir].y;
					tt[2] = gdr1[ir].z;
					// 应力张量σ_ij是一个3×3的对称矩阵，即σ_ij = σ_ji，
					// 为了避免重复计算，代码通过 m < l + 1 只计算矩阵的下三角部分（包括对角线）
					for(int l = 0;l< 3;l++)
					{
						for(int m = 0;m< l+1;m++)
						{
							// 设这个矩阵的9个元素的索引为0~8，
							// 则循环中索引ind依次取到0,3,4,6,7,8，对应σ_xx, σ_yx, σ_yy, σ_zx, σ_zy, σ_zz
							int ind = l*3 + m;
							// 每一点的贡献是两个方向梯度分量的乘积 tt[l] * tt[m] 再乘以势能项 v2xc
							// 这里的+=是因为外层有个遍历所有网格点ir的循环，
							// 而每一个local_stress_gga[ind]都要累加所有网格点的贡献，所以要用+=。
							// 而乘以 ModuleBase::e2（2.0），是因为abacus使用Rydberg单位制，
							// 需要将能量单位从 Hartree 转换为 Rydberg。
							// 在Hartree单位制下，1 Hartree = e^2 / (4πε₀ a₀)，在此单位制下，e^2=1，4πε₀=a₀=1
							// 在Rydberg单位制下，1 Rydberg = (1/2) * e^2 / (4πε₀ a₀)，
							// 由于1 Hartree = 2 Rydberg，所以如果以Rydberg作为能量的基准单位，
							// 在 Hartree 单位制中为1的能量数值，在 Rydberg 单位制中就变成了2，因此e^2=2.0。
							// 但是这里好像还没处理离散化后的体积元dr和体积 V 的归一化问题，可能在后续代码中会处理。
							local_stress_gga[ind] += tt[l] * tt[m] * ModuleBase::e2 * v2xc;
						}
					}
				}
				else
				{
					// first term of the gradient correction:
					// D(rho*Exc)/D(rho)
					// 梯度校正的第一项：d(rho*Exc)/d(rho)
					v(0, ir) += ModuleBase::e2 * v1xc;
					// cout << "v    " << v(0, ir) << endl;
					
					// h contains
					// D(rho*Exc) / D(|grad rho|) * (grad rho) / |grad rho|
					// h1 = d(rho*Exc)/d(|grad rho|) * (grad rho) / |grad rho| ,
					// 其中，v2xc = d(rho*Exc)/d(|grad rho|) / |grad rho|
					h1[ir] = ModuleBase::e2 * v2xc * gdr1[ir];
					
					// local_vtxcgc为本线程的梯度校正势能（只对价电子部分）
					// local_vtxcgc = d(rho*Exc)/d(rho) * (rho - rho_core)
					local_vtxcgc += ModuleBase::e2* v1xc * ( rhotmp1[ir] - chr->rho_core[ir] );
					// local_etxcgc为本线程的梯度校正能量
					// segno为密度符号，保证能量符号与密度一致，当密度为负时，segno=-1，当密度为正时，segno=1
					local_etxcgc += ModuleBase::e2* sxc  * segno;
				}
			} // end arho > epsr
		}
	}// end nspin0 == 1
	// 自旋极化体系
	else // spin polarized case
	{
#ifdef _OPENMP
#pragma omp for
#endif
		for(int ir=0; ir<rhopw->nrxx; ir++)
		{
			if(use_libxc)
			{
#ifdef USE_LIBXC
				double sxc, v1xcup, v1xcdw, v2xcup, v2xcdw, v2xcud;
				if(func_type == 3 || func_type == 5) //the gradcorr part to stress of mGGA
				{
					double v3xcup, v3xcdw;
					double atau1 = chr->kin_r[0][ir]/2.0;
					double atau2 = chr->kin_r[1][ir]/2.0;
					XC_Functional_Libxc::tau_xc_spin(
						func_id,
						rhotmp1[ir], rhotmp2[ir], gdr1[ir], gdr2[ir], 
						atau1, atau2, sxc, v1xcup, v1xcdw, v2xcup, v2xcdw, v2xcud, v3xcup, v3xcdw);
				}
				else
				{
					XC_Functional_Libxc::gcxc_spin_libxc(
						func_id,
						rhotmp1[ir], rhotmp2[ir], gdr1[ir], gdr2[ir], 
						sxc, v1xcup, v1xcdw, v2xcup, v2xcdw, v2xcud);
				}
				if(is_stress)
				{
					double tt1[3],tt2[3];
					{
						tt1[0] = gdr1[ir].x;
						tt1[1] = gdr1[ir].y;
						tt1[2] = gdr1[ir].z;
						tt2[0] = gdr2[ir].x;
						tt2[1] = gdr2[ir].y;
						tt2[2] = gdr2[ir].z;
					}
					for(int l = 0;l< 3;l++)
					{
						for(int m = 0;m< l+1;m++)
						{
							int ind = l*3 + m;
							local_stress_gga [ind] += ( tt1[l] * tt1[m] * v2xcup + 
									tt2[l] * tt2[m] * v2xcdw + 
									(tt1[l] * tt2[m] +
									tt2[l] * tt1[m] ) * v2xcud ) * ModuleBase::e2;
						}
					}
				}
				else
				{
					// first term of the gradient correction : D(rho*Exc)/D(rho)
					v(0,ir) += ModuleBase::e2 * v1xcup;
					v(1,ir) += ModuleBase::e2 * v1xcdw;
				
					// h contains D(rho*Exc)/D(|grad rho|) * (grad rho) / |grad rho|
					h1[ir] += ModuleBase::e2 * ( v2xcup * gdr1[ir] + v2xcud * gdr2[ir] );
					h2[ir] += ModuleBase::e2 * ( v2xcdw * gdr2[ir] + v2xcud * gdr1[ir] );

					local_vtxcgc = local_vtxcgc + ModuleBase::e2 * v1xcup * ( rhotmp1[ir] - chr->rho_core[ir] * fac );
					local_vtxcgc = local_vtxcgc + ModuleBase::e2 * v1xcdw * ( rhotmp2[ir] - chr->rho_core[ir] * fac );
					local_etxcgc = local_etxcgc + ModuleBase::e2 * sxc;
				}
#endif
			}
			else
			{
				double v1cup = 0.0;
				double v1cdw = 0.0;
				double v2cup = 0.0;
				double v2cdw = 0.0;
				double v1xup = 0.0;
				double v1xdw = 0.0;
				double v2xup = 0.0;
				double v2xdw = 0.0;
				double v2cud = 0.0;
				double v2c = 0.0;
				double sx = 0.0;
				double sc = 0.0;
				// rh 为当前网格点上两个自旋通道密度之和（含核区电子密度）
				double rh = rhotmp1[ir] + rhotmp2[ir];
				// grho2a：自旋向上密度梯度的模平方（x^2 + y^2 + z^2）
				grho2a = gdr1[ir].norm2();
				// grho2b：自旋向下密度梯度的模平方（x^2 + y^2 + z^2）
				grho2b = gdr2[ir].norm2();
				// 调用gcx_spin函数，计算自旋极化体系下的交换能量密度（sx）及其相关势（v1xup, v1xdw, v2xup, v2xdw）
				XC_Functional::gcx_spin(rhotmp1[ir], rhotmp2[ir], grho2a, grho2b,
					sx, v1xup, v1xdw, v2xup, v2xdw);
				
				if(rh > epsr)
				{
					if(igcc_is_lyp)
					{
						ModuleBase::WARNING_QUIT("XC_Functional","igcc_is_lyp is not available now.");
					}
					else
					{
						// 自旋极化率
						double zeta = ( rhotmp1[ir] - rhotmp2[ir] ) / rh;
						// neg[ir]：由 noncolin_rho 计算，表示该点自旋分量与量子化轴的方向（+1 或 -1）
						if(PARAM.inp.nspin==4&&(PARAM.globalv.domag||PARAM.globalv.domag_z)) { zeta = fabs(zeta) * neg[ir];
}
						// 自旋总密度梯度的模平方
						const double grh2 = (gdr1[ir]+gdr2[ir]).norm2();
						// 调用 gcc_spin 函数，计算自旋极化体系下的关联能量密度（sc）及其关联势（v1cup, v1cdw, v2c）
						XC_Functional::gcc_spin(rh, zeta, grh2, sc, v1cup, v1cdw, v2c);
						v2cup = v2c;
						v2cdw = v2c;
						v2cud = v2c;
					}
				}
				else
				{
					sc = 0.0;
					v1cup = 0.0;
					v1cdw = 0.0;
					v2c = 0.0;
					v2cup = 0.0;
					v2cdw = 0.0;
					v2cud = 0.0;
				}

				if(is_stress)
				{
					double tt1[3],tt2[3];
					{
						// 将自旋向上和自旋向下的密度梯度分量存入临时数组 tt1 和 tt2 中
						tt1[0] = gdr1[ir].x;
						tt1[1] = gdr1[ir].y;
						tt1[2] = gdr1[ir].z;
						tt2[0] = gdr2[ir].x;
						tt2[1] = gdr2[ir].y;
						tt2[2] = gdr2[ir].z;
					}
					for(int l = 0;l< 3;l++)
					{
						for(int m = 0;m< l+1;m++)
						{
							int ind = l*3 + m;
							//    exchange
							// 计算交换部分对应力张量的贡献（将自旋向上和自旋向下的分量分别考虑）
							local_stress_gga [ind] += tt1[l] * tt1[m] * ModuleBase::e2 * v2xup + 
									tt2[l] * tt2[m] * ModuleBase::e2 * v2xdw;
							//    correlation
							// 计算关联部分对应力张量的贡献（将自旋向上和自旋向下的分量分别考虑，以及它们的混合项）
							local_stress_gga [ind] += ( tt1[l] * tt1[m] * v2cup + 
									tt2[l] * tt2[m] * v2cdw + 
									(tt1[l] * tt2[m] +
									tt2[l] * tt1[m] ) * v2cud ) * ModuleBase::e2;
						}
					}
				}
				else
				{
					// first term of the gradient correction : D(rho*Exc)/D(rho)
					// 自旋向上和自旋向下梯度校正的第一项：d(rho*Exc)/d(rho)，累加每个网格点
					v(0,ir) = v(0,ir) + ModuleBase::e2 * ( v1xup + v1cup );
					v(1,ir) = v(1,ir) + ModuleBase::e2 * ( v1xdw + v1cdw );
				
					// h contains D(rho*Exc)/D(|grad rho|) * (grad rho) / |grad rho|
					// h：包含 d(rho*Exc)/d(|grad rho|) * (grad rho) / |grad rho|
					// 自旋向上分量和自旋向下分量分别计算，交换和关联部分、以及它们的混合项都考虑在内
					h1[ir] = ModuleBase::e2 * ( ( v2xup + v2cup ) * gdr1[ir] + v2cud * gdr2[ir] );
					h2[ir] = ModuleBase::e2 * ( ( v2xdw + v2cdw ) * gdr2[ir] + v2cud * gdr1[ir] );

					// local_vtxcgc为本线程的梯度校正势能（只对价电子部分），包含交换和关联部分、自旋向上和向下分量
					// local_etxcgc为本线程的梯度校正能量（包含交换和关联部分）
					local_vtxcgc = local_vtxcgc + ModuleBase::e2 * ( v1xup + v1cup ) * ( rhotmp1[ir] - chr->rho_core[ir] * fac );
					local_vtxcgc = local_vtxcgc + ModuleBase::e2 * ( v1xdw + v1cdw ) * ( rhotmp2[ir] - chr->rho_core[ir] * fac );
					local_etxcgc = local_etxcgc + ModuleBase::e2 * ( sx + sc );
				}
			}
		}// end ir

	}
#ifdef _OPENMP
	// #pragma omp critical用于用于创建一个临界区（Critical Section），
	// 互斥执行：它保证在同一时刻只有一个线程能执行大括号 { ... } 内部的代码块，
	// 防止数据竞争：在之前的代码中，每个线程都在并行计算自己的局部变量（如 local_stress_gga、local_etxcgc），
	// 这一步需要将这些个人结果加到全局变量（如 stress_gga、etxcgc）中，
	// 如果没有这个指令，多个线程同时写同一个内存位置会导致计算结果错误（即“数据竞争”）。
	// (xc_functional_gradcorr_reduce)：这是给这个临界区起的名字，方便调试和识别。
	#pragma omp critical(xc_functional_gradcorr_reduce)
	{
		// 如果计算应力，则将每个线程的局部应力张量累加到全局应力张量中
		if(is_stress)
		{
			for(int l = 0;l< 3;l++)
			{
				for(int m = 0;m< l+1;m++)
				{
					int ind = l*3 + m;
					stress_gga [ind] += local_stress_gga [ind];
				}
			}
		}
		// 如果不计算应力，则将每个线程的局部梯度校正势和能量累加到全局变量中
		else
		{
			vtxcgc += local_vtxcgc;
			etxcgc += local_etxcgc;
		}
	}
// 前面开启的 #pragma omp parallel 并行区域在这里结束
}
#endif

	//std::cout << "\n vtxcgc=" << vtxcgc;
	//std::cout << "\n etxcgc=" << etxcgc << std::endl;

	if(!is_stress)
	{
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
		for(int ir=0; ir<rhopw->nrxx; ir++)
		{
			// 将之前均匀分配到自旋通道1的核区密度从 rhotmp1 中减去，
			// 恢复 rhotmp1 只包含价电子部分的密度。
			rhotmp1[ir] -= fac * chr->rho_core[ir];
		}
		// 对于自旋极化体系，还需要将核区密度从自旋通道2的密度中减去
		if(nspin0==2)
		{
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
			for(int ir=0; ir<rhopw->nrxx; ir++)
			{
				rhotmp2[ir] -= fac * chr->rho_core[ir];
			}
		}
		
		// second term of the gradient correction :
		// \sum_alpha (D / D r_alpha) ( D(rho*Exc)/D(grad_alpha rho) )

		// dh is in real sapce.
		double* dh = new double[rhopw->nrxx];

		for(int is=0; is<nspin0; is++)
		{
			// 对每个自旋通道，调用 grad_dot 函数计算h的散度，结果存储在 dh 中
			if(is==0) {XC_Functional::grad_dot(h1,dh,rhopw,ucell->tpiba);
}
			if(is==1) {XC_Functional::grad_dot(h2,dh,rhopw,ucell->tpiba);
}
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
			for(int ir=0; ir<rhopw->nrxx; ir++) {
				// 将散度项从交换-相关势中减去
				v(is, ir) -= dh[ir];
}
		
			double sum = 0.0;
			// 计算散度dh与对应自旋通道价电子密度的乘积在实空间的积分（累加）
			if(is==0)
			{
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum) schedule(static, 256)
#endif
				for(int ir=0; ir<rhopw->nrxx; ir++) {
					sum += dh[ir] * rhotmp1[ir];
}
			}
			else if(is==1)
			{
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum) schedule(static, 256)
#endif
				for(int ir=0; ir<rhopw->nrxx; ir++) {
					sum += dh[ir] * rhotmp2[ir];
}
			}
			// 从梯度校正势能中减去该积分值
			vtxcgc -= sum;
		}
		
		delete[] dh;

		// vtxc和etxc是主调用者传入的引用变量，是总势和总能；vtxcgc和etxcgc是当前函数内计算得到的梯度校正势和能量，
		// 将本函数计算得到的梯度校正势和能量累加到总势和总能中
		vtxc += vtxcgc;
		etxc += etxcgc;

		// 对于非共线自旋情况，需要对GGA势进行自旋分量的重组
		if(PARAM.inp.nspin == 4 && (PARAM.globalv.domag||PARAM.globalv.domag_z))
		{
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
			for(int is=0;is<PARAM.inp.nspin;is++)
			{
				for(int ir=0;ir<rhopw->nrxx;ir++)
				{
					// 对于有效的自旋通道（is < nspin0），将当前计算得到的 GGA 势能存入 vgg，以便后续处理。
					if(is<nspin0) { vgg[is][ir] = v(is,ir);
}
					// 恢复势能数组 v 为原始值（即未加GGA修正前的势），为后续的自旋分量重组做准备。
					v(is,ir) = vsave[is][ir];
				}
			}
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
			for(int ir=0;ir<rhopw->nrxx;ir++)
			{
				// 将自旋向上和向下通道的GGA势能平均，累加到总密度分量的势能上。
				v(0,ir) += 0.5 * (vgg[0][ir] + vgg[1][ir]);
				// 计算自旋矢量模长 amag，判断该点是否有显著的自旋极化。
				double amag = sqrt(pow(chr->rho[1][ir],2)+pow(chr->rho[2][ir],2)+pow(chr->rho[3][ir],2));
				if(amag>1e-12)
				{
					// 将自旋通道的势能差，按自旋密度方向分配到各自旋分量上，实现自旋分量势能的重组。
					for(int i=1;i<4;i++) {
						v(i,ir)+= neg[ir] * 0.5 *(vgg[0][ir]-vgg[1][ir])*chr->rho[i][ir]/amag;
}
				}
			}
			// 这个for循环将对角化后的自旋通道势能（即“自旋向上/下”）重新组合为原始的四分量（总密度 + 三个自旋分量）形式，
			// 保证非共线自旋体系下势能的物理自洽性，这种重组方式确保了自旋极化方向和大小的信息被正确地反映到势能分量中。
		}
	}
	// deacllocate
	delete[] rhotmp1;
	delete[] rhogsum1;
	delete[] gdr1;
	if(!is_stress) { delete[] h1;
}

	if(PARAM.inp.nspin==2)
	{
		delete[] rhotmp2;
		delete[] rhogsum2;
		delete[] gdr2;
		if(!is_stress) { delete[] h2;
}
	}
	if(PARAM.inp.nspin == 4 && (PARAM.globalv.domag||PARAM.globalv.domag_z))
	{
		delete[] neg;
		if(!is_stress) 
		{
			for(int i=0; i<nspin0; i++) { delete[] vgg[i];
}
			delete[] vgg;
			for(int i=0; i<PARAM.inp.nspin; i++) { delete[] vsave[i];
}
			delete[] vsave;
			delete[] h2;
		}
		delete[] rhotmp2;
		delete[] rhogsum2;
		delete[] gdr2;
	}

	return;
}

// template是C++模板声明，表示后面的函数（或类）是一个泛型模板函数，可以接受不同类型的参数
// T：类型参数，代表数据类型（如 double、std::complex<double> 等）。
// Device：类型参数，代表计算设备（如 CPU、GPU）。
// Real：类型参数，代表实数类型（如 float、double）。
template <typename T, typename Device, typename Real>
// 计算波函数梯度
void XC_Functional::grad_wfc(
    const int ik,
    const Real tpiba,
    const ModulePW::PW_Basis_K* wfc_basis,
	const T* rhog,
    T* grad)
{
	// using ct_Device 这是类型别名声明，用于简化后续代码中的类型书写。
	// ct::PsiToContainer<Device>::type 是一个类型萃取（type trait），
	// 根据模板参数 Device（比如 CPU 或 GPU），自动选择对应的容器类型（如张量的存储方式）。
	// ct_Device 就是后续代码中实际用到的设备类型（比如 CpuDevice 或 GpuDevice）
    using ct_Device = typename ct::PsiToContainer<Device>::type;
	// 表示第 ik 个 k 点上的平面波数（npw_k）。wfc_basis->npwk 是一个数组，存储每个 k 点的平面波数量。
	const int npw_k = wfc_basis->npwk[ik];
	
	// 创建了一个张量对象 porter，用于存储波函数梯度计算中的中间结果。
	// ct::Tensor(...)：构造一个张量对象，类似于多维数组，支持不同数据类型和设备（如CPU/GPU）。
	// ct::DataTypeToEnum<T>::value：根据模板参数 T（如 double 或 std::complex<double>），自动选择张量的数据类型。
	// ct::DeviceTypeToEnum<ct_Device>::value：根据设备类型（如 CpuDevice 或 GpuDevice），自动选择张量的存储设备。
	// {wfc_basis->nmaxgr}：指定张量的形状，这里是一维，长度为 nmaxgr，即最大梯度分量数。
	// std::move用于将临时创建的张量对象的所有权转移给 porter，提高效率，避免不必要的拷贝。
	// porter 是后续波函数梯度计算的临时存储空间，支持在不同设备和数据类型下高效运算。
	// auto：让编译器会根据右侧表达式的类型，自动推断 porter 的实际类型，无需你手动写出完整类型名,
	// 这在模板和复杂类型场景下非常方便，能让代码更简洁易读
	auto porter = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<ct_Device>::value, {wfc_basis->nmaxgr}));
	// 创建了一个张量映射对象gcar，用于在后续波函数梯度计算中，提供G空间（倒空间）格矢的坐标数据，
	// 并根据需要将其转移到指定计算设备（如CPU或GPU）。
	// ct::TensorMap(...)：将已有的原始数据（这里是 wfc_basis->gcar，即G矢量坐标数组）映射为张量对象，方便后续高效运算。
	// wfc_basis->gcar[0][0]：指向G矢量坐标数据的首地址。
	// ct::DataType::DT_DOUBLE：指定数据类型为 double。
	// ct::DeviceType::CpuDevice：初始设备为CPU。
	// {wfc_basis->nks * wfc_basis->npwk_max, 3}：张量的形状，表示有 nks * npwk_max 个G矢量，每个矢量有3个分量（x, y, z）。
	// .to_device<ct_Device>()：将张量数据转移到目标设备（如CPU或GPU），ct_Device 是根据模板参数自动推断的设备类型。
	auto gcar = ct::TensorMap(
		&wfc_basis->gcar[0][0], ct::DataType::DT_DOUBLE, ct::DeviceType::CpuDevice, {wfc_basis->nks * wfc_basis->npwk_max, 3}).to_device<ct_Device>();
	// 创建了一个张量映射对象 kvec_c，用于在后续波函数梯度计算中，提供所有 k 点的坐标数据，
	// 并根据需要将其转移到指定计算设备（如 CPU 或 GPU）。
	// {wfc_basis->nks, 3}：张量的形状，表示有 nks 个 k 点，每个 k 点有 3 个分量（x, y, z）。
	auto kvec_c = ct::TensorMap(
		&wfc_basis->kvec_c[0][0],ct::DataType::DT_DOUBLE, ct::DeviceType::CpuDevice, {wfc_basis->nks, 3}).to_device<ct_Device>();
	
	// 创建了一个用于波函数梯度计算的算子对象 xc_functional_grad_wfc_solver
	// hamilt::xc_functional_grad_wfc_op<T, Device>()：这是一个工厂函数或类模板实例化，根据数据类型和设备类型，
	// 返回一个适合当前环境的波函数梯度计算算子。
	auto xc_functional_grad_wfc_solver 
		= hamilt::xc_functional_grad_wfc_op<T, Device>();

	// 调用函数xc_functional_grad_wfc_solver()计算波函数在三个方向（x, y, z）的梯度
	// ipol：方向索引（0,1,2），分别对应 x、y、z。
	for(int ipol=0; ipol<3; ipol++) {
		xc_functional_grad_wfc_solver(
            ik, ipol, npw_k, wfc_basis->npwk_max, // Integers
			tpiba,	// Double
            gcar.template data<Real>(),   // Array of Real
            kvec_c.template data<Real>(), // Array of double
			rhog, porter.data<T>());    // Array of std::complex<double>

		// bring the gdr from G --> R
		Device * ctx = nullptr;
		wfc_basis->recip_to_real(ctx, porter.data<T>(), porter.data<T>(), ik);

		xc_functional_grad_wfc_solver(
            ipol, wfc_basis->nrxx,	// Integers
			porter.data<T>(), grad);	// Array of std::complex<double>
    }
}


void XC_Functional::grad_rho(const std::complex<double>* rhog,
                             ModuleBase::Vector3<double>* gdr,
                             const ModulePW::PW_Basis* rho_basis,
                             const double tpiba)
{
	std::complex<double> *gdrtmp = new std::complex<double>[rho_basis->nmaxgr];

	// the formula is : rho(r)^prime = \int iG * rho(G)e^{iGr} dG
	// 依次处理x、y、z三个方向，最终每个实空间网格点都得到一个三维梯度向量。
	for(int i = 0 ; i < 3 ; ++i)
	{
		// calculate the charge density gradient in reciprocal space.
#ifdef _OPENMP
// OpenMP 的并行指令，用于让后面的 for 循环在多核CPU上并行执行
// static 表示每个线程分配到的循环迭代次数是固定的、均匀的。1024 表示每个线程一次分配1024个循环迭代
// 你不需要手动写多线程的创建、分配、同步等复杂代码。
// 只要用支持OpenMP的编译器（如g++/clang++加上 -fopenmp），
// 编译器看到这句指令后，会自动把后面的for循环分成多份，分配给多个CPU核心并行执行
#pragma omp parallel for schedule(static, 1024)
#endif
		for(int ig=0; ig<rho_basis->npw; ig++) {
			// 微分（梯度）的傅里叶变换就是在倒空间乘以 i*G
			// 在倒空间计算每个点的梯度分量：gdrtmp[ig] = i * rhog[ig] * G_i
			gdrtmp[ig] = ModuleBase::IMAG_UNIT * rhog[ig] * rho_basis->gcar[ig][i];
}

		// bring the gdr from G --> R
		// 逆傅里叶变换回实空间
		rho_basis->recip2real(gdrtmp, gdrtmp);

		// remember to multily 2pi/a0, which belongs to G vectors.
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
		for(int ir=0; ir<rho_basis->nrxx; ir++) {
			// tpiba为2pi/a0（a0是晶格常数）
			// 倒格矢向量rho_basis->gcar是以 2pi/a0 为单位存储的，通常是无量纲的数字，所以最后要乘以 tpiba
			gdr[ir][i] = gdrtmp[ir].real() * tpiba;
}
	}

	delete[] gdrtmp;
	return;
}


void XC_Functional::grad_dot(const ModuleBase::Vector3<double>* h, double* dh, const ModulePW::PW_Basis* rho_basis, const double tpiba)
{
	std::complex<double> *aux = new std::complex<double>[rho_basis->nmaxgr];
	std::complex<double> *gaux = new std::complex<double>[rho_basis->npw];

	for(int i = 0 ; i < 3 ; ++i)
	{
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
		for(int ir = 0; ir < rho_basis->nrxx; ++ir) {
			aux[ir] = std::complex<double>( h[ir][i], 0.0);
}

		// bring to G space.
		rho_basis->real2recip(aux,aux);
		if (i == 0)
		{
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
			for(int ig = 0; ig < rho_basis->npw; ++ig) {
				gaux[ig] =  ModuleBase::IMAG_UNIT * aux[ig] * rho_basis->gcar[ig][i];
}
		}
		else
		{
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
			for(int ig = 0; ig < rho_basis->npw; ++ig) {
				gaux[ig] +=  ModuleBase::IMAG_UNIT * aux[ig] * rho_basis->gcar[ig][i];
}
		}
	}

	// bring back to R space
	rho_basis->recip2real(gaux,aux);

#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
	for(int ir=0; ir<rho_basis->nrxx; ir++) {
		dh[ir] = aux[ir].real() * tpiba;
}
	
	delete[] aux;	
	delete[] gaux;
	return;
}

void XC_Functional::noncolin_rho(double *rhoout1, double *rhoout2, double *neg,
	const double*const*const rho, const int nrxx, const double* ux_, const bool lsign_)
{
	//this function diagonalizes the spin density matrix and gives as output the
	//spin up and spin down components of the charge.
	//If lsign is true up and dw are with respect to the fixed quantization axis 
	//ux, otherwise rho + |m| is always rhoup and rho-|m| is always rhodw.
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
	for(int ir = 0;ir<nrxx;ir++)
	{
		neg[ir] = 1.0;
	}
	if(lsign_)
	{
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
		for(int ir = 0;ir<nrxx;ir++)
		{
			if(rho[1][ir]*ux_[0] + rho[2][ir]*ux_[1] + rho[3][ir]*ux_[2]>0) { neg[ir] = 1.0;
			} else { neg[ir] = -1.0;
}
		}
	}
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(int ir = 0;ir<nrxx;ir++)
	{
		double amag = sqrt(pow(rho[1][ir],2)+pow(rho[2][ir],2)+pow(rho[3][ir],2));
		rhoout1[ir] = 0.5 * (rho[0][ir] + neg[ir] * amag);
		rhoout2[ir] = 0.5 * (rho[0][ir] - neg[ir] * amag);
	}
	return;
}

template void XC_Functional::grad_wfc<std::complex<double>, base_device::DEVICE_CPU, double>(
    const int ik,
    const double tpiba,
    const ModulePW::PW_Basis_K* wfc_basis,
    const std::complex<double>* rhog,
    std::complex<double>* grad);
#if __CUDA || __ROCM
template void XC_Functional::grad_wfc<std::complex<double>, base_device::DEVICE_GPU, double>(
    const int ik,
    const double tpiba,
    const ModulePW::PW_Basis_K* wfc_basis,
    const std::complex<double>* rhog,
    std::complex<double>* grad);
#endif // __CUDA || __ROCM