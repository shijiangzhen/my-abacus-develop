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
					v(0, ir) += ModuleBase::e2 * v1xc;
					// cout << "v    " << v(0, ir) << endl;
					
					// h contains
					// D(rho*Exc) / D(|grad rho|) * (grad rho) / |grad rho|
					h1[ir] = ModuleBase::e2 * v2xc * gdr1[ir];
					
					local_vtxcgc += ModuleBase::e2* v1xc * ( rhotmp1[ir] - chr->rho_core[ir] );
					local_etxcgc += ModuleBase::e2* sxc  * segno;
				}
			} // end arho > epsr
		}
	}// end nspin0 == 1
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
				double rh = rhotmp1[ir] + rhotmp2[ir];
				grho2a = gdr1[ir].norm2();
				grho2b = gdr2[ir].norm2();
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
						double zeta = ( rhotmp1[ir] - rhotmp2[ir] ) / rh;
						if(PARAM.inp.nspin==4&&(PARAM.globalv.domag||PARAM.globalv.domag_z)) { zeta = fabs(zeta) * neg[ir];
}
						const double grh2 = (gdr1[ir]+gdr2[ir]).norm2();
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
							local_stress_gga [ind] += tt1[l] * tt1[m] * ModuleBase::e2 * v2xup + 
									tt2[l] * tt2[m] * ModuleBase::e2 * v2xdw;
							//    correlation
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
					v(0,ir) = v(0,ir) + ModuleBase::e2 * ( v1xup + v1cup );
					v(1,ir) = v(1,ir) + ModuleBase::e2 * ( v1xdw + v1cdw );
				
					// h contains D(rho*Exc)/D(|grad rho|) * (grad rho) / |grad rho|
					h1[ir] = ModuleBase::e2 * ( ( v2xup + v2cup ) * gdr1[ir] + v2cud * gdr2[ir] );
					h2[ir] = ModuleBase::e2 * ( ( v2xdw + v2cdw ) * gdr2[ir] + v2cud * gdr1[ir] );

					local_vtxcgc = local_vtxcgc + ModuleBase::e2 * ( v1xup + v1cup ) * ( rhotmp1[ir] - chr->rho_core[ir] * fac );
					local_vtxcgc = local_vtxcgc + ModuleBase::e2 * ( v1xdw + v1cdw ) * ( rhotmp2[ir] - chr->rho_core[ir] * fac );
					local_etxcgc = local_etxcgc + ModuleBase::e2 * ( sx + sc );
				}
			}
		}// end ir

	}
#ifdef _OPENMP
	#pragma omp critical(xc_functional_gradcorr_reduce)
	{
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
		else
		{
			vtxcgc += local_vtxcgc;
			etxcgc += local_etxcgc;
		}
	}
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
			rhotmp1[ir] -= fac * chr->rho_core[ir];
		}
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
			if(is==0) {XC_Functional::grad_dot(h1,dh,rhopw,ucell->tpiba);
}
			if(is==1) {XC_Functional::grad_dot(h2,dh,rhopw,ucell->tpiba);
}
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
			for(int ir=0; ir<rhopw->nrxx; ir++) {
				v(is, ir) -= dh[ir];
}
		
			double sum = 0.0;
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
			vtxcgc -= sum;
		}
		
		delete[] dh;

		vtxc += vtxcgc;
		etxc += etxcgc;

		if(PARAM.inp.nspin == 4 && (PARAM.globalv.domag||PARAM.globalv.domag_z))
		{
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
			for(int is=0;is<PARAM.inp.nspin;is++)
			{
				for(int ir=0;ir<rhopw->nrxx;ir++)
				{
					if(is<nspin0) { vgg[is][ir] = v(is,ir);
}
					v(is,ir) = vsave[is][ir];
				}
			}
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
			for(int ir=0;ir<rhopw->nrxx;ir++)
			{
				v(0,ir) += 0.5 * (vgg[0][ir] + vgg[1][ir]);
				double amag = sqrt(pow(chr->rho[1][ir],2)+pow(chr->rho[2][ir],2)+pow(chr->rho[3][ir],2));
				if(amag>1e-12)
				{
					for(int i=1;i<4;i++) {
						v(i,ir)+= neg[ir] * 0.5 *(vgg[0][ir]-vgg[1][ir])*chr->rho[i][ir]/amag;
}
				}
			}
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

template <typename T, typename Device, typename Real>
void XC_Functional::grad_wfc(
    const int ik,
    const Real tpiba,
    const ModulePW::PW_Basis_K* wfc_basis,
	const T* rhog,
    T* grad)
{
    using ct_Device = typename ct::PsiToContainer<Device>::type;
	const int npw_k = wfc_basis->npwk[ik];
	
	auto porter = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<ct_Device>::value, {wfc_basis->nmaxgr}));
	auto gcar = ct::TensorMap(
		&wfc_basis->gcar[0][0], ct::DataType::DT_DOUBLE, ct::DeviceType::CpuDevice, {wfc_basis->nks * wfc_basis->npwk_max, 3}).to_device<ct_Device>();
	auto kvec_c = ct::TensorMap(
		&wfc_basis->kvec_c[0][0],ct::DataType::DT_DOUBLE, ct::DeviceType::CpuDevice, {wfc_basis->nks, 3}).to_device<ct_Device>();
	
	auto xc_functional_grad_wfc_solver 
		= hamilt::xc_functional_grad_wfc_op<T, Device>();

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