#include "pw_basis.h"
#include "source_base/global_function.h"
#include "source_base/timer.h"
#include "typeinfo"
namespace ModulePW
{
/**
 * @brief gather planes and scatter sticks
 * @param in: (nplane,fftny,fftnx)
 * @param out: (nz,nst)
 * @note in and out should be in different places
 * @note in[] will be changed
 */
template <typename T>
void PW_Basis::gatherp_scatters(std::complex<T>* in, std::complex<T>* out) const
{
    //ModuleBase::timer::tick(this->classname, "gatherp_scatters");
    
    if(this->poolnproc == 1) //In this case nst=nstot, nz = nplane, 
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int is = 0 ; is < this->nst ; ++is)
        {
            int ixy = this->istot2ixy[is];
            //int ixy = (ixy / fftny)*ny + ixy % fftny;
            std::complex<T> *outp = &out[is*nz];
            std::complex<T> *inp = &in[ixy*nz];
            for(int iz = 0 ; iz < this->nz ; ++iz)
            {
                outp[iz] = inp[iz];
            }
        }
        //ModuleBase::timer::tick(this->classname, "gatherp_scatters");
        return;
    }
#ifdef __MPI
    //change (nplane fftnxy) to (nplane,nstot)
    // Hence, we can send them at one time.
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int istot = 0;istot < nstot; ++istot)
	{
		int ixy = this->istot2ixy[istot];
        //int ixy = (ixy / fftny)*ny + ixy % fftny;
        std::complex<T> *outp = &out[istot*nplane];
        std::complex<T> *inp = &in[ixy*nplane];
		for (int iz = 0; iz < nplane; ++iz)
		{
			outp[iz] = inp[iz];
		}
	}

    //exchange data
    //(nplane,nstot) to (numz[ip],ns, poolnproc)
	if(typeid(T) == typeid(double))
	{
		MPI_Alltoallv(out, numr, startr, MPI_DOUBLE_COMPLEX, in, numg, startg, MPI_DOUBLE_COMPLEX, this->pool_world);
	}
	else if(typeid(T) == typeid(float))
	{
		MPI_Alltoallv(out, numr, startr, MPI_COMPLEX, in, numg, startg, MPI_COMPLEX, this->pool_world);
	}

    // change (nz,ns) to (numz[ip],ns, poolnproc)
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int ip = 0; ip < this->poolnproc ;++ip)
	{
		for (int is = 0; is < this->nst; ++is)
		{
            int nzip = this->numz[ip];
            std::complex<T> *outp0 = &out[startz[ip]];
            std::complex<T> *inp0 = &in[startg[ip]];
            std::complex<T> *outp = &outp0[is * nz];
            std::complex<T> *inp = &inp0[is * nzip ];
			for (int izip = 0; izip < nzip; ++izip)
			{
				outp[izip] = inp[izip];
			}
		}
	}
#endif
    //ModuleBase::timer::tick(this->classname, "gatherp_scatters");
    return;
}

/**
 * @brief gather sticks and scatter planes
 * @param in: (nz,nst)
 * @param out: (nplane,fftny,fftnx)
 * @note in and out should be in different places
 * @note in[] will be changed
 */
template <typename T>
void PW_Basis::gathers_scatterp(std::complex<T>* in, std::complex<T>* out) const
{
    // ModuleBase::timer::tick(this->classname, "gathers_scatterp");
    // 单进程，在这种情况下，nst=nstot，即所有的 stick 都在同一个进程上
    if(this->poolnproc == 1) //In this case nrxx=fftnx*fftny*nz, nst = nstot, 
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096/sizeof(T))
#endif
        // 将输出数组 out 全部清零
        // nrxx 是输出数组的总长度（通常等于 FFT 网格的总点数，如 fftnx * fftny * nz）。
        for(int i = 0; i < this->nrxx; ++i)
        {
            out[i] = std::complex<T>(0, 0);
        }

#ifdef _OPENMP
#pragma omp parallel for
#endif
 /*
 istot2ixy是Stick（柱）布局与平面（xy 网格）布局之间的映射表。
 Stick（柱）编号 is：在 Stick-Z（柱-纵向）布局下，所有 stick（即每一对（G_x,G_y））被编号为is=0,1,2,...,nst-1。
 平面网格编号 ixy：在 xy 平面布局下，所有(ix,iy)网格点被线性编号为ixy=ix*fftny + iy（或类似方式）。
 istot2ixy[is] 的作用：告诉你第 is 个 stick（柱）在 xy 网格中的编号是 ixy，即 stick 的数据应该搬运到 xy 网格的哪个位置。

 假设：N_x=2, N_y=2, N_z=3,即 xy 平面有 2x2=4 个stick（柱），每个 stick 上有 3 个 z 方向网格点。
 stick编号is从0到3，xy网格编号ixy也从0到3，映射关系如下：
    is | (ix,iy) | ixy=ix*N_y+iy
    -----------------------
    0  |  (0,0)  |  0
    1  |  (0,1)  |  1       
    2  |  (1,0)  |  2
    3  |  (1,1)  |  3
 如果 stick 的顺序和 xy 网格顺序一致，则 istot2ixy[is] = is。
 如果 stick 顺序与 xy 网格顺序不同，比如 stick 编号是按能量截断球内的实际分布顺序排列的，则 istot2ixy 就是一个乱序的映射表，
 例如：
    is | (ix,iy) | ixy
    -----------------------
    0  |  (1,0)  |  2
    1  |  (0,1)  |  1       
    2  |  (1,1)  |  3
    3  |  (0,0)  |  0
 这样，istot2ixy[0]=2, istot2ixy[1]=1, istot2ixy[2]=3, istot2ixy[3]=0。
 
 istot2ixy保证了 stick（柱）数据能正确地重排到 xy 平面，为后续 FFT 运算做准备。
 其内容由 stick 的实际排列和 xy 网格的编号方式共同决定。
 */
        // 这个循环将每个 stick 的 z 方向数据搬运到对应的 xy 网格位置，
        // 实现了 stick（柱）布局到 xy 平面网格布局的数据重排，
        // 把 z 方向逆FFT后的数据放到 FFTW 需要的 xy 平面顺序，为后续的 xy 方向逆FFT做准备。
        // 外层循环遍历所有 stick（柱），is 是 stick 的编号。
        for(int is = 0 ; is < this->nst ; ++is)
        {
            int ixy = istot2ixy[is];
            //int ixy = (ixy / fftny)*ny + ixy % fftny;
            // outp指向输出数组中第ixy个xy平面网格点的数据起始位置
            std::complex<T> *outp = &out[ixy*nz];
            // inp指向输入数组中第is个stick的数据起始位置
            std::complex<T> *inp = &in[is*nz];
            // 该循环当前 stick 上的所有 z 方向数据（inp[iz]）搬运到输出数组的对应位置（outp[iz]）。
            for(int iz = 0 ; iz < this->nz ; ++iz)
            {
                outp[iz] = inp[iz];
            }
        }
        // ModuleBase::timer::tick(this->classname, "gathers_scatterp");
        return;
    }

 // MPI并行,在这种情况下，nst < nstot，即所有的 stick 分布在多个进程上，每个进程只负责其中一部分 stick
#ifdef __MPI
    // change (nz,ns) to (numz[ip],ns, poolnproc)
    // Hence, we can send them at one time. 
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
 /*
 poolnproc：池内进程数（MPI ranks 数）。
 nz：每根 stick 在 z 方向的完整长度（全 z 网格点数）。
 nst：当前进程上的 stick 数量。
 numz[ip]：第 ip 个进程负责的 z 点数（z 方向切分后的块长度），这里记为 nzip。
 startz[ip]：在输入数组 in 中，第 ip 个进程对应的 z 子区间的起始偏移（以元素为单位的线性偏移）。
 startg[ip]：在输出数组 out 中，给第 ip 个进程预留的打包区的起始偏移。

 2) 这段拷贝在做什么（核心思想）
    输入 in 的布局（Stick-Z）
    对每根 stick（编号 is），都有一段连续的 z 数据：

    stick is 的起始地址：&in[ is * nz ]
    它包含 nz 个点：in[is*nz + 0 ... is*nz + nz-1]
    但在 MPI 模式下，z 方向通常会被分块。对某个目标进程 ip，我们只需要 z 的一段长度 nzip，从该段的起点开始取：

    这段 z 子区间的起点通过 startz[ip] 体现：inp0 = &in[startz[ip]]
    然后再跳到 stick is：inp = &inp0[is * nz]
    因此，inp[0 ... nzip-1] 对应的是：

    stick is 上，属于进程 ip 的那 nzip 个 z 点数据。

    输出 out 的布局（按 ip 打包、每 stick 只存 nzip）
    输出是按 ip 分区打包的：对每个 ip，连续存放所有 stick 的 z 子块，每根 stick 只占 nzip 个元素：

    outp0 = &out[startg[ip]]：找到 ip 的打包区起点
    stick is 的输出起点：outp = &outp0[ is * nzip ]
    拷贝 nzip 个元素：outp[izip] = inp[izip]

 3) 一句话总结其作用
    这段双重循环在做 “按进程 ip 把进程负责的每根 stick 的 z 子段打包到连续缓冲区 out” 的工作，
    典型用途是为后续的 MPI 通信（如 Alltoall/Alltoallv）或者后续阶段需要的不同布局（例如xyFFT 所需的分布/转置）做准备。

 注意，inp0只解决了 “z 子区间从哪里开始”，但它还没有指定 “是哪一根 stick（is）”。
 inp0[is * nz] 这一步是在 stick 维度上再做一次跳转，用来定位到 第 is 根 stick 上、同一段 z 子区间的起点。

    关键点：in 的内存布局是按 stick 分块的
    在这段 MPI 分支里，in（输入）可以按一维理解为：

    stick 0：in[0 ... nz-1]
    stick 1：in[nz ... 2*nz-1]
    stick 2：in[2*nz ... 3*nz-1]
    …
    也就是 stick 索引是最外层块，块内是 z 连续。

    *inp = &inp0[is * nz ];
    这在 inp0 的基础上，再跳到 第 is 根 stick 对应的块里。
    因为每根 stick 的块长度是 nz，所以第 is 根 stick 相对 stick0 的偏移就是 is * nz。
    所以 inp 指向的是：stick=is 且 z=startz[ip] 的起点。

    假设：

    nz = 8（每根 stick 有 8 个 z 点）
    nst = 3（3 根 stick）
    对某个 ip：startz[ip] = 3，nzip = 2（该进程要 z=3,4 两个点）
    那么 in 的线性布局是：

    stick0：in[0..7]
    stick1：in[8..15]
    stick2：in[16..23]
    此时：

    inp0 = &in[3] → stick0 的 z=3 起点
    若 is = 2：inp = inp0 + 2*nz = &in[3 + 16] = &in[19]
    这正是 stick2 的 z=3 起点，接下来拷贝 nzip=2 个元素就是 z=3、4。

    为什么不能只用 inp0？
    因为你要对 每根 stick 都拷贝它在该 ip 对应的那段 z 子区间。
    inp0 固定在 stick0 上，不加 is * nz 就永远只能读 stick0 的数据。
 */
    // 每个进程都要执行这一整个大的for循环，每个进程负责的stick不同，执行完后每个进程都有自己独立的out数组，且内容完全不同。
    for (int ip = 0; ip < this->poolnproc ;++ip)
	{
        // this->nst 不是“所有的 Stick 总数”，而是分配给当前 MPI 进程的 Stick 数量,
        // 也就是说每个进程只处理自己负责的那部分 Stick。
		for (int is = 0; is < this->nst; ++is)
		{
            int nzip = this->numz[ip];
            std::complex<T> *outp0 = &out[startg[ip]];
            std::complex<T> *inp0 = &in[startz[ip]];
            std::complex<T> *outp = &outp0[is * nzip];
            std::complex<T> *inp = &inp0[is * nz ];
			for (int izip = 0; izip < nzip; ++izip)
			{
                // outp是指针，实际改变的是out数组的内容
				outp[izip] = inp[izip];
			}
		}
	}
/*
这里有一个非常具体的例子来说明 for 循环结束后，每个进程的 out 数组里到底存了什么。

1. 设定场景 

假设我们需要进行 FFT，系统配置如下：

进程数 (MPI Ranks):2 (Rank 0 和 Rank 1)。
总 Stick 数 (Global Sticks):4 根 (S0, S1, S2, S3)。
分配方式:
  Rank 0 持有: S0, S1 (nst=2)。
  Rank 1 持有: S2, S3 (nst=2)。
Z 方向网格总点数 (nz):4 (索引 0, 1, 2, 3)。
Z 方向切分 (Plane Distribution目标):
  Rank 0 将负责: Z = 0, 1 (上半截)。
  Rank 1 将负责: Z = 2, 3 (下半截)。

---

2. 初始状态 (Input in)

每个进程的内存中，输入的 in 数组按 Stick 排列（stick-major）：

Rank 0 的 in 数组 (只包含 S0, S1 的全 Z 数据):
| 索引 |           内容             |       说明        |
| ---------------------------------------------------- |
| 0-3 | S0_z0, S0_z1, S0_z2, S0_z3 | Stick 0 的全部数据 |
| 4-7 | S1_z0, S1_z1, S1_z2, S1_z3 | Stick 1 的全部数据 |

Rank 1 的 in 数组 (只包含 S2, S3 的全 Z 数据):*
| 索引 |           内容             |       说明        |
| ---------------------------------------------------- |
| 0-3 | S2_z0, S2_z1, S2_z2, S2_z3 | Stick 2 的全部数据 |
| 4-7 | S3_z0, S3_z1, S3_z2, S3_z3 | Stick 3 的全部数据 |

---
 3. 执行 For 循环 (Local Packing)

每个进程都要执行上面的一整个for循环。
循环逻辑：遍历每个目标进程 (ip)，再遍历自己持有的每根 Stick (is)，切下属于那个进程的 Z 片段。

Rank 0 的操作过程：

1.  当 ip = 0 (打包给自己的):
      Rank 0 需要 Z=0,1。
      对 S0: 切下 S0_z0, S0_z1。
      对 S1: 切下 S1_z0, S1_z1。
      写入 out 的 Buffer 0 区。

2.  当 ip = 1 (打包给 Rank 1 的):
      Rank 1 需要 Z=2,3。
      对 S0: 切下 S0_z2, S0_z3。
      对 S1: 切下 S1_z2, S1_z3。
      写入 out 的 Buffer 1 区。

Rank 1 的操作过程：

1.  当 ip = 0 (打包给 Rank 0 的):
      Rank 0 需要 Z=0,1。
      对 S2: 切下 S2_z0, S2_z1。
      对 S3: 切下 S3_z0, S3_z1。
      写入 out 的 Buffer 0 区。
2.  当 ip = 1 (打包给自己的):
      Rank 1 需要 Z=2,3。
      对 S2: 切下 S2_z2, S2_z3。
      对 S3: 切下 S3_z2, S3_z3。
      写入 out 的 Buffer 1 区。

---
4. 最终状态 (Output out after For Loop)

这就是循环结束后，各个进程私有内存里 out 数组的快照：

Rank 0 的 out 数组 (准备发件)

| 分区 (按目标ip) |        数据内容 (Values)     |                    物理含义                    |
| ----------------------------------------------------------------------------------------------|
|   发给 Rank 0  |  S0_z0, S0_z1, S1_z0, S1_z1  | 我持有的Stick(0,1)中，属于Rank 0负责的层(z0,z1) |
|   发给 Rank 1  |  S0_z2, S0_z3, S1_z2, S1_z3  | 我持有的Stick(0,1)中，属于Rank 1负责的层(z2,z3) |

Rank 1 的 out 数组 (准备发件)

| 分区 (按目标ip) |        数据内容 (Values)     |                    物理含义                    |
| ----------------------------------------------------------------------------------------------|
|   发给 Rank 0  |  S2_z0, S2_z1, S3_z0, S3_z1  | 我持有的Stick(2,3)中，属于Rank 0负责的层(z0,z1) |
|   发给 Rank 1  |  S2_z2, S2_z3, S3_z2, S3_z3  | 我持有的Stick(2,3)中，属于Rank 1负责的层(z2,z3) |

---
总结

Rank 0 的 out 只有 S0, S1 的切片。
Rank 1 的 out 只有 S2, S3 的切片。
它们的内容完全不同，就像两个不同的快递分拣员，各自只分拣自己手头那堆货物。
下一步 (MPI_Alltoallv)：Rank 0 会收到Rank 1的第一部分 (S2_z0...)，加上自己留下的 (S0_z0...)，Rank 0 就凑齐了所有 Stick 在 Z=0,1 层的数据。
*/

	//exchange data
    //(numz[ip],ns, poolnproc) to (nplane,nstot)
    
/*
调用MPI_Alltoallv函数（全对全矢量通信），这是MPI中最复杂的通信模式之一。它的作用是让所有进程互相交换不同大小的数据块。
它将每个进程本地打包好的、分属于不同进程的数据块，通过网络分发给对应的目标进程，
从而完成从 “按 Stick 分布” 到 “按 Z平面切片分布” 的物理数据转移。
1.为什么需要这一步？
发送前（Out）：在前面的双重循环中，每个进程已经把属于其他进程的 z-stick 数据片段打包到了 out 数组中（按目标进程 ID 排列）。
通信中：每个进程把自己持有的、属于其他人的数据“发”出去，同时从其他人那里“收”回属于自己的数据。
接收后（In）：通信结束后，数据被写入 in 数组。此时 in 中的数据不再是自己原来负责的那部分 stick，
             而是所有其他进程发给自己的数据（这些数据拼起来就是自己负责的 xy 平面的部分 z 切片）。
2. 参数解析
发送端（Send）：
out: 发送缓冲区，里面存的是打包好的数据。
numg: 数组，numg[ip] 表示要发给第 ip 号进程的数据元素个数。
startg: 数组，startg[ip] 表示发给第 ip 号进程的数据在 out 中的起始位置（偏移量）。
接收端（Recv）：
in: 接收缓冲区，用于存放收到的数据。注意这里复用了输入指针 in 作为接收缓冲区（因为发送完后旧的 in 数据就没用了）。
numr: 数组，numr[ip] 表示期望从第 ip 号进程接收的数据元素个数。
startr: 数组，startr[ip] 表示从第 ip 号进程收到的数据应该存放在 in 数组的哪个起始位置。

结合上面那个具体的例子（Rank 0 持有 S0,S1；Rank 1 持有 S2,S3），我们继续详解 MPI_Alltoallv 的调用：

1. 通信前的状态（准备发件）
在上一部 for 循环结束后，每个进程的 out 数组（发送缓冲区）已经填好了“我要发给谁什么数据”。

Rank 0 的 out（手里拿着 S0, S1）:
给 Rank 0 (自己): [S0_z0, S0_z1, S1_z0, S1_z1] (数据量 4)
给 Rank 1 (老王): [S0_z2, S0_z3, S1_z2, S1_z3] (数据量 4)
Rank 1 的 out（手里拿着 S2, S3）:
给 Rank 0 (老李): [S2_z0, S2_z1, S3_z0, S3_z1] (数据量 4)
给 Rank 1 (自己): [S2_z2, S2_z3, S3_z2, S3_z3] (数据量 4)
注意：此时 Rank 0 只有 Stick 0/1 的切片，Rank 1 只有 Stick 2/3 的切片。

2. MPI_Alltoallv 执行过程（快递交换）
这行代码一下令，MPI 就像一个快递中心，开始在后台疯狂搬运数据：

Rank 0 发货:
把“给 Rank 1”的那块数据 [S0_z2, S0_z3, S1_z2, S1_z3] 发送出去。
（“给 Rank 0”的那块留给自己）。
Rank 1 发货:
把“给 Rank 0”的那块数据 [S2_z0, S2_z1, S3_z0, S3_z1] 发送出去。
（“给 Rank 1”的那块留给自己）。
关键点： 所有人都同时是寄件人和收件人。

3. 通信后的状态（收件拆包）
代码执行完后，查看接收缓冲区 in。注意，此时 in 里的旧数据（完整的 Stick）已经被覆盖，变成了新收到的数据。

Rank 0 的 in（接收区）:
Rank 0 负责的是 Z=0,1 (上半平面)。它收到了所有人在 Z=0,1 这一层的数据。

来自 Rank 0 (自己留下的): [S0_z0, S0_z1, S1_z0, S1_z1]
来自 Rank 1 (新收到的): [S2_z0, S2_z1, S3_z0, S3_z1]
合体结果: [S0_z0..z1, S1_z0..z1, S2_z0..z1, S3_z0..z1]
物理意义: Rank 0 现在拥有了 所有 Stick (S0到S3) 在 Z=0,1 层切面上的数据。这是一个完整的“上半部平面”。
Rank 1 的 in（接收区）:
Rank 1 负责的是 Z=2,3 (下半平面)。

来自 Rank 0 (新收到的): [S0_z2, S0_z3, S1_z2, S1_z3]
来自 Rank 1 (自己留下的): [S2_z2, S2_z3, S3_z2, S3_z3]
合体结果: [S0_z2..z3, S1_z2..z3, S2_z2..z3, S3_z2..z3]
物理意义: Rank 1 现在拥有了 所有 Stick (S0到S3) 在 Z=2,3 层切面上的数据。这是一个完整的“下半部平面”。

4. 总结
这一行 MPI_Alltoallv 实现了物理上的 数据转置（Transpose）：

输入: 每个进程拥有 部分 Stick 的 全部 Z。
输出: 每个进程拥有 全部 Stick 的 部分 Z。
这就是为什么 FFT 算法能够从“Z方向变换”（需要 Stick 连续）过渡到“XY平面变换”（需要 Z 平面连续）的关键步骤。
*/
	if(typeid(T) == typeid(double))
	{
		MPI_Alltoallv(out, numg, startg, MPI_DOUBLE_COMPLEX, in, numr, startr, MPI_DOUBLE_COMPLEX, this->pool_world);
	}
	else if(typeid(T) == typeid(float))
	{
		MPI_Alltoallv(out, numg, startg, MPI_COMPLEX, in, numr, startr, MPI_COMPLEX, this->pool_world);
	}

#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096/sizeof(T))
#endif
    // 作用： 把整个输出 out 数组清空（置为0）。
    // 原因： out 数组是最终的平面 FFT 网格缓冲区，大小是 nrxx（即 nx * ny * nplane），
    // 由于 Stick 可能并没有填满所有的 XY 网格（比如只占了球型的截断区域，其他角落是空的），
    // 所以必须先清零背景，确保那些没有 Stick 的位置是 0。
    for(int i = 0; i < this->nrxx; ++i)
    {
        out[i] = std::complex<T>(0, 0);
    }
    //change (nplane,nstot) to (nplane fftnxy)
#ifdef _OPENMP
#pragma omp parallel for
#endif
/*
下面的for循环将从所有进程收到的、紧凑排列的 Stick 数据，“散射”到物理的 xy 平面网格上的正确位置，
为接下来的二维 FFT（在 xy 平面上进行）做最后准备。

1. 上下文状态（MPI 通信后）

in 数组现在的状态：
经过 MPI_Alltoallv 后，in 数组里装的数据已经变了。它现在保存的是当前进程负责的 Z 平面（nplane 层） 上，
来自全系统所有 Stick（nstot 根） 的数据。
布局：它是紧凑排列的。先是 Global Stick 0 的 nplane 个点，然后是 Global Stick 1 的 nplane 个点……
    直到 Global Stick nstot-1。
形状：逻辑上可以看作 (nstot, nplane) 的二维数组。

out 数组现在的状态：在此代码块之前，out 已经被全部清零。
含义：out 代表了实际的物理 FFT 网格（xy 平面）。
形状：大小为 nrxx，即 nx * ny * nplane


这段代码位于 MPI_Alltoallv 通信之后，执行的是 “本地解包与重排”（Local Unpacking / Scatter） 操作。

它的作用是将从所有进程收到的、紧凑排列的 Stick 数据，“散射”（Scatter） 到物理的 xy 平面网格上的正确位置，为接下来的二维 FFT（在 xy 平面上进行）做最后准备。

详细解释
1. 上下文状态（MPI 通信后）

in 数组现在的状态：
经过 MPI_Alltoallv 后，in 数组里装的数据已经变了。它现在保存的是当前进程负责的 Z 平面（nplane 层） 上，来自全系统所有 Stick（nstot 根） 的数据。

布局：它是紧凑排列的。先是 Global Stick 0 的 nplane 个点，然后是 Global Stick 1 的 nplane 个点……直到 Global Stick nstot-1。
形状：逻辑上可以看作 (nstot, nplane) 的二维数组。
out 数组现在的状态：
在此代码块之前，out 已经被全部清零。

含义：out 代表了实际的物理 FFT 网格（xy 平面）。
形状：大小为 nrxx，即 nx * ny * nplane。

2. 形象比喻

in (收纳箱)：像是刚从快递点取回来的大包裹。里面的 Stick 数据是一根挨着一根紧紧捆在一起的，为了节省空间，没有任何空隙。
            虽然你知道每根 Stick 代表什么（通过编号 istot），但它们并没有摆在原本的空间位置上。
out (展示架/FFT网格)：是一个巨大的棋盘（xy网格）。大部分格子是空的（被置为 0，因为那里没有 G 向量）。
本段代码 (摆放员)：
拿起第 0 根 Stick，查表发现它属于 (0,0) 格子，于是把它插到 (0,0) 去。
拿起第 1 根 Stick，查表发现它属于 (0,1) 格子，把它插过去。
...
拿起第 100 根 Stick，查表发现它属于 (10,5) 格子，把它插过去。

3. 总结
这一步把“紧凑的逻辑 Stick 列表” 还原成了 “稀疏的物理 FFT 网格”。
*/
    // 遍历全系统所有的 Stick
	for (int istot = 0;istot < nstot; ++istot)
	{
        // 1. 查表：找到第 istot 根 Stick 在物理 xy 网格上的坐标
		int ixy = this->istot2ixy[istot];
        //int ixy = (ixy / fftny)*ny + ixy % fftny;

        // 2. 定位输出位置 (outp)
        // out 是按 xy 网格排列的。ixy 是网格点索引，nplane 是当前进程负责的层厚度。
        // 这指向了网格点 ixy 对应的 Z 柱子的起始位置。
        std::complex<T> *outp = &out[ixy * nplane];

        // 3. 定位输入位置 (inp)
        // in 是按 Stick 紧凑排列的。跳过前面 isot 根 stick 的数据。
        std::complex<T> *inp = &in[istot * nplane];

        // 4. 拷贝数据 (Z 方向的一小段)
        // 把这根 Stick 在当前进程负责的 nplane 层数据，填入网格的正确位置。
		for (int iz = 0; iz < nplane; ++iz)
		{
			outp[iz] = inp[iz];
		}
    }
#endif
    // ModuleBase::timer::tick(this->classname, "gathers_scatterp");
    return;
}



}
