#include "sep_cell.h"

#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_base/parallel_common.h"
#include "source_base/tool_title.h"

#include <algorithm>
#include <string>
#include <vector>

// namespace GlobalC
// {
// Sep_Cell sep_cell;
// }

Sep_Cell::Sep_Cell() noexcept : ntype(0), omega(0.0), tpiba2(0.0)
{
}

Sep_Cell::~Sep_Cell() noexcept = default;

void Sep_Cell::init(const int ntype_in)
{
    this->ntype = ntype_in;
    this->seps.resize(ntype);
    this->sep_enable.resize(ntype);
    std::fill(this->sep_enable.begin(), this->sep_enable.end(), false);
}

void Sep_Cell::set_omega(const double omega_in, const double tpiba2_in)
{
    this->omega = omega_in;
    this->tpiba2 = tpiba2_in;
}

/**
 * read sep potential files
 *
 * need to add following lines in STRU file, and order of elements must match ATOMIC_SPECIES.
 * SEP_FILES
 * symbol is_enable r_in r_out r_power enhence_a
 *
 * example
 * Li 0
 * F  1 F_pbe_50.sep 0.0 2.0 20.0 1.0
 */
int Sep_Cell::read_sep_potentials(std::ifstream& ifpos,
                                  const std::string& pp_dir,
                                  std::ofstream& ofs_running,
                                  std::vector<std::string>& ucell_atom_label)
{
    ModuleBase::TITLE("Sep_Cell", "read_sep_potentials");

    if (!ModuleBase::GlobalFunc::SCAN_BEGIN(ifpos, "SEP_FILES"))
    {
        GlobalV::ofs_running << "Cannot find SEP_FILES section in STRU" << std::endl;
        return false;
    }

    ifpos.ignore(300, '\n');

    for (int i = 0; i < this->ntype; ++i)
    {
        std::string one_line, atom_label;
        std::getline(ifpos, one_line);
        std::stringstream ss(one_line);

        // read the label of the atom
        bool enable_tmp;
        ss >> atom_label >> enable_tmp;

        // Validate atom label
        // 检查原子顺序是否与 ATOMIC_SPECIES 匹配，确保势能与原子类型对应
        if (atom_label != ucell_atom_label[i])
        {
            GlobalV::ofs_running << "Sep potential and atom order do not match. "
                                 << "Expected: " << ucell_atom_label[i] << ", Got: " << atom_label << std::endl;
            return false;
        }
        this->sep_enable[i] = enable_tmp;
        if (this->sep_enable[i])
        {
            this->seps[i].is_enable = this->sep_enable[i];
            std::string sep_filename;
            // 对每个使能的原子类型，读取自能势文件名和参数
            ss >> sep_filename;
            ss >> this->seps[i].r_in >> this->seps[i].r_out >> this->seps[i].r_power >> this->seps[i].enhence_a;
            std::string sep_addr = pp_dir + sep_filename;
            std::ifstream sep_ifs(sep_addr.c_str(), std::ios::in);
            if (!sep_ifs)
            {
                GlobalV::ofs_running << "Cannot find sep potential file: " << sep_addr << std::endl;
                return false;
            }
            // 读取自能势数据
            this->seps[i].read_sep(sep_ifs);
        }
    }

    return true;
}

#ifdef __MPI
// bcast_sep_cell() 的作用是：
// 在并行计算环境下，将主进程读取到的 DFT-1/2 自能势数据同步到所有进程，保证每个进程都能正确参与后续的自洽计算。
// 具体流程是：先广播原子类型数，再为每个原子类型广播使能标志和具体自能势数据。
// 好处是：只需主进程读取和解析输入文件，其他进程通过 MPI 广播获得一致的数据，避免重复IO和解析，提高并行效率和一致性。

// 广播（broadcast）在MPI中指的是：把一份数据从一个进程（通常是主进程，rank 0）发送给所有其他进程，让大家都拥有这份数据。
// 因为在并行计算中，通常只有主进程负责读取输入文件或生成某些数据。为了让所有进程都能用到这些数据，
// 就需要把它们“广播”出去。这样可以避免每个进程都去读文件，节省时间和资源，并保证数据一致。
void Sep_Cell::bcast_sep_cell()
{
    // 调用了一个标题输出函数，标记当前正在广播 `Sep_Cell` 数据
    ModuleBase::TITLE("Sep_Cell", "bcast_sep_cell");
    // 广播原子类型数 `ntype`，确保所有进程都获得一致的原子类型数量。
    Parallel_Common::bcast_int(this->ntype);

    if (GlobalV::MY_RANK != 0)
    {
        // 如果当前进程不是主进程（rank 0），
        // 则根据广播得到的 `ntype`，分配自能势数据结构 `seps` 和使能标志 `sep_enable` 的空间。
        this->seps.resize(this->ntype);
        this->sep_enable.resize(this->ntype);
    }

    // 对每个原子类型循环
    for (int i = 0; i < this->ntype; ++i)
    {
        bool tmp = false; // 定义临时变量 `tmp`，用于存储当前原子类型的使能标志。
        if (GlobalV::MY_RANK == 0)
        {
            // 如果是主进程（rank 0），将本地的 `sep_enable[i]` 赋值给 `tmp`。
            tmp = this->sep_enable[i];
        }
        Parallel_Common::bcast_bool(tmp);// 将主进程的 `tmp` 广播到所有进程。
        if (GlobalV::MY_RANK != 0)
        {
            // 非主进程收到广播后，将 `tmp` 赋值给本地的 `sep_enable[i]`，保证所有进程的使能标志一致。
            this->sep_enable[i] = tmp;
        }
        // 进一步广播每个原子类型的自能势具体数据，确保所有进程都拥有完整的自能势信息。
        // 在 MPI 广播（如 MPI_Bcast）的实际用法中，
        // 所有进程（包括主进程和非主进程）都会调用同一个广播函数，并且这个函数会自动在所有进程之间同步数据
        // 所以好像并不需要手动接收或赋值。
        this->seps[i].bcast_sep();
    }
}
#endif // __MPI

// #ifdef 和 #endif 是C/C++中的条件编译指令，用于控制某段代码是否被编译进最终的程序。
// 这里就是说只有定义了宏 __MPI 时，编译器才会编译 bcast_sep_cell() 函数的代码，实现并行功能，否则就是串行版本。
// 编译命令中加上-D__MPI，就会让编译器自动在所有源文件的最前面加了一句 #define __MPI，
// 所有用 #ifdef __MPI ... #endif 包裹的代码都会被编译器识别为“需要编译”，于是并行相关的代码就会被包含进最终的程序。