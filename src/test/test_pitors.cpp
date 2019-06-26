#include "files.h"
#include "test/ff.h"
#include "test/rt.h"
#include "test/test.h"

m_tinker_using_namespace;
using namespace test;

static const char* pitorsterm_only = R"**(
pitorsterm  only
)**";

static int usage =
    gpu::use_xyz | gpu::use_energy | gpu::use_grad | gpu::use_virial;

static const double ref_g_pitors_trpcage[][3] = {
    {0.0000, 0.0000, 0.0000},    {-0.0001, -0.0007, 0.0004},
    {0.0000, -0.0002, 0.0001},   {0.0001, 0.0007, -0.0004},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.8811, 0.7120, -0.1836},   {-0.0309, -0.0574, 0.0431},
    {-0.8683, -0.7016, 0.1810},  {-0.0286, 0.0052, -0.0245},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {-1.0806, -0.9685, 0.3687},  {1.1272, 1.0103, -0.3846},
    {-0.0000, 0.0000, 0.0000},   {-0.0393, -0.1247, -0.0196},
    {-0.0040, -0.0243, -0.0079}, {0.0391, 0.1237, 0.0200},
    {0.0001, 0.0009, -0.0005},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {-0.0030, 0.0004, 0.0042},
    {-0.0801, 0.3087, -0.1700},  {-0.0230, 0.0822, -0.0244},
    {0.0412, -0.4482, 0.1501},   {0.0462, 0.1598, 0.0250},
    {0.0000, 0.0000, 0.0000},    {0.0662, 0.1091, 0.0175},
    {-0.0460, -0.0799, -0.0071}, {-0.0834, -0.1326, -0.0211},
    {-0.0831, -0.1389, -0.0222}, {0.0913, 0.1486, 0.0193},
    {0.0779, 0.1305, 0.0172},    {0.0453, 0.0756, 0.0122},
    {-0.0967, -0.1601, -0.0219}, {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {-0.0331, -0.0531, -0.0074},
    {0.0076, 0.0130, 0.0019},    {0.0444, 0.0719, 0.0094},
    {0.0094, 0.0160, 0.0022},    {0.0000, 0.0000, 0.0000},
    {0.0115, 0.0052, -0.0031},   {-0.1363, 0.0794, -0.2017},
    {-0.0124, -0.0799, -0.0124}, {0.0725, 0.4087, 0.0503},
    {0.0756, -0.5676, 0.1782},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {-0.0009, -0.0044, -0.0004},
    {-0.1363, 0.3585, -0.2927},  {-0.0184, 0.1356, -0.0376},
    {0.0691, -0.8144, 0.2298},   {0.0811, 0.5198, 0.0798},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {-0.3144, 0.1106, -0.2295},  {0.0277, 0.0057, -0.0001},
    {0.3222, -0.1133, 0.2352},   {-0.0211, -0.0081, 0.0059},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.4041, -0.1450, 0.3254},   {-0.4185, 0.1501, -0.3370},
    {-0.0051, 0.0176, -0.0045},  {-0.2055, 0.3113, -0.3362},
    {-0.0076, -0.1141, -0.0291}, {0.0650, 0.5598, 0.0982},
    {0.1629, -1.0061, 0.2758},   {0.0000, 0.0000, 0.0000},
    {-0.0382, 0.3650, -0.2171},  {-0.0587, 0.4264, -0.2503},
    {0.0098, -0.1183, 0.0664},   {0.0036, -0.0408, 0.0242},
    {0.0380, -0.2449, 0.1567},   {-0.0033, 0.0172, -0.0103},
    {0.0582, -0.4340, 0.2528},   {-0.0391, 0.2912, -0.1833},
    {-0.0260, 0.1705, -0.1026},  {0.0199, -0.1297, 0.0805},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0247, -0.2146, 0.1320},   {0.0095, -0.0756, 0.0466},
    {0.0225, -0.1482, 0.0892},   {-0.0265, 0.1713, -0.1060},
    {0.0144, -0.0919, 0.0557},   {-0.0089, 0.0564, -0.0346},
    {-0.0027, -0.0031, 0.0036},  {-0.0500, -0.6129, -0.1259},
    {0.0000, -0.0028, 0.0009},   {-0.0019, 0.0155, -0.0046},
    {0.0618, 0.7129, 0.1556},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0004, -0.0003, 0.0000},
    {-0.1677, -0.3846, -0.0143}, {-0.0342, -0.0705, -0.0020},
    {0.1683, 0.3649, 0.0192},    {-0.0024, 0.0193, -0.0059},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0036, 0.0021, -0.0043},
    {-0.0957, -0.4183, -0.0262}, {0.0254, 0.0001, 0.0011},
    {-0.0821, 0.0271, -0.0095},  {0.2079, 0.4634, 0.0419},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {-0.0007, -0.0085, 0.0004},
    {-0.0833, -0.0786, 0.0591},  {-0.0245, -0.0060, 0.0122},
    {0.1770, 0.0510, -0.0561},   {-0.1096, 0.0373, -0.0064},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {-0.0080, -0.0030, -0.0025}, {1.2093, -3.4348, 0.8514},
    {-0.6333, 0.1666, -0.0062},  {-1.4453, 3.4878, -0.8195},
    {0.2179, 0.0598, -0.0643},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.6738, -0.1563, -0.0113},
    {0.1029, -3.6469, 1.4464},   {0.0021, -0.0117, 0.0223},
    {0.0010, 0.0014, -0.1161},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {-0.1005, 3.5294, -1.2879},  {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {-0.0036, 0.0113, 0.0000},
    {-0.7508, -0.1734, 1.5137},  {-0.1620, -0.1002, 0.2245},
    {0.7620, 0.1753, -1.4245},   {0.0098, 0.0029, -0.1480},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {-0.0107, 0.0255, 0.0592},   {-0.7728, -0.2979, 1.1570},
    {0.0179, 0.0285, -0.0493},   {-0.1136, -0.1097, 0.2759},
    {1.0379, 0.4763, -1.6763},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0048, -0.0048, -0.0054},
    {-0.0573, 0.0689, -0.0470},  {-0.0363, -0.0170, 0.0440},
    {0.1814, 0.0597, -0.2495},   {-0.1429, -0.1500, 0.3441},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {-0.0004, 0.0033, 0.0057},   {0.0250, -0.5051, 3.5067},
    {-0.1130, 0.2332, -0.2414},  {-0.2320, 0.4521, -3.3948},
    {0.2310, 0.0863, -0.3120},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.1255, -0.2825, 0.2297},   {1.7685, 1.1683, 3.8141},
    {-0.0470, -0.0027, -0.3280}, {-1.9147, -0.6194, -0.4965},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0396, -0.5539, -3.1795},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0777, 0.0198, 0.3504},    {3.8766, -1.6442, 0.0297},
    {-0.3906, -0.0222, -0.1163}, {-1.9839, 2.1756, 0.0993},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {-1.8960, -0.4301, -0.1204},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.3869, 0.0985, 0.1213},    {1.3503, -2.4533, -0.6262},
    {-0.0164, 0.0668, -0.0942},  {-0.0246, -0.1759, 0.4664},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {-1.2440, 2.4740, 0.1360},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0206, -0.0271, 0.0070},   {0.0249, 0.2250, -0.5079},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {-0.0299, -0.2703, 0.6102},  {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},    {0.0000, 0.0000, 0.0000}};
TEST_CASE("Pitors-Trpcage", "[ff][epitors][trpcage]") {
  const char* k = "test_trpcage.key";
  const char* x1 = "test_trpcage.xyz";
  const char* p = "amoebapro13.prm";

  std::string k0 = trpcage_key;
  k0 += pitorsterm_only;
  file fke(k, k0);

  file fx1(x1, trpcage_xyz);
  file fpr(p, amoebapro13_prm);

  const char* argv[] = {"dummy", x1};
  int argc = 2;
  test_begin_1_xyz(argc, argv);
  gpu::use_data = usage;
  tinker_gpu_data_create();

  const double eps_e = 0.0001;
  const double ref_e = 5.0769;
  const int ref_count = 37;
  const double eps_g = 0.0001;
  const double eps_v = 0.001;
  const double ref_v[][3] = {
      {-3.014, -1.132, 2.354}, {-1.132, 1.167, -0.679}, {2.354, -0.679, 1.847}};

  COMPARE_BONED_FORCE(gpu::epitors, gpu::ept, ref_e, eps_e, gpu::npitors,
                      ref_count, gpu::gx, gpu::gy, gpu::gz,
                      ref_g_pitors_trpcage, eps_g, gpu::vir_ept, ref_v, eps_v);

  tinker_gpu_data_destroy();
  test_end();
}
