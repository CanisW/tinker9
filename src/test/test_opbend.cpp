#include "files.h"
#include "test/ff.h"
#include "test/rt.h"
#include "test/test.h"

m_tinker_using_namespace;
using namespace test;

static const char* opbendterm_only = R"**(
opbendterm  only
)**";

static int usage =
    gpu::use_xyz | gpu::use_energy | gpu::use_grad | gpu::use_virial;

static const double ref_g_opbend_trpcage[][3] = {
    {0.0000, 0.0000, 0.0000},     {0.1615, 1.3064, -0.7696},
    {-0.7692, -7.2061, 4.4734},   {0.1732, 1.7496, -1.0103},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {1.6187, 1.3032, -0.3215},    {-6.0156, -4.8556, 1.3806},
    {2.1317, 1.8166, -0.4764},    {3.1242, 2.5098, -0.8736},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {-0.4388, -0.3985, 0.1519},   {-0.4202, -0.3754, 0.1390},
    {1.1009, 9.5399, -5.5036},    {-0.5742, -2.9124, 1.1106},
    {1.2388, 3.8187, 0.6659},     {-0.2502, -0.8077, -0.1320},
    {-0.2834, -3.0804, 1.6049},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {-1.7502, -5.8233, -0.9455},
    {0.3433, 2.4066, -0.1229},    {0.5133, -4.7431, 1.6365},
    {-0.1241, 1.2419, -0.4070},   {0.5291, 1.9461, 0.3176},
    {0.0000, 0.0000, 0.0000},     {0.1591, 0.2606, 0.0462},
    {-0.3904, -0.6421, -0.1044},  {0.0310, 0.0497, 0.0091},
    {0.0247, 0.0398, 0.0072},     {0.0172, 0.0285, 0.0048},
    {0.0119, 0.0197, 0.0036},     {-0.1112, -0.1840, -0.0256},
    {0.0573, 0.0950, 0.0124},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0652, 0.1032, 0.0146},
    {0.0643, 0.1118, 0.0166},     {0.0343, 0.0551, 0.0071},
    {0.0367, 0.0628, 0.0084},     {0.0000, 0.0000, 0.0000},
    {-0.7007, 5.5574, -1.8243},   {-0.0496, -2.3657, 0.2621},
    {0.8120, 4.5532, 0.6066},     {-0.2396, -1.4365, -0.1828},
    {0.2486, -1.7080, 0.5190},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {-0.6161, -3.5948, -0.5326},
    {0.0013, 2.0288, -0.2833},    {0.5452, -5.4972, 1.5832},
    {-0.1707, 1.7770, -0.4837},   {0.1320, 0.8867, 0.1389},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0417, -0.0147, 0.0305},    {0.8118, -0.3514, 0.6091},
    {0.0544, -0.0191, 0.0397},    {-3.4017, 1.2548, -2.7080},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {1.2546, -0.4580, 1.0516},    {1.2391, -0.4115, 0.9771},
    {-0.5320, 3.9868, -1.1567},   {0.0077, -1.5114, 0.0448},
    {0.4302, 3.7294, 0.7085},     {-0.1187, -1.0949, -0.1964},
    {0.1572, -0.9347, 0.2511},    {0.0000, 0.0000, 0.0000},
    {0.0081, -0.0750, 0.0448},    {-0.0393, 0.3575, -0.2142},
    {0.0402, -0.3592, 0.2191},    {0.0276, -0.2164, 0.1269},
    {0.0048, -0.0364, 0.0220},    {-0.0034, 0.0290, -0.0163},
    {-0.0308, 0.2070, -0.1238},   {0.0037, -0.0219, 0.0133},
    {0.0609, -0.3895, 0.2370},    {-0.0508, 0.3220, -0.1965},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {-0.0166, 0.1421, -0.0860},   {-0.0078, 0.0614, -0.0379},
    {0.0040, -0.0260, 0.0156},    {0.0023, -0.0148, 0.0092},
    {-0.0173, 0.1143, -0.0701},   {0.0145, -0.0941, 0.0570},
    {-0.3569, -3.5369, -0.7431},  {-0.0362, 1.6152, -0.1014},
    {0.3569, -2.9022, 0.8626},    {-0.1447, 1.1352, -0.3270},
    {0.0775, 0.9776, 0.2176},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {-0.0877, 0.6972, -0.2186},
    {-0.6219, -1.2388, -0.0823},  {2.4272, 5.2247, 0.3218},
    {-0.8026, -1.8026, -0.1083},  {-0.0141, 0.1161, -0.0359},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {-1.5166, -3.2293, -0.2576},
    {1.3635, 0.1146, 0.1750},     {-1.6729, 0.5418, -0.1468},
    {1.4775, -0.5038, 0.1812},    {0.2867, 0.6475, 0.0607},
    {0.0000, 0.0000, 0.0000},     {1.9750, 7.2518, 1.4477},
    {-6.8838, -26.3237, -3.8070}, {1.9870, 9.7686, 1.2812},
    {2.9219, 9.3033, 1.0781},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {-5.5477, 1.8972, -0.3616},
    {0.5813, -1.2075, 0.5661},    {5.4239, 1.5347, -1.6933},
    {-1.9378, -0.5824, 0.5881},   {2.5184, -0.7958, 0.1168},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {-2.4308, -0.6363, 0.7441},   {0.7391, -1.2245, 0.2542},
    {-1.8424, 6.5083, -1.5547},   {0.6513, -1.6721, 0.4231},
    {0.2556, 0.0706, -0.0750},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.7544, -6.8566, 2.2587},
    {-0.2117, 1.5515, 1.8343},    {0.1086, 0.0827, -7.0616},
    {-0.0054, 0.0428, 3.3673},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.1024, 1.6842, -0.7499},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.1573, 0.0139, -2.3038},
    {-0.8649, -0.1706, 2.8960},   {3.6336, 0.9805, -6.3118},
    {-0.9695, -0.2567, 1.8139},   {-0.1384, -0.0719, 1.9841},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {-3.8497, -1.4802, 6.2760},   {0.7375, 0.3022, -1.2169},
    {0.0053, -0.0210, 0.0126},    {-0.0879, -0.0844, 0.2137},
    {1.1167, 0.5532, -1.8024},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.6410, 0.6821, -1.5575},
    {-0.4955, -0.3089, 0.8926},   {1.5090, 0.5010, -2.0272},
    {-0.3693, -0.1245, 0.5075},   {-0.2774, -0.2953, 0.6651},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {-1.9150, -0.6878, 2.5730},   {0.5288, 0.0229, 0.4019},
    {-0.1459, 0.2663, -4.8271},   {0.0939, -0.1490, 1.3216},
    {0.6079, 0.2362, -0.8198},    {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {-0.0172, -0.0787, 0.0419},   {0.0917, 0.3812, -0.2048},
    {-0.0893, -0.3629, 0.1990},   {0.0646, 0.2398, -0.1344},
    {-0.0936, -0.2873, 0.1497},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {-0.0228, -0.0939, 0.0520},
    {-0.0127, -0.0461, 0.0262},   {-0.0124, -0.0447, 0.0258},
    {0.0448, 0.1465, -0.0792},    {0.0468, 0.1459, -0.0762},
    {0.0157, 0.4317, 4.6422},     {-0.9294, -0.4362, -1.2722},
    {1.5774, 0.5352, 0.7022},     {-1.1146, -0.3729, -0.3101},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {-0.0018, -0.2616, -1.1094},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {2.9363, 0.5927, -0.0133},    {-2.2131, 0.8976, -0.0838},
    {3.0083, -2.8560, -0.0659},   {-1.3536, 1.4008, 0.0779},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {-1.2965, -0.2385, -0.0033},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.1374, -1.2768, -0.1222},   {-0.4474, 0.4805, 0.7345},
    {0.1212, 0.3668, -1.0970},    {-0.0543, -0.3724, 1.0111},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {-0.3816, 0.8547, 0.0313},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.1354, 1.5775, -3.5107},    {-0.1677, -0.9129, 1.9064},
    {0.2778, 1.3345, -2.3704},    {-0.0968, -0.4863, 0.8563},
    {-0.0735, -0.7410, 1.6057},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},     {-0.1075, -0.4905, 0.8721}};
TEST_CASE("Opbend-Trpcage", "[ff][eopbend][allinger][trpcage]") {
  const char* k = "test_trpcage.key";
  const char* x1 = "test_trpcage.xyz";
  const char* p = "amoebapro13.prm";

  std::string k0 = trpcage_key;
  k0 += opbendterm_only;
  file fke(k, k0);

  file fx1(x1, trpcage_xyz);
  file fpr(p, amoebapro13_prm);

  const char* argv[] = {"dummy", x1};
  int argc = 2;
  test_begin_1_xyz(argc, argv);
  gpu::use_data = usage;
  tinker_gpu_data_create();

  const double eps_e = 0.0001;
  const double ref_e = 4.3016;
  const int ref_count = 189;
  const double eps_g = test_get_eps2(0.0004, 0.0001);
  const double eps_v = 0.001;
  const double ref_v[][3] = {{-1.646, 0.890, -0.224},
                             {0.890, 3.006, -1.049},
                             {-0.224, -1.049, -1.359}};

  COMPARE_BONED_FORCE(gpu::eopbend, gpu::eopb, ref_e, eps_e, gpu::nopbend,
                      ref_count, gpu::gx, gpu::gy, gpu::gz,
                      ref_g_opbend_trpcage, eps_g, gpu::vir_eopb, ref_v, eps_v);

  tinker_gpu_data_destroy();
  test_end();
}
