#include "search/common/temperature.h"

#include <gtest/gtest.h>

#include "utils/random.h"

namespace lczero {

TEST(TemperatureTest, EffectiveTauDecayAndCutoff) {
  TempParams p{1.0, 10, 0, 0.0, 0.0, 0.0};
  EXPECT_DOUBLE_EQ(EffectiveTau(p, 1), 1.0);
  EXPECT_NEAR(EffectiveTau(p, 5), 0.6, 1e-9);
  EXPECT_DOUBLE_EQ(EffectiveTau(p, 11), 0.0);
  p.temp_cutoff_move = 5;
  p.temp_endgame = 0.3;
  EXPECT_DOUBLE_EQ(EffectiveTau(p, 5), 0.3);
}

TEST(TemperatureTest, SampleProbabilityShift) {
  TempParams p{0.0, 0, 0, 0.0, 0.0, 0.0};
  std::vector<double> base{1.0, 4.0};
  std::vector<double> wp{0.5, 0.5};
  int count = 0;
  for (int i = 0; i < 5000; ++i) {
    int idx = SampleWithTemperature(base, wp, p, 1.0, Random::Get(), 0);
    if (idx == 1) ++count;
  }
  double freq = static_cast<double>(count) / 5000.0;
  EXPECT_NEAR(freq, 4.0 / 5.0, 0.05);

  count = 0;
  for (int i = 0; i < 5000; ++i) {
    int idx = SampleWithTemperature(base, wp, p, 0.5, Random::Get(), 0);
    if (idx == 1) ++count;
  }
  freq = static_cast<double>(count) / 5000.0;
  EXPECT_NEAR(freq, 16.0 / 17.0, 0.05);
}

TEST(TemperatureTest, CutoffAndVisitOffset) {
  TempParams p{0.0, 0, 0, 0.0, 0.1, 0.0};
  std::vector<double> base{1.0, 1.0};
  std::vector<double> wp{1.0, 0.8};
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(0, SampleWithTemperature(base, wp, p, 1.0, Random::Get(), 0));
  }
  p.visit_offset = 2.0;
  std::vector<double> base2{1.0};
  EXPECT_EQ(0, SampleWithTemperature(base2, std::span<const double>(), p, 1.0,
                                     Random::Get(), 0));
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

