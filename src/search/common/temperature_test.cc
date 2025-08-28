#include "search/common/temperature.h"

#include <gtest/gtest.h>

#include "utils/random.h"

namespace lczero {

TEST(TemperatureTest, EffectiveTauDecayAndCutoff) {
  // temperature = 1.0, decay over 10 moves
  TemperatureParams p{1.0f, 10, 0, 0.0f, 0, 0.0f, 0.0f};
  // fullmove 1 -> ply 0
  EXPECT_FLOAT_EQ(EffectiveTau(p, 0), 1.0f);
  // fullmove 5 -> moves_played = 4 -> ply ~8
  EXPECT_NEAR(EffectiveTau(p, 8), 0.6f, 1e-6f);
  // fullmove 11 -> moves_played = 10 -> ply ~20 -> tau should be 0
  EXPECT_FLOAT_EQ(EffectiveTau(p, 20), 0.0f);
  p.temp_cutoff_move = 5;
  p.temp_endgame = 0.3f;
  // fullmove 5 -> trigger cutoff when moves+1 >= 5, so ply 8 (moves=4) should
  // hit endgame temperature
  EXPECT_FLOAT_EQ(EffectiveTau(p, 8), 0.3f);
}

TEST(TemperatureTest, SampleProbabilityShift) {
  TemperatureParams p{0.0f, 0, 0, 0.0f, 0, 0.0f, 0.0f};
  std::vector<double> base{1.0, 4.0};
  std::vector<double> wp{0.5, 0.5};
  int count = 0;
  for (int i = 0; i < 5000; ++i) {
    int idx = SampleWithTemperature(base, wp, p, 1.0f, Random::Get(), 0);
    if (idx == 1) ++count;
  }
  double freq = static_cast<double>(count) / 5000.0;
  EXPECT_NEAR(freq, 4.0 / 5.0, 0.05);

  count = 0;
  for (int i = 0; i < 5000; ++i) {
    int idx = SampleWithTemperature(base, wp, p, 0.5f, Random::Get(), 0);
    if (idx == 1) ++count;
  }
  freq = static_cast<double>(count) / 5000.0;
  EXPECT_NEAR(freq, 16.0 / 17.0, 0.05);
}

TEST(TemperatureTest, CutoffAndVisitOffset) {
  TemperatureParams p{0.0f, 0, 0, 0.0f, 0, 0.1f, 0.0f};
  std::vector<double> base{1.0, 1.0};
  std::vector<double> wp{1.0, 0.8};
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(0, SampleWithTemperature(base, wp, p, 1.0f, Random::Get(), 0));
  }
  p.visit_offset = 2.0;
  std::vector<double> base2{1.0};
  EXPECT_EQ(0, SampleWithTemperature(base2, std::span<const double>(), p, 1.0f,
                                     Random::Get(), 0));
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

