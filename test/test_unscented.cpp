#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "unscented.hpp"

namespace unscented
{
TEST_CASE("Static dimensions")
{
  using UKF32 = UKF<Eigen::Vector3d, 3, Eigen::Vector2d, 2>;

  CHECK(UKF32::N == 3);
  CHECK(UKF32::M == 2);
  CHECK(UKF32::NUM_SIGMA_POINTS == 7);

  using UKF96 =
      UKF<Eigen::Matrix<double, 9, 1>, 9, Eigen::Matrix<double, 6, 1>, 6>;

  CHECK(UKF96::N == 9);
  CHECK(UKF96::M == 6);
  CHECK(UKF96::NUM_SIGMA_POINTS == 19);
}

TEST_CASE("Weights")
{
  using UKF32 = UKF<Eigen::Vector3d, 3, Eigen::Vector2d, 2>;
  UKF32 ukf_32;

  SECTION("Alpha 1, Beta 2, Kappa 0")
  {
    ukf_32.set_weight_coefficients(1.0, 2.0, 0.0);

    const auto mean_weights = ukf_32.get_mean_sigma_weights();
    const auto cov_weights = ukf_32.get_covariance_sigma_weights();

    REQUIRE(mean_weights.size() == UKF32::NUM_SIGMA_POINTS);
    REQUIRE(cov_weights.size() == UKF32::NUM_SIGMA_POINTS);

    // All mean weights add to 1.0
    const auto mean_total =
        std::accumulate(mean_weights.begin(), mean_weights.end(), 0.0);
    CHECK(mean_total == Approx(1.0));

    // Actual values of all mean weights
    CHECK(mean_weights[0] == Approx(0.0));
    for (auto i = 1; i < UKF32::NUM_SIGMA_POINTS; ++i)
    {
      CHECK(mean_weights[1] == Approx(1.0 / 6.0));
    }

    // All cov weights add to 3.0
    const auto cov_total =
        std::accumulate(cov_weights.begin(), cov_weights.end(), 0.0);
    CHECK(cov_total == Approx(3.0));

    // Actual values of all cov weights
    CHECK(cov_weights[0] == Approx(2.0));
    for (auto i = 1; i < UKF32::NUM_SIGMA_POINTS; ++i)
    {
      CHECK(cov_weights[1] == Approx(1.0 / 6.0));
    }
  }

  SECTION("Alpha 0.001, Beta 2, Kappa 0")
  {
    ukf_32.set_weight_coefficients(0.001, 2.0, 0.0);

    const auto mean_weights = ukf_32.get_mean_sigma_weights();
    const auto cov_weights = ukf_32.get_covariance_sigma_weights();

    REQUIRE(mean_weights.size() == UKF32::NUM_SIGMA_POINTS);
    REQUIRE(cov_weights.size() == UKF32::NUM_SIGMA_POINTS);

    // All mean weights add to 1.0
    const auto mean_total =
        std::accumulate(mean_weights.begin(), mean_weights.end(), 0.0);
    CHECK(mean_total == Approx(1.0));

    // Actual values of all mean weights
    CHECK(mean_weights[0] == Approx(-999999.0));
    for (auto i = 1; i < UKF32::NUM_SIGMA_POINTS; ++i)
    {
      CHECK(mean_weights[1] == Approx(1000000.0 / 6.0));
    }

    // All cov weights add to 4.0
    const auto cov_total =
        std::accumulate(cov_weights.begin(), cov_weights.end(), 0.0);
    CHECK(cov_total == Approx(4.0));

    // Actual values of all cov weights
    CHECK(cov_weights[0] == Approx(-999999.0));
    for (auto i = 1; i < UKF32::NUM_SIGMA_POINTS; ++i)
    {
      CHECK(cov_weights[1] == Approx(1000000.0 / 6.0));
    }
  }

  SECTION("Alpha 1, Beta 1, Kappa 1")
  {
    ukf_32.set_weight_coefficients(1.0, 1.0, 1.0);

    const auto mean_weights = ukf_32.get_mean_sigma_weights();
    const auto cov_weights = ukf_32.get_covariance_sigma_weights();

    REQUIRE(mean_weights.size() == UKF32::NUM_SIGMA_POINTS);
    REQUIRE(cov_weights.size() == UKF32::NUM_SIGMA_POINTS);

    // All mean weights add to 1.0
    const auto mean_total =
        std::accumulate(mean_weights.begin(), mean_weights.end(), 0.0);
    CHECK(mean_total == Approx(1.0));

    // Actual values of all mean weights
    CHECK(mean_weights[0] == Approx(0.25));
    for (auto i = 1; i < UKF32::NUM_SIGMA_POINTS; ++i)
    {
      CHECK(mean_weights[1] == Approx(0.75 / 6.0));
    }

    // All cov weights add to 2.0
    const auto cov_total =
        std::accumulate(cov_weights.begin(), cov_weights.end(), 0.0);
    CHECK(cov_total == Approx(2.0));

    // Actual values of all cov weights
    CHECK(cov_weights[0] == Approx(1.25));
    for (auto i = 1; i < UKF32::NUM_SIGMA_POINTS; ++i)
    {
      CHECK(cov_weights[1] == Approx(0.75 / 6.0));
    }
  }
}
} // namespace unscented
