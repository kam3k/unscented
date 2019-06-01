#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "unscented/components.hpp"
#include "unscented/ukf.hpp"

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
  using UKF = UKF<Eigen::Vector3d, 3, Eigen::Vector2d, 2>;
  UKF ukf;

  SECTION("Alpha 1, Beta 2, Kappa 0")
  {
    ukf.set_weight_coefficients(1.0, 2.0, 0.0);

    const auto mean_weights = ukf.get_mean_sigma_weights();
    const auto cov_weights = ukf.get_covariance_sigma_weights();

    REQUIRE(mean_weights.size() == UKF::NUM_SIGMA_POINTS);
    REQUIRE(cov_weights.size() == UKF::NUM_SIGMA_POINTS);

    // All mean weights add to 1.0
    const auto mean_total =
        std::accumulate(mean_weights.begin(), mean_weights.end(), 0.0);
    CHECK(mean_total == Approx(1.0));

    // Actual values of all mean weights
    CHECK(mean_weights[0] == Approx(0.0));
    for (auto i = 1; i < UKF::NUM_SIGMA_POINTS; ++i)
    {
      CHECK(mean_weights[1] == Approx(1.0 / 6.0));
    }

    // All cov weights add to 3.0
    const auto cov_total =
        std::accumulate(cov_weights.begin(), cov_weights.end(), 0.0);
    CHECK(cov_total == Approx(3.0));

    // Actual values of all cov weights
    CHECK(cov_weights[0] == Approx(2.0));
    for (auto i = 1; i < UKF::NUM_SIGMA_POINTS; ++i)
    {
      CHECK(cov_weights[1] == Approx(1.0 / 6.0));
    }
  }

  SECTION("Alpha 0.001, Beta 2, Kappa 0")
  {
    ukf.set_weight_coefficients(0.001, 2.0, 0.0);

    const auto mean_weights = ukf.get_mean_sigma_weights();
    const auto cov_weights = ukf.get_covariance_sigma_weights();

    REQUIRE(mean_weights.size() == UKF::NUM_SIGMA_POINTS);
    REQUIRE(cov_weights.size() == UKF::NUM_SIGMA_POINTS);

    // All mean weights add to 1.0
    const auto mean_total =
        std::accumulate(mean_weights.begin(), mean_weights.end(), 0.0);
    CHECK(mean_total == Approx(1.0));

    // Actual values of all mean weights
    CHECK(mean_weights[0] == Approx(-999999.0));
    for (auto i = 1; i < UKF::NUM_SIGMA_POINTS; ++i)
    {
      CHECK(mean_weights[1] == Approx(1000000.0 / 6.0));
    }

    // All cov weights add to 4.0
    const auto cov_total =
        std::accumulate(cov_weights.begin(), cov_weights.end(), 0.0);
    CHECK(cov_total == Approx(4.0));

    // Actual values of all cov weights
    CHECK(cov_weights[0] == Approx(-999999.0));
    for (auto i = 1; i < UKF::NUM_SIGMA_POINTS; ++i)
    {
      CHECK(cov_weights[1] == Approx(1000000.0 / 6.0));
    }
  }

  SECTION("Alpha 1, Beta 1, Kappa 1")
  {
    ukf.set_weight_coefficients(1.0, 1.0, 1.0);

    const auto mean_weights = ukf.get_mean_sigma_weights();
    const auto cov_weights = ukf.get_covariance_sigma_weights();

    REQUIRE(mean_weights.size() == UKF::NUM_SIGMA_POINTS);
    REQUIRE(cov_weights.size() == UKF::NUM_SIGMA_POINTS);

    // All mean weights add to 1.0
    const auto mean_total =
        std::accumulate(mean_weights.begin(), mean_weights.end(), 0.0);
    CHECK(mean_total == Approx(1.0));

    // Actual values of all mean weights
    CHECK(mean_weights[0] == Approx(0.25));
    for (auto i = 1; i < UKF::NUM_SIGMA_POINTS; ++i)
    {
      CHECK(mean_weights[1] == Approx(0.75 / 6.0));
    }

    // All cov weights add to 2.0
    const auto cov_total =
        std::accumulate(cov_weights.begin(), cov_weights.end(), 0.0);
    CHECK(cov_total == Approx(2.0));

    // Actual values of all cov weights
    CHECK(cov_weights[0] == Approx(1.25));
    for (auto i = 1; i < UKF::NUM_SIGMA_POINTS; ++i)
    {
      CHECK(cov_weights[1] == Approx(0.75 / 6.0));
    }
  }
}

TEST_CASE("Sigma points")
{
  SECTION("State is a vector space")
  {
    using UKF = UKF<Eigen::Vector3d, 3, Eigen::Vector2d, 2>;
    UKF ukf;

    ukf.set_weight_coefficients(1.0, 1.0, 1.0);
    ukf.set_state(UKF::State(1, 2, 3));
    ukf.set_state_covariance(UKF::N_by_N::Identity());
    ukf.generate_sigma_points();
    const auto& sigma_points = ukf.get_sigma_points();

    REQUIRE(sigma_points.size() == UKF::NUM_SIGMA_POINTS);
    CHECK(sigma_points[0].x() == Approx(1.0));
    CHECK(sigma_points[0].y() == Approx(2.0));
    CHECK(sigma_points[0].z() == Approx(3.0));
    CHECK(sigma_points[1].x() == Approx(3.0));
    CHECK(sigma_points[1].y() == Approx(2.0));
    CHECK(sigma_points[1].z() == Approx(3.0));
    CHECK(sigma_points[2].x() == Approx(1.0));
    CHECK(sigma_points[2].y() == Approx(4.0));
    CHECK(sigma_points[2].z() == Approx(3.0));
    CHECK(sigma_points[3].x() == Approx(1.0));
    CHECK(sigma_points[3].y() == Approx(2.0));
    CHECK(sigma_points[3].z() == Approx(5.0));
    CHECK(sigma_points[4].x() == Approx(-1.0));
    CHECK(sigma_points[4].y() == Approx(2.0));
    CHECK(sigma_points[4].z() == Approx(3.0));
    CHECK(sigma_points[5].x() == Approx(1.0));
    CHECK(sigma_points[5].y() == Approx(0.0));
    CHECK(sigma_points[5].z() == Approx(3.0));
    CHECK(sigma_points[6].x() == Approx(1.0));
    CHECK(sigma_points[6].y() == Approx(2.0));
    CHECK(sigma_points[6].z() == Approx(1.0));
  }

  SECTION("State is not a vector space")
  {
    using UKF = UKF<Angle, 1, Angle, 1>;
    UKF ukf;

    ukf.set_weight_coefficients(1.0, 1.0, 1.0);
    ukf.set_state(UKF::State(1.0));
    ukf.set_state_covariance(UKF::N_by_N::Identity());
    ukf.generate_sigma_points();
    const auto& sigma_points = ukf.get_sigma_points();

    REQUIRE(sigma_points.size() == UKF::NUM_SIGMA_POINTS);
    CHECK(sigma_points[0].theta() == Approx(1.0));
    CHECK(sigma_points[1].theta() == Approx(2.0));
    CHECK(sigma_points[2].theta() == Approx(0.0));
  }
}

TEST_CASE("Predict")
{
  SECTION("State is a vector space")
  {
    using UKF = UKF<Eigen::Vector3d, 3, Eigen::Vector2d, 2>;

    SECTION("Linear system model")
    {
      UKF ukf;

      ukf.set_weight_coefficients(1.0, 1.0, 1.0);
      ukf.set_state(UKF::State(1, 2, 3));
      ukf.set_state_covariance(UKF::N_by_N::Identity());
      ukf.set_process_covariance(UKF::N_by_N::Identity() * 1e-3);

      // System model adds one to all state elements
      auto sys_model = [](UKF::State& state) { state += UKF::State::Ones(); };

      ukf.predict(sys_model);

      const auto& state = ukf.get_state();
      const auto& cov = ukf.get_state_covariance();

      CHECK(state.x() == Approx(2.0));
      CHECK(state.y() == Approx(3.0));
      CHECK(state.z() == Approx(4.0));

      CHECK(cov(0, 0) == Approx(1.0 + 1e-3));
      CHECK(cov(0, 1) == Approx(0.0));
      CHECK(cov(0, 2) == Approx(0.0));
      CHECK(cov(1, 0) == Approx(0.0));
      CHECK(cov(1, 1) == Approx(1.0 + 1e-3));
      CHECK(cov(1, 2) == Approx(0.0));
      CHECK(cov(2, 0) == Approx(0.0));
      CHECK(cov(2, 1) == Approx(0.0));
      CHECK(cov(2, 2) == Approx(1.0 + 1e-3));
    }

    // SECTION("Nonlinear system model")
    // {
    //   UKF ukf;

    //   ukf.set_weight_coefficients(1.0, 1.0, 1.0);
    //   ukf.set_state(UKF::State(1, 2, 3));
    //   ukf.set_state_covariance(UKF::N_by_N::Identity());
    //   ukf.set_process_covariance(UKF::N_by_N::Identity() * 1e-3);

    //   // System model adds one to all state elements
    //   auto sys_model = [](UKF::State& state) {
    //     state += UKF::State::Ones();
    //   };

    //   ukf.predict(sys_model);

    //   const auto& state = ukf.get_state();
    //   const auto& cov = ukf.get_state_covariance();

    //   CHECK(state.x() == Approx(2.0));
    //   CHECK(state.y() == Approx(3.0));
    //   CHECK(state.z() == Approx(4.0));

    //   CHECK(cov(0, 0) == Approx(1.0 + 1e-3));
    //   CHECK(cov(0, 1) == Approx(0.0));
    //   CHECK(cov(0, 2) == Approx(0.0));
    //   CHECK(cov(1, 0) == Approx(0.0));
    //   CHECK(cov(1, 1) == Approx(1.0 + 1e-3));
    //   CHECK(cov(1, 2) == Approx(0.0));
    //   CHECK(cov(2, 0) == Approx(0.0));
    //   CHECK(cov(2, 1) == Approx(0.0));
    //   CHECK(cov(2, 2) == Approx(1.0 + 1e-3));
    // }
  }
}

// SECTION("Custom state mean function")
// {
//   UKF ukf;

//   // Custom state mean function clips the z value of the state to be
//   // between 0 and 1
//   auto custom_state_mean_func = [](const UKF::SigmaPoints& states,
//                                    const UKF::SigmaWeights& weights) {
//     Eigen::Vector3d weighted_state = states[0] * weights[0];
//     for (std::size_t i = 1; i < states.size(); ++i)
//     {
//       weighted_state = weighted_state + states[i] * weights[i];
//     }
//     if (weighted_state.z() > 1)
//     {
//       weighted_state.z() = 1.0;
//     }
//     else if (weighted_state.z() < 0)
//     {
//       weighted_state.z() = 0.0;
//     }
//     return weighted_state;
//   };
//   ukf.set_state_mean_function(custom_state_mean_func);

//   ukf.set_weight_coefficients(1.0, 1.0, 1.0);
//   ukf.set_state(UKF::State(1, 2, 0.5));
//   ukf.set_state_covariance(UKF::N_by_N::Identity());
//   ukf.set_process_covariance(UKF::N_by_N::Identity() * 1e-3);

//   // System model adds one to all state elements
//   auto sys_model = [](UKF::State& state) {
//     state += UKF::State::Ones();
//   };

//   ukf.predict(sys_model);

//   const auto& state = ukf.get_state();
//   const auto& cov = ukf.get_state_covariance();

//   CHECK(state.x() == Approx(2.0));
//   CHECK(state.y() == Approx(3.0));
//   CHECK(state.z() == Approx(1.0)); // clipped

//   CHECK(cov(0, 0) == Approx(1.0 + 1e-3));
//   CHECK(cov(0, 1) == Approx(0.0));
//   CHECK(cov(0, 2) == Approx(0.0));
//   CHECK(cov(1, 0) == Approx(0.0));
//   CHECK(cov(1, 1) == Approx(1.0 + 1e-3));
//   CHECK(cov(1, 2) == Approx(0.0));
//   CHECK(cov(2, 0) == Approx(0.0));
//   CHECK(cov(2, 1) == Approx(0.0));
//   CHECK(cov(2, 2) == Approx(1.0 + 1e-3));
// }
} // namespace unscented
