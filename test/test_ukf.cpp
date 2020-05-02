#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "unscented/primitives.hpp"
#include "unscented/ukf.hpp"

namespace unscented
{
TEST_CASE("Static dimensions")
{
  using UKF_32 = UKF<Vector<3>, Vector<2>>;

  CHECK(UKF_32::N == 3);
  CHECK(UKF_32::M == 2);
  CHECK(UKF_32::NUM_SIGMA_POINTS == 7);

  using UKF_96 = UKF<Vector<9>, Vector<6>>;

  CHECK(UKF_96::N == 9);
  CHECK(UKF_96::M == 6);
  CHECK(UKF_96::NUM_SIGMA_POINTS == 19);

  using UKF_UnitComplex = UKF<UnitComplex, UnitComplex>;

  CHECK(UKF_UnitComplex::N == 1);
  CHECK(UKF_UnitComplex::M == 1);
  CHECK(UKF_UnitComplex::NUM_SIGMA_POINTS == 3);

  using UKF_SO2 = UKF<SO2, SO2>;

  CHECK(UKF_SO2::N == 1);
  CHECK(UKF_SO2::M == 1);
  CHECK(UKF_SO2::NUM_SIGMA_POINTS == 3);

  using UKF_SE2 = UKF<SE2, SE2>;

  CHECK(UKF_SE2::N == 3);
  CHECK(UKF_SE2::M == 3);
  CHECK(UKF_SE2::NUM_SIGMA_POINTS == 7);

  using UKF_UnitQuaternion = UKF<UnitQuaternion, UnitQuaternion>;

  CHECK(UKF_UnitQuaternion::N == 3);
  CHECK(UKF_UnitQuaternion::M == 3);
  CHECK(UKF_UnitQuaternion::NUM_SIGMA_POINTS == 7);

  using UKF_SO3 = UKF<SO3, SO3>;

  CHECK(UKF_SO3::N == 3);
  CHECK(UKF_SO3::M == 3);
  CHECK(UKF_SO3::NUM_SIGMA_POINTS == 7);

  using UKF_SE3 = UKF<SE3, SE3>;

  CHECK(UKF_SE3::N == 6);
  CHECK(UKF_SE3::M == 6);
  CHECK(UKF_SE3::NUM_SIGMA_POINTS == 13);
}

TEST_CASE("Weights")
{
  using UKF = UKF<Vector<3>, Vector<2>>;
  UKF ukf;

  SECTION("Alpha 1, Beta 2, Kappa 0")
  {
    ukf.weight_coefficients(1.0, 2.0, 0.0);

    const auto mean_weights = ukf.mean_sigma_weights();
    const auto cov_weights = ukf.covariance_sigma_weights();

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
    ukf.weight_coefficients(0.001, 2.0, 0.0);

    const auto mean_weights = ukf.mean_sigma_weights();
    const auto cov_weights = ukf.covariance_sigma_weights();

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
    ukf.weight_coefficients(1.0, 1.0, 1.0);

    const auto mean_weights = ukf.mean_sigma_weights();
    const auto cov_weights = ukf.covariance_sigma_weights();

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
    using UKF = UKF<Vector<3>, Vector<2>>;
    UKF ukf;

    ukf.weight_coefficients(1.0, 1.0, 1.0);
    ukf.state(UKF::State(1, 2, 3));
    ukf.state_covariance(UKF::N_by_N::Identity());
    ukf.generate_sigma_points();
    const auto& sigma_points = ukf.sigma_points();

    REQUIRE(UKF::NUM_SIGMA_POINTS == 7);
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
    using UKF = UKF<UnitComplex, UnitComplex>;
    UKF ukf;

    // No wrapping
    ukf.weight_coefficients(1.0, 1.0, 1.0);
    double init_angle = 1.0; // radians
    ukf.state(UKF::State(init_angle));
    ukf.state_covariance(UKF::N_by_N::Identity());
    ukf.generate_sigma_points();
    const auto& no_wrap_sigma_points = ukf.sigma_points();

    REQUIRE(UKF::NUM_SIGMA_POINTS == 3);
    REQUIRE(no_wrap_sigma_points.size() == UKF::NUM_SIGMA_POINTS);
    CHECK(no_wrap_sigma_points[0].angle() == Approx(init_angle));
    CHECK(no_wrap_sigma_points[1].angle() ==
          Approx(init_angle + std::sqrt(2.0)));
    CHECK(no_wrap_sigma_points[2].angle() ==
          Approx(init_angle - std::sqrt(2.0)));

    // Wrapping
    ukf.weight_coefficients(1.0, 1.0, 1.0);
    init_angle = M_PI - 0.1; // radians
    ukf.state(UKF::State(init_angle));
    ukf.state_covariance(UKF::N_by_N::Identity());
    ukf.generate_sigma_points();
    const auto& sigma_points = ukf.sigma_points();

    REQUIRE(sigma_points.size() == UKF::NUM_SIGMA_POINTS);
    CHECK(sigma_points[0].angle() == Approx(init_angle));
    CHECK(sigma_points[1].angle() ==
          Approx(init_angle + std::sqrt(2.0) - 2 * M_PI));
    CHECK(sigma_points[2].angle() == Approx(init_angle - std::sqrt(2.0)));
  }
}

TEST_CASE("Predict")
{
  SECTION("State is a vector space")
  {
    using UKF = UKF<Vector<3>, Vector<2>>;
    UKF ukf;

    SECTION("Linear system model")
    {
      ukf.weight_coefficients(1.0, 1.0, 1.0);
      ukf.state(UKF::State(1, 2, 3));
      ukf.state_covariance(UKF::N_by_N::Identity());
      ukf.process_covariance(UKF::N_by_N::Identity() * 1e-3);

      // System model adds one to all state elements
      auto sys_model = [](UKF::State& state) { state += UKF::State::Ones(); };

      ukf.predict(sys_model);

      const auto& state = ukf.state();
      const auto& cov = ukf.state_covariance();

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

    SECTION("Nonlinear system model")
    {
      ukf.weight_coefficients(1.0, 1.0, 1.0);
      ukf.state(UKF::State(1, 2, 3));
      ukf.state_covariance(UKF::N_by_N::Identity());
      ukf.process_covariance(UKF::N_by_N::Identity() * 1e-3);

      // System model has a product with state elements
      auto sys_model = [](UKF::State& state) {
        const auto state_in = state;
        state(0) += 0.1 * state_in(1);
        state(1) += 1.0;
        state(2) += 0.01 * state_in(0) * state_in(1);
      };

      ukf.predict(sys_model);

      const auto& state = ukf.state();
      const auto& cov = ukf.state_covariance();

      CHECK(state.x() == Approx(1.2));
      CHECK(state.y() == Approx(3.0));
      CHECK(state.z() == Approx(3.02));

      CHECK(cov(0, 0) == Approx(1.0 + 1e-3 + 0.01));
      CHECK(cov(0, 1) == Approx(0.1));
      CHECK(cov(0, 2) == Approx(0.021));
      CHECK(cov(1, 0) == Approx(0.1));
      CHECK(cov(1, 1) == Approx(1.0 + 1e-3));
      CHECK(cov(1, 2) == Approx(0.01));
      CHECK(cov(2, 0) == Approx(0.021));
      CHECK(cov(2, 1) == Approx(0.01));
      CHECK(cov(2, 2) == Approx(1.0 + 1e-3 + 0.0005));
    }
  }

  // SECTION("State is a not a vector space")
  // {
  //   using UKF = UKF<Vector<3>, 3, Vector<2>, 2>;

  //   SECTION("Linear system model")
  //   {
  //     UKF ukf;

  //     ukf.weight_coefficients(1.0, 1.0, 1.0);
  //     ukf.state(UKF::State(1, 2, 3));
  //     ukf.state_covariance(UKF::N_by_N::Identity());
  //     ukf.process_covariance(UKF::N_by_N::Identity() * 1e-3);

  //     // System model adds one to all state elements
  //     auto sys_model = [](UKF::State& state) { state += UKF::State::Ones();
  //     };

  //     ukf.predict(sys_model);

  //     const auto& state = ukf.state();
  //     const auto& cov = ukf.state_covariance();

  //     CHECK(state.x() == Approx(2.0));
  //     CHECK(state.y() == Approx(3.0));
  //     CHECK(state.z() == Approx(4.0));

  //     CHECK(cov(0, 0) == Approx(1.0 + 1e-3));
  //     CHECK(cov(0, 1) == Approx(0.0));
  //     CHECK(cov(0, 2) == Approx(0.0));
  //     CHECK(cov(1, 0) == Approx(0.0));
  //     CHECK(cov(1, 1) == Approx(1.0 + 1e-3));
  //     CHECK(cov(1, 2) == Approx(0.0));
  //     CHECK(cov(2, 0) == Approx(0.0));
  //     CHECK(cov(2, 1) == Approx(0.0));
  //     CHECK(cov(2, 2) == Approx(1.0 + 1e-3));
  //   }

  //   SECTION("Nonlinear system model")
  //   {
  //     UKF ukf;

  //     ukf.weight_coefficients(1.0, 1.0, 1.0);
  //     ukf.state(UKF::State(1, 2, 3));
  //     ukf.state_covariance(UKF::N_by_N::Identity());
  //     ukf.process_covariance(UKF::N_by_N::Identity() * 1e-3);

  //     // System model has a product with state elements
  //     auto sys_model = [](UKF::State& state) {
  //       const auto state_in = state;
  //       state(0) += 0.1 * state_in(1);
  //       state(1) += 1.0;
  //       state(2) += 0.01 * state_in(0) * state_in(1);
  //     };

  //     ukf.predict(sys_model);

  //     const auto& state = ukf.state();
  //     const auto& cov = ukf.state_covariance();

  //     CHECK(state.x() == Approx(1.2));
  //     CHECK(state.y() == Approx(3.0));
  //     CHECK(state.z() == Approx(3.02));

  //     CHECK(cov(0, 0) == Approx(1.0 + 1e-3 + 0.01));
  //     CHECK(cov(0, 1) == Approx(0.1));
  //     CHECK(cov(0, 2) == Approx(0.021));
  //     CHECK(cov(1, 0) == Approx(0.1));
  //     CHECK(cov(1, 1) == Approx(1.0 + 1e-3));
  //     CHECK(cov(1, 2) == Approx(0.01));
  //     CHECK(cov(2, 0) == Approx(0.021));
  //     CHECK(cov(2, 1) == Approx(0.01));
  //     CHECK(cov(2, 2) == Approx(1.0 + 1e-3 + 0.0005));
  //   }
  // }
}

// SECTION("Custom state mean function")
// {
//   UKF ukf;

//   // Custom state mean function clips the z value of the state to be
//   // between 0 and 1
//   auto custom_state_mean_func = [](const UKF::SigmaPoints& states,
//                                    const UKF::SigmaWeights& weights) {
//     Vector<3> weighted_state = states[0] * weights[0];
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
//   ukf.state_mean_function(custom_state_mean_func);

//   ukf.weight_coefficients(1.0, 1.0, 1.0);
//   ukf.state(UKF::State(1, 2, 0.5));
//   ukf.state_covariance(UKF::N_by_N::Identity());
//   ukf.process_covariance(UKF::N_by_N::Identity() * 1e-3);

//   // System model adds one to all state elements
//   auto sys_model = [](UKF::State& state) {
//     state += UKF::State::Ones();
//   };

//   ukf.predict(sys_model);

//   const auto& state = ukf.state();
//   const auto& cov = ukf.state_covariance();

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
