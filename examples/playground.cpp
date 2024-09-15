#include "unscented/primitives.h"
#include "unscented/ukf.h"

int main()
{
  // Define state type
  using State = unscented::Vector<4>; // position, velocity, altitude, climb rate

  // Define measurement type
  using Range = unscented::Scalar; // floating point
  using Elevation = unscented::Angle; // -pi to pi
  using Measurement = unscented::Compound<unscented::Scalar, unscented::Angle>;

  // Initialize UKF
  using UKF = unscented::UKF<State, Measurement>;
  UKF ukf;

  // Set initial state and its covariance
  State state{0.0, 100.0, 2000.0, -5.0};
  ukf.set_state(state);
  UKF::N_by_N P;
  P << 10.0, 0.00, 0.00, 0.00,
       0.00, 5.00, 0.00, 0.00,
       0.00, 0.00, 25.0, 0.00,
       0.00, 0.00, 0.00, 0.50;
  ukf.set_state_covariance(P);

  // Set up process and measurement covariance
  UKF::N_by_N Q;
  Q << 0.20, 0.14, 0.00, 0.00,
       0.14, 0.09, 0.00, 0.00,
       0.00, 0.00, 0.20, 0.14,
       0.00, 0.00, 0.14, 0.09;
  ukf.set_process_covariance(Q);
  UKF::M_by_M R;
  R << 10.0, 0.00,
       0.00, 0.10;

  // Set up system model
  auto system_model = [](State& state, double dt) 
  {
    Eigen::Matrix4d F;
    F << 1.0,  dt, 0.0, 0.0, 
         0.0, 1.0, 0.0, 0.0, 
         0.0, 0.0, 1.0,  dt, 
         0.0, 0.0, 0.0, 1.0;
    state = F * state;
  };

  // Set up measurement model
  auto measurement_model = [](const State& state)
  {
    const auto expected_range =
        std::sqrt(std::pow(state[0], 2) + std::pow(state[2], 2));
    const auto expected_elevation =
        unscented::Angle(std::atan2(state[2], state[0]));
    return Measurement{expected_range, expected_elevation};
  };

  // Run the filter
  // Get current timestep (dt) from clock, measurement (meas) from sensor
  // ...

  // Predict with system model
  // ukf.predict(system_model, dt);

  // If desired...
  // auto a_priori_state = ukf.get_state();
  // auto a_priori_cov = ukf.get_state_covariance();
  // auto sigma_points = ukf.get_sigma_points();

  // Correct with measurement model
  // ukf.correct(measurement_model, meas);

  // If desired...
  // auto a_posteriori_state = ukf.get_state();
  // auto a_posteriori_cov = ukf.get_state_covariance();
  // auto exp_meas = ukf.get_expected_measurement();
  // auto exp_meas_cov = ukf.get_expected_measurement_covariance();
  // auto cross_cov = ukf.get_cross_covariance();
  // auto K = ukf.get_kalman_gain();
  // auto innovation = ukf.get_innovation();
  // auto meas_sigma_points = ukf.get_measurement_sigma_points();

  return 0;
}
