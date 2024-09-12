#include "matplot/freestanding/axes_functions.h"
#include "unscented/primitives.h"
#include "unscented/ukf.h"

#include <iostream>
#include <random>
#include <matplot/matplot.h>

//////////////////////////////////////////////////////////////////////////////
// Set up the state
//////////////////////////////////////////////////////////////////////////////

using State = unscented::Vector<4>;

enum StateElements
{
  POSITION = 0,
  VELOCITY,
  ALTITUDE,
  CLIMB_RATE
};

//////////////////////////////////////////////////////////////////////////////
// Set up the system model
//////////////////////////////////////////////////////////////////////////////

void system_model(State& state, double dt)
{
  Eigen::Matrix4d F;
  // clang-format off
  F << 1.0,  dt, 0.0, 0.0, 
       0.0, 1.0, 0.0, 0.0, 
       0.0, 0.0, 1.0,  dt, 
       0.0, 0.0, 0.0, 1.0;
  // clang-format on
  state = F * state;
}

//////////////////////////////////////////////////////////////////////////////
// Set up the measurement
//////////////////////////////////////////////////////////////////////////////

using Range = unscented::Scalar;
using Elevation = unscented::Angle;
using Measurement = unscented::Compound<Range, Elevation>;

//////////////////////////////////////////////////////////////////////////////
// Set up the measurement model
//////////////////////////////////////////////////////////////////////////////

Measurement measurement_model(const State& state)
{
  Measurement meas;
  meas.data = {
      std::sqrt(std::pow(state[POSITION], 2) + std::pow(state[ALTITUDE], 2)),
      unscented::Angle(std::atan2(state[ALTITUDE], state[POSITION]))};
  return meas;
}

int main()
{
  //////////////////////////////////////////////////////////////////////////////
  // Set up the filter
  //////////////////////////////////////////////////////////////////////////////

  // The airplane state has four degrees of freedom (position, velocity,
  // altitude, climb rate) and the radar measurement has two degrees of freedom
  // (range, elevation)
  using UKF = unscented::UKF<State, Measurement>;
  UKF ukf;

  // Simulation parameters
  const auto SIM_DURATION = 360.0; // seconds
  const auto DT = 3.0; // seconds

  // Calculate process noise covariance Q using the discrete constant white
  // noise model (Bar-Shalom. “Estimation with Applications To Tracking and
  // Navigation”. John Wiley & Sons, 2001. Page 274.). To simplify, one can
  // simply pick (or tune) appropriate values on the diagonal of Q.
  const auto PROCESS_VAR = 100;
  unscented::Vector<2> G(0.5 * DT * DT, DT);
  UKF::N_by_N Q;
  Q.block<2, 2>(0, 0) = G * G.transpose() * PROCESS_VAR;
  Q.block<2, 2>(2, 2) = G * G.transpose() * PROCESS_VAR;
  ukf.set_process_covariance(Q);

  // Calculate measurement noise covariance R (standard deviations chosen
  // somewhat arbitrarily)
  const auto RANGE_STD_DEV = 5.0; // meters
  const auto ELEVATION_STD_DEV = 0.5 * M_PI / 180.0; // radians
  UKF::M_by_M R;
  R << std::pow(RANGE_STD_DEV, 2), 0.0, 0.0, std::pow(ELEVATION_STD_DEV, 2);
  ukf.set_measurement_covariance(R);

  // Set initial state estimate and its covariance
  State true_state(0, 100, 1000, 0);
  State initial_state_estimate(0, 90, 1100, 0);
  ukf.set_state(initial_state_estimate);
  UKF::N_by_N P;
  P(0, 0) = std::pow(300.0, 2); // m^2
  P(1, 1) = std::pow(30.0, 2); // (m/s)^2
  P(2, 2) = std::pow(150.0, 2); // m^2
  P(3, 3) = std::pow(3.0, 2); // (m/s)^2
  ukf.set_state_covariance(P);

  //////////////////////////////////////////////////////////////////////////////
  // Set up the simulation
  //////////////////////////////////////////////////////////////////////////////

  // Create vectors to store the state errors and standard deviations over time
  // to be used for plotting at the end of the simulation
  std::vector<double> position_errors;
  std::vector<double> velocity_errors;
  std::vector<double> altitude_errors;
  std::vector<double> climb_rate_errors;
  std::vector<double> position_std_devs;
  std::vector<double> velocity_std_devs;
  std::vector<double> altitude_std_devs;
  std::vector<double> climb_rate_std_devs;

  // Setup random number generation
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<double> vel_noise(0.0, 10.0);
  std::normal_distribution<double> range_noise(0.0, RANGE_STD_DEV);
  std::normal_distribution<double> elevation_noise(0.0, ELEVATION_STD_DEV);

  //////////////////////////////////////////////////////////////////////////////
  // Run the simulation
  //////////////////////////////////////////////////////////////////////////////

  double sim_time = 0.0;
  std::vector<double> sim_time_history;
  while (sim_time <= SIM_DURATION)
  {
    // One minute into the simulation, set a non-zero climb rate
    if (sim_time > 60.0)
    {
      true_state[CLIMB_RATE] = 300.0 / 60; // 300 m/min
    }

    // Update the true position and altitude, adding some noise to the velocity
    // to make up for deficiencies in the constant velocity model (i.e., it's
    // unlikely the velocity was constant throughout the full time step)
    true_state[POSITION] += (true_state[VELOCITY] + vel_noise(gen)) * DT;
    true_state[ALTITUDE] += (true_state[CLIMB_RATE] + vel_noise(gen)) * DT;

    // Simulate a measurement based on the true state
    auto meas = measurement_model(true_state);
    auto& [r, e] = meas.data;
    r.value += range_noise(gen);
    e = unscented::Angle(e.get_angle() + elevation_noise(gen));

    // Update the filter estimates
    ukf.predict(system_model, DT);
    ukf.correct(measurement_model, meas);

    const auto& sp = ukf.get_measurement_sigma_points();
    const auto& y_hat = ukf.get_expected_measurement();
    const auto& P_yy = ukf.get_measurement_covariance();
    const auto& P_xy = ukf.get_cross_covariance();
    const auto& K = ukf.get_kalman_gain();
    const auto& inn = ukf.get_innovation();
    const auto& P = ukf.get_state_covariance();

    // Record all the current estimated values in the history
    const auto& est_state = ukf.get_state();
    position_errors.push_back(est_state[POSITION] - true_state[POSITION]);
    velocity_errors.push_back(est_state[VELOCITY] - true_state[VELOCITY]);
    altitude_errors.push_back(est_state[ALTITUDE] - true_state[ALTITUDE]);
    climb_rate_errors.push_back(est_state[CLIMB_RATE] - true_state[CLIMB_RATE]);
    const auto& est_cov = ukf.get_state_covariance();
    position_std_devs.push_back(std::sqrt(est_cov(POSITION, POSITION)));
    velocity_std_devs.push_back(std::sqrt(est_cov(VELOCITY, VELOCITY)));
    altitude_std_devs.push_back(std::sqrt(est_cov(ALTITUDE, ALTITUDE)));
    climb_rate_std_devs.push_back(std::sqrt(est_cov(CLIMB_RATE, CLIMB_RATE)));
    sim_time_history.push_back(sim_time);

    // Move time forward
    sim_time += DT;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Plot the results
  //////////////////////////////////////////////////////////////////////////////

  // Create +/- 2 std dev vectors
  const auto num_std_devs = 2;
  const auto num_pts = position_std_devs.size();
  std::vector<double> position_plus_2_std_devs(num_pts);
  std::vector<double> position_minus_2_std_devs(num_pts);
  std::vector<double> velocity_plus_2_std_devs(num_pts);
  std::vector<double> velocity_minus_2_std_devs(num_pts);
  std::vector<double> altitude_plus_2_std_devs(num_pts);
  std::vector<double> altitude_minus_2_std_devs(num_pts);
  std::vector<double> climb_rate_plus_2_std_devs(num_pts);
  std::vector<double> climb_rate_minus_2_std_devs(num_pts);
  for (std::size_t i = 0; i < num_pts; ++i)
  {
    position_plus_2_std_devs[i] = num_std_devs * position_std_devs[i];
    position_minus_2_std_devs[i] = -num_std_devs * position_std_devs[i];
    velocity_plus_2_std_devs[i] = num_std_devs * velocity_std_devs[i];
    velocity_minus_2_std_devs[i] = -num_std_devs * velocity_std_devs[i];
    altitude_plus_2_std_devs[i] = num_std_devs * altitude_std_devs[i];
    altitude_minus_2_std_devs[i] = -num_std_devs * altitude_std_devs[i];
    climb_rate_plus_2_std_devs[i] = num_std_devs * climb_rate_std_devs[i];
    climb_rate_minus_2_std_devs[i] = -num_std_devs * climb_rate_std_devs[i];
  }

  // Position errors
  matplot::figure();
  matplot::plot(sim_time_history, position_errors, "b");
  matplot::hold(true);
  matplot::plot(sim_time_history, position_plus_2_std_devs, "r--");
  matplot::plot(sim_time_history, position_minus_2_std_devs, "r--");
  matplot::title("Position Error");
  matplot::xlabel("Time (s)");
  matplot::ylabel("Error (m)");

  // Velocity errors
  matplot::figure();
  matplot::plot(sim_time_history, velocity_errors, "b");
  matplot::hold(true);
  matplot::plot(sim_time_history, velocity_plus_2_std_devs, "r--");
  matplot::plot(sim_time_history, velocity_minus_2_std_devs, "r--");
  matplot::title("Velocity Error");
  matplot::xlabel("Time (s)");
  matplot::ylabel("Error (m/s)");

  // Altitude errors
  matplot::figure();
  matplot::plot(sim_time_history, altitude_errors, "b");
  matplot::hold(true);
  matplot::plot(sim_time_history, altitude_plus_2_std_devs, "r--");
  matplot::plot(sim_time_history, altitude_minus_2_std_devs, "r--");
  matplot::title("Altitude Error");
  matplot::xlabel("Time (s)");
  matplot::ylabel("Error (m)");

  // Climb rate errors
  matplot::figure();
  matplot::plot(sim_time_history, climb_rate_errors, "b");
  matplot::hold(true);
  matplot::plot(sim_time_history, climb_rate_plus_2_std_devs, "r--");
  matplot::plot(sim_time_history, climb_rate_minus_2_std_devs, "r--");
  matplot::title("Climb Rate Error");
  matplot::xlabel("Time (m/s)");
  matplot::ylabel("Error (m)");

  matplot::show();

  return 0;
}
