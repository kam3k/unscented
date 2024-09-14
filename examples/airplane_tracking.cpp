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
  const auto expected_range = 
      std::sqrt(std::pow(state[POSITION], 2) + std::pow(state[ALTITUDE], 2));
  const auto expected_elevation = 
      unscented::Angle(std::atan2(state[ALTITUDE], state[POSITION]));
  return {expected_range, expected_elevation};
}

int main()
{
  //////////////////////////////////////////////////////////////////////////////
  // Set up the filter
  //////////////////////////////////////////////////////////////////////////////

  // Initialize the UKF and set the weights
  using UKF = unscented::UKF<State, Measurement>;
  UKF ukf;
  ukf.set_weight_coefficients(0.1, 2.0, -1.0);

  // Simulation parameters
  const auto SIM_DURATION = 360.0; // seconds
  const auto DT = 3.0; // seconds

  // Calculate process noise covariance Q using the discrete constant white
  // noise model (Bar-Shalom. “Estimation with Applications To Tracking and
  // Navigation”. John Wiley & Sons, 2001. Page 274.). To simplify, one can
  // simply pick (or tune) appropriate values on the diagonal of Q.
  const auto PROCESS_VAR = 0.1;
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

  // Create vectors to store the state and state errors and standard deviations
  // over time to be used for plotting at the end of the simulation
  std::vector<double> true_positions;
  std::vector<double> true_velocities;
  std::vector<double> true_altitudes;
  std::vector<double> true_climb_rates;
  std::vector<double> est_positions;
  std::vector<double> est_velocities;
  std::vector<double> est_altitudes;
  std::vector<double> est_climb_rates;
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
  std::normal_distribution<double> vel_noise(0.0, 0.02);
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
    auto& [range, elevation] = meas.data;
    range.value += range_noise(gen);
    elevation = unscented::Angle(elevation.get_angle() + elevation_noise(gen));

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
    true_positions.push_back(true_state[POSITION]);
    true_velocities.push_back(true_state[VELOCITY]);
    true_altitudes.push_back(true_state[ALTITUDE]);
    true_climb_rates.push_back(true_state[CLIMB_RATE]);
    est_positions.push_back(est_state[POSITION]);
    est_velocities.push_back(est_state[VELOCITY]);
    est_altitudes.push_back(est_state[ALTITUDE]);
    est_climb_rates.push_back(est_state[CLIMB_RATE]);
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

  // Create +/- n std dev vectors
  const auto num_std_devs = 2;
  const auto num_pts = position_std_devs.size();
  std::vector<double> position_positive_n_std_devs(num_pts);
  std::vector<double> position_negative_n_std_devs(num_pts);
  std::vector<double> velocity_positive_n_std_devs(num_pts);
  std::vector<double> velocity_negative_n_std_devs(num_pts);
  std::vector<double> altitude_positive_n_std_devs(num_pts);
  std::vector<double> altitude_negative_n_std_devs(num_pts);
  std::vector<double> climb_rate_positive_n_std_devs(num_pts);
  std::vector<double> climb_rate_negative_n_std_devs(num_pts);
  for (std::size_t i = 0; i < num_pts; ++i)
  {
    position_positive_n_std_devs[i] = num_std_devs * position_std_devs[i];
    position_negative_n_std_devs[i] = -num_std_devs * position_std_devs[i];
    velocity_positive_n_std_devs[i] = num_std_devs * velocity_std_devs[i];
    velocity_negative_n_std_devs[i] = -num_std_devs * velocity_std_devs[i];
    altitude_positive_n_std_devs[i] = num_std_devs * altitude_std_devs[i];
    altitude_negative_n_std_devs[i] = -num_std_devs * altitude_std_devs[i];
    climb_rate_positive_n_std_devs[i] = num_std_devs * climb_rate_std_devs[i];
    climb_rate_negative_n_std_devs[i] = -num_std_devs * climb_rate_std_devs[i];
  }
  
  // Position
  matplot::figure();
  matplot::plot(sim_time_history, true_positions, "k");
  matplot::hold(true);
  matplot::plot(sim_time_history, est_positions, "b");
  matplot::title("Position");
  matplot::xlabel("Time (s)");
  matplot::ylabel("Position (m)");
  
  // Velocity
  matplot::figure();
  matplot::plot(sim_time_history, true_velocities, "k");
  matplot::hold(true);
  matplot::plot(sim_time_history, est_velocities, "b");
  matplot::title("Velocity");
  matplot::xlabel("Time (s)");
  matplot::ylabel("Velocities (m/s)");
  
  // Altitude
  matplot::figure();
  matplot::plot(sim_time_history, true_altitudes, "k");
  matplot::hold(true);
  matplot::plot(sim_time_history, est_altitudes, "b");
  matplot::title("Altitude");
  matplot::xlabel("Time (s)");
  matplot::ylabel("Altitude (m)");
  
  // Climb rate
  matplot::figure();
  matplot::plot(sim_time_history, true_climb_rates, "k");
  matplot::hold(true);
  matplot::plot(sim_time_history, est_climb_rates, "b");
  matplot::title("Climb rate");
  matplot::xlabel("Time (s)");
  matplot::ylabel("Climb rate (m/s)");

  // Position errors
  matplot::figure();
  matplot::plot(sim_time_history, position_errors, "b");
  matplot::hold(true);
  matplot::plot(sim_time_history, position_positive_n_std_devs, "r--");
  matplot::plot(sim_time_history, position_negative_n_std_devs, "r--");
  matplot::title("Position Error");
  matplot::xlabel("Time (s)");
  matplot::ylabel("Error (m)");

  // Velocity errors
  matplot::figure();
  matplot::plot(sim_time_history, velocity_errors, "b");
  matplot::hold(true);
  matplot::plot(sim_time_history, velocity_positive_n_std_devs, "r--");
  matplot::plot(sim_time_history, velocity_negative_n_std_devs, "r--");
  matplot::title("Velocity Error");
  matplot::xlabel("Time (s)");
  matplot::ylabel("Error (m/s)");

  // Altitude errors
  matplot::figure();
  matplot::plot(sim_time_history, altitude_errors, "b");
  matplot::hold(true);
  matplot::plot(sim_time_history, altitude_positive_n_std_devs, "r--");
  matplot::plot(sim_time_history, altitude_negative_n_std_devs, "r--");
  matplot::title("Altitude Error");
  matplot::xlabel("Time (s)");
  matplot::ylabel("Error (m)");

  // Climb rate errors
  matplot::figure();
  matplot::plot(sim_time_history, climb_rate_errors, "b");
  matplot::hold(true);
  matplot::plot(sim_time_history, climb_rate_positive_n_std_devs, "r--");
  matplot::plot(sim_time_history, climb_rate_negative_n_std_devs, "r--");
  matplot::title("Climb Rate Error");
  matplot::xlabel("Time (m/s)");
  matplot::ylabel("Error (m)");

  matplot::show();

  return 0;
}
