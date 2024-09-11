#include "unscented/primitives.h"
#include "unscented/ukf.h"

#include <iostream>
#include <random>
#include <utility>

using State = unscented::Vector<4>;

enum StateElements
{
  POSITION = 0,
  VELOCITY,
  ALTITUDE,
  CLIMB_RATE
};

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

using Range = unscented::Scalar;
using Elevation = unscented::Angle;
using Measurement = unscented::Compound<Range, Elevation>;

Measurement measurement_model(const State& state)
{
  Measurement meas;
  meas.data = {
      std::sqrt(std::pow(state[POSITION], 2) + std::pow(state[ALTITUDE], 2)),
      unscented::Angle(std::atan2(state[ALTITUDE], state[POSITION]))};
  return meas;
}

int main() {
  //////////////////////////////////////////////////////////////////////////////
  // Setup the filter
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

  // Create vectors that will hold the histories of the true and estimated
  // states, populating them both with the initial states. This is simply
  // recorded so the results over time can be plotted at the end of the
  // simulation.
  std::vector<State> true_state_history;
  true_state_history.push_back(true_state);
  std::vector<State> estimated_state_history;
  estimated_state_history.push_back(ukf.get_state());
  std::vector<UKF::N_by_N> estimated_state_cov_history;

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
    true_state_history.push_back(true_state);

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

    // Record all the current values in the history
    estimated_state_history.push_back(ukf.get_state());
    estimated_state_cov_history.push_back(ukf.get_state_covariance());
    sim_time_history.push_back(sim_time);

    // Move time forward
    sim_time += DT;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Output the results
  //////////////////////////////////////////////////////////////////////////////

  std::cout << "time,true_pos,est_pos,est_pos_std_dev,true_vel,est_vel,est_vel_"
               "std_dev,true_alt,est_alt,est_alt_std_dev,true_climb,est_climb,"
               "est_climb_std_dev\n";
  for (auto i = 0; i < sim_time_history.size(); ++i)
  {
    const auto& timestamp = sim_time_history[i];
    const auto& true_state = true_state_history[i];
    const auto& est_state = estimated_state_history[i];
    const auto& est_state_cov = estimated_state_cov_history[i];
    std::cout << timestamp << "," << true_state[POSITION] << ","
              << est_state[POSITION] << ","
              << std::sqrt(est_state_cov(POSITION, POSITION)) << ","
              << true_state[VELOCITY] << "," << est_state[VELOCITY] << ","
              << std::sqrt(est_state_cov(VELOCITY, VELOCITY)) << ","
              << true_state[ALTITUDE] << "," << est_state[ALTITUDE] << ","
              << std::sqrt(est_state_cov(ALTITUDE, ALTITUDE)) << ","
              << true_state[CLIMB_RATE] << "," << est_state[CLIMB_RATE] << ","
              << std::sqrt(est_state_cov(CLIMB_RATE, CLIMB_RATE)) << "\n";
  }
 return 0;
 }

// int main() {
//   using MState = unscented::Compound<unscented::Vector<2>, unscented::Angle>;
//   MState t;
//   t.data =
//       std::make_tuple(unscented::Vector<2>{-0.1, 0.2}, unscented::Angle{0.2});
//   std::cout << std::get<0>(t.data).transpose() << " "
//             << " a: " << std::get<1>(t.data).get_angle() << "\n";
//   auto x = operator+(t, unscented::Vector<3>{1.0, 2.0, 3.0});
//   // auto x = t + unscented::Vector<3>{1.0, 2.0, 3.0};
//   std::cout << std::get<0>(x.data).transpose() << " "
//             << " a: " << std::get<1>(x.data).get_angle() << "\n";
//   auto v = x - t;
//   std::cout << "v: " << v.transpose() << "\n";

// return 0;

// }
