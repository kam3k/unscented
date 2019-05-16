#include "unscented.hpp"

#include "matplotlibcpp.h"

#include <iostream>
#include <random>

namespace plt = matplotlibcpp;

double wrap_angle(double angle)
{
  angle = fmod(angle + M_PI, 2 * M_PI);
  if (angle < 0.0)
  {
    angle += 2 * M_PI;
  }
  return angle - M_PI;
}

using AirplaneState = Eigen::Vector4d;
enum StateElements
{
  POSITION = 0,
  VELOCITY,
  ALTITUDE,
  CLIMB_RATE
};

struct RadarMeasurement
{
  RadarMeasurement() = default;
  RadarMeasurement(double r, double e) : range(r), elevation(e) {}
  double range;
  double elevation;
};

RadarMeasurement operator+(const RadarMeasurement& lhs,
                           const RadarMeasurement& rhs)
{
  return RadarMeasurement(lhs.range + rhs.range,
                          wrap_angle(lhs.elevation + rhs.elevation));
}

Eigen::Vector2d operator-(const RadarMeasurement& lhs,
                          const RadarMeasurement& rhs)
{
  return Eigen::Vector2d(lhs.range - rhs.range,
                         wrap_angle(lhs.elevation - rhs.elevation));
}

RadarMeasurement operator*(const RadarMeasurement& meas, double scale)
{
  return RadarMeasurement(meas.range * scale, meas.elevation * scale);
}

void system_model(AirplaneState& state, double dt)
{
  Eigen::Matrix4d F;
  F << 1.0,  dt, 0.0, 0.0,
       0.0, 1.0, 0.0, 0.0,
       0.0, 0.0, 1.0,  dt,
       0.0, 0.0, 0.0, 1.0;
  state = F * state;
}

RadarMeasurement measurement_model(const AirplaneState& state)
{
  const auto range =
      std::sqrt(std::pow(state[POSITION], 2) + std::pow(state[ALTITUDE], 2));
  const auto elevation = std::atan2(state[ALTITUDE], state[POSITION]);
  return {range, elevation};
}

int main()
{
  using UKF = unscented::UKF<AirplaneState, 4, RadarMeasurement, 2>;

  // Simulation parameters
  const auto SIM_DURATION = 360.0;
  const auto DT = 3.0;

  // Create UKF
  UKF ukf;

  // Calculate process noise covariance Q
  const auto PROCESS_VAR = 0.1;
  Eigen::Vector2d G(0.5 * DT * DT, DT);
  Eigen::Matrix4d Q;
  Q.block(0, 0, 1, 1) = G * G.transpose() * PROCESS_VAR;
  Q.block(2, 2, 3, 3) = G * G.transpose() * PROCESS_VAR;
  ukf.set_process_covariance(Q);

  // Calculate measurement noise covariance R
  const auto RANGE_STD_DEV = 5.0; // meters
  const auto ELEVATION_STD_DEV = 0.5 * M_PI / 180.0; // radians
  UKF::M_by_M R;
  R << std::pow(RANGE_STD_DEV, 2), 0.0, 0.0, std::pow(ELEVATION_STD_DEV, 2);
  ukf.set_measurement_covariance(R);

  // Set initial state estimate
  ukf.set_state({0, 90, 1100, 0});
  UKF::N_by_N P = UKF::N_by_N::Zero();
  P(0, 0) = std::pow(300.0, 2);
  P(1, 1) = std::pow(30.0, 2);
  P(2, 2) = std::pow(150.0, 2);
  P(3, 3) = std::pow(3.0, 2);
  ukf.set_state_covariance(P);

  // Create vectors that will hold the histories of the true and estimated
  // states, populating them both with the initial states
  AirplaneState true_state(0, 100, 1000, 0);
  std::vector<AirplaneState> true_state_history;
  true_state_history.push_back(true_state);
  std::vector<AirplaneState> estimated_state_history;
  estimated_state_history.push_back(ukf.get_state());

  // Setup random number generation
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<double> vel_noise(0.0, 0.02);
  std::normal_distribution<double> range_noise(0.0, RANGE_STD_DEV);
  std::normal_distribution<double> elevation_noise(0.0, ELEVATION_STD_DEV);

  // Run the simulation
  double sim_time = 0.0;
  std::vector<double> sim_time_history = {sim_time};
  while (sim_time < SIM_DURATION)
  {
    // One minute into the simulation, set a non-zero climb rate
    if (sim_time > 60.0)
    {
      true_state[CLIMB_RATE] = 300.0/60; // 300 m/min 
    }

    true_state[POSITION] += (true_state[VELOCITY] + vel_noise(gen)) * DT;
    true_state[ALTITUDE] += (true_state[CLIMB_RATE] + vel_noise(gen))* DT;
    true_state_history.push_back(true_state);

    ukf.predict(system_model, DT);

    auto meas = measurement_model(true_state);
    meas.range += range_noise(gen);
    meas.elevation += elevation_noise(gen);
    std::cout << meas.range << " " << meas.elevation << std::endl;

    ukf.correct(measurement_model, meas);

    estimated_state_history.push_back(ukf.get_state());

    sim_time += DT;
    sim_time_history.push_back(sim_time);
  }

  // Plot results
  std::vector<double> true_positions;
  std::vector<double> true_velocities;
  std::vector<double> true_altitudes;
  std::vector<double> true_climb_rates;
  for (const auto& state : true_state_history)
  {
    true_positions.push_back(state[POSITION]);
    true_velocities.push_back(state[VELOCITY]);
    true_altitudes.push_back(state[ALTITUDE]);
    true_climb_rates.push_back(state[CLIMB_RATE]);
  }
  std::vector<double> estimated_positions;
  std::vector<double> estimated_velocities;
  std::vector<double> estimated_altitudes;
  std::vector<double> estimated_climb_rates;
  for (const auto& state : estimated_state_history)
  {
    estimated_positions.push_back(state[POSITION]);
    estimated_velocities.push_back(state[VELOCITY]);
    estimated_altitudes.push_back(state[ALTITUDE]);
    estimated_climb_rates.push_back(state[CLIMB_RATE]);
  }

  plt::subplot(4, 1, 1);
  plt::plot(sim_time_history, true_positions, "k-", sim_time_history, estimated_positions, "r--");
  plt::subplot(4, 1, 2);
  plt::plot(sim_time_history, true_velocities, "k-", sim_time_history, estimated_velocities, "r--");
  // plt::ylim(96, 102);
  plt::subplot(4, 1, 3);
  plt::plot(sim_time_history, true_altitudes, "k-", sim_time_history, estimated_altitudes, "r--");
  // plt::ylim(960, 1020);
  plt::subplot(4, 1, 4);
  plt::plot(sim_time_history, true_climb_rates, "k-", sim_time_history, estimated_climb_rates, "r--");
  plt::show();
}
