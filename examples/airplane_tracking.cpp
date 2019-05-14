#include "unscented.hpp"

#include <iostream>
#include <random>

double wrap_angle(double angle)
{
  angle = fmod(angle + M_PI, 2 * M_PI);
  if (angle < 0.0)
  {
    angle += 2 * M_PI;
  }
  return angle - M_PI;
}

using AirplaneState = Eigen::Vector3d;

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
  Eigen::Matrix3d F;
  F << 1.0,  dt, 0.0,
       0.0, 1.0, 0.0,
       0.0, 0.0, 1.0;
  state = F * state;
}

RadarMeasurement measurement_model(const AirplaneState& state)
{
  const auto range = std::sqrt(std::pow(state[0], 2) + std::pow(state[2], 2));
  const auto elevation = std::atan2(state[2], state[0]);
  return {range, elevation};
}

int main()
{
  using UKF = unscented::UKF<AirplaneState, 3, RadarMeasurement, 2>;

  // Simulation parameters
  const auto SIM_DURATION = 360.0;
  const auto DT = 3.0;

  // Create UKF
  UKF ukf;

  // Calculate process noise covariance Q
  const auto PROCESS_VAR = 0.1;
  Eigen::Vector2d G(0.5 * DT * DT, DT);
  Eigen::Matrix3d Q;
  Q.block(0, 0, 1, 1) = G * G.transpose() * PROCESS_VAR;
  Q(2, 2) = 0.1;
  ukf.set_process_covariance(Q);

  // Calculate measurement noise covariance R
  const auto RANGE_STD_DEV = 5.0; // meters
  const auto ELEVATION_STD_DEV = 0.5 * M_PI / 180.0; // radians
  UKF::M_by_M R;
  R << std::pow(RANGE_STD_DEV, 2), 0.0, 0.0, std::pow(ELEVATION_STD_DEV, 2);
  ukf.set_measurement_covariance(R);

  // Set initial state estimate
  ukf.set_state({0, 90, 1100});
  UKF::N_by_N P = UKF::N_by_N::Zero();
  P(0, 0) = std::pow(300.0, 2);
  P(1, 1) = std::pow(30.0, 2);
  P(2, 2) = std::pow(150.0, 2);
  ukf.set_state_covariance(P);

  // Create vectors that will hold the histories of the true and estimated
  // states, populating them both with the initial states
  AirplaneState true_state(0, 100, 1000);
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
  while (sim_time < SIM_DURATION)
  {
    const auto true_position_change =
        (true_state[1] + vel_noise(gen))* DT;
    true_state[0] += true_position_change;
    true_state_history.push_back(true_state);

    ukf.predict(system_model, DT);

    auto meas = measurement_model(true_state);
    meas.range += range_noise(gen);
    meas.elevation += elevation_noise(gen);

    ukf.correct(measurement_model, meas);

    estimated_state_history.push_back(ukf.get_state());
    std::cout << ukf.get_state().transpose() << std::endl;

    sim_time += DT;
  }
}
