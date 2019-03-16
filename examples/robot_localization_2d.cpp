#include "unscented.hpp"

#include <random>
#include <iostream>

float wrapAngle(float angle)
{
  angle = fmod(angle + M_PI, 2 * M_PI);
  if (angle < 0.0f)
  {
    angle += 2 * M_PI;
  }
  return angle - M_PI;
}

struct RobotState
{
  RobotState() = default;

  RobotState(float x, float y, float theta) : x(x), y(y), theta(theta)
  {
  }

  RobotState(const Eigen::Vector3f& vec)
    : x(vec.x()), y(vec.y()), theta(vec.z())
  {
  }

  float x = 0.0f;
  float y = 0.0f;
  float theta = 0.0f;
};

RobotState operator+(const RobotState& rhs, const RobotState& lhs)
{
  return {rhs.x + lhs.x, rhs.y + lhs.y, wrapAngle(rhs.theta + lhs.theta)};
}

Eigen::Vector3f operator-(const RobotState& rhs, const RobotState& lhs)
{
  return Eigen::Vector3f(rhs.x - lhs.x, rhs.y - lhs.y,
                         wrapAngle(rhs.theta - lhs.theta));
}

RobotState operator*(const RobotState& state, float scale)
{
  return RobotState(state.x * scale, state.y * scale, state.theta * scale);
}

std::ostream& operator<<(std::ostream& os, const RobotState& state)
{
  os << "(" << state.x << ", " << state.y << ", " << state.theta << ")";
  return os;
}

struct RobotMeasurement
{
  RobotMeasurement() = default;

  RobotMeasurement(float range, float bearing) : range(range), bearing(bearing)
  {
  }

  float range;
  float bearing;
};

RobotMeasurement operator+(const RobotMeasurement& rhs, const RobotMeasurement& lhs)
{
  return RobotMeasurement(rhs.range + lhs.range,
                          wrapAngle(rhs.bearing + lhs.bearing));
}

Eigen::Vector2f operator-(const RobotMeasurement& rhs,
                          const RobotMeasurement& lhs)
{
  return Eigen::Vector2f(rhs.range - lhs.range,
                         wrapAngle(rhs.bearing - lhs.bearing));
}

RobotMeasurement operator*(const RobotMeasurement& meas, float scale)
{
  return RobotMeasurement(meas.range * scale, meas.bearing * scale);
}

void systemModel(RobotState& state, float velocity, float angular_velocity, float T)
{
  state.x += velocity * std::cos(state.theta) * T;
  state.y += velocity * std::sin(state.theta) * T;
  state.theta = wrapAngle(state.theta + angular_velocity * T);
}

RobotMeasurement measurementModel(const RobotState& state, const Eigen::Vector2f& landmark)
{
  const auto expected_range = std::sqrt(std::pow(landmark.x() - state.x, 2) +
                                        std::pow(landmark.y() - state.y, 2));
  const auto expected_bearing =
      std::atan2(landmark.y() - state.y, landmark.x() - state.x);
  return RobotMeasurement(expected_range, expected_bearing);
}

int main()
{
  using UKF = unscented::UKF<RobotState, 3, RobotMeasurement, 2>;

  // Setup the simulation
  const auto SIM_DURATION = 5.0f; // total simulation duration
  const auto T = 0.02f; // time between prediction steps (i.e., input period)
  const auto MEAS_PERIOD = 0.1f; // time between measurements
  const std::size_t NUM_LANDMARKS = 4; // total number of landmarks
  const std::array<Eigen::Vector2f, NUM_LANDMARKS> LANDMARKS = {{
      {1.0f, 4.0f},
      {2.0f, 0.5f},
      {2.5f, 3.5f},
      {4.0f, 2.0f}}}; // positions of landmarks
  const auto VEL_STD_DEV = 0.1f;
  const auto ANG_VEL_STD_DEV = 0.071f;
  UKF::State state_true(1.0f, 1.0f, 0.0f);

  // Function to simulate a range/bearing measurement given the current robot
  // state and a landmark
  auto simulate_measurement = [](const RobotState& state,
                                 const Eigen::Vector2f& landmark) {
    const auto range = std::sqrt(std::pow(landmark.x() - state.x, 2.0f) +
                                 std::pow(landmark.y() - state.y, 2.0f));
    const auto bearing =
        wrapAngle(std::atan2(landmark.y() - state.y, landmark.x() - state.x) -
                  state.theta);
    return RobotMeasurement(range, bearing);
  };

  // Setup the UKF (initial state, state covariance, system covariance,
  // measurement covariance)
  UKF ukf;
  UKF::State state_init(1.1f, 0.9f, M_PI / 12);
  ukf.setState(state_init);
  UKF::N_by_N P_init;
  P_init << 0.500, 0.000, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.000, 0.250;
  ukf.setStateCovariance(P_init);
  UKF::N_by_N Q;
  Q << 0.002, 0.000, 0.000,
       0.000, 0.002, 0.000,
       0.000, 0.000, 0.002;
  ukf.setProcessCovariance(Q);
  UKF::M_by_M R;
  R << 0.063, 0.000,
       0.000, 0.007;
  ukf.setMeasurementCovariance(R);

  // Setup random number generation
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<float> vel_noise(0.0f, VEL_STD_DEV);
  std::normal_distribution<float> ang_vel_noise(0.0f, ANG_VEL_STD_DEV);
  std::normal_distribution<float> range_noise(0.0f, std::sqrt(R(0, 0)));
  std::normal_distribution<float> bearing_noise(0.0f, std::sqrt(R(1, 1)));

  // Run the simulation
  float sim_time = 0.0f;
  float last_meas_time = 0.0f;
  while (sim_time < SIM_DURATION)
  {
    // Get the simulated (true) inputs
    const auto velocity = 0.5f;
    const auto angular_velocity = 0.01f;

    // Move the true state forward in time
    systemModel(state_true, velocity, angular_velocity, T);

    // Get the noisy inputs by perturbing the true inputs
    const auto velocity_noisy = velocity + vel_noise(gen);
    const auto angular_velocity_noisy = angular_velocity + ang_vel_noise(gen);

    // Predict the estimated state forward in time
    ukf.predict(systemModel, velocity_noisy, angular_velocity_noisy, T);
    std::cout << state_true << " " << ukf.getState() << " \n";

    // Check if it is time for a measurent
    if (sim_time - last_meas_time >= MEAS_PERIOD)
    {
      for (const auto& landmark : LANDMARKS)
      {
        // Get the simulated (true) measurement
        const auto& measurement = simulate_measurement(state_true, landmark);

        // Get the noisy measurement by perturbing the true measurement
        RobotMeasurement measurement_noisy(
            measurement.range + range_noise(gen),
            measurement.bearing + bearing_noise(gen));

        // Correct the estimated state
        ukf.correct(measurementModel, measurement_noisy, landmark);
      }

      last_meas_time = sim_time;
    }

    // Move the simulation time forward
    sim_time += T;
  }
}

