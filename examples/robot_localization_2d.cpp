#include "unscented/ukf.hpp"

#include <iostream>
#include <random>

double wrapAngle(double angle)
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

  RobotState(double x, double y, double theta) : x(x), y(y), theta(theta)
  {
  }

  RobotState(const Eigen::Vector3d& vec)
    : x(vec.x()), y(vec.y()), theta(vec.z())
  {
  }

  double x = 0.0;
  double y = 0.0;
  double theta = 0.0;
};

RobotState operator+(const RobotState& rhs, const RobotState& lhs)
{
  return {rhs.x + lhs.x, rhs.y + lhs.y, wrapAngle(rhs.theta + lhs.theta)};
}

Eigen::Vector3d operator-(const RobotState& rhs, const RobotState& lhs)
{
  return Eigen::Vector3d(rhs.x - lhs.x, rhs.y - lhs.y,
                         wrapAngle(rhs.theta - lhs.theta));
}

RobotState operator*(const RobotState& state, double scale)
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

  RobotMeasurement(double range, double bearing)
    : range(range), bearing(bearing)
  {
  }

  double range;
  double bearing;
};

RobotMeasurement operator+(const RobotMeasurement& rhs,
                           const RobotMeasurement& lhs)
{
  return RobotMeasurement(rhs.range + lhs.range,
                          wrapAngle(rhs.bearing + lhs.bearing));
}

Eigen::Vector2d operator-(const RobotMeasurement& rhs,
                          const RobotMeasurement& lhs)
{
  return Eigen::Vector2d(rhs.range - lhs.range,
                         wrapAngle(rhs.bearing - lhs.bearing));
}

RobotMeasurement operator*(const RobotMeasurement& meas, double scale)
{
  return RobotMeasurement(meas.range * scale, meas.bearing * scale);
}

std::ostream& operator<<(std::ostream& os, const RobotMeasurement& meas)
{
  os << "(" << meas.range << ", " << meas.bearing << ")";
  return os;
}

void systemModel(RobotState& state, double velocity, double angular_velocity,
                 double T)
{
  state.x += velocity * std::cos(state.theta) * T;
  state.y += velocity * std::sin(state.theta) * T;
  state.theta = wrapAngle(state.theta + angular_velocity * T);
}

RobotMeasurement measurementModel(const RobotState& state,
                                  const Eigen::Vector2d& landmark)
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
  const auto SIM_DURATION = 5.0; // total simulation duration
  const auto T = 0.02; // time between prediction steps (i.e., input period)
  const auto MEAS_PERIOD = 0.1; // time between measurements
  const std::size_t NUM_LANDMARKS = 4; // total number of landmarks
  const std::array<Eigen::Vector2d, NUM_LANDMARKS> LANDMARKS = {
      {{1.0, 4.0},
       {2.0, 0.5},
       {2.5, 3.5},
       {4.0, 2.0}}}; // positions of landmarks
  const auto VEL_STD_DEV = 0.1;
  const auto ANG_VEL_STD_DEV = 0.071;
  UKF::State state_true(1.0, 1.0, 0.0);

  // Function to simulate a range/bearing measurement given the current robot
  // state and a landmark
  auto simulate_measurement = [](const RobotState& state,
                                 const Eigen::Vector2d& landmark) {
    const auto range = std::sqrt(std::pow(landmark.x() - state.x, 2.0) +
                                 std::pow(landmark.y() - state.y, 2.0));
    const auto bearing =
        wrapAngle(std::atan2(landmark.y() - state.y, landmark.x() - state.x) -
                  state.theta);
    return RobotMeasurement(range, bearing);
  };

  // Setup the UKF (initial state, state covariance, system covariance,
  // measurement covariance)
  UKF ukf;
  UKF::State state_init(1.1, 0.9, M_PI / 12);
  ukf.set_state(state_init);
  UKF::N_by_N P_init;
  P_init << 0.500, 0.000, 0.000, 0.000, 0.500, 0.000, 0.000, 0.000, 0.250;
  ukf.set_state_covariance(P_init);
  UKF::N_by_N Q;
  Q << 0.002, 0.000, 0.000, 0.000, 0.002, 0.000, 0.000, 0.000, 0.002;
  ukf.set_process_covariance(Q);
  UKF::M_by_M R;
  R << 0.063, 0.000, 0.000, 0.007;
  ukf.set_measurement_covariance(R);

  // Setup random number generation
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<double> vel_noise(0.0, VEL_STD_DEV);
  std::normal_distribution<double> ang_vel_noise(0.0, ANG_VEL_STD_DEV);
  std::normal_distribution<double> range_noise(0.0, std::sqrt(R(0, 0)));
  std::normal_distribution<double> bearing_noise(0.0, std::sqrt(R(1, 1)));

  // Run the simulation
  double sim_time = 0.0;
  double last_meas_time = 0.0;
  while (sim_time < SIM_DURATION)
  {
    // Get the simulated (true) inputs
    const auto velocity = 0.5;
    const auto angular_velocity = 0.01;

    // Move the true state forward in time
    systemModel(state_true, velocity, angular_velocity, T);

    // Get the noisy inputs by perturbing the true inputs
    const auto velocity_noisy = velocity + vel_noise(gen);
    const auto angular_velocity_noisy = angular_velocity + ang_vel_noise(gen);

    // Predict the estimated state forward in time
    ukf.predict(systemModel, velocity_noisy, angular_velocity_noisy, T);
    std::cout << state_true << " " << ukf.get_state() << " \n";

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

