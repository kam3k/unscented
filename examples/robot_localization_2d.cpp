#include "unscented.hpp"

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
  const auto SIM_DURATION = 30.0f; // total simulation duration
  const auto T = 0.02f; // time between prediction steps (i.e., input period)
  const auto MEAS_PERIOD = 0.1f; // time between measurements
  const std::size_t NUM_LANDMARKS = 4; // total number of landmarks
  const std::array<Eigen::Vector2f, NUM_LANDMARKS> LANDMARKS = {{
      {1.0f, 4.0f},
      {2.0f, 0.5f},
      {2.5f, 3.5f},
      {4.0f, 2.0f}}}; // positions of landmarks
  const auto VEL_VAR = 0.01f;
  const auto ANG_VEL_VAR = 0.005f;
  UKF::State state_true(1.0f, 1.0f, 0.0f);

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

  // Run the simulation
  float sim_time = 0.0f;
  float last_meas_time = 0.0f;
  while (sim_time < SIM_DURATION)
  {
    // Get the simulated inputs

    // Predict the estimated state forward in time

    // Move the true state forward in time

    // Check if it is time for a measurent
    if (sim_time - last_meas_time >= MEAS_PERIOD)
    {
      // Get the simulated measurement

      // Correct the estimated state

      last_meas_time = sim_time;
    }
    sim_time += T;
  }
}

