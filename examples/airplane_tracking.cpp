#include "unscented.hpp"

#include <random>

///////////////////////////////////////////////////////////////////////////////
// Utility functions
///////////////////////////////////////////////////////////////////////////////

/** 
 * @brief Given an angle in radians, wraps it to [-pi, pi)
 * 
 * @param[in] angle
 * 
 * @return Angle wrapped to range [-pi, pi)
 */
double wrap_angle(double angle)
{
  angle = fmod(angle + M_PI, 2 * M_PI);
  if (angle < 0.0)
  {
    angle += 2 * M_PI;
  }
  return angle - M_PI;
}

///////////////////////////////////////////////////////////////////////////////
// State and system model
///////////////////////////////////////////////////////////////////////////////

/** State is simple a 4-vector */
using AirplaneState = Eigen::Vector4d;

/** The elements of the state vector */
enum StateElements
{
  POSITION = 0,
  VELOCITY,
  ALTITUDE,
  CLIMB_RATE
};

/**
 * @brief The airplane's system model is the constant velocity model; that is,
 * the velocity is assumed to be constant over the duration of a timestep dt,
 * and the positions are simply updated using the first-order euler method
 *
 * @param[in,out] state The state to be updated
 * @param[in] dt The timestep (seconds)
 */
void system_model(AirplaneState& state, double dt)
{
  Eigen::Matrix4d F;
  F << 1.0,  dt, 0.0, 0.0,
       0.0, 1.0, 0.0, 0.0,
       0.0, 0.0, 1.0,  dt,
       0.0, 0.0, 0.0, 1.0;
  state = F * state;
}

///////////////////////////////////////////////////////////////////////////////
// Measurement and measurement model
///////////////////////////////////////////////////////////////////////////////

/** 
 * @brief The radar measurement consists of a range and elevation
 */
struct RadarMeasurement
{
  /** 
   * @brief Default constructor (required by filter)
   */
  RadarMeasurement() = default;

  /**
   * @brief Constructor populating range and elevation (not required by filter,
   * included for convenience)
   *
   * @param[in] r Measured range to airplane
   * @param[in] e Measured elevation of airplane
   */
  RadarMeasurement(double r, double e) : range(r), elevation(e) {}

  /** Range to airplane */
  double range;

  /** Elevation of airplane */
  double elevation;
};

/**
 * @brief Adds two measurements together and produces another measurement. This
 * is required by the filter unless a custom measurement mean function is
 * provided.
 *
 * @param[in] lhs
 * @param[in] rhs
 *
 * @return Sum of two measurements, carefully wrapping the summed elevation if
 * necessary
 */
RadarMeasurement operator+(const RadarMeasurement& lhs,
                           const RadarMeasurement& rhs)
{
  return RadarMeasurement(lhs.range + rhs.range,
                          wrap_angle(lhs.elevation + rhs.elevation));
}

/**
 * @brief Substracts the rhs measurement from the lhs measurement, with the
 * result put into a vector of size equal to the DOF of the measurement
 * (required by the filter)
 *
 * @param[in] lhs
 * @param[in] rhs
 *
 * @return Difference of two measurements in a vector, carefully wrapping the
 * elevation difference if necessary
 */
Eigen::Vector2d operator-(const RadarMeasurement& lhs,
                          const RadarMeasurement& rhs)
{
  return Eigen::Vector2d(lhs.range - rhs.range,
                         wrap_angle(lhs.elevation - rhs.elevation));
}

/**
 * @brief Multiples a measurement by a scale factor. This is required by the
 * filter unless a custom measurement mean function is provided.
 *
 * @param[in] meas
 * @param[in] scale
 *
 * @return A scaled measurement
 */
RadarMeasurement operator*(const RadarMeasurement& meas, double scale)
{
  return RadarMeasurement(meas.range * scale, meas.elevation * scale);
}

/**
 * @brief The measurement model maps a state to an expected measurement. In this
 * case, we can use trigonometry to map the position and altitude of the
 * airplane to its range and elevation. This is essentially a Cartesian-to-polar
 * conversion.
 *
 * @param[in] state The state to map to an expected measurement
 *
 * @return The expected measurement
 */
RadarMeasurement measurement_model(const AirplaneState& state)
{
  const auto range =
      std::sqrt(std::pow(state[POSITION], 2) + std::pow(state[ALTITUDE], 2));
  const auto elevation = std::atan2(state[ALTITUDE], state[POSITION]);
  return {range, elevation};
}

int main()
{
  //////////////////////////////////////////////////////////////////////////////
  // Setup the filter
  //////////////////////////////////////////////////////////////////////////////

  // The airplane state has four degrees of freedom (position, velocity,
  // altitude, climb rate) and the radar measurement has two degrees of freedom
  // (range, elevation)
  using UKF = unscented::UKF<AirplaneState, 4, RadarMeasurement, 2>;
  UKF ukf;

  // Simulation parameters
  const auto SIM_DURATION = 360.0; // seconds
  const auto DT = 3.0; // seconds

  // Calculate process noise covariance Q using the discrete constant white
  // noise model (Bar-Shalom. “Estimation with Applications To Tracking and
  // Navigation”. John Wiley & Sons, 2001. Page 274.). To simplify, one can
  // simply pick (or tune) appropriate values on the diagonal of Q.
  const auto PROCESS_VAR = 0.1;
  Eigen::Vector2d G(0.5 * DT * DT, DT);
  Eigen::Matrix4d Q;
  Q.block(0, 0, 1, 1) = G * G.transpose() * PROCESS_VAR;
  Q.block(2, 2, 3, 3) = G * G.transpose() * PROCESS_VAR;
  ukf.set_process_covariance(Q);

  // Calculate measurement noise covariance R (standard deviations chosen
  // somewhat arbitrarily)
  const auto RANGE_STD_DEV = 5.0; // meters
  const auto ELEVATION_STD_DEV = 0.5 * M_PI / 180.0; // radians
  UKF::M_by_M R;
  R << std::pow(RANGE_STD_DEV, 2), 0.0, 0.0, std::pow(ELEVATION_STD_DEV, 2);
  ukf.set_measurement_covariance(R);

  // Set initial state estimate and its covariance
  AirplaneState true_state(0, 100, 1000, 0);
  AirplaneState initial_state_estimate(0, 90, 1100, 0);
  ukf.set_state(initial_state_estimate);
  UKF::N_by_N P = UKF::N_by_N::Zero();
  P(0, 0) = std::pow(300.0, 2); // m^2
  P(1, 1) = std::pow(30.0, 2); // (m/s)^2
  P(2, 2) = std::pow(150.0, 2); // m^2
  P(3, 3) = std::pow(3.0, 2); // (m/s)^2
  ukf.set_state_covariance(P);

  // Create vectors that will hold the histories of the true and estimated
  // states, populating them both with the initial states. This is simply
  // recorded so the results over time can be plotted at the end of the
  // simulation.
  std::vector<AirplaneState> true_state_history;
  true_state_history.push_back(true_state);
  std::vector<AirplaneState> estimated_state_history;
  estimated_state_history.push_back(ukf.get_state());
  std::vector<UKF::N_by_N> estimated_state_cov_history;

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
  std::vector<double> sim_time_history = {sim_time};
  while (sim_time < SIM_DURATION)
  {
    // One minute into the simulation, set a non-zero climb rate
    if (sim_time > 60.0)
    {
      true_state[CLIMB_RATE] = 300.0/60; // 300 m/min 
    }

    // Update the true position and altitude, adding some noise to the velocity
    // to make up for deficiencies in the constant velocity model (i.e., it's
    // unlikely the velocity was constant throughout the full time step)
    true_state[POSITION] += (true_state[VELOCITY] + vel_noise(gen)) * DT;
    true_state[ALTITUDE] += (true_state[CLIMB_RATE] + vel_noise(gen))* DT;
    true_state_history.push_back(true_state);

    // Simulate a measurement based on the true state
    auto meas = measurement_model(true_state);
    meas.range += range_noise(gen);
    meas.elevation += elevation_noise(gen);

    // Update the filter estimates
    ukf.predict(system_model, DT);
    ukf.correct(measurement_model, meas);

    // Move time forward and record all the current values in the history
    sim_time += DT;
    estimated_state_history.push_back(ukf.get_state());
    estimated_state_cov_history.push_back(ukf.get_state_covariance());
    sim_time_history.push_back(sim_time);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Plot the results
  //////////////////////////////////////////////////////////////////////////////

#include "matplotlibcpp.h"
#include <valarray>

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
  
  const auto hist_size = sim_time_history.size();
  std::valarray<double> estimated_positions(hist_size);
  std::valarray<double> estimated_positions_std_devs(hist_size);
  std::valarray<double> estimated_velocities(hist_size);
  std::valarray<double> estimated_velocities_std_devs(hist_size);
  std::valarray<double> estimated_altitudes(hist_size);
  std::valarray<double> estimated_altitudes_std_devs(hist_size);
  std::valarray<double> estimated_climb_rates(hist_size);
  std::valarray<double> estimated_climb_rates_std_devs(hist_size);
  for (auto i = 0; i < sim_time_history.size(); ++i)
  {
    const auto& state = estimated_state_history[i];
    const auto& state_cov = estimated_state_cov_history[i];
    estimated_positions[i] = state[POSITION];
    estimated_positions_std_devs[i] = std::sqrt(state_cov(POSITION, POSITION));
    estimated_velocities[i] = state[VELOCITY];
    estimated_velocities_std_devs[i] = std::sqrt(state_cov(VELOCITY, VELOCITY));
    estimated_altitudes[i] = state[ALTITUDE];
    estimated_altitudes_std_devs[i] = std::sqrt(state_cov(ALTITUDE, ALTITUDE));
    estimated_climb_rates[i] = state[CLIMB_RATE];
    estimated_climb_rates_std_devs[i] =
        std::sqrt(state_cov(CLIMB_RATE, CLIMB_RATE));
  }

  namespace plt = matplotlibcpp;
  plt::subplot(4, 1, 1);
  plt::plot(
      sim_time_history, true_positions, "k-", sim_time_history,
      estimated_positions, "b-", sim_time_history,
      std::valarray<double>(estimated_positions + estimated_positions_std_devs),
      "g--", sim_time_history,
      std::valarray<double>(estimated_positions - estimated_positions_std_devs),
      "g--");
  plt::xlabel("Time [s]");
  plt::ylabel("Position [m]");
  plt::subplot(4, 1, 2);
  plt::plot(sim_time_history, true_velocities, "k-", sim_time_history,
            estimated_velocities, "b-", sim_time_history,
            std::valarray<double>(estimated_velocities +
                                  estimated_velocities_std_devs),
            "g--", sim_time_history,
            std::valarray<double>(estimated_velocities -
                                  estimated_velocities_std_devs),
            "g--");
  plt::xlabel("Time [s]");
  plt::ylabel("Velocity [m/s]");
  plt::subplot(4, 1, 3);
  plt::plot(
      sim_time_history, true_altitudes, "k-", sim_time_history,
      estimated_altitudes, "b-", sim_time_history,
      std::valarray<double>(estimated_altitudes + estimated_altitudes_std_devs),
      "g--", sim_time_history,
      std::valarray<double>(estimated_altitudes - estimated_altitudes_std_devs),
      "g--");
  plt::xlabel("Time [s]");
  plt::ylabel("Altitude [m]");
  plt::subplot(4, 1, 4);
  plt::plot(sim_time_history, true_climb_rates, "k-", sim_time_history,
            estimated_climb_rates, "b-", sim_time_history,
            std::valarray<double>(estimated_climb_rates +
                                  estimated_climb_rates_std_devs),
            "g--", sim_time_history,
            std::valarray<double>(estimated_climb_rates -
                                  estimated_climb_rates_std_devs),
            "g--");
  plt::title("Airplane Climb Rate");
  plt::xlabel("Time [s]");
  plt::ylabel("Climb Rate [m/s]");

  plt::show();
}
