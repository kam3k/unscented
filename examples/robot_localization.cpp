#include "unscented/primitives.h"
#include "unscented/ukf.hpp"

#include <iostream>
#include <random>
#include <matplot/matplot.h>

using State = unscented::Pose2d;

void system_model(State& state, double velocity, double angular_velocity, double dt) {
  auto& [position, heading] = state.data;
  position.x() += velocity * std::cos(heading.get_angle()) * dt;
  position.y() += velocity * std::sin(heading.get_angle()) * dt;
  heading = unscented::Angle(heading.get_angle() + angular_velocity * dt);
}

using Measurement = unscented::Compound<unscented::Scalar, unscented::Angle>;

Measurement measurement_model(const State& state,
                              const unscented::Vector<2> landmark_position)
{
  const auto& [position, heading] = state.data;
  const unscented::Vector<2> relative_position = (landmark_position - position);
  const auto expected_range = relative_position.norm();
  const auto expected_bearing = unscented::Angle(
      std::atan2(relative_position.y(), relative_position.x()));
  Measurement expected_measurement;
  expected_measurement.data = {expected_range, expected_bearing};
  return expected_measurement;
}

int main()
{
  using UKF = unscented::UKF<State, Measurement>;

  // Set up the simulation
  const auto SIM_DURATION = 10.0; // total simulation duration
  const auto DT = 0.02; // time between prediction steps (i.e., input period)
  const auto MEAS_PERIOD = 0.1; // time between measurements
  const std::size_t NUM_LANDMARKS = 4; // total number of landmarks
  const std::array<Eigen::Vector2d, NUM_LANDMARKS> LANDMARKS = {
      {{1.0, 4.0},
       {2.0, 0.5},
       {2.5, 3.5},
       {4.0, 2.0}}}; // positions of landmarks
  const auto VEL_STD_DEV = 0.1;
  const auto ANG_VEL_STD_DEV = 0.071;
  State state_true{unscented::Vector<2>(1.0, 1.0), unscented::Angle(0.0)};

  // Setup the UKF (initial state, state covariance, system covariance,
  // measurement covariance)
  UKF ukf;
  UKF::State state_init(unscented::Vector<2>(1.1, 0.9),
                        unscented::Angle(M_PI / 12));
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

  // Set up random number generation
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<double> vel_noise(0.0, VEL_STD_DEV);
  std::normal_distribution<double> ang_vel_noise(0.0, ANG_VEL_STD_DEV);
  std::normal_distribution<double> range_noise(0.0, std::sqrt(R(0, 0)));
  std::normal_distribution<double> bearing_noise(0.0, std::sqrt(R(1, 1)));

  // Set up vectors to store the results for plotting
  std::vector<double> sim_times;
  std::vector<double> x_true;
  std::vector<double> y_true;
  std::vector<double> x_est;
  std::vector<double> y_est;
  std::vector<double> x_position_errors;
  std::vector<double> y_position_errors;
  std::vector<double> heading_errors;
  std::vector<double> x_position_std_devs;
  std::vector<double> y_position_std_devs;
  std::vector<double> heading_std_devs;

  // Run the simulation
  double sim_time = 0.0;
  double last_meas_time = 0.0;
  while (sim_time < SIM_DURATION)
  {
    // Get the simulated (true) inputs
    const auto velocity = 0.5;
    const auto angular_velocity = 0.01;

    // Move the true state forward in time
    system_model(state_true, velocity, angular_velocity, DT);

    // Get the noisy inputs by perturbing the true inputs
    const auto velocity_noisy = velocity + vel_noise(gen);
    const auto angular_velocity_noisy = angular_velocity + ang_vel_noise(gen);

    // Predict the estimated state forward in time
    ukf.predict(system_model, velocity_noisy, angular_velocity_noisy, DT);

    // Check if it is time for a measurent
    if (sim_time - last_meas_time >= MEAS_PERIOD)
    {
      for (const auto& landmark : LANDMARKS)
      {
        // Get the simulated (true) measurement
        const auto& measurement = measurement_model(state_true, landmark);
        const auto& [range, bearing] = measurement.data;

        // Get the noisy measurement by perturbing the true measurement
        Measurement measurement_noisy(
            unscented::Scalar(range.value + range_noise(gen)),
            unscented::Angle(bearing.get_angle() + bearing_noise(gen)));

        // Correct the estimated state
        ukf.correct(measurement_model, measurement_noisy, landmark);
      }
      last_meas_time = sim_time;
    }

    // Record results for this iteration
    const auto& [true_position, true_heading] = state_true.data;
    const auto& [est_position, est_heading] = ukf.get_state().data;
    const auto& P = ukf.get_state_covariance();
    sim_times.push_back(sim_time);
    x_true.push_back(true_position.x());
    y_true.push_back(true_position.y());
    x_est.push_back(est_position.x());
    y_est.push_back(est_position.y());
    x_position_errors.push_back(est_position.x() - true_position.x());
    y_position_errors.push_back(est_position.y() - true_position.y());
    heading_errors.push_back(unscented::Angle(est_heading - true_heading).get_angle());
    x_position_std_devs.push_back(std::sqrt(P(0, 0)));
    y_position_std_devs.push_back(std::sqrt(P(1, 1)));
    heading_std_devs.push_back(std::sqrt(P(2, 2)));

    // Move the simulation time forward
    sim_time += DT;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Plot the results
  //////////////////////////////////////////////////////////////////////////////

  // Create +/- n std dev vectors
  const auto num_std_devs = 2;
  const auto num_pts = x_position_std_devs.size();
  std::vector<double> x_position_positive_n_std_devs(num_pts);
  std::vector<double> x_position_negative_n_std_devs(num_pts);
  std::vector<double> y_position_positive_n_std_devs(num_pts);
  std::vector<double> y_position_negative_n_std_devs(num_pts);
  std::vector<double> heading_positive_n_std_devs(num_pts);
  std::vector<double> heading_negative_n_std_devs(num_pts);
  for (std::size_t i = 0; i < num_pts; ++i)
  {
    x_position_positive_n_std_devs[i] = num_std_devs * x_position_std_devs[i];
    x_position_negative_n_std_devs[i] = -num_std_devs * x_position_std_devs[i];
    y_position_positive_n_std_devs[i] = num_std_devs * y_position_std_devs[i];
    y_position_negative_n_std_devs[i] = -num_std_devs * y_position_std_devs[i];
    heading_positive_n_std_devs[i] = num_std_devs * heading_std_devs[i];
    heading_negative_n_std_devs[i] = -num_std_devs * heading_std_devs[i];
  }

  // Trajectories
  matplot::figure();
  matplot::plot(x_true, y_true, "b");
  matplot::hold(true);
  matplot::plot(x_est, y_est, "r");
  matplot::axis(matplot::equal);

  matplot::show();
}

