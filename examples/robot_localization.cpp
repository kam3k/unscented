#include "matplot/freestanding/plot.h"
#include "unscented/primitives.h"
#include "unscented/ukf.hpp"

#include <matplot/matplot.h>
#include <cstddef>
#include <iostream>
#include <random>

using State = unscented::Pose2d;

void system_model(State& state, double velocity, double steering_angle,
                  double wheelbase, double dt)
{
  auto& [position, heading] = state.data;
  const auto dist = velocity * dt;

  // If the robot is not turning, do simple forward motion
  const auto is_robot_turning = std::abs(steering_angle) > 0.001;
  if (!is_robot_turning)
  {
    position.x() += dist * std::cos(heading.get_angle());
    position.y() += dist * std::sin(heading.get_angle());
    return;
  }

  // Robot is turning
  const auto beta = (dist / wheelbase) * std::tan(steering_angle);
  const auto r = wheelbase / std::tan(steering_angle); // turning radius
  const auto sinh = std::sin(heading.get_angle());
  const auto sinhb = std::sin(heading.get_angle() + beta);
  const auto cosh = std::cos(heading.get_angle());
  const auto coshb = std::cos(heading.get_angle() + beta);
  position.x() += -r * sinh + r * sinhb;
  position.y() += r * cosh - r * coshb;
  heading = unscented::Angle(heading.get_angle() + beta);
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
  return {expected_range, expected_bearing};
}

int main()
{
  using UKF = unscented::UKF<State, Measurement>;

  // Set up the simulation
  const auto DT = 0.1; // s (i.e., input period)
  const auto WHEELBASE = 0.5; // m
  const auto VEL_STD_DEV = 0.1; // m/s
  const auto STEERING_ANGLE_STD_DEV = M_PI / 180.0; // rad
  const auto RANGE_STD_DEV = 0.3; // m
  const auto BEARING_STD_DEV = 0.1; // rad
  const auto MAX_MEAS_RANGE = 10.0;
  const std::size_t NUM_LANDMARKS = 4; // total number of landmarks
  const std::array<unscented::Vector<2>, NUM_LANDMARKS> LANDMARKS = {
      {{5.0, 10.0},
       {10.0, 5.0},
       {30.0, 25.0},
       {60.0, 45.0}}}; // positions of landmarks
  State state_true{unscented::Vector<2>(2.0, 6.0), unscented::Angle(0.3)};

  // Setup the UKF (initial state, state covariance, system covariance,
  // measurement covariance)
  UKF ukf;
  UKF::State state_init(unscented::Vector<2>(2.1, 5.8), unscented::Angle(0.25));
  ukf.set_state(state_init);
  // clang-format off
  UKF::N_by_N P;
  P << 0.100, 0.000, 0.000, 
       0.000, 0.100, 0.000, 
       0.000, 0.000, 0.050;
  UKF::N_by_N Q;
  Q << 0.0001, 0.0000, 0.0000, 
       0.0000, 0.0001, 0.0000, 
       0.0000, 0.0000, 0.0001;
  UKF::M_by_M R;
  const auto r_var = RANGE_STD_DEV * RANGE_STD_DEV;
  const auto b_var = BEARING_STD_DEV * BEARING_STD_DEV;
  R << r_var, 0.000,
       0.000, b_var;
  // clang-format on
  ukf.set_state_covariance(P);
  ukf.set_process_covariance(Q);
  ukf.set_measurement_covariance(R);

  // Set up random number generation
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<double> vel_noise(0.0, VEL_STD_DEV);
  std::normal_distribution<double> steer_noise(0.0, STEERING_ANGLE_STD_DEV);
  std::normal_distribution<double> range_noise(0.0, RANGE_STD_DEV);
  std::normal_distribution<double> bearing_noise(0.0, BEARING_STD_DEV);

  // Set up vectors to store the results for plotting
  std::vector<State> est_states;
  std::vector<State> true_states;
  std::vector<UKF::N_by_N> covs;

  // Set up the commands
  std::vector<double> velocities(500, 1.1);
  std::vector<double> steering_angles(200, 0.01);

  // Run the simulation
  const std::size_t NUM_STEPS = 750;
  for (std::size_t i = 0; i < NUM_STEPS; ++i)
  {
    // Generate a velocity and steering angle
    const auto v = 1.1;
    auto steering_angle = 0.02;
    if (i > 200)
    {
      steering_angle = -0.02;
    }
    if (i > 400)
    {
      steering_angle = 0.04;
    }
    if (i > 450)
    {
      steering_angle = -0.03;
    }
    if (i > 600)
    {
      steering_angle = 0.04;
    }

    // Update true state
    system_model(state_true, v, steering_angle, WHEELBASE, DT);

    // Get a velocity and steering angle
    const auto v_noisy = v + vel_noise(gen);
    const auto steer_noisy = steering_angle + steer_noise(gen);

    // Predict
    ukf.predict(system_model, v_noisy, steer_noisy, WHEELBASE, DT);

    // Correct
    for (const auto& landmark : LANDMARKS)
    {
      // Get the measurement 
      const auto& measurement = measurement_model(state_true, landmark);
      const auto& [range, bearing] = measurement.data;

      // Only use the measurement if the landmark is within range of the robot
      if (range.value > MAX_MEAS_RANGE) {
        continue;
      }

      // Generate a noisy measurement by perturbing the true one
      Measurement measurement_noisy(
          unscented::Scalar(range.value + range_noise(gen)),
          unscented::Angle(bearing.get_angle() + bearing_noise(gen)));

      // Correct with the noisy measurement
      ukf.correct(measurement_model, measurement_noisy, landmark);
    }

    // Record results for this iteration
    true_states.push_back(state_true);
    est_states.push_back(ukf.get_state());
    covs.push_back(ukf.get_state_covariance());
  }

  //////////////////////////////////////////////////////////////////////////////
  // Plot the results
  //////////////////////////////////////////////////////////////////////////////

  // Get the estimated x-y trajectory
  std::vector<double> xs_est;
  std::vector<double> ys_est;
  for (const auto& state : est_states) {
    const auto& [pos, head] = state.data;
    xs_est.push_back(pos.x());
    ys_est.push_back(pos.y());
  }

  // Get the true x-y trajectory
  std::vector<double> xs_true;
  std::vector<double> ys_true;
  for (const auto& state : true_states) {
    const auto& [pos, head] = state.data;
    xs_true.push_back(pos.x());
    ys_true.push_back(pos.y());
  }

  matplot::figure();
  matplot::plot(xs_true, ys_true, "k");
  matplot::hold(true);
  matplot::plot(xs_est, ys_est, "b");
  matplot::title("Trajectory");
  for (const auto& landmark : LANDMARKS) {
    auto rect =
        matplot::rectangle(landmark.x() - 0.5, landmark.y() - 0.5, 1.0, 1.0);
    rect->fill(true);
    rect->color({0.8f, 0.f, 0.f, 1.f});
  }
  matplot::xlabel("x (m)");
  matplot::ylabel("y (m)");
  matplot::axis("equal");

  matplot::show();

  return 0;
}

