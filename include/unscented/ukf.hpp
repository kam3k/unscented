#ifndef UNSCENTED_UKF_HPP
#define UNSCENTED_UKF_HPP

#include "unscented/ukf.h"

#include <cassert>
#include <iostream>
#include <numeric>

namespace unscented
{
// Need to explicitly provide the definitions of N, M, and NUM_SIGMA_POINTS
// despite their declaration and initialization being in the .h file (see
// https://stackoverflow.com/q/8016780 for more info)
template <typename STATE, typename MEAS>
constexpr std::size_t UKF<STATE, MEAS>::N;

template <typename STATE, typename MEAS>
constexpr std::size_t UKF<STATE, MEAS>::M;

template <typename STATE, typename MEAS>
constexpr std::size_t UKF<STATE, MEAS>::NUM_SIGMA_POINTS;

template <typename STATE, typename MEAS>
UKF<STATE, MEAS>::UKF()
  : P_(N_by_N::Identity())
  , Q_(N_by_N::Identity())
  , R_(M_by_M::Identity())
  , Pyy_(M_by_M::Identity())
{
  calculate_weights();
}

template <typename STATE, typename MEAS>
template <typename SYS_MODEL, typename... PARAMS>
void UKF<STATE, MEAS>::predict(const SYS_MODEL& system_model, PARAMS... params)
{
  generate_sigma_points();

  // Transform each sigma point through the system model
  for (auto& sigma_point : sigma_points_)
  {
    system_model(sigma_point, params...);
  }

  // The (a priori) state is the weighted mean of the transformed sigma points
  x_ = calculate_mean_manifold(sigma_points_, sigma_weights_mean_);

  // Calculate the state covariance
  P_ = Q_;
  for (std::size_t i = 0; i < NUM_SIGMA_POINTS; ++i)
  {
    const N_by_1 diff = sigma_points_[i] - x_;
    P_ += sigma_weights_cov_[i] * (diff * diff.transpose());
  }
}

template <typename STATE, typename MEAS>
template <typename MEAS_MODEL, typename... PARAMS>
void UKF<STATE, MEAS>::correct(const MEAS_MODEL& meas_model, PARAMS... params)
{
  generate_sigma_points();

  // Transform each sigma point through the measurement model
  for (std::size_t i = 0; i < NUM_SIGMA_POINTS; ++i)
  {
    meas_sigma_points_[i] = meas_model(sigma_points_[i], params...);
  }

  // The expected measurement is the weighted mean of the measurement sigma
  // points
  y_hat_ = calculate_mean_manifold(meas_sigma_points_, sigma_weights_mean_);

  // Calculate the expected measurement covariance
  Pyy_ = R_;
  for (std::size_t i = 0; i < NUM_SIGMA_POINTS; ++i)
  {
    const M_by_1 diff = meas_sigma_points_[i] - y_hat_;
    Pyy_ += sigma_weights_cov_[i] * diff * diff.transpose();
  }

  // Calculate the cross covariance between the state and expected measurement
  // (would love to have Python's zip(...) functionality here)
  Pxy_ = N_by_M::Zero();
  for (std::size_t i = 0; i < NUM_SIGMA_POINTS; ++i)
  {
    Pxy_ += sigma_weights_cov_[i] * (sigma_points_[i] - x_) *
            (meas_sigma_points_[i] - y_hat_).transpose();
  }

  // Calculate the Kalman gain and innovation
  K_ = Pxy_ * Pyy_.inverse();
  innovation_ = y_ - y_hat_;

  // Update the state covariance (temporary)
  P_ -= K_ * Pyy_ * K_.transpose();

  // Update the state mean
  generate_sigma_points(K_ * innovation_);
  x_ = calculate_mean_manifold(sigma_points_, sigma_weights_mean_);

  // Update the state covariance
  P_ = N_by_N::Zero();
  for (std::size_t i = 0; i < NUM_SIGMA_POINTS; ++i)
  {
    const N_by_1 diff = sigma_points_[i] - x_;
    P_ += sigma_weights_cov_[i] * diff * diff.transpose();
  }
}

template <typename STATE, typename MEAS>
template <typename MEAS_MODEL, typename... PARAMS>
void UKF<STATE, MEAS>::correct(const MEAS_MODEL& meas_model, MEAS meas,
                               PARAMS... params)
{
  set_measurement(std::move(meas));
  correct(meas_model, params...);
}

template <typename STATE, typename MEAS>
void UKF<STATE, MEAS>::generate_sigma_points(
    const N_by_1& delta /*= N_by_1::Zero()*/)
{
  // Calculate the (weighted) matrix square root of the state covariance
  // matrix
  cholesky_.compute(P_);
  const N_by_N& sqrt_P = eta_ * cholesky_.matrixL().toDenseMatrix();

  // First sigma point is the perturbed state mean
  sigma_points_[0] = x_ + delta;

  // Next N sigma points are the current state mean offset by the columns
  // of sqrt_P, and the N sigma points after that are the same offsets
  // but negated
  for (std::size_t i = 0; i < N; ++i)
  {
    const N_by_1& offset = sqrt_P.col(i);
    sigma_points_[i + 1] = x_ + (delta + offset);
    sigma_points_[N + i + 1] = x_ + (delta - offset);
  }
}

template <typename STATE, typename MEAS>
void UKF<STATE, MEAS>::set_state(const STATE& state)
{
  x_ = state;
}

template <typename STATE, typename MEAS>
void UKF<STATE, MEAS>::set_state(STATE&& state)
{
  x_ = std::move(state);
}

template <typename STATE, typename MEAS>
const STATE& UKF<STATE, MEAS>::get_state() const
{
  return x_;
}

template <typename STATE, typename MEAS>
void UKF<STATE, MEAS>::set_measurement(const MEAS& measurement)
{
  y_ = measurement;
}

template <typename STATE, typename MEAS>
void UKF<STATE, MEAS>::set_measurement(MEAS&& measurement)
{
  y_ = std::move(measurement);
}

template <typename STATE, typename MEAS>
const MEAS& UKF<STATE, MEAS>::get_measurement() const
{
  return y_;
}

template <typename STATE, typename MEAS>
void UKF<STATE, MEAS>::set_state_covariance(const N_by_N& state_covariance)
{
  P_ = state_covariance;
}

template <typename STATE, typename MEAS>
void UKF<STATE, MEAS>::set_state_covariance(N_by_N&& state_covariance)
{
  P_ = std::move(state_covariance);
}

template <typename STATE, typename MEAS>
const typename UKF<STATE, MEAS>::N_by_N&
UKF<STATE, MEAS>::get_state_covariance() const
{
  return P_;
}

template <typename STATE, typename MEAS>
void UKF<STATE, MEAS>::set_process_covariance(const N_by_N& process_covariance)
{
  Q_ = process_covariance;
}

template <typename STATE, typename MEAS>
void UKF<STATE, MEAS>::set_process_covariance(N_by_N&& process_covariance)
{
  Q_ = std::move(process_covariance);
}

template <typename STATE, typename MEAS>
const typename UKF<STATE, MEAS>::N_by_N&
UKF<STATE, MEAS>::get_process_covariance() const
{
  return Q_;
}

template <typename STATE, typename MEAS>
void UKF<STATE, MEAS>::set_measurement_covariance(
    const M_by_M& measurement_covariance)
{
  R_ = measurement_covariance;
}

template <typename STATE, typename MEAS>
void UKF<STATE, MEAS>::set_measurement_covariance(
    M_by_M&& measurement_covariance)
{
  R_ = std::move(measurement_covariance);
}

template <typename STATE, typename MEAS>
const typename UKF<STATE, MEAS>::M_by_M&
UKF<STATE, MEAS>::get_measurement_covariance() const
{
  return R_;
}

template <typename STATE, typename MEAS>
const MEAS& UKF<STATE, MEAS>::get_expected_measurement() const
{
  return y_hat_;
}

template <typename STATE, typename MEAS>
const typename UKF<STATE, MEAS>::M_by_M&
UKF<STATE, MEAS>::get_expected_measurement_covariance() const
{
  return Pyy_;
}

template <typename STATE, typename MEAS>
const typename UKF<STATE, MEAS>::N_by_M&
UKF<STATE, MEAS>::get_cross_covariance() const
{
  return Pxy_;
}

template <typename STATE, typename MEAS>
const typename UKF<STATE, MEAS>::N_by_M& UKF<STATE, MEAS>::get_kalman_gain()
    const
{
  return K_;
}

template <typename STATE, typename MEAS>
const typename UKF<STATE, MEAS>::M_by_1& UKF<STATE, MEAS>::get_innovation()
    const
{
  return innovation_;
}

template <typename STATE, typename MEAS>
const typename UKF<STATE, MEAS>::SigmaPoints&
UKF<STATE, MEAS>::get_sigma_points() const
{
  return sigma_points_;
}

template <typename STATE, typename MEAS>
const typename UKF<STATE, MEAS>::MeasurementSigmaPoints&
UKF<STATE, MEAS>::get_measurement_sigma_points() const
{
  return meas_sigma_points_;
}

template <typename STATE, typename MEAS>
void UKF<STATE, MEAS>::set_weight_coefficients(double alpha, double beta,
                                               double kappa)
{
  alpha_ = alpha;
  beta_ = beta;
  kappa_ = kappa;
  calculate_weights();
}

template <typename STATE, typename MEAS>
const typename UKF<STATE, MEAS>::SigmaWeights&
UKF<STATE, MEAS>::get_mean_sigma_weights() const
{
  return sigma_weights_mean_;
}

template <typename STATE, typename MEAS>
const typename UKF<STATE, MEAS>::SigmaWeights&
UKF<STATE, MEAS>::get_covariance_sigma_weights() const
{
  return sigma_weights_cov_;
}

template <typename STATE, typename MEAS>
void UKF<STATE, MEAS>::calculate_weights()
{
  lambda_ = alpha_ * alpha_ * (N + kappa_) - N;

  assert(N + lambda_ > 1e-6);

  eta_ = std::sqrt(N + lambda_);

  const auto w_mean_0 = lambda_ / (N + lambda_);
  const auto w_cov_0 = w_mean_0 + (1.0 - alpha_ * alpha_ + beta_);
  const auto w_i = 1.0 / (2.0 * (N + lambda_));

  sigma_weights_mean_[0] = w_mean_0;
  sigma_weights_cov_[0] = w_cov_0;

  for (std::size_t i = 1; i < NUM_SIGMA_POINTS; ++i)
  {
    sigma_weights_mean_[i] = w_i;
    sigma_weights_cov_[i] = w_i;
  }
}

template <typename MANIFOLD, std::size_t ARRAY_SIZE>
MANIFOLD calculate_mean_manifold(
    const std::array<MANIFOLD, ARRAY_SIZE>& manifolds,
    const std::array<double, ARRAY_SIZE>& weights)
{
  static const int MAX_ITERATIONS = 10000;
  static const double EPS = 1e-6;

  auto reference_manifold = manifolds[0];
  Eigen::Matrix<double, MANIFOLD::DOF, 1> mean_vec;
  int iteration_count = 0;

  do
  {
    mean_vec = Eigen::Matrix<double, MANIFOLD::DOF, 1>::Zero();
    for (std::size_t i = 0; i < ARRAY_SIZE; ++i)
    {
      mean_vec += weights[i] * (manifolds[i] - reference_manifold);
    }
    reference_manifold = reference_manifold + mean_vec;
    ++iteration_count;
  } while (mean_vec.norm() > EPS && iteration_count < MAX_ITERATIONS);

  assert(iteration_count < MAX_ITERATIONS &&
         "Calculating mean manifold did not converge");

  return reference_manifold;
}
} // namespace unscented

#endif
