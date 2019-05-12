#include "unscented.h"

#include <cassert>
#include <numeric>

namespace unscented
{
  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::UKF()
  {
    P_ = N_by_N::Identity();
    Q_ = N_by_N::Identity();
    R_ = M_by_M::Identity();

    calculate_weights();

    // Default state mean function is simply the weighted average of all states
    state_mean_function_ = [this](const SigmaPoints& states,
                                  const SigmaWeights& weights) {
      STATE weighted_state = states[0] * weights[0];
      return std::inner_product(states.begin() + 1, states.end(),
                                weights.begin() + 1, weighted_state);
    };

    // Default meas mean function is simply the weighted average of all
    // measurements
    meas_mean_function_ = [this](const MeasurementSigmaPoints& measurements,
                                 const SigmaWeights& weights) {
      MEAS weighted_meas = measurements[0] * weights[0];
      return std::inner_product(measurements.begin() + 1, measurements.end(),
                                weights.begin() + 1, weighted_meas);
    };
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  template <typename SYS_MODEL, typename... PARAMS>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::predict(
      const SYS_MODEL& system_model, PARAMS... params)
  {
    generate_sigma_points();

    // Transform each sigma point through the system model
    for (std::size_t i = 0; i < N; ++i)
    {
      system_model(sigma_points_[i], params...);
    }

    // The (a priori) state is the weighted mean of the transformed sigma points
    x_ = state_mean_function_(sigma_points_, sigma_weights_mean_);

    // Calculate the state covariance
    P_ = Q_ +
         std::inner_product(sigma_points_.begin(), sigma_points_.end(),
                            sigma_weights_cov_.begin(), N_by_N(N_by_N::Zero()),
                            std::plus<N_by_N>(),
                            [this](const STATE& state, double weight) {
                              const N_by_1 diff = state - x_;
                              return N_by_N(diff * diff.transpose() * weight);
                            });
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  template <typename MEAS_MODEL, typename... PARAMS>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::correct(
      const MEAS_MODEL& meas_model, PARAMS... params)
  {
    generate_sigma_points();

    // Transform each sigma point through the measurement model
    for (std::size_t i = 0; i < N; ++i)
    {
      meas_sigma_points_[i] = meas_model(sigma_points_[i], params...);
    }

    // The expected measurement is the weighted mean of the measurement sigma
    // points
    y_hat_ = meas_mean_function_(meas_sigma_points_, sigma_weights_mean_);

    // Calculate the expected measurement covariance
    Pyy_ = R_ + std::inner_product(meas_sigma_points_.begin(),
                                   meas_sigma_points_.end(),
                                   sigma_weights_cov_.begin(),
                                   M_by_M(M_by_M::Zero()), std::plus<M_by_M>(),
                                   [this](const MEAS& meas, double weight) {
                                     const M_by_1 diff = meas - y_hat_;
                                     return diff * diff.transpose() * weight;
                                   });

    // Calculate the cross covariance between the state and expected measurement
    // (would love to have Python's zip(...) functionality here)
    Pxy_ = N_by_M::Zero();
    for (std::size_t i = 0; i < N; ++i)
    {
      Pxy_ += sigma_weights_cov_[i] * (sigma_points_[i] - x_) *
              (meas_sigma_points_[i] - y_hat_).transpose();
    }

    // Calculate the Kalman gain and innovation
    K_ = Pxy_ * Pyy_.inverse();
    innovation_ = y_ - y_hat_;

    // Update the state mean and covariance
    x_ = x_ + STATE(K_ * innovation_);
    P_ -= K_ * Pyy_ * K_.transpose();
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  template <typename MEAS_MODEL, typename... PARAMS>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::correct(
      const MEAS_MODEL& meas_model, MEAS meas, PARAMS... params)
  {
    set_measurement(std::move(meas));
    correct(meas_model, params...);
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::set_state(const STATE& state)
  {
    x_ = state;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::set_state(STATE&& state)
  {
    x_ = std::move(state);
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  const STATE& UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::get_state() const
  {
    return x_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::set_measurement(
      const MEAS& measurement)
  {
    y_ = measurement;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::set_measurement(
      MEAS&& measurement)
  {
    y_ = std::move(measurement);
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  const MEAS& UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::get_measurement() const
  {
    return y_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::set_state_covariance(
      const N_by_N& state_covariance)
  {
    P_ = state_covariance;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::set_state_covariance(
      N_by_N&& state_covariance)
  {
    P_ = std::move(state_covariance);
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::N_by_N&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::get_state_covariance() const
  {
    return P_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::set_process_covariance(
      const N_by_N& process_covariance)
  {
    Q_ = process_covariance;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::set_process_covariance(
      N_by_N&& process_covariance)
  {
    Q_ = std::move(process_covariance);
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::N_by_N&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::get_process_covariance() const
  {
    return Q_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::set_measurement_covariance(
      const M_by_M& measurement_covariance)
  {
    R_ = measurement_covariance;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::set_measurement_covariance(
      M_by_M&& measurement_covariance)
  {
    R_ = std::move(measurement_covariance);
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::M_by_M&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::get_measurement_covariance() const
  {
    return R_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  const MEAS& UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::get_expected_measurement()
      const
  {
    return y_hat_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::M_by_M&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::get_expected_measurement_covariance()
      const
  {
    return Pyy_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::N_by_M&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::get_cross_covariance() const
  {
    return Pxy_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::N_by_M&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::get_kalman_gain() const
  {
    return K_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::M_by_1&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::get_innovation() const
  {
    return innovation_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::SigmaPoints&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::get_sigma_points() const
  {
    return sigma_points_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::MeasurementSigmaPoints&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::get_measurement_sigma_points() const
  {
    return meas_sigma_points_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::set_weight_coefficients(
      double alpha, double beta, double kappa)
  {
    alpha_ = alpha;
    beta_ = beta;
    kappa_ = kappa;
    calculate_weights();
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::SigmaWeights&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::get_mean_sigma_weights() const
  {
    return sigma_weights_mean_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::SigmaWeights&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::get_covariance_sigma_weights() const
  {
    return sigma_weights_cov_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::set_state_mean_function(
      StateMeanFunction state_mean_function)
  {
    state_mean_function_ = std::move(state_mean_function);
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::StateMeanFunction&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::get_state_mean_function() const
  {
    return state_mean_function_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::set_measurement_mean_function(
      MeasurementMeanFunction meas_mean_function)
  {
    meas_mean_function_ = std::move(meas_mean_function);
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::MeasurementMeanFunction&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::get_measurement_mean_function() const
  {
    return meas_mean_function_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::calculate_weights()
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

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF>::generate_sigma_points()
  {
    // Calculate the (weighted) matrix square root of the state covariance
    // matrix
    cholesky_.compute(P_);
    const N_by_N& sqrt_P = eta_ * cholesky_.matrixL().toDenseMatrix();

    // First sigma point is the current state mean
    sigma_points_[0] = x_;

    // Next N sigma points are the current state mean perturbed by the columns
    // of sqrt_P, and the N sigma points after that are the same perturbations
    // but negated
    for (std::size_t i = 0; i < N; ++i)
    {
      const N_by_1& perturb = sqrt_P.col(i);
      sigma_points_[i + 1] = x_ + STATE(perturb);
      sigma_points_[N + i + 1] = x_ + STATE(-perturb);
    }
  }
} // namespace unscented
