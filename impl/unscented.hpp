#include "unscented.h"

#include <cassert>
#include <numeric>

namespace unscented
{
  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::UKF()
  {
    P_ = N_by_N::Identity();
    Q_ = N_by_N::Identity();
    R_ = M_by_M::Identity();

    calculateWeights();

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
            std::size_t MEAS_DOF, typename SCALAR>
  template <typename... PARAMS>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::predict(
      const std::function<void(STATE&, PARAMS...)>& /*system_model*/, PARAMS...)
  {
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  template <typename... PARAMS>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::correct(
      const std::function<MEAS(const STATE&, PARAMS...)>& /*meas_model*/,
      PARAMS...)
  {
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  template <typename... PARAMS>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::correct(
      const std::function<MEAS(const STATE&, PARAMS...)>& /*meas_model*/,
      MEAS /*meas*/, PARAMS...)
  {
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::setState(
      const STATE& state)
  {
    x_ = state;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::setState(STATE&& state)
  {
    x_ = std::move(state);
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  const STATE& UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::getState() const
  {
    return x_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::setMeasurement(
      const MEAS& measurement)
  {
    y_ = measurement;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::setMeasurement(
      MEAS&& measurement)
  {
    y_ = std::move(measurement);
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  const MEAS& UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::getMeasurement()
      const
  {
    return y_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::setStateCovariance(
      const N_by_N& state_covariance)
  {
    P_ = state_covariance;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::setStateCovariance(
      N_by_N&& state_covariance)
  {
    P_ = std::move(state_covariance);
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::N_by_N&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::getStateCovariance() const
  {
    return P_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::setProcessCovariance(
      const N_by_N& process_covariance)
  {
    Q_ = process_covariance;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::setProcessCovariance(
      N_by_N&& process_covariance)
  {
    Q_ = std::move(process_covariance);
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::N_by_N&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::getProcessCovariance() const
  {
    return Q_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::setMeasurementCovariance(
      const M_by_M& measurement_covariance)
  {
    R_ = measurement_covariance;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::setMeasurementCovariance(
      M_by_M&& measurement_covariance)
  {
    R_ = std::move(measurement_covariance);
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::M_by_M&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::getMeasurementCovariance()
      const
  {
    return R_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  const MEAS&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::getExpectedMeasurement() const
  {
    return y_hat_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::M_by_M&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF,
      SCALAR>::getExpectedMeasurementCovariance() const
  {
    return Py_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::M_by_N&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::getCrossCovariance() const
  {
    return Pxy_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::M_by_N&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::getKalmanGain() const
  {
    return K_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::M_by_1&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::getInnovation() const
  {
    return innovation_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::SigmaPoints&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::getSigmaPoints() const
  {
    return sigma_points_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::setWeightCoefficients(
      SCALAR alpha, SCALAR beta, SCALAR kappa)
  {
    alpha_ = alpha;
    beta_ = beta;
    kappa_ = kappa;
    calculateWeights();
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::SigmaWeights&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::getMeanSigmaWeights() const
  {
    return sigma_weights_mean_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::SigmaWeights&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::getCovarianceSigmaWeights()
      const
  {
    return sigma_weights_cov_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::setStateMeanFunction(
      StateMeanFunction state_mean_function)
  {
    state_mean_function_ = std::move(state_mean_function);
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF,
                     SCALAR>::StateMeanFunction&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::getStateMeanFunction() const
  {
    return state_mean_function_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  void
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::setMeasurementMeanFunction(
      MeasurementMeanFunction meas_mean_function)
  {
    meas_mean_function_ = std::move(meas_mean_function);
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  const typename UKF<STATE, STATE_DOF, MEAS, MEAS_DOF,
                     SCALAR>::MeasurementMeanFunction&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::getMeasurementMeanFunction()
      const
  {
    return meas_mean_function_;
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::calculateWeights()
  {
    lambda_ = alpha_ * alpha_ * (N + kappa_) - N;

    assert(N + lambda_ > 1e-6);

    eta_ = std::sqrt(N + lambda_);

    const auto w_mean_0 = lambda_ / (N + lambda_);
    const auto w_cov_0 = w_mean_0 + (1.0 - alpha_ * alpha_ + beta_);
    const auto w_i = 1.0 / (2.0 * (N + lambda_));

    sigma_weights_mean_[0] = w_mean_0;
    sigma_weights_cov_[0] = w_cov_0;

    for (auto i = 1; i < NUM_SIGMA_POINTS; ++i)
    {
      sigma_weights_mean_[i] = w_i;
      sigma_weights_cov_[i] = w_i;
    }
  }

  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, SCALAR>::generateSigmaPoints()
  {
    // Calculate the (weighted) matrix square root of the state covariance matrix
    cholesky.compute(P_);
    const N_by_N& sqrt_P = eta_ * cholesky.matrixL().toDenseMatrix();

    // First sigma point is the current state mean
    sigma_points_[0] = x_;

    // Next N sigma points are the current state mean perturbed by the columns
    // of sqrt_P, and the N sigma points after that are the same perturbations
    // but subtracted
    for (std::size_t i = 0; i < N; ++i)
    {
      STATE perturb(sqrt_P.col(i));
      sigma_points_[i + 1] = x_ + STATE(sqrt_P.col(i));
      sigma_points_[N + i + 1] = x_ + STATE(-sqrt_P.col(i));
    }
  }
} // namespace unscented
