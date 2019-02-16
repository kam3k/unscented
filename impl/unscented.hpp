#include "unscented.h"

namespace unscented
{
  template <typename STATE, typename STATE_DOF, typename MEAS,
            typename MEAS_DOF, typename T = float>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, T>::setState(const STATE& state)
  {
    x_ = state;
  }

  template <typename STATE, typename STATE_DOF, typename MEAS,
            typename MEAS_DOF, typename T = float>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, T>::setState(STATE&& state)
  {
    x_ = std::move(state);
  }

  template <typename STATE, typename STATE_DOF, typename MEAS,
            typename MEAS_DOF, typename T = float>
  const STATE& UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, T>::getState() const
  {
    return x_;
  }

  template <typename STATE, typename STATE_DOF, typename MEAS,
            typename MEAS_DOF, typename T = float>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, T>::setMeasurement(
      const MEAS& measurement)
  {
    y_ = measurement;
  }

  template <typename STATE, typename STATE_DOF, typename MEAS,
            typename MEAS_DOF, typename T = float>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, T>::setMeasurement(
      MEAS&& measurement)
  {
    y_ = std::move(measurement);
  }

  template <typename STATE, typename STATE_DOF, typename MEAS,
            typename MEAS_DOF, typename T = float>
  const MEAS& UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, T>::getMeasurement() const
  {
    return y_;
  }

  template <typename STATE, typename STATE_DOF, typename MEAS,
            typename MEAS_DOF, typename T = float>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, T>::setStateCovariance(
      const N_by_N& state_covariance)
  {
    P_ = state_covariance;
  }

  template <typename STATE, typename STATE_DOF, typename MEAS,
            typename MEAS_DOF, typename T = float>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, T>::setStateCovariance(
      N_by_N&& state_covariance)
  {
    P_ = std::move(state_covariance);
  }

  template <typename STATE, typename STATE_DOF, typename MEAS,
            typename MEAS_DOF, typename T = float>
  const N_by_N& UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, T>::getStateCovariance()
      const
  {
    return P_;
  }

  template <typename STATE, typename STATE_DOF, typename MEAS,
            typename MEAS_DOF, typename T = float>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, T>::setProcessCovariance(
      const N_by_N& process_covariance)
  {
    Q_ = process_covariance;
  }

  template <typename STATE, typename STATE_DOF, typename MEAS,
            typename MEAS_DOF, typename T = float>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, T>::setProcessCovariance(
      N_by_N&& process_covariance)
  {
    Q_ = std::move(process_covariance);
  }

  template <typename STATE, typename STATE_DOF, typename MEAS,
            typename MEAS_DOF, typename T = float>
  const N_by_N& UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, T>::getProcessCovariance()
      const
  {
    return Q_;
  }

  template <typename STATE, typename STATE_DOF, typename MEAS,
            typename MEAS_DOF, typename T = float>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, T>::setMeasurementCovariance(
      const M_by_M& measurement_covariance)
  {
    R_ = measurement_covariance;
  }

  template <typename STATE, typename STATE_DOF, typename MEAS,
            typename MEAS_DOF, typename T = float>
  void UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, T>::setMeasurementCovariance(
      M_by_M&& measurement_covariance)
  {
    R_ = std::move(measurement_covariance);
  }

  template <typename STATE, typename STATE_DOF, typename MEAS,
            typename MEAS_DOF, typename T = float>
  const M_by_M&
  UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, T>::getMeasurementCovariance() const
  {
    return R_;
  }

  template <typename STATE, typename STATE_DOF, typename MEAS,
            typename MEAS_DOF, typename T = float>
  const M_by_M& UKF<STATE, STATE_DOF, MEAS, MEAS_DOF,
                    T>::getExpectedMeasurementCovariance() const
  {
    return Py_;
  }

  template <typename STATE, typename STATE_DOF, typename MEAS,
            typename MEAS_DOF, typename T = float>
  const M_by_N& UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, T>::getCrossCovariance()
      const
  {
    return Pxy_;
  }

  template <typename STATE, typename STATE_DOF, typename MEAS,
            typename MEAS_DOF, typename T = float>
  const M_by_N& UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, T>::getKalmanGain() const
  {
    return K_;
  }

  template <typename STATE, typename STATE_DOF, typename MEAS,
            typename MEAS_DOF, typename T = float>
  const M_by_1& UKF<STATE, STATE_DOF, MEAS, MEAS_DOF, T>::getInnovation() const
  {
    return innovation_;
  }
} // namespace unscented
