#include <Eigen/Core>
#include <Eigen/Cholesky>

namespace unscented
{
  template <typename STATE, typename STATE_DOF, typename MEAS,
            typename MEAS_DOF, typename T = float>
  class UKF
  {
  public:
    using N_by_1 = Eigen::Matrix<T, STATE_DOF, 1>;

    using N_by_N = Eigen::Matrix<T, STATE_DOF, STATE_DOF>;

    using M_by_1 = Eigen::Matrix<T, MEAS_DOF, 1>;

    using M_by_M = Eigen::Matrix<T, MEAS_DOF, MEAS_DOF>;

    using M_by_N = Eigen::Matrix<T, STATE_DOF, MEAS_DOF>;

    void setState(const STATE& state);

    void setState(STATE&& state);

    const STATE& getState() const;

    void setMeasurement(const MEAS& measurement);

    void setMeasurement(MEAS&& measurement);

    const MEAS& getMeasurement() const;

    void setStateCovariance(const N_by_N& state_covariance);

    void setStateCovariance(N_by_N&& state_covariance);

    const N_by_N& getStateCovariance() const;

    void setProcessCovariance(const N_by_N& process_covariance);

    void setProcessCovariance(N_by_N&& process_covariance);

    const N_by_N& getProcessCovariance() const;

    void setMeasurementCovariance(const M_by_M& measurement_covariance);

    void setMeasurementCovariance(M_by_M&& measurement_covariance);

    const M_by_M& getMeasurementCovariance() const;

    const M_by_M& getExpectedMeasurementCovariance() const;

    const M_by_N& getCrossCovariance() const;

    const M_by_N& getKalmanGain() const;

    const M_by_1& getInnovation() const;

  private:
    STATE x_;

    MEAS y_;

    N_by_N P_;

    N_by_N Q_;

    M_by_M R_;

    M_by_M Py_;

    M_by_N Pxy_;

    M_by_N K_;

    M_by_1 innovation_;
  };
} // namespace unscented
