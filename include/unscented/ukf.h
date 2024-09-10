#ifndef UNSCENTED_UKF_H
#define UNSCENTED_UKF_H

#include <Eigen/Dense>

#include <array>

namespace unscented
{
template <typename STATE, typename MEAS>
class UKF
{
public:
  static constexpr std::size_t N = STATE::DOF;

  static constexpr std::size_t M = MEAS::DOF;

  static constexpr std::size_t NUM_SIGMA_POINTS = 2 * N + 1;

  using State = STATE;

  using Measurement = MEAS;

  using N_by_1 = Eigen::Matrix<double, N, 1>;

  using N_by_N = Eigen::Matrix<double, N, N>;

  using M_by_1 = Eigen::Matrix<double, M, 1>;

  using M_by_M = Eigen::Matrix<double, M, M>;

  using N_by_M = Eigen::Matrix<double, N, M>;

  using SigmaPoints = std::array<STATE, NUM_SIGMA_POINTS>;

  using SigmaWeights = std::array<double, NUM_SIGMA_POINTS>;

  using MeasurementSigmaPoints = std::array<MEAS, NUM_SIGMA_POINTS>;

  UKF();

  template <typename SYS_MODEL, typename... PARAMS>
  void predict(const SYS_MODEL& system_model, PARAMS...);

  template <typename MEAS_MODEL, typename... PARAMS>
  void correct(const MEAS_MODEL& meas_model, PARAMS...);

  template <typename MEAS_MODEL, typename... PARAMS>
  void correct(const MEAS_MODEL& meas_model, MEAS meas, PARAMS...);

  void generate_sigma_points(const N_by_1& delta = N_by_1::Zero());

  void set_state(const STATE& state);

  void set_state(STATE&& state);

  const STATE& get_state() const;

  void set_measurement(const MEAS& measurement);

  void set_measurement(MEAS&& measurement);

  const MEAS& get_measurement() const;

  void set_state_covariance(const N_by_N& state_covariance);

  void set_state_covariance(N_by_N&& state_covariance);

  const N_by_N& get_state_covariance() const;

  void set_process_covariance(const N_by_N& process_covariance);

  void set_process_covariance(N_by_N&& process_covariance);

  const N_by_N& get_process_covariance() const;

  void set_measurement_covariance(const M_by_M& measurement_covariance);

  void set_measurement_covariance(M_by_M&& measurement_covariance);

  const M_by_M& get_measurement_covariance() const;

  const MEAS& get_expected_measurement() const;

  const M_by_M& get_expected_measurement_covariance() const;

  const N_by_M& get_cross_covariance() const;

  const N_by_M& get_kalman_gain() const;

  const M_by_1& get_innovation() const;

  const SigmaPoints& get_sigma_points() const;

  const MeasurementSigmaPoints& get_measurement_sigma_points() const;

  void set_weight_coefficients(double alpha, double beta, double kappa);

  const SigmaWeights& get_mean_sigma_weights() const;

  const SigmaWeights& get_covariance_sigma_weights() const;

private:
  void calculate_weights();

  STATE x_;

  MEAS y_;

  MEAS y_hat_;

  N_by_N P_;

  N_by_N Q_;

  M_by_M R_;

  M_by_M Pyy_;

  N_by_M Pxy_;

  N_by_M K_;

  M_by_1 innovation_;

  SigmaPoints sigma_points_;

  MeasurementSigmaPoints meas_sigma_points_;

  SigmaWeights sigma_weights_mean_;

  SigmaWeights sigma_weights_cov_;

  double alpha_ = 0.001;

  double beta_ = 2.0;

  double kappa_ = 0.0;

  double lambda_;

  double eta_;

  Eigen::LLT<N_by_N> cholesky_;
};

template <typename MANIFOLD, std::size_t ARRAY_SIZE>
MANIFOLD calculate_mean_manifold(
    const std::array<MANIFOLD, ARRAY_SIZE>& manifolds,
    const std::array<double, ARRAY_SIZE>& weights);
} // namespace unscented

#endif
