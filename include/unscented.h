#include <Eigen/Dense>

#include <array>

namespace unscented
{
  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF>
  class UKF
  {
  public:
    static constexpr auto NUM_SIGMA_POINTS = 2 * STATE_DOF + 1;

    static constexpr auto N = STATE_DOF;

    static constexpr auto M = MEAS_DOF;

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

    using StateMeanFunction =
        std::function<STATE(const SigmaPoints&, const SigmaWeights&)>;

    using MeasurementMeanFunction =
        std::function<MEAS(const MeasurementSigmaPoints&, const SigmaWeights&)>;

    UKF();

    template <typename SYS_MODEL, typename... PARAMS>
    void predict(const SYS_MODEL& system_model, PARAMS...);

    template <typename MEAS_MODEL, typename... PARAMS>
    void correct(const MEAS_MODEL& meas_model, PARAMS...);

    template <typename MEAS_MODEL, typename... PARAMS>
    void correct(const MEAS_MODEL& meas_model, MEAS meas, PARAMS...);

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

    void set_state_mean_function(StateMeanFunction state_mean_function);

    const StateMeanFunction& get_state_mean_function() const;

    void set_measurement_mean_function(
        MeasurementMeanFunction meas_mean_function);

    const MeasurementMeanFunction& get_measurement_mean_function() const;

  private:
    void calculate_weights();

    void generate_sigma_points();

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

    StateMeanFunction state_mean_function_;

    MeasurementMeanFunction meas_mean_function_;

    Eigen::LLT<N_by_N> cholesky_;
  };
} // namespace unscented
