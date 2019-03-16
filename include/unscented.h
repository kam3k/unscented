#include <Eigen/Dense>

#include <array>

namespace unscented
{
  template <typename STATE, std::size_t STATE_DOF, typename MEAS,
            std::size_t MEAS_DOF, typename SCALAR = float>
  class UKF
  {
  public:
    static constexpr auto NUM_SIGMA_POINTS = 2 * STATE_DOF + 1;

    static constexpr auto N = STATE_DOF;

    static constexpr auto M = MEAS_DOF;

    using State = STATE;

    using Measurement = MEAS;

    using N_by_1 = Eigen::Matrix<SCALAR, N, 1>;

    using N_by_N = Eigen::Matrix<SCALAR, N, N>;

    using M_by_1 = Eigen::Matrix<SCALAR, M, 1>;

    using M_by_M = Eigen::Matrix<SCALAR, M, M>;

    using N_by_M = Eigen::Matrix<SCALAR, N, M>;

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

    const MEAS& getExpectedMeasurement() const;

    const M_by_M& getExpectedMeasurementCovariance() const;

    const N_by_M& getCrossCovariance() const;

    const N_by_M& getKalmanGain() const;

    const M_by_1& getInnovation() const;

    const SigmaPoints& getSigmaPoints() const;

    const MeasurementSigmaPoints& getMeasurementSigmaPoints() const;

    void setWeightCoefficients(double alpha, double beta, double kappa);

    const SigmaWeights& getMeanSigmaWeights() const;

    const SigmaWeights& getCovarianceSigmaWeights() const;

    void setStateMeanFunction(StateMeanFunction state_mean_function);

    const StateMeanFunction& getStateMeanFunction() const;

    void setMeasurementMeanFunction(MeasurementMeanFunction meas_mean_function);

    const MeasurementMeanFunction& getMeasurementMeanFunction() const;

  private:
    void calculateWeights();

    void generateSigmaPoints();

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

    Eigen::LLT<N_by_N> cholesky;
  };
} // namespace unscented
