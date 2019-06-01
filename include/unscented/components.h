#include <Eigen/Dense>

namespace unscented
{
using Vector1d = Eigen::Matrix<double, 1, 1>;

using Vector1f = Eigen::Matrix<float, 1, 1>;

class Angle
{
public:
  Angle() = default;

  Angle(double a, double b);

  explicit Angle(double theta);

  explicit Angle(const Vector1d& vec);

  double theta() const;

  friend Angle operator*(const Angle&, double);

private:
  double a = 0.0;
  double b = 0.0;
};

Angle operator+(const Angle& lhs, const Angle& rhs);

Vector1d operator-(const Angle& lhs, const Angle& rhs);

Angle operator*(const Angle& angle, double scale);
} // namespace unscented
