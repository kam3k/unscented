#include "unscented/components.h"

namespace unscented
{
///////////////////////////////////////////////////////////////////////////////
// Vectors
///////////////////////////////////////////////////////////////////////////////
template <std::size_t DIM, std::size_t NUM_VECS>
Eigen::Matrix<double, DIM, 1> vector_mean_function(
    const std::array<Eigen::Matrix<double, DIM, 1>, NUM_VECS>& vectors,
    const std::array<double, NUM_VECS>& weights)
{
  Eigen::Matrix<double, DIM, 1> mean_vector;
  for (std::size_t i = 0; i < NUM_VECS; ++i)
  {
    mean_vector += vectors[i] * weights[i];
  }
  return mean_vector;
}

///////////////////////////////////////////////////////////////////////////////
// UnitComplex
///////////////////////////////////////////////////////////////////////////////
UnitComplex::UnitComplex(const Vector1& vec) : UnitComplex(vec(0))
{
}

UnitComplex::UnitComplex(double angle)
  : UnitComplex(std::cos(angle), std::sin(angle))
{
}

UnitComplex::UnitComplex(double a_in, double b_in) : a(a_in), b(b_in)
{
  static const double EPS = 1e-6;
  const auto sq_norm = a * a + b * b;
  if (std::abs(1 - sq_norm) > EPS)
  {
    const auto norm = std::sqrt(sq_norm);
    a /= norm;
    b /= norm;
  }
}

double UnitComplex::angle() const
{
  return std::atan2(b, a);
}

UnitComplex operator+(const UnitComplex& lhs, const UnitComplex& rhs)
{
  return UnitComplex(lhs.angle() + rhs.angle());
}

Vector1 operator-(const UnitComplex& lhs, const UnitComplex& rhs)
{
  return Vector1(UnitComplex(rhs.angle() - lhs.angle()).angle());
}

UnitComplex unit_complex_mean_function(
    const std::array<UnitComplex, 2 * UnitComplex::DOF + 1>& states,
    const std::array<double, 2 * UnitComplex::DOF + 1>& weights)
{
  double a{0.0};
  double b{0.0};
  for (std::size_t i = 0; i < (2 * UnitComplex::DOF + 1); ++i)
  {
    a += weights[i] * states[i].a;
    b += weights[i] * states[i].b;
  }
  return UnitComplex(a, b);
}
} // namespace unscented
