#include "unscented/primitives.h"

namespace unscented
{
template <typename PRIMITIVE, std::size_t ARRAY_SIZE>
PRIMITIVE mean_function(const std::array<PRIMITIVE, ARRAY_SIZE>& primitives,
                        const std::array<double, ARRAY_SIZE>& weights)
{
  PRIMITIVE mean_primitive;
  for (std::size_t i = 0; i < ARRAY_SIZE; ++i)
  {
    mean_primitive = mean_primitive + primitives[i] * weights[i];
  }
  return mean_primitive;
}

///////////////////////////////////////////////////////////////////////////////
// UnitComplex
///////////////////////////////////////////////////////////////////////////////
UnitComplex::UnitComplex(const Vector<1>& vec) : UnitComplex(vec(0))
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

Vector<1> operator-(const UnitComplex& lhs, const UnitComplex& rhs)
{
  return Vector<1>(UnitComplex(rhs.angle() - lhs.angle()).angle());
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
