#include "unscented/primitives.h"

namespace unscented
{
///////////////////////////////////////////////////////////////////////////////
// UnitComplex
///////////////////////////////////////////////////////////////////////////////
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

UnitComplex operator+(const UnitComplex& lhs, double angle)
{
  return UnitComplex(lhs.angle() + angle);
}

UnitComplex operator+(const UnitComplex& lhs,
                      const Vector<UnitComplex::DOF>& vec)
{
  return lhs + vec(0);
}

Vector<UnitComplex::DOF> operator-(const UnitComplex& lhs,
                                   const UnitComplex& rhs)
{
  return Vector<UnitComplex::DOF>(
      UnitComplex(rhs.angle() - lhs.angle()).angle());
}
} // namespace unscented
