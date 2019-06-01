#include "unscented/components.h"

namespace unscented
{
Angle::Angle(double a, double b) : a(a), b(b)
{
}

Angle::Angle(double theta)
{
  a = std::cos(theta);
  b = std::sin(theta);
}

Angle::Angle(const Vector1d& vec) : Angle(vec(0))
{
}

double Angle::theta() const
{
  return std::atan2(b, a);
}

Angle operator+(const Angle& lhs, const Angle& rhs)
{
  return Angle(lhs.theta() + rhs.theta());
}

Vector1d operator-(const Angle& lhs, const Angle& rhs)
{
  return Vector1d(Angle(rhs.theta() - lhs.theta()).theta());
}

Angle operator*(const Angle& angle, double scale)
{
  return Angle(scale * angle.a, scale * angle.b);
}
} // namespace unscented
