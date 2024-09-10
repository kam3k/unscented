#ifndef UNSCENTED_PRIMITIVES_HPP
#define UNSCENTED_PRIMITIVES_HPP

#include "unscented/primitives.h"

namespace unscented
{
///////////////////////////////////////////////////////////////////////////////
// Angle
///////////////////////////////////////////////////////////////////////////////
Angle::Angle(const Vector<Angle::DOF>& vec) : Angle(vec[0])
{
}

Angle::Angle(double angle) : Angle(std::cos(angle), std::sin(angle))
{
}

Angle::Angle(double cos_angle, double sin_angle)
  : cos_angle_(cos_angle), sin_angle_(sin_angle)
{
  static const double EPS = 1e-6;
  const auto sq_norm = cos_angle_ * cos_angle_ + sin_angle_ * sin_angle_;
  if (std::abs(1.0 - sq_norm) > EPS)
  {
    const auto norm = std::sqrt(sq_norm);
    cos_angle_ /= norm;
    sin_angle_ /= norm;
  }
}

double Angle::get_angle() const
{
  return std::atan2(sin_angle_, cos_angle_);
}

Vector<Angle::DOF> Angle::get_vector() const
{
  return Vector<Angle::DOF>{get_angle()};
}

Angle operator+(const Angle& lhs, const Vector<Angle::DOF>& vec)
{
  return Angle(lhs.get_vector() + vec);
}

Vector<Angle::DOF> operator-(const Angle& lhs, const Angle& rhs)
{
  return Vector<Angle::DOF>(
      Angle(rhs.get_angle() - lhs.get_angle()).get_angle());
}
} // namespace unscented

#endif
