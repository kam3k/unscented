#ifndef UNSCENTED_PRIMITIVES_H
#define UNSCENTED_PRIMITIVES_H

#include <Eigen/Dense>

namespace unscented
{
///////////////////////////////////////////////////////////////////////////////
// Vectors
///////////////////////////////////////////////////////////////////////////////
template <std::size_t DIM>
using Vector = Eigen::Matrix<double, DIM, 1>;

///////////////////////////////////////////////////////////////////////////////
// Angle
///////////////////////////////////////////////////////////////////////////////
class Angle
{
public:
  static constexpr std::size_t DOF = 1;

  Angle() = default;

  explicit Angle(const Vector<DOF>& vec);

  explicit Angle(double angle);

  Angle(double cos_angle, double sin_angle);

  double get_angle() const;

  Vector<DOF> get_vector() const;

private:
  double cos_angle_{1.0};

  double sin_angle_{0.0};
};

Angle operator+(const Angle& lhs, const Vector<Angle::DOF>& vec);

Vector<Angle::DOF> operator-(const Angle& lhs, const Angle& rhs);

///////////////////////////////////////////////////////////////////////////////
// SE(2)
///////////////////////////////////////////////////////////////////////////////
class SE2
{
public:
  static constexpr std::size_t DOF = 3;

  SE2() = default;

  explicit SE2(const Vector<DOF>& vec);

  explicit SE2(const Eigen::Affine2d& affine);

  SE2(const Vector<2>& position, const Angle& angle);

  SE2(const Vector<2>& position, double angle);

  SE2(double x, double y, const Angle& angle);

  SE2(double x, double y, double angle);

private:
  Eigen::Affine2d affine{Eigen::Affine2d::Identity()};
};

SE2 operator+(const SE2& lhs, const Vector<SE2::DOF>& vec);

Vector<SE2::DOF> operator-(const SE2& lhs, const SE2& rhs);

///////////////////////////////////////////////////////////////////////////////
// UnitQuaternion
///////////////////////////////////////////////////////////////////////////////
class UnitQuaternion
{
public:
  static constexpr std::size_t DOF = 3;

  UnitQuaternion() = default;

  explicit UnitQuaternion(const Vector<DOF>& vec);

  explicit UnitQuaternion(const Eigen::Quaterniond& q);

  UnitQuaternion(double w, double x, double y, double z);

private:
  Eigen::Quaterniond q{Eigen::Quaterniond::Identity()};
};

UnitQuaternion operator+(const UnitQuaternion& lhs,
                         const Vector<UnitQuaternion::DOF>& vec);

Vector<UnitQuaternion::DOF> operator-(const UnitQuaternion& lhs,
                                      const UnitQuaternion& rhs);

///////////////////////////////////////////////////////////////////////////////
// SE(3)
///////////////////////////////////////////////////////////////////////////////
class SE3
{
public:
  static constexpr std::size_t DOF = 6;

  SE3() = default;

  explicit SE3(const Vector<DOF>& vec);

  explicit SE3(const Eigen::Affine3d& affine);

  SE3(const Vector<3>& position, const UnitQuaternion& q);

  SE3(const Vector<3>& position, const Eigen::Quaterniond& q);

  SE3(double x, double y, double z, const UnitQuaternion& q);

  SE3(double x, double y, double z, const Eigen::Quaterniond& q);

private:
  Eigen::Affine3d affine{Eigen::Affine3d::Identity()};
};

SE3 operator+(const SE3& lhs, const Vector<SE3::DOF>& vec);

Vector<SE3::DOF> operator-(const SE3& lhs, const SE3& rhs);
} // namespace unscented

#endif
