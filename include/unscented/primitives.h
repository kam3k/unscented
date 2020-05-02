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
// UnitComplex
///////////////////////////////////////////////////////////////////////////////
struct UnitComplex
{
  static constexpr std::size_t DOF = 1;

  UnitComplex() = default;

  explicit UnitComplex(double angle);

  UnitComplex(double a_in, double b_in);

  double angle() const;

  double a{1.0};

  double b{0.0};
};

UnitComplex operator+(const UnitComplex& lhs, double angle);

UnitComplex operator+(const UnitComplex& lhs,
                      const Vector<UnitComplex::DOF>& vec);

Vector<UnitComplex::DOF> operator-(const UnitComplex& lhs, const UnitComplex& rhs);

///////////////////////////////////////////////////////////////////////////////
// SO2
///////////////////////////////////////////////////////////////////////////////
struct SO2
{
  static constexpr std::size_t DOF = 1;

  SO2() = default;

  explicit SO2(const Vector<1>& vec);

  explicit SO2(const Eigen::Matrix2d& rotation_matrix);

  explicit SO2(const UnitComplex& unit_complex);

  explicit SO2(double angle);

  UnitComplex unit_complex() const;

  double angle() const;

  Eigen::Matrix2d rotation_matrix{Eigen::Matrix2d::Identity()};
};

SO2 operator+(const SO2& lhs, const SO2& rhs);

Vector<1> operator-(const SO2& lhs, const SO2& rhs);

SO2 SO2_mean_function(const std::array<SO2, 2 * SO2::DOF + 1>& states,
                      const std::array<double, 2 * SO2::DOF + 1>& weights);

///////////////////////////////////////////////////////////////////////////////
// SE(2)
///////////////////////////////////////////////////////////////////////////////
struct SE2
{
  static constexpr std::size_t DOF = 3;

  SE2() = default;

  explicit SE2(const Vector<3>& vec);

  explicit SE2(const Eigen::Affine2d& affine);

  SE2(const Vector<2>& position, const SO2& rotation);

  SE2(const Vector<2>& position, const UnitComplex& unit_complex);

  SE2(const Vector<2>& position, double angle);

  SE2(const Vector<2>& position, double a, double b);

  SE2(double x, double y, const SO2& rotation);

  SE2(double x, double y, const UnitComplex& unit_complex);

  SE2(double x, double y, double angle);

  SE2(double x, double y, double a, double b);

  Eigen::Affine2d affine{Eigen::Affine2d::Identity()};
};

SE2 operator+(const SE2& lhs, const SE2& rhs);

Vector<3> operator-(const SE2& lhs, const SE2& rhs);

SE2 SE2_mean_function(const std::array<SE2, 2 * SE2::DOF + 1>& states,
                      const std::array<double, 2 * SE2::DOF + 1>& weights);

///////////////////////////////////////////////////////////////////////////////
// UnitQuaternion
///////////////////////////////////////////////////////////////////////////////
struct SO3; // forward declaration so UnitQuaternion can use it

struct UnitQuaternion
{
  static constexpr std::size_t DOF = 3;

  UnitQuaternion() = default;

  explicit UnitQuaternion(const Vector<3>& vec);

  explicit UnitQuaternion(const Eigen::Quaterniond& q);

  explicit UnitQuaternion(const SO3& so3);

  explicit UnitQuaternion(const Eigen::Matrix3d& rotation_matrix);

  UnitQuaternion(double w, double x, double y, double z);

  Eigen::Quaterniond q{Eigen::Quaterniond::Identity()};
};

UnitQuaternion operator+(const UnitQuaternion& lhs, const UnitQuaternion& rhs);

Vector<3> operator-(const UnitQuaternion& lhs, const UnitQuaternion& rhs);

UnitQuaternion unit_quaternion_mean_function(
    const std::array<UnitQuaternion, 2 * UnitQuaternion::DOF + 1>& states,
    const std::array<double, 2 * UnitQuaternion::DOF + 1>& weights);

///////////////////////////////////////////////////////////////////////////////
// SO3
///////////////////////////////////////////////////////////////////////////////
struct SO3
{
  static constexpr std::size_t DOF = 3;

  SO3() = default;

  explicit SO3(const Vector<3>& vec);

  explicit SO3(const Eigen::Matrix3d& rotation_matrix);

  explicit SO3(const UnitQuaternion& q);

  explicit SO3(const Eigen::Quaterniond& q);

  Eigen::Matrix3d rotation_matrix{Eigen::Matrix3d::Identity()};
};

SO3 operator+(const SO3& lhs, const SO3& rhs);

Vector<3> operator-(const SO3& lhs, const SO3& rhs);

SO3 SO3_mean_function(const std::array<SO3, 2 * SO3::DOF + 1>& states,
                      const std::array<double, 2 * SO3::DOF + 1>& weights);

///////////////////////////////////////////////////////////////////////////////
// SE(3)
///////////////////////////////////////////////////////////////////////////////
struct SE3
{
  static constexpr std::size_t DOF = 6;

  SE3() = default;

  explicit SE3(const Vector<6>& vec);

  explicit SE3(const Eigen::Affine3d& affine);

  SE3(const Vector<3>& position, const SO3& rotation);

  SE3(const Vector<3>& position, const UnitQuaternion& q);

  SE3(const Vector<3>& position, const Eigen::Quaterniond& q);

  SE3(const Vector<3>& position, const Eigen::Matrix3d& rotation_matrix);

  SE3(double x, double y, double z, const SO3& rotation);

  SE3(double x, double y, double z, const UnitQuaternion& q);

  SE3(double x, double y, double z, const Eigen::Quaterniond& q);

  SE3(double x, double y, double z, const Eigen::Matrix3d& rotation_matrix);

  Eigen::Affine3d affine{Eigen::Affine3d::Identity()};
};

SE3 operator+(const SE3& lhs, const SE3& rhs);

Vector<6> operator-(const SE3& lhs, const SE3& rhs);

SE3 SE3_mean_function(const std::array<SE3, 2 * SE3::DOF + 1>& states,
                      const std::array<double, 2 * SE3::DOF + 1>& weights);
} // namespace unscented

#endif
