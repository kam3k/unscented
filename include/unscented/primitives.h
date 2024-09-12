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
// Scalars
///////////////////////////////////////////////////////////////////////////////
struct Scalar
{
  Scalar() = default;
  Scalar(double r); // not explicit on purpose to allow implicit conversion
  static constexpr std::size_t DOF = 1;
  double value;
};

Scalar operator+(const Scalar& lhs, const Vector<Scalar::DOF>& vec);

Vector<Scalar::DOF> operator-(const Scalar& lhs, const Scalar& rhs);

///////////////////////////////////////////////////////////////////////////////
// Compounds
///////////////////////////////////////////////////////////////////////////////
namespace detail
{
// Helper to calculate total DOF of the elements of a tuple
template <typename Tuple, std::size_t... I>
constexpr std::size_t total_dof_impl(std::index_sequence<I...>);

template <typename Tuple>
constexpr std::size_t total_dof();

// Apply the addition of a vector to a state tuple
template <typename Tuple, std::size_t... I>
Tuple apply_addition(const Tuple& state,
                     const unscented::Vector<total_dof<Tuple>()>& vec,
                     std::index_sequence<I...>);

// Apply the subtraction of two state tuples to get a vector
template <typename Tuple, std::size_t... I>
unscented::Vector<total_dof<Tuple>()> apply_subtraction(
    const Tuple& lhs, const Tuple& rhs, std::index_sequence<I...>);
} // namespace detail

template <typename... Ts>
struct Compound
{
  using Tuple = std::tuple<Ts...>;
  static constexpr std::size_t TupleSize = std::tuple_size_v<Tuple>;
  static constexpr std::size_t DOF = detail::total_dof<Tuple>();
  Compound() = default;
  Compound(const Ts&... args) : data(std::make_tuple(args...)) {}
  Compound(Ts&&... args) : data(std::make_tuple(args...)) {}
  Tuple data;
};

template <typename... Ts>
Compound<Ts...> operator+(const Compound<Ts...>& state,
                          const unscented::Vector<Compound<Ts...>::DOF>& vec);

template <typename... Ts>
unscented::Vector<Compound<Ts...>::DOF> operator-(const Compound<Ts...>& lhs,
                                                  const Compound<Ts...>& rhs);

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
// SE(2) / Pose2d
///////////////////////////////////////////////////////////////////////////////
using SE2 = Compound<Vector<2>, Angle>;
using Pose2d = SE2;

///////////////////////////////////////////////////////////////////////////////
// SE(3) / Pose3d
///////////////////////////////////////////////////////////////////////////////
using SE3 = Compound<Vector<3>, UnitQuaternion>;
using Pose3d = SE3;

} // namespace unscented

#include <unscented/primitives.hpp>

#endif
