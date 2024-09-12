#ifndef UNSCENTED_PRIMITIVES_HPP
#define UNSCENTED_PRIMITIVES_HPP

#include <utility>
#include "unscented/primitives.h"

namespace unscented
{
///////////////////////////////////////////////////////////////////////////////
// Scalars
///////////////////////////////////////////////////////////////////////////////
Scalar::Scalar(double r) : value(r)
{
}

Scalar operator+(const Scalar& lhs, const Vector<Scalar::DOF>& vec)
{
  return lhs.value + vec[0];
}

Vector<Scalar::DOF> operator-(const Scalar& lhs, const Scalar& rhs)
{
  return Vector<Scalar::DOF>{lhs.value - rhs.value};
}

///////////////////////////////////////////////////////////////////////////////
// Compound states (tuples)
///////////////////////////////////////////////////////////////////////////////
namespace detail
{
// Helper to calculate total DOF of the elements of a tuple
template <typename Tuple, std::size_t... I>
constexpr std::size_t total_dof_impl(std::index_sequence<I...>)
{
  return (std::tuple_element_t<I, Tuple>::DOF + ...);
}

template <typename Tuple>
constexpr std::size_t total_dof()
{
  return total_dof_impl<Tuple>(
      std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

// Apply the addition of a vector to a state tuple
template <typename Tuple, std::size_t... I>
Tuple apply_addition(const Tuple& state,
                     const unscented::Vector<total_dof<Tuple>()>& vec,
                     std::index_sequence<I...>)
{
  Tuple result;
  std::size_t offset = 0;

  // Define a lambda to process each index
  auto apply = [&](auto index)
  {
    constexpr std::size_t DOF = std::tuple_element_t<index, Tuple>::DOF;
    std::get<index>(result) =
        std::get<index>(state) + vec.template segment<DOF>(offset);
    offset += DOF;
  };

  (apply(std::integral_constant<std::size_t, I>{}), ...);

  return result;
}

// Apply the subtraction of two state tuples to get a vector
template <typename Tuple, std::size_t... I>
unscented::Vector<total_dof<Tuple>()> apply_subtraction(
    const Tuple& lhs, const Tuple& rhs, std::index_sequence<I...>)
{
  unscented::Vector<total_dof<Tuple>()> result;
  std::size_t offset = 0;

  // Define a lambda to process each index
  auto apply = [&](auto index)
  {
    constexpr std::size_t DOF = std::tuple_element_t<index, Tuple>::DOF;
    result.template segment<DOF>(offset) =
        std::get<index>(lhs) - std::get<index>(rhs);
    offset += DOF;
  };

  (apply(std::integral_constant<std::size_t, I>{}), ...);

  return result;
}
} // namespace detail

template <typename... Ts>
Compound<Ts...> operator+(const Compound<Ts...>& state,
                          const unscented::Vector<Compound<Ts...>::DOF>& vec)
{
  Compound<Ts...> ret;
  ret.data = detail::apply_addition<typename Compound<Ts...>::Tuple>(
      state.data, vec, std::make_index_sequence<Compound<Ts...>::TupleSize>{});
  return ret;
}

template <typename... Ts>
unscented::Vector<Compound<Ts...>::DOF> operator-(const Compound<Ts...>& lhs,
                                                  const Compound<Ts...>& rhs)
{
  return detail::apply_subtraction<typename Compound<Ts...>::Tuple>(
      lhs.data, rhs.data,
      std::make_index_sequence<Compound<Ts...>::TupleSize>{});
}

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
  return Angle(lhs.get_angle() - rhs.get_angle()).get_vector();
}
} // namespace unscented

#endif
