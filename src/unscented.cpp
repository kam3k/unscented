#include "unscented.hpp"

namespace unscented
{
  namespace _detail
  {
    float wrapAngle(float angle)
    {
      angle = fmod(angle + M_PI, 2 * M_PI);
      if (angle < 0.0f)
      {
        angle += 2 * M_PI;
      }
      return angle - M_PI;
    }
  } // namespace _detail

  class UnicycleState
  {
  public:
    UnicycleState() = default;

    UnicycleState(float x, float y, float theta) : x_(x), y_(y), theta_(theta)
    {
    }

    UnicycleState(const Eigen::Vector3f& perturb)
      : x_(perturb.x()), y_(perturb.y()), theta_(perturb.z())
    {
    }

    UnicycleState operator+(const UnicycleState& other)
    {
      return {x_ + other.x_, y_ + other.y_,
              _detail::wrapAngle(theta_ + other.theta_)};
    }

    Eigen::Vector3f operator-(const UnicycleState& other)
    {
      return {x_ - other.x_, y_ - other.y_,
              _detail::wrapAngle(theta_ - other.theta_)};
    }

    UnicycleState operator*(float scale) const
    {
      return {x_ * scale, y_ * scale, theta_ * scale};
    }

  private:
    float x_ = 0.0f;
    float y_ = 0.0f;
    float theta_ = 0.0f;
  };

  template class UKF<Eigen::Matrix<float, 6, 1>, 6, Eigen::Vector3f, 3, float>;
  template class UKF<UnicycleState, 3, UnicycleState, 3, float>;
} // namespace unscented
