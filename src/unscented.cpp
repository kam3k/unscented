#include "unscented.hpp"

namespace unscented
{
  class UnicycleState
  {
  public:
    UnicycleState() = default;

    UnicycleState(float x, float y, float theta) : x_(x), y_(y), theta_(theta)
    {
    }

    UnicycleState& operator+(const UnicycleState& other)
    {
      x_ += other.x_;
      y_ += other.y_;
      theta_ = wrapAngle(theta_ + other.theta_);
      return *this;
    }

    UnicycleState& operator+(const Eigen::Vector3f& perturbation)
    {
      return *this + UnicycleState(perturbation.x(), perturbation.y(),
                                   perturbation.z());
    }

  private:
    float wrapAngle(float angle)
    {
      angle = fmod(angle + M_PI, 2 * M_PI);
      if (angle < 0.0f)
      {
        angle += 2 * M_PI;
      }
      return angle - M_PI;
    }

    float x_ = 0.0f;
    float y_ = 0.0f;
    float theta_ = 0.0f;
  };

  template class UKF<Eigen::Matrix<float, 6, 1>, 6, Eigen::Vector3f, 3, float>;
}
