#include "unscented.hpp"

namespace unscented
{
  template class UKF<Eigen::Matrix<float, 6, 1>, 6, Eigen::Vector3f, 3, float>;
} // namespace unscented
