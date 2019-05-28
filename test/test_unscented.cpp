#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "unscented.hpp"

namespace unscented
{
TEST_CASE("Temp")
{
  UKF<Eigen::Vector3d, 3, Eigen::Vector2d, 2> ukf;
}
} // namespace unscented
