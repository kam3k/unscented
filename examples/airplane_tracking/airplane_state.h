#ifndef UNSCENTED_AIRPLANE_TRACKING_AIRPLANE_STATE_H
#define UNSCENTED_AIRPLANE_TRACKING_AIRPLANE_STATE_H

#include "unscented/primitives.hpp"
#include "unscented/ukf.hpp"

/** State is simple a 4-vector */
using AirplaneState = unscented::Vector<4>;

/** The elements of the state vector */
enum StateElements
{
  POSITION = 0,
  VELOCITY,
  ALTITUDE,
  CLIMB_RATE
};

/**
 * @brief The airplane's system model is the constant velocity model; that is,
 * the velocity is assumed to be constant over the duration of a timestep dt,
 * and the positions are simply updated using the first-order euler method
 *
 * @param[in,out] state The state to be updated
 * @param[in] dt The timestep (seconds)
 */
void system_model(AirplaneState& state, double dt)
{
  Eigen::Matrix4d F;
  // clang-format off
  F << 1.0,  dt, 0.0, 0.0, 
       0.0, 1.0, 0.0, 0.0, 
       0.0, 0.0, 1.0,  dt, 
       0.0, 0.0, 0.0, 1.0;
  // clang-format on
  state = F * state;
}

#endif
