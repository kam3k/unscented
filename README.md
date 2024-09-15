<p align="center">
  <img src="media/unscented.png" />
</p>

A flexible and powerful unscented Kalman filter library (C++17 or later) that makes no assumptions about what you're estimating or how you're measuring it.

## Table of contents
* [What is an unscented Kalman filter?](#what-is-an-unscented-kalman-filter)
* [Quickstart](#quickstart)
* [Features](#features)
* [Installation](#installation)
* [Documentation](#documentation)
* [Examples](#examples)
	* [Airplane tracking](#airplane-tracking)
	* [Robot localization](#robot-localization)
	* [Orientation estimation](#orientation-esitmation)
* [Background](#background)
	* [Random variables](#random-variables) 
	* [The Kalman filter](#the-kalman-filter)
	* [The unscented transformation](#the-unscented-transformation)
	* [The unscented Kalman filter](#the-unscented-kalman-filter)
* [License and contributing](#license-and-contributing)

## What is an unscented Kalman filter?

A Kalman filter (KF) is an algorithm that takes a series of measurements over time (e.g., positions, temperatures, distances, pressures, velocities, etc.) and produces estimates of an evolving unknown state (e.g., the position and heading of a vehicle) with accuracy better than each individual measurement. It accomplishes this by modelling the uncertainty of all concerned variables (e.g., the noise in sensor measurements) and combining them in such a way that optimises the resulting estimates of the state.

For example, a KF is well-suited for estimating the state of a car (e.g., its position, speed, turning rate, etc.) by combining measurements from a GPS device, the car's speedometer and steering angle, and a model of how a car moves (e.g., cars do not tend to move laterally, rather they move in the direction their wheels are facing). Combining these elements in a KF provides a better estimate of the car's state than any of the individual measurements.

An unscented Kalman filter (UKF) is a nonlinear extension to the KF that handles problems where the kinematics or dynamics of the state and/or the relationship between the measurements and the state are nonlinear. In all but the simplest applications, these nonlinearities exist. In fact, it is rare that KF is implemented in its pure linear form. A tip-of-the-iceberg explanation of UKFs, written to provide an intuitive feel for the algorithm, is [available below](#background).

The KF became well known during NASA's Apollo program in the 1960s, where it helped navigate spacecraft on missions to the moon. It has since been used in a plethora of applications; including weather forecasting, autopilot, speech enhancement, vehicle tracking, economics, and [many more](https://en.wikipedia.org/wiki/Kalman_filter#Applications).

## Showcase

Unscented provides some batteries-included tools that make getting up and running straightforward. The [features](#features), [documentation](#documentation), and [examples](#examples) sections below provide a more in-depth look at unscented. However, to get a look at how it works, below is a showcase of setting up a and running simple filter. In this showcase, the state of an airplane is being estimated by a constant-velocity motion model and a range and elevation sensor on the ground.
```cpp
#include "unscented/primitives.h"
#include "unscented/ukf.h"

int main()
{
  // Define state type
  using State = unscented::Vector<4>; // position, velocity, altitude, climb rate

  // Define measurement type
  using Range = unscented::Scalar; // floating point
  using Elevation = unscented::Angle; // -pi to pi
  using Measurement = unscented::Compound<unscented::Scalar, unscented::Angle>;

  // Initialize UKF
  using UKF = unscented::UKF<State, Measurement>;
  UKF ukf;

  // Set initial state and its covariance
  State state{0.0, 100.0, 2000.0, -5.0};
  ukf.set_state(state);
  UKF::N_by_N P;
  P << 10.0, 0.00, 0.00, 0.00,
       0.00, 5.00, 0.00, 0.00,
       0.00, 0.00, 25.0, 0.00,
       0.00, 0.00, 0.00, 0.50;
  ukf.set_state_covariance(P);

  // Set up process and measurement covariance
  UKF::N_by_N Q;
  Q << 0.20, 0.14, 0.00, 0.00,
       0.14, 0.09, 0.00, 0.00,
       0.00, 0.00, 0.20, 0.14,
       0.00, 0.00, 0.14, 0.09;
  ukf.set_process_covariance(Q);
  UKF::M_by_M R;
  R << 10.0, 0.00,
       0.00, 0.10;

  // Set up system model
  auto system_model = [](State& state, double dt) 
  {
    Eigen::Matrix4d F;
    F << 1.0,  dt, 0.0, 0.0, 
         0.0, 1.0, 0.0, 0.0, 
         0.0, 0.0, 1.0,  dt, 
         0.0, 0.0, 0.0, 1.0;
    state = F * state;
  };

  // Set up measurement model
  auto measurement_model = [](const State& state)
  {
    const auto expected_range =
        std::sqrt(std::pow(state[0], 2) + std::pow(state[2], 2));
    const auto expected_elevation =
        unscented::Angle(std::atan2(state[2], state[0]));
    return Measurement{expected_range, expected_elevation};
  };

  // Run the filter
  // Get current timestep (dt) from clock, measurement (meas) from sensor
  // ...

  // Predict with system model
  ukf.predict(system_model, dt);

  // If desired...
  // auto a_priori_state = ukf.get_state();
  // auto a_priori_cov = ukf.get_state_covariance();
  // auto sigma_points = ukf.get_sigma_points();

  // Correct with measurement model
  ukf.correct(measurement_model, meas);

  // If desired...
  // auto a_posteriori_state = ukf.get_state();
  // auto a_posteriori_cov = ukf.get_state_covariance();
  // auto exp_meas = ukf.get_expected_measurement();
  // auto exp_meas_cov = ukf.get_expected_measurement_covariance();
  // auto cross_cov = ukf.get_cross_covariance();
  // auto K = ukf.get_kalman_gain();
  // auto innovation = ukf.get_innovation();
  // auto meas_sigma_points = ukf.get_measurement_sigma_points();

  return 0;
}

```

## Features

The features of *unscented* follow from its goals and motivation for being written. Some inspiration for *unscented* came from the excellent Python library [FilterPy](https://github.com/rlabbe/filterpy), which includes an excellent [UKF library](https://filterpy.readthedocs.io/en/latest/kalman/UnscentedKalmanFilter.html) with many of the same features as *unscented*. There are a number of C++ libraries available that include UKF implementations (e.g., [1](https://github.com/sfwa/ukf), [2](http://jeremyfix.github.io/easykf/), [3](http://verdandi.sourceforge.net/doc-1.6/unscented_kalman_filter.php), [4](https://libraries.io/github/preritj/Unscented-Kalman-Filter), [5](https://github.com/mithi/fusion-ukf), [6](https://libraries.io/github/Veilkrand/Unscented-Kalman-Filter-Sensor-Fusion)), but none that meet all the features described below.

##### Flexible

There are no constraints on the mathematical spaces of the state and measurements in *unscented*. Some (many) libraries require the state and measurements to live in vector spaces. On the other hand, with *unscented* the state and measurements can be any arbitrary type as long as they meet some minimum requirements. This allows one to use, for example, a 3D affine transformation for the pose of a vehicle (often parameterised as a 4x4 transformation matrix), non-minimal representations of rotations, unit vectors, etc. Note that if your state and/or measurements live in a vector space, there is no need to create custom types, one can simply use those provided by the library (which are simply aliases to vectors in Eigen).

Furthermore, *unscented* allows one to easily use multiple system models, multiple measurement models, and custom functions for calculating means (e.g., the average of a bunch of angles isn't their sum divided by the number of angles).

##### Batteries included

In addition to being flexible, *unscented* includes implementations of many standard state and measurement primitives, with a focus on geometric primitives. These include different parameterizations for angles, rotations, and affine transformations (in both 2D and 3D). It is hoped that users creating new primitives will contribute them back to *unscented* so that others can benefit from them.

##### Expansive API

It is good practice to encapsulate code and only provide a public interface that is required for users to make use of a library. That being said, *unscented* opens up its doors a little to allow users to peek at its internals and even modify steps of the UKF. For example, the results of all intermediate calculations are available (e.g., the innovation of the latest measurement and its covariance, the latest sigma points). One can even generate sigma points or manually update the state or its covariance. This promotes using *unscented* as a base for extending the UKF or using it for research or in a non-standard application.

##### Well-documented

Software that is clearly written and well documented tends to get used more than other software, even if that other software is more cleverly written, faster, or even has more features. It is a goal of *unscented* to fall firmly in the former category, offering plenty of [examples](#examples), [documentation](#documentation), and a formal API. It is not a goal of *unscented* to be the fastest or smallest implementation (although it should be fast enough for the vast majority of applications), but it should be one of the easiest to use, fix, and contribute to. Put differently, unscented was not written for a single use-case for the author, rather it is written to be used by a larger community.

##### Tested

There is a disturbingly large amount of software out there that contains no form of testing beyond "it worked in first application in which it was needed". Put differently, new users are guinea pigs who are immediately wandering into untested waters. It is hoped that along with its unit tests, the readability and documentation of *unscented* will make it easy for new contributors to help keep things in good shape and contribute their own tests for new features.

## Installation

## Documentation

## Examples

### Airplane tracking

### Robot localization

### Orientation estimation

## Background

### Random variables

### The Kalman filter

### The unscented transformation

#### Why "unscented"?

Jeffrey Uhlmann, the creator of the unscented transformation, was a humble guy that didn't want his name attached to his invention. Uhlmann [arbitrarily picked the name](https://medium.com/@anthony_sarkis/what-is-a-kalman-filter-and-why-is-there-an-unscented-version-bc5f6e77c509) *unscented* transformation from a stick of unscented deodorant on his colleague's desk. It was chosen in part to test Uhlmann's theory that people simply accept technical terms without much thought (it turns out he was right in this case). Afterwards, a running joke was made that the term was selected because the previously established nonlinear extension to the Kalman filter (the aptly named [extended Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter#Extended_Kalman_filter)) had *stinky* performance, unlike the new unscented approach.

### The unscented Kalman filter

## License and contributing
