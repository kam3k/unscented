![unscented](media/unscented.png)

A flexible and powerful unscented Kalman filter C++11 library that makes no assumptions about what you're estimating or how you're measuring it.

## Table of contents
* [What's an unscented Kalman filter?](#whats-an-unscented-kalman-filter)
	* [Sounds good, it looks like I'm in the right spot](#sounds-good-it-looks-like-im-in-the-right-spot)
	* [Okay, but why do we need the UKF?](#okay-but-why-do-we-need-the-ukf)
  * [What makes the UKF "unscented"?](#what-makes-the-ukf-unscented)
* [Why another Kalman filter library?](#why-another-kalman-filter-library)
* [Usage](#usage)
* [Installation](#installation)
* [Examples](#examples)
	* [Airplane tracking](#airplane-tracking)
	* [Robot localization](#robot-localization)
	* [Estimating orientations](#estimating-orientations)
* [The unscented transformation](#the-unscented-transformation)
* [License and contributing](#license-and-contributing)

## What's an unscented Kalman filter?

The unscented Kalman filter (UKF) is an extension to the well-used [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter) (KF) for nonlinear system and/or measurement models. It uses the [unscented transformation](#the-unscented-transformation) to approximate passing probability distributions through nonlinear functions. There are a lot of [helpful books](https://academic.csuohio.edu/simond/estimation/), [blog posts](https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/), and [encyclopedia articles](https://en.wikipedia.org/wiki/Unscented_transform) with all the technical details about these processes, so rather than re-hashing those, this section will provide an intuitive feel for the type of problems KFs solve, and why the UKF is needed in some (most) of these scenarios. If you're comfortable with how KFs work, go ahead and skip straight to the [usage](#usage), or perhaps the [motivation](#why-another-kalman-filter-library) for why this library was created. Or if you really trust me, just go ahead and [install](#installation) it.

### Sounds good, it looks like I'm in the right spot

Are you sure you know that? Richard Feynman [once wrote](http://www.feynmanlectures.caltech.edu/I_06.html)

> The *most* we can know is in terms of probabilities

which is to say, you can never know anything for certain. For example, there is always some uncertainty in the position of something relative to something else. Maybe your car's GPS says you're at 742 Evergreen Terrace in the town of Springfield, but that's only accurate to several metres. That's fine if you're looking for an answer to the question "where in town am I?", but less helpful if you want to know how close you are to the passing oncoming traffic. 

This uncertainty is one reason why self-driving vehicles are equipped with sensors in addition to GPS to determine where they are. But then how do we combine the information from these different sensors to get the best estimate of our location? That's where KFs come in. They combine a model of how a car moves with uncertain measurements from sensor(s) to estimate not only the position of the car but other important characteristics like velocity, acceleration, heading, etc. These characteristics being estimated are called the *state*, and the data from the sensor(s) are called the *measurements*. Both the state and measurements are modelled as random variables; that is, they have uncertainty. It is really important to at least have a [basic understanding](http://marcgallant.ca/2015/12/16/you-dont-know-where-your-robot-is/) of random variables (specifically Gaussian random variables) to understand how a Kalman filter works and why it is necessary.

### Show me the algorithm already

Alright, alright, let's show how this works in practice using a really simplified example. Suppose your car is driving in a straight line and you have the following information (keeping in mind these all have uncertainty):

* The position at which your car started
* The speed of the car from the speedometer
* A clock
* A GPS

You want to use this information to estimate the (one-dimensional) position and speed of the car as you drive. That is to say, the *state* is the position and speed of the car, and data from the speedometer and GPS are the *measurements*.

But wait, why do we need a KF to estimate position and speed if we already have sensors that measure those things (GPS and speedometer, respectively)? There are a few reasons. First, your measurements could be coming in at a relatively slow rate (for example, once per second). What about all the time between measurements? Secondly, and perhaps most importantly, these measurements are *noisy*. One GPS measurement from the next can jump by several metres, even when the vehicle is sitting still. The KF helps smooth this out by giving you a chance to impose constraints on how a car moves (e.g., cars don't tend to move laterally, unless you have really [fancy wheels](https://en.wikipedia.org/wiki/Omni_wheel)).

For the example above, here is the pseudocode of how a simple KF would estimate the state (position and speed) given the measurements (GPS and speedometer):

```cpp
state = initial_state
previous_time = current_time()

while (driving)
  measurement = read_sensors()
  time_step = current_time() - previous_time
  state = predict(state, time_step)
  state = correct(state, measurement)
  previous_time = previous_time + time_step
```

The meat and potatoes of the KF algorithm are in the `predict(…)` and `correct(…)` functions shown below:

```cpp
function predict(state, time_step)
  state.position = state.position + state.speed * time_step
  state.covariance = ... // increases
  return state
  
function correct(state, measurement)
  expected_measurement = state // expected measurement of position and speed
  innovation = measurement - expected_measurement // difference between actual and expected
  kalman_gain = state.covariance / (state.covariance + measurement.covariance)
  state = state + kalman_gain * innovation
  state.covariance = ... // decreases
  return state
```

The `predict(…)` function propagates the state forward in time based on a *system model*. That is to say, it predicts what the state will be after the given time step. Here, we've selected a very simple system model; that is, the change in position is the speed integrated over the time step. Note that we've made the assumption that the speed is constant over the duration of the time step. This means the system model does not update the speed, and we use [Euler's method](https://en.wikipedia.org/wiki/Euler_method) to perform numerical integration (in this case, this is just a fancy way of saying that change is position is equal to speed multiplied by time).

An important thing to note here is that the system model is *linear* with respect to the state variables (position and speed). That is to say, the equation on the right hand side of the equal sign is a linear combination of the state variables. This is very important if you recall that the state is not a single number, rather it is a (Gaussian) random variable. What makes this important is that if you pass a Gaussian random variable through a linear function (like our linear system model), you get a Gaussian random variable as the output. Not shown is the covariance (i.e., uncertainty) calculation (it requires some matrix math), but I assure you it's quite simple (see the [resources provided above](#whats-an-unscented-kalman-filter) for details). What's important is that the uncertainty *increases* in prediction step. We are just propagating the state forward in time without any external measurements, so naturally we become more uncertain of it.

The `correct(…)` function updates the predicted state with measurement(s) based on a *measurement model*. That is to say, it attempts to correct the increased uncertainty introducted by the prediction step using measurements. Here the measurement model is extremely simple: it's the state itself. We have a GPS to measure the position and a speedometer to measure the speed. You can imagine a case where it is not the state itself; for example, if we had a laser scanner that could measure the position of a known landmark. Whatever measurement you have, you need to be able to relate it to the elements of the state. More specifically, you need to be able to calculate what you *expect the measurement to be* based on the current state estimate. This calculation is what the measurement model does. In this example, the measurement model is just the state itself, which, like the system model, is linear. As described above, this is good news for our Gaussian random variables.

The *innovation* is just the difference between what we measured and what we expected to measure. We then update the state based on the innovation by first scaling it by the *Kalman gain*. The Kalman gain serves two purposes: to convert the innovation into a change in state (no conversion needed in this example), and to scale how much of the innovation gets applied to the state, as shown in the calculation above. (Note that I've taken some liberties with the notation, this is actually a matrix calculation.) Consider what happens when the covariance (uncertainty) of the measurement is really high compared to the current state covariance. In this case, the Kalman gain approaches zero, which means the measurement has little effect on the state. Conversely, when the measurement covariance is small, the Kalaman gain approaches one, which results in the state being "pulled" sharply towards the measurement. Usually it is somewhere between these two extremes.

And there we have it. We cycle back and forth between predicting the state forward in time, and then correcting it with measurements. See the [usage](#usage) section below for an actual implementation of this example using the *unscented* library.

### Okay, but why do we need the UKF?

We achieved the impossible in the previous example. We found ourselves a system to estimate that had both a linear system model and linear measurement model, so our Gaussian random variables were always Gaussian. It turns it was our simplifications what made this true, and most real-world scenarios will have nonlinearities. For example, what about the heading of the car? Surely we don't want to drive straight forever. Now our state goes from (one-dimensional) position and speed to two-dimensional position (x, y), heading, and speed. So let's add that to our state and re-write the system model:

```cpp
state.position.x = state.position.x + state.speed * time_step * cos(state.heading)
state.position.y = state.position.y + state.speed * time_step * sin(state.heading)
```

Note that I've made the assumption that, like speed, the heading is constant over the time step. More importantly, the system model is no longer linear! Our poor Gaussian random variables will be in shambles after passing though this model. As a result, we need a way to estimate what the Gaussian distribution looks like after passing through this model. Some sort of transformation that can reconstruct the distrubution. Aha! How about the [unscented transformation](#the-unscented-transformation)?

### What makes the UKF "unscented"?

Jeffrey Uhlmann, the creator of the unscented transformation on which the UKF is based, was a humble guy that didn't want his name attached to his invention. Uhlmann [arbitrarily picked the name](https://medium.com/@anthony_sarkis/what-is-a-kalman-filter-and-why-is-there-an-unscented-version-bc5f6e77c509) *unscented* transformation from the word "unscented" on a stick of deodorant on his colleague's desk. It was chosen in part to test Uhlmann's theory that people simply accept technical terms without much thought (it turns out he was right in this case). Afterwards, a running joke was made that the term was selected because the performance of another extension to the Kalman filter (the aptly named [extended Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter#Extended_Kalman_filter)) was *stinky*, and the unscented version wasn't.

## Why another Kalman filter library?

## Installation

## Usage

## API

## Examples

### Airplane tracking

### Robot localization

### Estimating orientations

## The unscented transformation

## License and contributing



