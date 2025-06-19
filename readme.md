# mujoco_drone


## Kinematics
Rotors are arraged as shown:

![alt text](drone_model.drawio.svg)

![alt text](drone_model_3d.drawio.svg)

$\omega_i$: angular velocity of rotor $i$

$f_i$: force created by rotor around its axis,  $f_i=k_f \omega_i^2$

$\tau_i$: torque created by the robot around its axis, $\tau_{i}=k_{\tau} \omega_i^2$




## Dynamics
### Force and Moments
The resultant force applied to the body w.r.t. its own frame are:

$`
\mathbf{T}^B=\left[\begin{array}{l}
0 \\
0 \\
T
\end{array}\right]=\left[\begin{array}{c}
0 \\
0 \\
\sum_{i=1}^4 f_i
\end{array}\right]=\left[\begin{array}{c}
0 \\
0 \\
k_f \sum_{i=1}^4 \omega_i^2
\end{array}\right]
`$

$`
\tau^B=\left[\begin{array}{c}
\tau_x \\
\tau_y \\
\tau_z
\end{array}\right]=\left[\begin{array}{c}
-f_1 \cdot d-f_2 \cdot d+f_3 \cdot d+f_4 \cdot d \\
f_1 \cdot d-f_2 \cdot d+f_3 \cdot d-f_4 \cdot d \\
\tau_1-\tau_2-\tau_3+\tau_4
\end{array}\right]=\left[\begin{array}{c}
d k_f\left(-\omega_1^2-\omega_2^2+\omega_3^2+\omega_4^2\right) \\
d k_f\left(\omega_1^2-\omega_2^2+\omega_3^2-\omega_4^2\right) \\
k_\tau\left(\omega_1^2-\omega_2^2-\omega_3^2+\omega_4^2\right)
\end{array}\right]
`$

where $d$ is the distance from the rotor center to x-axis or y-aixs, here assuming they are the same.

### Equation of Motion

In the inertial frame, the acceleration of the quadrotor is due to thrust, gravity. We can obtain the thrust force in the inertial frame by using the rotation matrix $\mathbf{R}^I_{B}$ to map it from the body frame to the inertial frame. Thus, the linear part of the equation of motion is:

$`
m\ddot{\mathbf{p}} = \mathbf{f} + m\mathbf{g}
`$

$`
m \ddot{\mathbf{p}}=
\mathbf{R}_B^I \mathbf{f}^B + 
m\left[\begin{array}{c}
0 \\
0 \\
-g
\end{array}\right]
`$

where $\mathbf{p}$ is the position of the quadrotor in the inertial frame, $g$ is the gravity acceleration, and $\mathbf{f}$ is the thrust force in inertia frame and $\mathbf{f}^B$ is the thrust force in the body frame.

While it is convenient to have the linear equations of motion in the inertial frame, the rotational equations of motion is simpler to derive in the body frame: 

$`
\mathbf{I}^B \dot{\omega}^B+\omega^B \times\left(\mathbf{I}^B \omega^B\right)=\tau^B
`$

where $\omega^B$ is the angular velocity vector in the body frame, $I^B$ is the inertia matrix in the body frame, and $\tau^B$ is resultant torque from the rotors in the body frame. For drone with small angular velocity, it can be further simplifed since $\boldsymbol{\omega} \times(I \boldsymbol{\omega}) \approx 0$:

$`
\mathbf{I}^B \dot{\boldsymbol{\omega}}^B=\tau^B
`$

## PD control
The quadrotor has six states, three positions and three angles, but only four control inputs, the angular velocities of the four rotors. It is an underactuated system. 
We can choose to control the total thrust and torques in its body frame. They can be easily mapped back to the four rotor speeds. 
With the torques and total thrust in the body frame, we can design a simple controller to make the robot stabilise in the air: 

$`
\begin{aligned}
T & =\left(g+K_{z, D}\left(\dot{z}_d-\dot{z}\right)+K_{z, P}\left(z_d-z\right)\right) \frac{m}{C_\phi C_\theta}, \\
\tau_\phi & =\left(K_{\phi, D}\left(\dot{\phi}_d-\dot{\phi}\right)+K_{\phi, P}\left(\phi_d-\phi\right)\right) I_{x x}, \\
\tau_\theta & =\left(K_{\theta, D}\left(\dot{\theta}_d-\dot{\theta}\right)+K_{\theta, P}\left(\theta_d-\theta\right)\right) I_{y y}, \\
\tau_\psi & =\left(K_{\psi, D}\left(\dot{\psi}_d-\dot{\psi}\right)+K_{\psi, P}\left(\psi_d-\psi\right)\right) I_{z z},
\end{aligned}
`$

The mapping between the control variable and rotor angular speeds can be derived as:

$`
\begin{gathered}
T=k\left(\omega_1^2+\omega_2^2+\omega_3^2+\omega_4^2\right) \\
\tau_x=k d\left(-\omega_1^2-\omega_2^2+\omega_3^2+\omega_4^2\right) \\
\tau_y=k d\left(\omega_1^2-\omega_2^2+\omega_3^2-\omega_4^2\right) \\
\tau_z=b\left(\omega_1^2-\omega_2^2-\omega_3^2+\omega_4^2\right)
\end{gathered}
`$

solving the system, we have:

$`
\begin{gathered}
 \omega_1^2=\frac{1}{4}\left(\frac{T}{k}-\frac{\tau_x}{k d}+\frac{\tau_y}{k d}+\frac{\tau_z}{b}\right) \\
 \omega_2^2=\frac{1}{4}\left(\frac{T}{k}-\frac{\tau_x}{k d}-\frac{\tau_y}{k d}-\frac{\tau_z}{b}\right) \\
 \omega_3^2=\frac{1}{4}\left(\frac{T}{k}+\frac{\tau_x}{k d}+\frac{\tau_y}{k d}-\frac{\tau_z}{b}\right) \\
 \omega_4^2=\frac{1}{4}\left(\frac{T}{k}+\frac{\tau_x}{k d}-\frac{\tau_y}{k d}+\frac{\tau_z}{b}\right)
 \end{gathered}
`$

## SE(3) control

Lee, Taeyoung, Melvin Leok, and N. Harris McClamroch. "Geometric tracking control of a quadrotor UAV on SE (3)." 49th IEEE conference on decision and control (CDC). IEEE, 2010.

* Position Controller on SO(3)

Given desired position, we can calculate required acceleration to reach the target:
$$
\mathbf{a}_{cmd} = k_p(\mathbf{p}^*-\mathbf{p})+k_d(\mathbf{v}^*-\mathbf{v})+\mathbf{a}^*
$$
based on the Newton's Equation:
$$
\mathbf{f}_{cmd} = m\mathbf{a}_{cmd} - m\mathbf{g}
$$

then 
$$
\mathbf{f}_{cmd}^B = \mathbf{R}_I^B \mathbf{f}
$$
The thrust would be: 
$$
f = \mathbf{f}_{cmd}^B \cdot [0, 0, 1]^T
$$

or can be simply the projection of $\mathbf{f}_{cmd}$ on to the $z$ axis of the body frame 
$
f = \mathbf{f}_{cmd} \cdot \mathbf{z}_B^I
$

* Attitude Controller on SO(3)

The desired orientation is determined by the direction of the command force $\mathbf{f}_{cmd}$ and the desired heading direction $\mathbf{h}_d$, 

The desired $z$ axis is simply the normalized thrust force: 
$$
\mathbf{z}_d = \hat{\mathbf{f}}_{cmd}
$$

The $y$ axis can be defined as: 
$$
\mathbf{y}_d = \mathbf{z}_d \times \hat{\mathbf{h}}_d 
$$
The $x$ axis is then defiend as: 
$$
\mathbf{x}_d = \mathbf{y}_d \times\mathbf{z}_d
$$

The final desired orientation matrix can be assembed as: 

$`
\mathbf{R}_d = \left[\begin{array}{ccc}
\mid & \mid & \mid \\
\mathbf{x}_d & \mathbf{y}_d & \mathbf{z}_d \\
\mid & \mid & \mid
\end{array}\right]
`$


Defines rotation error:

$$
\begin{gathered}
e_R=\frac{1}{2}\left(R_d^T R-R^T R_d\right)^{\vee} \\
e_{\Omega}=\Omega-R^T R_d \Omega_d
\end{gathered}
$$


Then the moment command is:

$$
M=-k_R e_R-k_{\Omega} e_{\Omega}+\Omega \times J \Omega-J\left(\hat{\Omega} R^T R_d \Omega_d-R^T R_d \dot{\Omega}_d\right)
$$

3. Total Control Inputs
- Thrust: $f$
- Moment: $M$