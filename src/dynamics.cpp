
#include <Eigen/Dense>
#include <cmath>

class SixDOF {
protected:
    Eigen::VectorXd state;  // 13-dimensional state vector
    double rho;  // air density
    Eigen::Vector3d omega_earth;  // Earth's angular velocity
    Eigen::Vector3d gravity;  // gravity vector
    double mass;  // mass of the vehicle
    
    // Cache for Euler angles
    mutable bool euler_cached;
    mutable Eigen::Vector3d cached_euler;

public:
    SixDOF(double rho = 1.225, 
           const Eigen::Vector3d& omega_earth = Eigen::Vector3d(7.2921150e-5, 0, 0),
           const Eigen::Vector3d& gravity = Eigen::Vector3d(0, 0, -9.81),
           double mass = 1.0)
        : state(Eigen::VectorXd::Zero(13)),
          rho(rho),
          omega_earth(omega_earth),
          gravity(gravity),
          mass(mass),
          euler_cached(false)
    {
        state(6) = 1.0;  // Default quaternion [1, 0, 0, 0]
    }

    virtual ~SixDOF() = default;

    // Pure virtual methods
    virtual Eigen::VectorXd getControls() const = 0;
    virtual Eigen::Vector3d getBodyForces() const = 0;
    virtual Eigen::Vector3d getBodyMoments() const = 0;

    // Getters and setters
    const Eigen::VectorXd& getState() const { return state; }
    
    Eigen::Matrix3d getInertia() const {
        return Eigen::Matrix3d::Identity();
    }

    Eigen::Vector3d getInertialPosition() const {
        return state.segment<3>(0);
    }

    void setInertialPosition(const Eigen::Vector3d& pos) {
        state.segment<3>(0) = pos;
    }

    Eigen::Vector3d getInertialVelocity() const {
        return state.segment<3>(3);
    }

    void setInertialVelocity(const Eigen::Vector3d& vel) {
        state.segment<3>(3) = vel;
    }

    Eigen::Vector4d getInertialAttitude() const {
        Eigen::Vector4d quat = state.segment<4>(6);
        return quat.normalized();
    }

    void setInertialAttitude(const Eigen::Vector4d& quat) {
        state.segment<4>(6) = quat.normalized();
        euler_cached = false;
    }

    Eigen::Vector3d getBodyAngularVelocity() const {
        return state.segment<3>(10);
    }

    // Coordinate transformations
    Eigen::Vector3d fromInertialToBody(const Eigen::Vector3d& vector) const {
        Eigen::Quaterniond q(getInertialAttitude());
        return q.matrix() * vector;
    }

    Eigen::Vector3d fromBodyToInertial(const Eigen::Vector3d& vector) const {
        Eigen::Quaterniond q(getInertialAttitude());
        return q.matrix().transpose() * vector;
    }

    // Aerodynamic properties
    Eigen::Vector3d getBodyVelocity() const {
        return fromInertialToBody(getInertialVelocity());
    }

    double getAirspeed() const {
        return std::max(getBodyVelocity().norm(), 1e-8);
    }

    double getAlpha() const {
        const Eigen::Vector3d v = getBodyVelocity();
        return std::atan2(v(2), v(0));
    }

    double getBeta() const {
        return std::asin(getBodyVelocity()(1) / getAirspeed());
    }

    double getQBar() const {
        return 0.5 * rho * std::pow(getAirspeed(), 2);
    }

    // Euler angles
    void computeEulerAngles() const {
        const Eigen::Vector4d q = getInertialAttitude();
        double w = q(0), x = q(1), y = q(2), z = q(3);
        
        double phi = std::atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
        double theta = std::asin(std::clamp(2 * (w * y - z * x), -1.0, 1.0));
        double psi = std::atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
        
        cached_euler = Eigen::Vector3d(phi, theta, psi);
        euler_cached = true;
    }

    double getPhi() const {
        if (!euler_cached) computeEulerAngles();
        return cached_euler(0);
    }

    double getTheta() const {
        if (!euler_cached) computeEulerAngles();
        return cached_euler(1);
    }

    double getPsi() const {
        if (!euler_cached) computeEulerAngles();
        return cached_euler(2);
    }

    // State derivatives
    Eigen::Vector3d getInertialForces() const {
        Eigen::Vector3d forces = fromBodyToInertial(getBodyForces());
        forces += gravity;
        
        Eigen::Vector3d r = getInertialPosition();
        Eigen::Vector3d centrifugal = omega_earth.cross(omega_earth.cross(r));
        forces += centrifugal;
        
        Eigen::Vector3d v = getInertialVelocity();
        Eigen::Vector3d coriolis = 2.0 * omega_earth.cross(v);
        forces += coriolis;
        
        return forces;
    }

    Eigen::Vector3d getInertialMoments() const {
        Eigen::Vector3d moments = fromBodyToInertial(getBodyMoments());
        
        Eigen::Vector3d omega_body = getBodyAngularVelocity();
        Eigen::Vector3d omega_inertial = fromBodyToInertial(omega_body);
        
        Eigen::Vector3d coriolis = 2.0 * omega_earth.cross(omega_inertial);
        Eigen::Vector3d centrifugal = omega_earth.cross(omega_earth.cross(omega_inertial));
        
        moments += coriolis + centrifugal;
        
        return moments;
    }

    Eigen::VectorXd getStateDot() const {
        Eigen::VectorXd state_dot(13);
        
        // Position derivative
        state_dot.segment<3>(0) = getInertialVelocity();
        
        // Velocity derivative
        state_dot.segment<3>(3) = getInertialForces() / mass;
        
        // Quaternion derivative
        Eigen::Vector4d q = getInertialAttitude();
        Eigen::Vector3d omega = getBodyAngularVelocity();
        Eigen::Matrix4d omega_matrix;
        omega_matrix << 0, -omega(0), -omega(1), -omega(2),
                       omega(0), 0, omega(2), -omega(1),
                       omega(1), -omega(2), 0, omega(0),
                       omega(2), omega(1), -omega(0), 0;
        state_dot.segment<4>(6) = 0.5 * omega_matrix * q;
        
        // Angular velocity derivative
        Eigen::Matrix3d I = getInertia();
        Eigen::Vector3d omega_dot = I.inverse() * 
            (getBodyMoments() - omega.cross(I * omega));
        state_dot.segment<3>(10) = omega_dot;
        
        return state_dot;
    }

    void step(double dt) {
        // Simple Euler integration
        state += dt * getStateDot();
        state.segment<4>(6).normalize();
        euler_cached = false;
    }
};
