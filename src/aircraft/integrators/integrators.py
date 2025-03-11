import numpy as np
from abc import ABC, abstractmethod

class Integrator(ABC):
    def __init__(self, f):
        self.f = f

    @abstractmethod
    def integrate(self, x0, tf, dt = 0.01):
        pass

class EulerIntegrator(Integrator):
    def euler_step(self, x, dt):
            return x + dt * self.f(x)
    
    def integrate(self, x0, dt, tf):
        x = x0
        steps = int(tf / dt)
        for _ in range(steps):
            x = self.euler_step(x, dt)
        return x
    
class RK4Integrator(Integrator):
    def rk4_step(self, x, dt):
            k1 = dt * self.f(x)
            k2 = dt * self.f(x + 0.5 * k1)
            k3 = dt * self.f(x + 0.5 * k2)
            k4 = dt * self.f(x + k3)
            return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def integrate(self, x0, dt, tf):
        x = x0
        steps = int(tf / dt)
        for _ in range(steps):
            x = self.rk4_step(x, dt)
        return x
    
class RK45Integrator(Integrator):
     
    def __init__(self, f, tol=1e-6, max_steps=1000):
        super().__init__(f)
        self.tol

    def rk45_step(self, x, dt):
            k1 = dt * self.f(x)
            k2 = dt * self.f(x + 0.2 * k1)
            k3 = dt * self.f(x + 0.3 * k2)
            k4 = dt * self.f(x + 0.6 * k3)
            k5 = dt * self.f(x + 0.8 * k4)
            k6 = dt * self.f(x + k5)
            return x + (k1 + 4 * k3 + k4) / 6 + (k2 + k5) / 30

    def integrate(self, x0, dt, tf):
        x = x0
        steps = int(tf / dt)
        for _ in range(steps):
            x = self.rk45_step(x, dt)
        return x

class AdaptiveRK45Integrator(Integrator):
    def __init__(self, f, tol=1e-6, max_steps=1000):
        super().__init__(f)
        self.tol = tol
        self.max_steps = max_steps

    def rk45_step(self, x, dt):
        k1 = dt * self.f(x)
        k2 = dt * self.f(x + 0.2 * k1)
        k3 = dt * self.f(x + 0.3 * k2)
        k4 = dt * self.f(x + 0.6 * k3)
        k5 = dt * self.f(x + 0.8 * k4)
        k6 = dt * self.f(x + k5)
        return x + (k1 + 4 * k3 + k4) / 6 + (k2 + k5) / 30
    
    def integrate(self, x0, dt, tf):
        x = x0
        steps = int(tf / dt)
        for _ in range(steps):
            x = self.rk45_step(x, dt)
        return x
    





