import numpy as np

from .baseLattice import BaseLattice


class StableDAR(BaseLattice):
    # ------------------------------------------------ #
    # Functions that overload abstract methods         #
    # ------------------------------------------------ #
    def decode(self, lar):
        """Extracts parcor coefficients from encoded version (e.g. LAR)

        lar : array containing the encoded coefficients

        returns:
        ki : array containing the decoded coefficients (same size as lar)

        """
        exp_lar = np.exp(lar)
        ki = (exp_lar - 1.0) / (exp_lar + 1.0)
        return ki

    def encode(self, ki):
        """Encodes parcor coefficients to LAR coefficients

        ki : array containing the original parcor coefficients

        returns:
        lar : array containing the encoded coefficients (same size as ki)

        """
        lar = np.log((1.0 + ki) / (1.0 - ki))
        return lar

    def common_gradient(self, p, ki):
        """Compute common factor in gradient. The gradient is computed as
        G[p] = sum from t=1 to T {g[p,t] * F(t)}
        where F(t) is the vector of driving signal and its powers
        g[p,t] = (e_forward[p, t] * e_backward[p-1, t-1]
                    + e_backward[p, t] * e_forward[p-1, t]) * phi'[k[p,t]]
        phi is the encoding function, and phi' is its derivative.

        p  : order corresponding to the current lattice cell
        ki : array containing the original parcor coefficients

        returns:
        g : array containing the factors (size (n_epochs, n_points - 1))

        """
        e_forward = self.forward_residual
        e_backward = self.backward_residual
        _, n_epochs, n_points = e_forward.shape

        g = e_forward[p, :, 1:n_points] * e_backward[p - 1, :, 0:n_points - 1]
        g += e_backward[p, :, 1:n_points] * e_forward[p - 1, :, 1:n_points]
        g *= 0.5 * (1.0 - ki[:, 1:n_points] ** 2)   # phi'[k[p,t]])
        return np.reshape(g, (n_epochs, n_points - 1))

    def common_hessian(self, p, ki):
        """Compute common factor in Hessian. The Hessian is computed as
        H[p] = sum from t=1 to T {F(t) * h[p,t] * F(t).T}
        where F(t) is the vector of driving signal and its powers
        h[p,t] =   (e_forward[p, t-1]**2 + e_backward[p-1, t-1]**2)
                    * phi'[k[p,t]]**2
                 + (e_forward[p, t] * e_backward[p-1, t-1]
                    e_backward[p, t] * e_forward[p-1, t]) * phi''[k[p,t]]
        phi is the encoding function, phi' is its first derivative,
        and phi'' is its second derivative.

        p  : order corresponding to the current lattice cell
        ki : array containing the original parcor coefficients

        returns:
        h : array containing the factors (size (n_epochs, n_points - 1))

        """
        e_forward = self.forward_residual
        e_backward = self.backward_residual
        _, n_epochs, n_points = e_forward.shape

        h1 = e_forward[p - 1, :, 1:n_points] ** 2
        h1 += e_backward[p - 1, :, 0:n_points - 1] ** 2
        h1 *= (0.5 * (1.0 - ki[:, 1:n_points] ** 2)) ** 2

        h2 = e_forward[p, :, 1:n_points] * e_backward[p - 1, :, 0:n_points - 1]
        h2 += e_backward[p, :, 1:n_points] * e_forward[p - 1, :, 1:n_points]
        h2 *= (-0.5 * ki[:, 1:n_points] * (1.0 - ki[:, 1:n_points] ** 2))

        return np.reshape(h1 + h2, (n_epochs, n_points - 1))
