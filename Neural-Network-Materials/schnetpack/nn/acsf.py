import torch
import numpy as np
from torch import nn
from torch.nn import functional

from schnetpack.nn.cutoff import CosineCutoff
from schnetpack.nn.activations import softplus_inverse
from typing import Callable, Dict, Set, List, Optional

__all__ = [
    "AngularDistribution",
    "BehlerAngular",
    "GaussianSmearing",
    "RadialDistribution",
    "BernsteinPolynomials",
    "ExponentialBernsteinPolynomials",
    "SphericalHarmonics",
]


class AngularDistribution(nn.Module):
    """
    Routine used to compute angular type symmetry functions between all atoms i-j-k, where i is the central atom.

    Args:
        radial_filter (callable): Function used to expand distances (e.g. Gaussians)
        angular_filter (callable): Function used to expand angles between triples of atoms (e.g. BehlerAngular)
        cutoff_functions (callable): Cutoff function
        crossterms (bool): Include radial contributions of the distances r_jk
        pairwise_elements (bool): Recombine elemental embedding vectors via an outer product. If e.g. one-hot encoding
            is used for the elements, this is equivalent to standard Behler functions
            (default=False).

    """

    def __init__(
        self,
        radial_filter: nn.Module,
        angular_filter: nn.Module,
        cutoff_functions: nn.Module = CosineCutoff,
        crossterms: bool = False,
        pairwise_elements: bool = False,
    ):
        super(AngularDistribution, self).__init__()
        self.radial_filter = radial_filter
        self.angular_filter = angular_filter
        self.cutoff_function = cutoff_functions
        self.crossterms = crossterms
        self.pairwise_elements = pairwise_elements

    def forward(self,
                r_ij: torch.Tensor,
                r_ik: torch.Tensor,
                r_jk: torch.Tensor,
                triple_masks: Optional[torch.Tensor] = None,
                elemental_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            r_ij (torch.Tensor): Distances to neighbor j
            r_ik (torch.Tensor): Distances to neighbor k
            r_jk (torch.Tensor): Distances between neighbor j and k
            triple_masks (torch.Tensor): Tensor mask for non-counted pairs (e.g. due to cutoff)
            elemental_weights (tuple of two torch.Tensor): Weighting functions for neighboring elements, first is for
                                                            neighbors j, second for k

        Returns:
            torch.Tensor: Angular distribution functions

        """

        nbatch, natoms, npairs = r_ij.size()

        # compute gaussilizated distances and cutoffs to neighbor atoms
        radial_ij = self.radial_filter(r_ij)
        radial_ik = self.radial_filter(r_ik)
        angular_distribution = radial_ij * radial_ik

        if self.crossterms:
            radial_jk = self.radial_filter(r_jk)
            angular_distribution = angular_distribution * radial_jk

        # Use cosine rule to compute cos( theta_ijk )
        cos_theta = (torch.pow(r_ij, 2) + torch.pow(r_ik, 2) - torch.pow(r_jk, 2)) / (
            2.0 * r_ij * r_ik
        )

        # Required in order to catch NaNs during backprop
        if triple_masks is not None:
            cos_theta[triple_masks == 0] = 0.0

        angular_term = self.angular_filter(cos_theta)

        if self.cutoff_function is not None:
            cutoff_ij = self.cutoff_function(r_ij).unsqueeze(-1)
            cutoff_ik = self.cutoff_function(r_ik).unsqueeze(-1)
            angular_distribution = angular_distribution * cutoff_ij * cutoff_ik

            if self.crossterms:
                cutoff_jk = self.cutoff_function(r_jk).unsqueeze(-1)
                angular_distribution = angular_distribution * cutoff_jk

        # Compute radial part of descriptor
        if triple_masks is not None:
            # Filter out nan divisions via boolean mask, since
            # angular_term = angular_term * triple_masks
            # is not working (nan*0 = nan)
            angular_term[triple_masks == 0] = 0.0
            angular_distribution[triple_masks == 0] = 0.0

        # Apply weights here, since dimension is still the same
        if elemental_weights is not None:
            if not self.pairwise_elements:
                Z_ij, Z_ik = elemental_weights
                Z_ijk = Z_ij * Z_ik
                angular_distribution = (
                    torch.unsqueeze(angular_distribution, -1)
                    * torch.unsqueeze(Z_ijk, -2).float()
                )
            else:
                # Outer product to emulate vanilla SF behavior
                Z_ij, Z_ik = elemental_weights
                B, A, N, E = Z_ij.size()
                pair_elements = Z_ij[:, :, :, :, None] * Z_ik[:, :, :, None, :]
                pair_elements = pair_elements + pair_elements.permute(0, 1, 2, 4, 3)
                # Filter out lower triangular components
                pair_filter = torch.triu(torch.ones(E, E)) == 1
                pair_elements = pair_elements[:, :, :, pair_filter]
                angular_distribution = torch.unsqueeze(
                    angular_distribution, -1
                ) * torch.unsqueeze(pair_elements, -2)

        # Dimension is (Nb x Nat x Nneighpair x Nrad) for angular_distribution and
        # (Nb x Nat x NNeigpair x Nang) for angular_term, where the latter dims are orthogonal
        # To multiply them:
        angular_distribution = (
            angular_distribution[:, :, :, :, None, :]
            * angular_term[:, :, :, None, :, None]
        )
        # For the sum over all contributions
        angular_distribution = torch.sum(angular_distribution, 2)
        # Finally, we flatten the last two dimensions
        angular_distribution = angular_distribution.view(nbatch, natoms, -1)

        return angular_distribution


class BehlerAngular(nn.Module):
    """
    Compute Behler type angular contribution of the angle spanned by three atoms:

    :math:`2^{(1-\zeta)} (1 + \lambda \cos( {\\theta}_{ijk} ) )^\zeta`

    Sets of zetas with lambdas of -1 and +1 are generated automatically.

    Args:
        zetas (set of int): Set of exponents used to compute angular Behler term (default={1})

    """

    def __init__(self, zetas: Set[int] ={1}):
        super(BehlerAngular, self).__init__()
        self.zetas = zetas

    def forward(self, cos_theta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cos_theta (torch.Tensor): Cosines between all pairs of neighbors of the central atom.

        Returns:
            torch.Tensor: Tensor containing values of the angular filters.
        """
        angular_pos = [
            2 ** (1 - zeta) * ((1.0 - cos_theta) ** zeta).unsqueeze(-1)
            for zeta in self.zetas
        ]
        angular_neg = [
            2 ** (1 - zeta) * ((1.0 + cos_theta) ** zeta).unsqueeze(-1)
            for zeta in self.zetas
        ]
        angular_all = angular_pos + angular_neg
        return torch.cat(angular_all, -1)


def gaussian_smearing(distances:torch.Tensor,
                      offset:torch.Tensor,
                      widths:torch.Tensor,
                      centered:bool = False) -> torch.Tensor:
    r"""Smear interatomic distance values using Gaussian functions.

    Args:
        distances (torch.Tensor): interatomic distances of (N_b x N_at x N_nbh) shape.
        offset (torch.Tensor): offsets values of Gaussian functions.
        widths: width values of Gaussian functions.
        centered (bool, optional): If True, Gaussians are centered at the origin and
            the offsets are used to as their widths (used e.g. for angular functions).
    Returns:
        torch.Tensor: smeared distances (N_b x N_at x N_nbh x N_g).

    """
    if not centered:
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        coeff = -0.5 / torch.pow(widths, 2)
        # Use advanced indexing to compute the individual components
        diff = distances[:, :, :, None] - offset[None, None, None, :]
    else:
        # if Gaussian functions are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # if Gaussian functions are centered, no offset is subtracted
        diff = distances[:, :, :, None]
    # compute smear distance values
        
    gauss = torch.exp(coeff * torch.pow(diff, 2))
    return gauss


class GaussianSmearing(nn.Module):
    r"""Smear layer using a set of Gaussian functions.

    Args:
        start (float, optional): center of first Gaussian function, :math:`\mu_0`.
        stop (float, optional): center of last Gaussian function, :math:`\mu_{N_g}`
        n_gaussians (int, optional): total number of Gaussian functions, :math:`N_g`.
        centered (bool, optional): If True, Gaussians are centered at the origin and
            the offsets are used to as their widths (used e.g. for angular functions).
        trainable (bool, optional): If True, widths and offset of Gaussian functions
            are adjusted during training process.

    """

    def __init__(
        self,
        start:float = 0.0,
        stop:float = 5.0,
        n_gaussians:float = 50,
        centered:bool = False,
        trainable:bool = False):

        super(GaussianSmearing, self).__init__()
        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered
        
        self.relevance = None

    def forward(self, distances:torch.Tensor) -> torch.Tensor:
        """Compute smeared-gaussian distance values.

        Args:
            distances (torch.Tensor): interatomic distance values of
                (N_b x N_at x N_nbh) shape.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh x N_g) shape.

        """
        return gaussian_smearing(
            distances, self.offsets, self.width, centered=self.centered)
            
    def relprop(self,
                input:Dict[str, torch.Tensor],
                relevance: torch.Tensor) -> torch.Tensor:
        """Compute layer relevance for its inputs.

        Args:
            input (dict of torch.Tensor): batch of input values.
            relevance (torch.tensor): relevance of outputs

        Returns:
            torch.Tensor : layer relevance of inputs

        """
        
        self.relevance = torch.sum(relevance, dim = 3)
        
        return self.relevance


class RadialDistribution(nn.Module):
    """
    Radial distribution function used e.g. to compute Behler type radial symmetry functions.

    Args:
        radial_filter (callable): Function used to expand distances (e.g. Gaussians)
        cutoff_function (callable): Cutoff function
    """

    def __init__(self,
                 radial_filter:nn.Module,
                 cutoff_function:nn.Module = CosineCutoff):
        super(RadialDistribution, self).__init__()
        self.radial_filter = radial_filter
        self.cutoff_function = cutoff_function

    def forward(self,
                r_ij:torch.Tensor,
                elemental_weights:Optional[torch.Tensor]=None,
                neighbor_mask:Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Args:
            r_ij (torch.Tensor): Interatomic distances
            elemental_weights (torch.Tensor): Element-specific weights for distance functions
            neighbor_mask (torch.Tensor): Mask to identify positions of neighboring atoms

        Returns:
            torch.Tensor: Nbatch x Natoms x Nfilter tensor containing radial distribution functions.
        """

        nbatch, natoms, nneigh = r_ij.size()

        radial_distribution = self.radial_filter(r_ij)

        # If requested, apply cutoff function
        if self.cutoff_function is not None:
            cutoffs = self.cutoff_function(r_ij)
            radial_distribution = radial_distribution * cutoffs.unsqueeze(-1)

        # Apply neighbor mask
        if neighbor_mask is not None:
            radial_distribution = radial_distribution * torch.unsqueeze(
                neighbor_mask, -1
            )

        # Weigh elements if requested
        if elemental_weights is not None:
            radial_distribution = (
                radial_distribution[:, :, :, :, None]
                * elemental_weights[:, :, :, None, :].float()
            )

        radial_distribution = torch.sum(radial_distribution, 2)
        radial_distribution = radial_distribution.view(nbatch, natoms, -1)
        return radial_distribution


class BernsteinPolynomials(nn.Module):
    r"""
    Radial basis functions based on Bernstein polynomials given by:
    b_{v,n}(x) = (n over v) * (x/cutoff)**v * (1-(x/cutoff))**(n-v)
    (see https://en.wikipedia.org/wiki/Bernstein_polynomial)
    Here, n = num_basis_functions-1 and v takes values from 0 to n. The basis
    functions are placed to optimally cover the range x = 0...cutoff.

    Args:
        n_basis_functions (int, optional): total number of basis functions, :math:`N_bf`.
        cutoff (float): Cutoff radius

    """

    def __init__(self,
                 cutoff: float,
                 n_basis_functions:int = 32):
        
        super(BernsteinPolynomials, self).__init__()
        logfactorial = np.zeros((n_basis_functions))
        for i in range(2, n_basis_functions):
            logfactorial[i] = logfactorial[i - 1] + np.log(i)
        v = np.arange(0, n_basis_functions)
        n = (n_basis_functions - 1) - v
        logbinomial = logfactorial[-1] - logfactorial[v] - logfactorial[n]
        
        # register buffers and parameters
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.register_buffer("logc", torch.tensor(logbinomial, dtype=torch.float32))
        self.register_buffer("n", torch.tensor(n, dtype=torch.float32))
        self.register_buffer("v", torch.tensor(v, dtype=torch.float32))
   
        self.relevance = None
         
    def forward(self, distances:torch.Tensor) -> torch.Tensor:
        """
        Evaluates radial basis functions given distances

        Args:
            distances (torch.Tensor): interatomic distance values of
                (N_b x N_at x N_nbh) shape.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh x N_bf) shape.

        """
        
        x = distances[:, :, :, None] / self.cutoff    # So that values are between 0 and 1
        '''
        Preventing nans
        Hopefully distances with zero values get masked at the end when acting with weights during cfconv
        Distance greater than cutoff after normalising with cutoff value should not occur
        '''
        x = torch.where((x < 1.0) * (x > 0.0), x, 0.5 * torch.ones_like(x))
        x = torch.log(x)
        x = self.logc[None, None, None, :] + self.n[None, None, None, :] * x + self.v[None, None, None, :] * torch.log(-torch.expm1(x))
        rbf = torch.exp(x)

        return rbf
        
    def relprop(self,
                input: Dict[str, torch.Tensor],
                relevance:torch.Tensor) -> torch.Tensor:
        """Compute layer relevance for its inputs.

        Args:
            input (dict of torch.Tensor): batch of input values.
            relevance (torch.tensor): relevance of outputs

        Returns:
            torch.Tensor : layer relevance of inputs

        """
        
        self.relevance = torch.sum(relevance, dim = 3)       
        return self.relevance


class ExponentialBernsteinPolynomials(nn.Module):
    r"""
    Radial basis functions based on exponential Bernstein polynomials given by:
    b_{v,n}(x) = (n over v) * exp(-alpha*x)**v * (1-exp(-alpha*x))**(n-v)
    (see https://en.wikipedia.org/wiki/Bernstein_polynomial)
    Here, n = num_basis_functions-1 and v takes values from 0 to n. This
    implementation operates in log space to prevent multiplication of very large
    (n over v) and very small numbers (exp(-alpha*x)**v and
    (1-exp(-alpha*x))**(n-v)) for numerical stability.
    NOTE: There is a problem for x = 0, as log(-expm1(0)) will be log(0) = -inf.
    This itself is not an issue, but the buffer v contains an entry 0 and
    0*(-inf)=nan. The correct behaviour could be recovered by replacing the nan
    with 0.0, but should not be necessary because issues are only present when
    r = 0, which will not occur with chemically meaningful inputs.

    Args:
        n_basis_functions (int, optional): total number of basis functions, :math:`N_bf`.
        no_basis_function_at_infinity (bool): If True, no basis function is put at exp(-alpha*x) = 0, i.e. x = infinity.
        ini_alpha (float): Initial value for scaling parameter alpha (Default value corresponds to 0.5 1/Bohr converted to 1/Angstrom).
        exp_weighting (bool): If True, basis functions are weighted with a factor exp(-alpha*r).

    """

    def __init__(self,
                 n_basis_functions: int,
                 no_basis_function_at_infinity: bool = False,
                 ini_alpha: float = 0.9448630629184640,
                 exp_weighting: bool = False,
    ):
    
        super(ExponentialBernsteinPolynomials, self).__init__()
        self.ini_alpha = ini_alpha
        self.exp_weighting = exp_weighting
        if no_basis_function_at_infinity:  # increase number of basis functions by one
            n_basis_functions += 1
            
        # compute values to initialize buffers
        logfactorial = np.zeros((n_basis_functions))
        for i in range(2, n_basis_functions):
            logfactorial[i] = logfactorial[i - 1] + np.log(i)
        v = np.arange(0, n_basis_functions)
        n = (n_basis_functions - 1) - v
        logbinomial = logfactorial[-1] - logfactorial[v] - logfactorial[n]
        if no_basis_function_at_infinity:  # remove last basis function at infinity
            v = v[:-1]
            n = n[:-1]
            logbinomial = logbinomial[:-1]
        
        # register buffers and parameters
        self.register_buffer("logc", torch.tensor(logbinomial, dtype=torch.float32))
        self.register_buffer("n", torch.tensor(n, dtype=torch.float32))
        self.register_buffer("v", torch.tensor(v, dtype=torch.float32))
        self.register_parameter( "_alpha", nn.Parameter(torch.tensor(1.0, dtype=torch.float32)))
        nn.init.constant_(self._alpha, softplus_inverse(self.ini_alpha))
        
        self.relevance = None
   
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Evaluates radial basis functions given distances
                       
        Args:
            distances (torch.Tensor): interatomic distance values of
                (N_b x N_at x N_nbh) shape.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh x N_bf) shape.

        """
        
        ''' softplus forces the exponential constant to be always positive during training '''
        alphar = -functional.softplus(self._alpha) * distances[:, :, :, None]  
        alphar = torch.where( alphar < 0.0, alphar, -0.5 * torch.ones_like(alphar))
        x = self.logc[None, None, None, :] + self.n[None, None, None, :] * alphar + self.v[None, None, None, :] * torch.log(-torch.expm1(alphar))
        rbf = torch.exp(x)
        
        if self.exp_weighting:
            return rbf * torch.exp(alphar)
        else:
            return rbf
                        
    def relprop(self,
                input: Dict[str, torch.Tensor],
                relevance: torch.Tensor) -> torch.Tensor:
        """Compute layer relevance for its inputs.

        Args:
            input (dict of torch.Tensor): batch of input values.
            relevance (torch.tensor): relevance of outputs

        Returns:
            torch.Tensor : layer relevance of inputs

        """
        
        self.relevance = torch.sum(relevance, dim = 3)        
        return self.relevance
            
            
class SphericalHarmonics(nn.Module):
    
    r"""
    Spherical Harmonics according to the hydrogenic orbitals
    s = 1
    px = x/r
    py = y/r
    pz = z/r
    dxy = \sqrt{3}xy/r**2
    dyz = \sqrt{3}yz/r**2
    dxz = \sqrt{3}xz/r**2
    dz2 = 1/2*(3(z/r)**2 - 1)
    dxy2 = \sqrt{3}/2 (x**2 - y**2)/r**2
    

    Args:
        s_orbitals_indicator (bool): If true then return s orbital poentials
        p_orbitals_indicator (bool): If true then return p orbital potentials
        d_orbitals_indicator (bool): If true then return d orbital potentials

    """

    def __init__(self,
                 s_oribitals_indicator:bool = True,
                 p_orbitals_indicator:bool = True,
                 d_orbitals_indicator:bool = False):
        
        super(SphericalHarmonics, self).__init__()
        self.s_orbitals_indicator = s_oribitals_indicator
        self.p_orbitals_indicator = p_orbitals_indicator
        self.d_orbitals_indicator = d_orbitals_indicator
               
    def forward(self, distance_vectors:torch.Tensor) -> List[torch.Tensor]:
    
        """
        Evaluates spherical basis functions given distance vectors
                       
        Args:
            distance_vectors (torch.Tensor): unit interatomic distance vectors of
                (N_b x N_at x N_nbh X 3) shape.

        Returns:
            list of torch.Tensor: a list where the size depends on what orbitals we require
                index 0 : evaluated s orbital values of (N_b X N_at X N_nbh X 1) shape
                index 1 : evaluated p orbital values of (N_b X N_at X N_nbh X 3) shape
                index 2 : evaluated d orbital values of (N_b X N_at X N_nbh X 5) shape

        """
        g_list:List[Optional[torch.Tensor]] = []

        if self.s_orbitals_indicator:
            s_orbitals = torch.ones_like(distance_vectors[:, :, :, [0]])
            g_list.append(s_orbitals)
        else:
            g_list.append(None)
        
        if self.p_orbitals_indicator:
            g_list.append(distance_vectors)
        else:
            g_list.append(None)
        
        if self.d_orbitals_indicator:
            g_d = torch.zeros(distance_vectors.shape[:-1] + (5,), device = distance_vectors.device)    # Appropriate shape
            x = distance_vectors[:, :, :, 0]
            y = distance_vectors[:, :, :, 1]
            z = distance_vectors[:, :, :, 2]
            
            g_d[:, :, :, 0] = np.sqrt(3)*x*y
            g_d[:, :, :, 1] = np.sqrt(3)*y*z
            g_d[:, :, :, 2] = np.sqrt(3)*x*z
            g_d[:, :, :, 3] = 1/2*(3*torch.pow(z, 2) - 1)
            g_d[:, :, :, 4] = np.sqrt(3)/2*(torch.pow(x, 2) - torch.pow(y, 2))
            
            g_list.append(g_d)
        else:
            g_list.append(None)
        
        return g_list
        
        

            
        
        

