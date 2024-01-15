import torch
import torch.nn as nn
import math
from typing import Optional  

__all__ = ["Attention"]

class Attention(nn.Module):
    r"""
    Efficient (linear scaling) approximation for attention described in
    Choromanski, K., et al. "Rethinking Attention with Performers.".
    
    Args:
        dim_qk (int):
            Dimension of query/key vectors.
        dim_v (int):
            Dimension of value vectors.
        num_random_featues (int):
            Number of random features for approximating attention matrix. If
            this is 0, the exact attention matrix is computed.
    """

    def __init__(
        self,
        dim_qk:int,
        dim_v:int,
        num_random_features: Optional[int] = None
    ):

        super(Attention, self).__init__()
        self.num_random_features = num_random_features
        if self.num_random_features is not None:
            omega = self._omega(num_random_features, dim_qk)
        else:
            omega = []
        self.register_buffer("omega", torch.tensor(omega, dtype=torch.float32))
        self.reset_parameters() 

    def reset_parameters(self) -> None:
        """ For compatibility with other modules. """
        pass

    def _omega(self,
               num_random_features:int,
               dim_qk:int) -> torch.Tensor:
        r""" Calculates the random feature matrix omega. Forms block orthogonal matrices
        
        Args:
            num_random_features(int): The number of rows in the random feature matrix
            dim_qk(int): The number of cols in the random feature matrix
            
        Returns:
            omega(np.ndarray dim_qk X num_random_features): The random feature matrix
        
        """
        nrows = num_random_features
        ncols = dim_qk
        
        nblocks = int(nrows / ncols)
        blocks = []
        for i in range(nblocks):
            block = torch.randn(size=(ncols, ncols))
            q, _ = torch.linalg.qr(block)
            blocks.append(torch.transpose(q, 0, 1))
        missing_rows = nrows - nblocks * ncols
        if missing_rows > 0:
            block = torch.randn(size=(ncols, ncols))
            q, _ = torch.linalg.qr(block)
            blocks.append(torch.transpose(q, 0, 1)[:missing_rows])
        norm = torch.linalg.norm(  # renormalize rows so they still follow N(0,1)
            torch.randn(size=(nrows, ncols)), axis=1, keepdims=True
        )
        omega = (norm * torch.vstack(blocks)).T
        
        return omega

    def _phi(
        self,
        X:torch.Tensor,
        atom_mask:torch.Tensor,
        is_query:bool,
        eps:float = 1e-4,
    ) -> torch.Tensor:
        r""" Normalize the input and project into random feature space. 
        
        Args:
            X (torch.tensor n_batch X n_atoms X n_features): The input array which contains the atomic features in the cols.
                                                             Should have been masked before to remove the false atoms
                                                             
            atom_mask (torch.tensor n_batch X n_atoms): The mask to remove the false atoms
            is_query:  True if query matrix is given otherwise False
                       If query matrix then subtract over the maximum of only over feature axis
                       If key matrix then subtract over the maximum of both atom and feature axis
            eps (float):  A small constant for avoiding division errors
        
        Returns:
            U (torch.tensor n_batch X natoms X n_random_features): The input projected into the random feature space
        
        """
        num_features = X.shape[-1]    
        num_random_features = self.omega.shape[-1]    
        U = (X / num_features ** 0.25) @ (self.omega)
        h = torch.sum(X ** 2, dim=-1, keepdim=True) / (2 * num_features ** 0.5)  # OLD
        # determine maximum (is subtracted to prevent numerical overflow)
        
        if is_query:
            maximum, _ = torch.max(U, dim=-1, keepdim=True)    # Maximum only over the features
        else:
            maximum, _ = torch.max(U, dim = -1, keepdim = True)    # Maximum over the features
            maximum, _ = torch.max(maximum, dim = -2, keepdim = True)    # Maximum over the atoms
            maximum = maximum.expand(-1, X.shape[1], 1)    # Resizing over the atom axis
            
        U = ((torch.exp(U - h - maximum) + eps) / math.sqrt(num_random_features)) * atom_mask.unsqueeze(-1)
          
        return U

    def _exact_attention(
        self,
        Q:torch.Tensor,
        K:torch.Tensor,
        V:torch.Tensor,
        atom_mask:torch.Tensor,
        eps:float = 1e-8,
    ) -> torch.Tensor:
        r""" Compute exact attention. 
        
        Args:
            Q (torch.Tensor): The Q query matrix for attention calculation. Should have been masked by atom mask before
            K (torch.Tensor): The K key matrix for attention calculation. Should have been masked by atom mask before
            V (torch.Tensor): The V value matrix for attention calculation. Should have been masked by atom mask before
            atom_mask (torch.Tensor): The atom mask to remove the incorporation of false atoms
            eps (Float): Constant for avoding dividing zero error
        
        Returns:
            attention (torch.Tensor nbatch X natoms X nfeatures): The attention matrix  
        
        """
                
        num_features = Q.shape[-1]
        A_temp = torch.bmm(Q, K.transpose(-2, -1))  # start formation of the attention A matrix
        A = torch.exp((A_temp - torch.max(A_temp)) / num_features ** 0.5)  # normalised attention matrix
        A_mask = atom_mask.unsqeeze(-1) * atom_mask.unsqueeze(-2)    # Mask to remove the attention terms with false atoms
        A = A * A_mask     # Since exp(0) = 1. We mask again to remove their attention contribution
        
        # Old Code
        
        #if num_batch > 1:  # mask out entries of different batches
        #    brow = batch_seg.view(1, -1).expand(A.shape[-2], -1)
        #    bcol = batch_seg.view(-1, 1).expand(-1, A.shape[-1])
        #    mask = torch.where(brow == bcol, torch.ones_like(A), torch.zeros_like(A))
        #    A = A * mask
        
        
        norm = torch.sum(A, -1, keepdim=True) + eps
        attention = torch.bmm(A/norm, V)    # Softmax kernelized
        return attention

    def _approximate_attention(
        self,
        Q:torch.Tensor,
        K:torch.Tensor,
        V:torch.Tensor,
        atom_mask:torch.Tensor,
        eps:float = 1e-8,
    ) -> torch.Tensor:
       r""" Compute approximate attention. 
       
       Args:
           Q (torch.Tensor): The Q query matrix for attention calculation
           K (torch.Tensor): The K key matrix for attention calculation
           V (torch.Tensor): The V value matrix for attention calculation
           atom_mask (torch.Tensor): To remove the contribution from the false atoms
           eps (Float): Constant for avoding dividing zero error  
       
       Returns:
           attention (torch.Tensor nbatch X natoms X nfeatures): The attention matrix  
       
       """
       Q = self._phi(Q, atom_mask, True)  # random projection of Q via softmax kernel
       K = self._phi(K, atom_mask, False)  # random projection of K via softmax kernel
       
       # Old Code
       '''
       if num_batch > 1:
           d = Q.shape[-1]
           n = batch_seg.shape[0]
      
           # compute norm
           idx = batch_seg.unsqueeze(-1).expand(-1, d)
           tmp = K.new_zeros(num_batch, d).scatter_add_(0, idx, K)
           norm = torch.gather(Q @ tmp.T, -1, batch_seg.unsqueeze(-1)) + eps
      
           # the ops below are equivalent to this loop (but more efficient):
           # return torch.cat([Q[b==batch_seg]@(
           #    K[b==batch_seg].transpose(-1,-2)@V[b==batch_seg])
           #    for b in range(num_batch)])/norm
           if mask is None:  # mask can be shared across multiple attentions
               one_hot = nn.functional.one_hot(batch_seg).to(
                  dtype=V.dtype, device=V.device
              )
               mask = one_hot @ one_hot.transpose(-1, -2)
           return ((mask * (K @ Q.transpose(-1, -2))).transpose(-1, -2) @ V) / norm
      
       else:
           norm = Q @ torch.sum(K, 0, keepdim=True).T + eps
           return (Q @ (K.T @ V)) / norm
       '''
           
       norm = torch.bmm(Q, torch.sum(K, 1, keepdim = True).transpose(-1, -2)) + eps
       temp = torch.bmm(K.transpose(-1, -2), V)
       attention = torch.bmm(Q, temp) / norm
       
       return attention

    def forward(
        self,
        Q:torch.Tensor,
        K:torch.Tensor,
        V:torch.Tensor,
        atom_mask:torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Compute attention for the given query, key and value vectors.
        N: Number of input values.
        dim_qk: Dimension of query/key vectors.
        dim_v: Dimension of value vectors.
        Arguments:
            Q (FloatTensor [n_batch, n_atoms, dim_qk]):
                Matrix of query vectors. Should have been masked by atom mask before
            K (FloatTensor [n_batch, n_atoms, dim_qk]):
                Matrix of key vectors. Should have been masked by atom mask before
            V (FloatTensor [n_batch, n_atoms, dim_v]):
                Matrix of value vectors. Should have been masked by atom mask before
            atom_mask (BoolTensor [n_batch, n_atoms]):
                
        Returns:
            y (FloatTensor [n_batch, n_atoms, dim_v]):
                Attention-weighted sum of value vectors.
        """
        if self.num_random_features is None:
            return self._exact_attention(Q, K, V, atom_mask)
        else:
            return self._approximate_attention(Q, K, V, atom_mask)
