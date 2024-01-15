import torch
from torch import nn

from schnetpack.nn import Dense
from schnetpack.nn.base import Aggregate
from typing import Callable, Dict, Optional

__all__ = ["CFConv", "AngularCFConv"]


class CFConv(nn.Module):
    r"""Continuous-filter convolution block used in SchNet module.

    Args:
        n_in (int): number of input (i.e. atomic embedding) dimensions.
        n_filters (int): number of filter dimensions.
        n_out (int): number of output dimensions.
        filter_network (nn.Module): filter block.
        cutoff_network (nn.Module, optional): if None, no cut off function is used.
        activation (callable, optional): if None, no activation function is used.
        normalize_filter (bool, optional): If True, normalize filter to the number
            of neighbors when aggregating.
        axis (int, optional): axis over which convolution should be applied.

    """

    def __init__(
        self,
        n_in:int,
        n_filters:int,
        n_out:int,
        filter_network: nn.Module,
        cutoff_network:Optional[nn.Module]=None,
        activation:Optional[Callable]=None,
        normalize_filter:bool=False,
        axis:int=2,
    ):
        super(CFConv, self).__init__()
        self.in2f = Dense(n_in, n_filters, bias=False, activation=None)
        self.f2out = Dense(n_filters, n_out, bias=True, activation=activation)
       
        self.filter_network = filter_network
        self.cutoff_network = cutoff_network
        self.agg = Aggregate(axis=axis, mean=normalize_filter)
        
        self.x_relevance = None
        self.f_relevance = None
        self.W_relevance = 0
        
    def forward(self,
                x:torch.Tensor,
                r_ij:torch.Tensor,
                neighbors:torch.Tensor,
                pairwise_mask:torch.Tensor,
                f_ij:Optional[torch.Tensor]=None) -> torch.Tensor:
        """Compute convolution block.

        Args:
            x (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, n_in) shape.
            r_ij (torch.Tensor): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (torch.Tensor): indices of neighbors of (N_b, N_a, N_nbh) shape.
            pairwise_mask (torch.Tensor): mask to filter out non-existing neighbors
                introduced via padding.
            f_ij (torch.Tensor, optional): expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.

        Returns:
            torch.Tensor: block output with (N_b, N_a, n_out) shape.

        """
        
        if f_ij is None:
            f_ij = r_ij.unsqueeze(-1)

        # pass expanded interactomic distances through filter block
        W = self.filter_network(f_ij)
        # apply cutoff
        if self.cutoff_network is not None:
            C = self.cutoff_network(r_ij)
            W = W * C.unsqueeze(-1)

        # pass initial embeddings through Dense layer
        y = self.in2f(x)
        # reshape y for element-wise multiplication by W
        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
        nbh = nbh.expand(-1, -1, y.size(2))
        y = torch.gather(y, 1, nbh)
        y = y.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)
        
        # element-wise multiplication, aggregating and Dense layer
        y = y * W
        y = self.agg(y, pairwise_mask)
        y = self.f2out(y)
        
        return y
        
    def relprop(self,
                x:torch.Tensor,
                r_ij:torch.Tensor,
                neighbors:torch.Tensor,
                pairwise_mask:torch.Tensor,
                relevance:torch.Tensor,
                f_ij:Optional[torch.Tensor]=None,
                gamma:float = 0.0,
                epsilon:float = 0.0) -> torch.Tensor:
        """Compute layer relevance for its inputs.

        Args:
            inputs (dict of torch.Tensor): batch of input values.
            relevance (torch.tensor): relevance of output interactions

        Returns:
            torch.Tensor : layer relevance of inputs.

        """
        
        if f_ij is None:
            f_ij = r_ij.unsqueeze(-1)

        # pass expanded interatomic distances through filter block
        filter_network_inputs_list = []    # Saving intermediatory results for relprop
        filter_network_input = f_ij
        
        for nn in self.filter_network:
            filter_network_inputs_list.append(filter_network_input)
            filter_network_input = nn(filter_network_input)
            
        W = self.filter_network(f_ij)
        # apply cutoff
        if self.cutoff_network is not None:
            C = self.cutoff_network(r_ij)
            W = W * C.unsqueeze(-1)

        # pass initial embeddings through Dense layer
        xf = self.in2f(x)
        # reshape y for element-wise multiplication by W
        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
        nbh = nbh.expand(-1, -1, xf.size(2))
        xf = torch.gather(xf, 1, nbh)
        xf = xf.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)
        nbh = nbh.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)
        
        # element-wise multiplication, aggregating and Dense layer
        y = l * W
        lf= self.agg(y, pairwise_mask)
        l = self.f2out(lf)
        
        lf_relevance = self.f2out.relprop(lf, relevance)
        y_relevance =  self.agg.relprop(y, lf_relevance, mask = pairwise_mask)
        
        ''' Checking if assumption is right '''
        print('Assumption : ',(torch.sign(xf)*torch.sign(W) == torch.sign(y_relevance)).any())
        
        ''' Breaking up the relevances for multiplication '''
        
        alpha = abs(W + 1e-9)/abs(xf + 1e-9)
        W_relevance = torch.sign(W)*alpha*torch.sqrt(abs(y_relevance))
        xf_relevance = torch.sign(xf)/alpha*torch.sqrt(abs(y_relevance))
        
        # Summing up relevances of atom embeddings along the neighbor axis
        xf_size = xf.size()
        x_relevance = torch.zeros(xf_size[0], xf_size[1], xf_size[3])
        
        batch_idx = torch.arange(xf_size[0])[:, None, None].expand(-1, xf_size[2], xf_size[3])
        filter_idx = torch.arange(xf_size[3])[None, None, :].expand(xf_size[0], xf_size[2], -1)
        
        for row in range(xf_size[1]):    # unique indices along rows (atom axis) for neighbor indices
            x_relevance[batch_idx, nbh[:, row, :, :], filter_idx] += xf_relevance[:, row, :, :]
        
        '''
        If coupled:
            Then then same weights used for every interaction block. So add up the weight contributions
            
        If not coupled:
            Then different weight filters used for each interaction block. So this particular cfconv is only
            called once. Therefore it effectivly only assigns the weight relevance value
        
        '''       
        self.W_relevance += W_relevance   
        f_relevance = self.W_relevance
        filter_network_inputs_list = reversed(filter_network_inputs_list)    # Reversing for backward relevance propagation
        for ii, nn in enumerate(reversed(list(self.filter_network))):    # Going in reverse order to propagate relevance
            f_relevance = nn.relprop(filter_network_inputs_list[ii], f_relevance)
        
        
        self.x_relevance = x_relevance
        self.f_relevance = f_relevance
                        
        return self.x_relevance, self.f_relevance, self.W_relevance
                
class AngularCFConv(nn.Module):
    r"""Continuous-filter convolution block used in SchNet module including angular informatin through spherical harmonics

    Args:
        n_in (int): number of input (i.e. atomic embedding) dimensions.
        n_filters_sblock: number of filter dimensions
        n_out (int): number of output dimensions.
        filter_network (nn.Module): filter block.
        cutoff_network (nn.Module, optional): if None, no cut off function is used.
        activation (callable, optional): if None, no activation function is used.
        normalize_filter (bool, optional): If True, normalize filter to the number
            of neighbors when aggregating.
        axis (int, optional): axis over which convolution should be applied.

    """

    def __init__(
        self,
        n_in:int,
        n_filters:int,
        n_out:int,
        filter_network_sblock:Optional[nn.Module] = None,
        filter_network_pblock:Optional[nn.Module] = None,
        filter_network_dblock:Optional[nn.Module] = None,
        cutoff_network:Optional[nn.Module]=None,
        activation:Optional[Callable]=None,
        normalize_filter:bool=False,
        axis:int=2,
        sblock_indicator:bool = True,
        pblock_indicator:bool = True,
        dblock_indicator:bool = False,
        projection_indicator:bool = False,
    ):
        super(AngularCFConv, self).__init__()
        
        self.sblock_indicator = sblock_indicator
        self.pblock_indicator = pblock_indicator
        self.dblock_indicator = dblock_indicator
        self.projection_indicator = projection_indicator
        self.n_filters = n_filters
        
        if sblock_indicator:      
            self.in2f_sblock = Dense(n_in, n_filters, bias=False, activation=activation)
        if pblock_indicator:
            self.in2f_pblock = Dense(n_in, n_filters, bias=False, activation=activation)
            
            if projection_indicator:
                self.P1_pblock = Dense(n_filters, n_filters, bias=False)    # To project into P1 space
                self.P2_pblock = Dense(n_filters, n_filters, bias=False)    # To project into P2 space
                
        if dblock_indicator:
            self.in2f_dblock = Dense(n_in, n_filters, bias=False, activation=activation)
            
            if projection_indicator:
                self.P1_dblock = Dense(n_filters, n_filters, bias=False)    # To project into P1 space
                self.P2_dblock = Dense(n_filters, n_filters, bias=False)    # To project into P2 space
        
        
        self.f2out = Dense(n_filters, n_out, bias=True, activation=activation)      
        self.filter_network_sblock = filter_network_sblock
        self.filter_network_pblock = filter_network_pblock
        self.filter_network_dblock = filter_network_dblock
        self.cutoff_network = cutoff_network
        self.agg = Aggregate(axis=axis, mean=normalize_filter)

    def forward(self,
                x:torch.Tensor,
                r_ij:torch.Tensor,
                neighbors:torch.Tensor,
                pairwise_mask:torch.Tensor,
                fsblock_ij:Optional[torch.Tensor]=None,
                fpblock_ij:Optional[torch.Tensor] = None,
                fdblock_ij:Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute convolution block.

        Args:
            x (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, n_in) shape.
            r_ij (torch.Tensor): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (torch.Tensor): indices of neighbors of (N_b, N_a, N_nbh) shape.
            pairwise_mask (torch.Tensor): mask to filter out non-existing neighbors
                introduced via padding.
            fsblock_ij (torch.Tensor, optional): expanded interatomic distances in a basis evaulated for s orbitals. (Nb, Na, N_nbh, 1, N_fil)
                If None, r_ij.unsqueeze(-1) is used.
            fpblock_ij (torch.Tensor, optional): expanded interatomic distances in a basis evaulated for p orbitals. (Nb, Na, N_nbh, 3, N_fil)
                If None, r_ij.unsqueeze(-1) is used.
            fdblock_ij (torch.Tensor, optional): expanded interatomic distances in a basis evaulated for d orbitals. (Nb, Na, N_nbh, 5, N_fil)
                If None, r_ij.unsqueeze(-1) is used.

        Returns:
            torch.Tensor: block output with (N_b, N_a, n_out) shape.

        """
        
        if self.cutoff_network is not None:
            C = self.cutoff_network(r_ij)
        else:
            C = torch.ones_like(r_ij)
         
        # reshape neighbors for element-wise multiplication by W
        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
        nbh = nbh.expand(-1, -1, self.n_filters) 
        
        # The resultant interaction vector
        y = 0
           
        # pass expanded interactomic distances through filter block
        if self.sblock_indicator:
            Ws = self.filter_network_sblock(fsblock_ij)  
            Ws = Ws * C[:, :, :, None, None] 
                            

            # pass initial embeddings through Dense layer
            ys = self.in2f_sblock(x)
            # reshape y for element-wise multiplication by W
            ys = torch.gather(ys, 1, nbh)
            ys = ys.view(nbh_size[0], nbh_size[1], nbh_size[2], 1, -1)
            
            # element-wise multiplication, aggregating and Dense layer
            ys = ys * Ws
            ys = ys.squeeze(dim = 3)    # Removing the orbital axis
            ys = self.agg(ys, pairwise_mask)

            y = y + ys
                                  
        if self.pblock_indicator:
            Wp = self.filter_network_pblock(fpblock_ij)
            Wp = Wp * C[:, :, :, None, None] 

            # pass initial embeddings through Dense layer
            yp = self.in2f_pblock(x)
            # reshape y for element-wise multiplication by W
            yp = torch.gather(yp, 1, nbh)
            yp = yp.view(nbh_size[0], nbh_size[1], nbh_size[2], 1, -1)
            
            # element-wise multiplication, aggregating and Dense layer
            yp = yp * Wp
            yp = self.agg(yp, pairwise_mask[..., None])    # Including orbital axis to pairwise mask
            
            if self.projection_indicator: 
                yp_P1 = self.P1_pblock(yp)
                yp_P2 = self.P2_pblock(yp)
                yp = torch.sum(yp_P1*yp_P2, 2)    # summing over the orbital axis. Resultant shape (Nb X Na X Nfil)
            else:
                yp = torch.sum(torch.pow(yp, 2), 2)    # summing over the orbital axis. Resultant shape (Nb X Na X Nfil)
            
                        
            y = y + yp
            
        if self.dblock_indicator:
            Wd = self.filter_network_dblock(fdblock_ij)
            Wd = Wd * C[:, :, :, None, None] 
            
            # pass initial embeddings through Dense layer
            yd = self.in2f_dblock(x)
            # reshape y for element-wise multiplication by W
            yd = torch.gather(yd, 1, nbh)
            yd = yd.view(nbh_size[0], nbh_size[1], nbh_size[2], 1, -1)
            
            # element-wise multiplication, aggregating and Dense layer
            yd = yd * Wd
            yd = self.agg(yd, pairwise_mask[..., None])    # Including orbital axis to pairwise mask
            
            if self.projection_indicator:           
                yd_P1 = self.P1_dblock(yd)
                yd_P2 = self.P2_dblock(yd)
                yd = torch.sum(yd_P1*yd_P2, 2)    # summing over the orbital axis. Resultand shape (Nb X Na X Nfil)
                
            else:
                yd = torch.sum(torch.pow(yd, 2), 2)    # summing over the orbital axis. Resultand shape (Nb X Na X Nfil) 
            
            y = y + yd
        
        y = self.f2out(y)
        
        return y
        