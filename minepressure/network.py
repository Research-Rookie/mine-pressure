"""Complete paper architecture; component training and inference are separate."""
from torch import nn
from .full_train_extreme_operator import FullTrainWideFaceOperator
from .asymmetric_boundary_flux_network import AsymmetricBoundaryFlux
from .conditional_normal_flux import ConditionalNormalFlux
from .backward_boundary_prototypes import TwoBackwardBoundaryCovers
from .pipeline import predict_arrays


class MinePressureNetwork(nn.Module):
    def __init__(self, numerical_statistics, normal_edges, body_feature_mean,
                 body_feature_scale, warning_states, *, parent_width, body_width):
        super().__init__()
        self.numeric = FullTrainWideFaceOperator(numerical_statistics)
        self.parent = AsymmetricBoundaryFlux(normal_edges, width=parent_width)
        self.body = ConditionalNormalFlux(normal_edges, body_feature_mean,
                                          body_feature_scale, width=body_width, local=True)
        self.warning = nn.ModuleList([
            TwoBackwardBoundaryCovers(state, transport=False)
            for state in warning_states
        ])

    def forward(self, sequence, summary_raw, anchor_raw, installed, *,
                preprocessing, probability_links, raw_cuts, engineering_limits,
                lower_limits, body_batch_size):
        """Array inference; use individual component forwards when fitting."""
        bundle = dict(numeric=self.numeric, parent=self.parent, body=self.body,
                      warning=self.warning, preprocessing=preprocessing,
                      probability_links=probability_links, raw_cuts=raw_cuts,
                      engineering_limits=engineering_limits, lower_limits=lower_limits,
                      body_batch_size=body_batch_size,
                      device=str(next(self.parameters()).device))
        return predict_arrays(bundle, sequence, summary_raw, anchor_raw,
                              installed)
