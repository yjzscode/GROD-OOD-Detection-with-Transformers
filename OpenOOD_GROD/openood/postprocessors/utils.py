from openood.utils import Config

from .base_postprocessor import BasePostprocessor
from .vim_postprocessor import VIMPostprocessor


def get_postprocessor(config: Config):
    postprocessors = {
        'ash': ASHPostprocessor,
        'cider': CIDERPostprocessor,
        'conf_branch': ConfBranchPostprocessor,
        'msp': BasePostprocessor,
        'ebo': EBOPostprocessor,
        'odin': ODINPostprocessor,
        'mds': MDSPostprocessor,
        'mds_ensemble': MDSEnsemblePostprocessor,
        'rmds': RMDSPostprocessor,
        'gmm': GMMPostprocessor,
        'patchcore': PatchcorePostprocessor,
        'openmax': OpenMax,
        'react': ReactPostprocessor,
        'vim': VIMPostprocessor,
        'gradnorm': GradNormPostprocessor,
        'godin': GodinPostprocessor,
        'gram': GRAMPostprocessor,
        'cutpaste': CutPastePostprocessor,
        'mls': MaxLogitPostprocessor,
        'npos': NPOSPostprocessor,
        'residual': ResidualPostprocessor,
        'klm': KLMatchingPostprocessor,
        'temperature_scaling': TemperatureScalingPostprocessor,
        'ensemble': EnsemblePostprocessor,
        'dropout': DropoutPostProcessor,
        'draem': DRAEMPostprocessor,
        'dsvdd': DSVDDPostprocessor,
        'mos': MOSPostprocessor,
        'mcd': MCDPostprocessor,
        'opengan': OpenGanPostprocessor,
        'knn': KNNPostprocessor,
        'dice': DICEPostprocessor,
        'ssd': SSDPostprocessor,
        'she': SHEPostprocessor,
        'rd4ad': Rd4adPostprocessor,
        'rts': RTSPostprocessor,
        'rotpred': RotPredPostprocessor,
        'rankfeat': RankFeatPostprocessor,
        'gen': GENPostprocessor,
        'relation': RelationPostprocessor,
        'grod': GRODPostprocessor
    }

    return postprocessors[config.postprocessor.name](config)
