from src.lerobot.datasets.utils import dataset_to_policy_features
from src.lerobot.bootcamp.vlm_rlpd import SACVLMPolicy


def make_sac(sac_cfg, ds_meta):
    features = dataset_to_policy_features(ds_meta.features)

    kwargs = {}
    if not sac_cfg.output_features:
        sac_cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    if not sac_cfg.input_features:
        sac_cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    kwargs["config"] = sac_cfg

    if sac_cfg.pretrained_path:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = sac_cfg.pretrained_path
        policy = SACVLMPolicy.from_pretrained(**kwargs)
    else:
        # Make a fresh policy.
        policy = SACVLMPolicy(**kwargs)

    return policy
