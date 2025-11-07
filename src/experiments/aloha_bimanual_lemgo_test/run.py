from experiments.aloha_bimanual.config import AlohaSingleDatasetConfig
from experiments.aloha_bimanual.config import AlohaSingleEnvConfig


if __name__ == "__main__":
    from lerobot.share.record import RecordConfig, record
    record(RecordConfig(env=AlohaSingleEnvConfig(), dataset=AlohaSingleDatasetConfig()))