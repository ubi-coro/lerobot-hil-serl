from experiments.aloha_bimanual.config import AlohaTestConfig, AlohaBimanualEnvConfig


if __name__ == "__main__":
    from lerobot.share.record import RecordConfig, record
    record(RecordConfig(env=AlohaBimanualEnvConfig(), dataset=AlohaTestConfig()))