# Main driver file
import numpy as np
import matplotlib.pyplot as plt
from Streamer import Streamer


def main() -> None:
    streamer = Streamer("netherlands-60m",
                        {"bootstrap_servers": "192.168.99.100:9092"},
                        topic_name="PerceptiveSentinel")
    streamer.start()


if __name__ == '__main__':
    main()
