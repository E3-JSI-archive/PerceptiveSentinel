# Main driver file
import numpy as np
import matplotlib.pyplot as plt
from Streamer import Streamer


def main() -> None:
    streamer = Streamer("netherlands")
    streamer.prepare()


if __name__ == '__main__':
    main()
