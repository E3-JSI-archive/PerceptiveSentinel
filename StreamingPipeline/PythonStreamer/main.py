# Main driver file
import numpy as np
import matplotlib.pyplot as plt
from Streamer import Streamer, TULIP_FIELD_COORDINATES, DEFAULT_CLOUD_DETECTION_SETTINGS


def main() -> None:
    dcds = DEFAULT_CLOUD_DETECTION_SETTINGS
    dcds.x_scale = 1
    dcds.y_scale = 1
    daq_settings = {
        "coordinates": TULIP_FIELD_COORDINATES,
        "start_date": "2017-03-20",  # Nice non-cloud date by hand
        "end_date": "2017-11-25",
        "res_x": 10,  # Full resolution
        "res_y": 10,
        "cloud_detection_settings": dcds,
    }

    # Construct a new streamer, operating on data named "netherlands-60m"
    streamer = Streamer("netherlands-60m_clouds",
                        kafka_config={
                            "bootstrap_servers": "192.168.99.100:9092"},
                        topic_name="PerceptiveSentinel",
                        daq_settings=daq_settings)
    streamer.start()


if __name__ == '__main__':
    main()
