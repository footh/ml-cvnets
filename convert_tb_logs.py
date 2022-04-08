import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse


parser = argparse.ArgumentParser(description='Convert tb logs to pandas dataframe')
parser.add_argument('path', type=str, help='Path to log files')
args = parser.parse_args()


# Extraction function
def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


df = tflog2pandas(args.path)
df.to_csv(f'{args.path.split("/")[0]}.csv')
