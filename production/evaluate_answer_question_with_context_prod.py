import os
from datetime import datetime, timezone, timedelta
from domino_eval_trace import *
from util import answer_question_with_context
import evaluators
import pandas as pd

"""
Runs the question_fulfillment evaluator
on the dataset built from production "rag_response" traces
"""

def get_domino_dataset_file_names() -> list[str]:
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).timestamp() * 1000
    return [str(yesterday)]

if __name__ == "__main__":
    # required arguments
    # NOTE: these are best practices that we will share through user facing docs
    yesterday = datetime.now(timezone.utc) - timedelta(days=2)
    evaluate_start_time_ts = datetime.fromisoformat(os.getenv("DOMINO_EVAL_START_TS", str(yesterday))).timestamp() * 1000

    # get dataset files with name that are >= evaluate_start_time_ts
    for fn in get_domino_dataset_file_names():
        if int(fn) >= evaluate_start_time_ts:
            df = pd.read_csv(fn)
            df['evaluation_score'] = df.apply(
                lambda row: evaluators.question_fullfillment_evaluator(row['inputs'], row['outputs'])['fullfilled'],
                axis=1
            )

            # write back to dataset scratchspace
            df.to_csv(fn, index=False)
