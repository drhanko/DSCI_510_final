# import praw
#
# reddit = praw.Reddit(
#     client_id="YOUR_ID",
#     client_secret="YOUR_SECRET",
#     user_agent="emotion_dataset"
# )
#
# subreddit = reddit.subreddit("BreakUps")
#
# for post in subreddit.hot(limit=10):
#     print(post.title)
#     print(post.selftext)
import pandas as pd
from datasets import load_dataset
#import data
empathetic_dialogues = load_dataset("empathetic_dialogues",trust_remote_code=True)
print("1")
print(empathetic_dialogues["train"][0])

daily_dialog = load_dataset("roskoN/dailydialog",trust_remote_code=True,download_mode="force_redownload")
print("2")
print(daily_dialog["train"][0])

go_emotions = load_dataset("go_emotions",trust_remote_code=True)
print("3")
print(go_emotions["train"][0])


rows = []

for idx, item in enumerate(daily_dialog["train"]):
    for turn_id, sentence in enumerate(item["dialog"]):
        rows.append({
            "dataset": "daily_dialog",
            "conversation_id": idx,
            "role": "user" if turn_id % 2 == 0 else "assistant",
            "text": sentence,
            "emotion": item["emotion"][turn_id]
        })

df_daily = pd.DataFrame(rows)
df_daily.to_csv("daily_dialog.csv", index=False)
