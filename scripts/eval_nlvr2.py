"""
copied from official NLVR2 github
(https://github.com/lil-lab/nlvr/tree/master/nlvr2)

python scripts/eval_nlvr2.py <output.csv> <annotation.json>
"""
import json
import sys

# Load the predictions file. Assume it is a CSV.
predictions = { }
for line in open(sys.argv[1]).readlines():
  if line:
    splits = line.strip().split(",")
    # We assume identifiers are in the format "split-####-#-#.png".
    identifier = splits[0]
    prediction = splits[1]
    predictions[identifier] = prediction

# Load the labeled examples.
labeled_examples = [json.loads(line) for line in open(sys.argv[2]).readlines() if line]

# If not, identify the ones that are missing, and exit.
total_num = len(labeled_examples)
if len(predictions) < total_num:
  print("Some predictions are missing!")
  print("Got " + str(len(predictions)) + " predictions but expected " + str(total_num))

  for example in labeled_examples:
      lookup = example["identifier"]
      if not lookup in predictions:
        print("Missing prediction for item " + str(lookup))
  exit()

# Get the precision by iterating through the examples and checking the value
# that was predicted.
# Also update the "consistency" dictionary that keeps track of whether all
# predictions for a given sentence were correct.
num_correct = 0.
consistency_dict = { }

for example in labeled_examples:
  anon_label = example["identifier"].split("-")
  anon_label[2] = ''
  anon_label = '-'.join(anon_label)
  if not anon_label in consistency_dict:
    consistency_dict[anon_label] = True
  lookup = example["identifier"]
  prediction = predictions[lookup]
  if prediction.lower() == example["label"].lower():
    num_correct += 1.
  else:
    consistency_dict[anon_label] = False

# Calculate consistency.
num_consistent = 0.
unique_sentence = len(consistency_dict)
for identifier, consistent in consistency_dict.items():
  if consistent:
    num_consistent += 1

# Report values.
print("accuracy=" + str(num_correct / total_num))
print("consistency=" + str(num_consistent / unique_sentence))
