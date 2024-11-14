# from datasets import load_dataset, DatasetDict

# # Load the dataset
# ds = load_dataset("kowndinya23/flan2022")

# # Create a new DatasetDict to hold the train and test splits
# split_ds = DatasetDict()

# # Iterate over each subset in ds and split into train and test sets
# for subset in ds:
#     train_test_split = ds[subset].train_test_split(test_size=0.2)
#     split_ds[subset] = DatasetDict({
#         'train': train_test_split['train'],
#         'test': train_test_split['test']
#     })

# # Save the split dataset to disk
# split_ds.save_to_disk('flan2022')


# # Count the unique template types
# unique_template_types = set(item['template_type'] for item in ds['flan2021'])
# total_unique_template_types = len(unique_template_types)

# print(f"Total number of unique template types: {total_unique_template_types}")