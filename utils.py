import os
import math
import time
import torch
import pickle
import argparse
import random
import datasets
import submodlib
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset, load_from_disk, DatasetDict
import datasets  # Ensure datasets module is imported
import submodlib.functions as submod_fn
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

random.seed(23)

SUBMIXTURES=[
    "flan2021",
    "t0",
    "niv2",
    "cot",
    "dialog"
]  #1841

TEMPLATE_TYPES=[
    "zs_opt",
    "zs_noopt",
    "fs_opt",
    "fs_noopt"
]

CONTEXT_LEN=4096
TEMPERATURE=1
HUB_TOKEN=None
HUB_USERNAME=None
dir = '/mnt/DATA/datasets'

def parse_args():
    parser=argparse.ArgumentParser(description="Get Instructo Dataset from FLAN 2022")
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=8,
        help="Number of tasks from FLAN 2022 to use for creating dataset"
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=25000,
        help="Number of instances from FLAN 2022 to use for creating dataset"
    )
    parser.add_argument(
        "--submod_fnc_tasks",
        type=str,
        choices=["fl", "gc", "logdet", "random"],
        default="gc",
        help="Submodular function to use for computing task budgets"
    )
    parser.add_argument(
        "--submod_fnc_instances",
        type=str,
        choices=["fl", "gc", "logdet", "random"],
        default="fl",
        help="Submodular function to use for getting subset of instances from each task"
    )
    parser.add_argument(
        "--HUB_TOKEN",
        type=str,
        default=None,
        help="Hugging Face Hub Token to upload the dataset"
    )
    parser.add_argument(
        "--HUB_USERNAME",
        type=str,
        default=None,
        help="Hugging Face Hub Username to upload the dataset"
    )

    args=parser.parse_args()

    # Some sanity checks

    # 1841 is the total number of tasks in FLAN 2022
    assert args.num_tasks > 0 and args.num_tasks < 1841 

    # 17587432 is the total number of instances in FLAN 2022 (CONTEXT_LEN=4096)
    assert args.num_instances > 0 and args.num_instances <= 17587432

    return args

def download_flan2022():
    print("Check if FLAN 2022 dataset is already downloaded. If not, download it and load it")
    if not os.path.exists('flan2022'):
        # Load the dataset
        ds = load_dataset("kowndinya23/flan2022")

        # Create a new DatasetDict to hold the train and test splits
        split_ds = DatasetDict()

        # Iterate over each subset in ds and split into train and test sets
        for subset in ds:
            train_test_split = ds[subset].train_test_split(test_size=0.2)
            split_ds[subset] = DatasetDict({
                'train': train_test_split['train'],
                'test': train_test_split['test']
            })

        # Save the split dataset to disk
        split_ds.save_to_disk(f'{dir}/flan2022')
        
    for submixture in SUBMIXTURES:
        print(f"Loading {submixture} dataset...")                   
        dataset = load_from_disk(f"{dir}/flan2022/{submixture}")
        print(dataset)

def compute_prompt_embeddings():
    # check if prompts_embeddings directory is present
    print("Check if prompts_embeddings directory is present and it contains embeddings for all submixtures. If not, compute embeddings for prompts.")
    already_computed=True
    if os.path.exists("prompts_embeddings"):
        print("\n==== Path already Exists ====\n")
        for submixture in SUBMIXTURES:
            if not os.path.exists(f"prompts_embeddings/{submixture}.npy"):
                already_computed=False
                break
    else:
        already_computed=False  # Set to False if directory does not exist

    if not already_computed:
        print("Embeddings not computed. Computing embeddings for prompts now...")
        os.makedirs("prompts_embeddings", exist_ok=True)
        print("Loading thenlper/gte-large model")
        model=SentenceTransformer("thenlper/gte-large")
        for submixture in SUBMIXTURES:
            print(f"Generating embeddings for {submixture} prompts")
            submixture_data=DatasetDict.load_from_disk(f"{dir}/flan2022/{submixture}")["train"]
            prompts=submixture_data["inputs"]
            pbar = tqdm(total=len(prompts), desc=f"Computing embeddings for {submixture}")
            prompts_embeddings = []
            start_time = time.time()
            for i in range(0, len(prompts), 64):
                batch = prompts[i:i+64]
                embeddings = model.encode(
                    sentences=batch,
                    batch_size=64,
                    device="cuda:0"
                )
                prompts_embeddings.extend(embeddings)
                pbar.update(len(batch))
            pbar.close()
            end_time = time.time()
            prompts_embeddings = np.array(prompts_embeddings)
            print(f"Time taken to compute embeddings for {submixture}:", end_time - start_time)
            print("Embeddings computed. Shape:", prompts_embeddings.shape)
            np.save(f"prompts_embeddings/{submixture}.npy", prompts_embeddings)
    else:
        print("Embeddings already computed")

def get_task_indices():
    task_indices={}
    try:
        print("Trying to load task indices from task_indices directory...")
        for submixture in SUBMIXTURES:
            with open(f"task_indices/{submixture}.pkl", "rb") as f:
                task_indices[submixture]=pickle.load(f)
    except:
        print("Task indices not found. Getting task indices now...")
        os.makedirs("task_indices", exist_ok=True)
        for submixture in SUBMIXTURES:
            print(f"Processing {submixture}")
            submixture_data=DatasetDict.load_from_disk(f"{dir}/flan2022/{submixture}")["train"]
            TASKS=list(set(submixture_data['task_name']))  # Convert set to list
            task_labels=submixture_data["task_name"]
            template_types=submixture_data["template_type"]
            data={}
            N=len(task_labels)
            pbar=tqdm(range(N))
            for i in range(N):
                task_name=task_labels[i]  # Use task_labels directly
                template_type=template_types[i]
                if task_name not in data:
                    data[task_name]={}
                if template_type not in data[task_name]:
                    data[task_name][template_type]=[]
                data[task_name][template_type].append(i)
                pbar.update(1)
            with open(f"task_indices/{submixture}.pkl", "wb") as f:
                pickle.dump(data, f)
            task_indices[submixture]=data
    return task_indices

def get_task_embeddings(task_indices):
    tasks=[]
    embeddings=[]
    try:
        print("Trying to load task embeddings from task_embeddings directory...")
        with open("task_embeddings/tasks.pkl", "rb") as f:
            tasks=pickle.load(f)
        embeddings=np.load("task_embeddings/tasks_embeddings.npy")
    except:
        print("Task embeddings not found. Computing task embeddings now...")
        os.makedirs("task_embeddings", exist_ok=True)
        for submixture in SUBMIXTURES:
            print(f"Processing {submixture}")
            submixture_embeddings=np.load(f"prompts_embeddings/{submixture}.npy")
            submixture_task_indices=task_indices[submixture]
            all_tasks=list(submixture_task_indices.keys())
            print(f"The total number of tasks in {submixture} is {len(all_tasks)}")
            pbar=tqdm(range(len(all_tasks)))
            for task in all_tasks:
                indices=[]
                for template_type in TEMPLATE_TYPES:
                    if template_type in submixture_task_indices[task]:
                        indices+=submixture_task_indices[task][template_type]
                tasks.append(task)
                embeddings.append(np.mean(submixture_embeddings[indices], axis=0))
                pbar.update(1)
        print("Saving task embeddings...")
        # concatenate all embeddings
        embeddings=np.concatenate([embeddings.reshape((1,-1)) for embeddings in embeddings], axis=0)
        with open("task_embeddings/tasks.pkl", "wb") as f:
            pickle.dump(tasks, f)
        np.save("task_embeddings/tasks_embeddings.npy", embeddings)
        print(f"total number of tasks: {len(tasks)}")
    return (tasks, embeddings)

def get_task_totals(task_indices):
    task_totals={}
    print("Getting task totals...")
    for submixture in SUBMIXTURES:
        submixture_task_indices=task_indices[submixture]
        for task in submixture_task_indices.keys():
            task_totals[task]=0
            for template_type in TEMPLATE_TYPES:
                if template_type in submixture_task_indices[task]:
                    task_totals[task]+=len(submixture_task_indices[task][template_type])
    return task_totals

def budget_split(budget, ratio):
    ratio_sum=sum(ratio)
    parts=[math.floor(budget*r/ratio_sum) for r in ratio]
    remainder=budget-sum(parts)
    for i in range(remainder):
        parts[i%len(ratio)]+=1
    return parts

def taylor_softmax_v1(x, dim=1, n=2, use_log=False):
    assert n % 2 == 0 and n > 0
    fn = torch.ones_like(x)
    denor = 1.
    for i in range(1, n + 1):
        denor *= i
        fn = fn + x.pow(i) / denor
    out = fn / fn.sum(dim=dim, keepdims=True)
    if use_log: out = out.log()
    return out

def get_task_budgets(tasks, task_totals, embeddings, submod_fnc, num_tasks, num_instances):
    print("Getting task budgets...")
    if submod_fnc=="random":
        indices=random.sample(list(range(len(tasks))), num_tasks)
        gains=[1 for _ in range(num_tasks)]
    else:
        print("Creating similarity kernel...")
        data_sijs=submodlib.helper.create_kernel(X=embeddings, metric="cosine", method="sklearn")
        if submod_fnc=="fl":
            print("Instantiating facility location function...")
            submod_obj=submod_fn.facilityLocation.FacilityLocationFunction(n=embeddings.shape[0], mode="dense", sijs=data_sijs, separate_rep=False)
        elif submod_fnc=="gc":
            print("Instantiating graph cut function...")
            submod_obj=submod_fn.graphCut.GraphCutFunction(n=embeddings.shape[0], mode="dense", ggsijs=data_sijs, lambdaVal=0.4, separate_rep=False)
        elif submod_fnc=="logdet":
            print("Instantiating log-determinant function...")
            submod_obj=submod_fn.logDeterminant.LogDeterminantFunction(n=embeddings.shape[0], mode="dense", sijs=data_sijs, lambdaVal=1)

        print("Running the lazy greedy algorithm...")
        greedyList=submod_obj.maximize(budget=num_tasks, optimizer="LazyGreedy", show_progress=True)

        indices=[idx for idx, _ in greedyList]
        gains=[gain for _, gain in greedyList]

    all_tasks_total=0
    for idx in indices:
        task=tasks[idx]
        all_tasks_total+=task_totals[task]
    print(f"The total number of instances in the selected tasks is {all_tasks_total}. The total number of instances in the subset is {num_instances}.")
    if all_tasks_total<num_instances:
        raise Exception("The total number of instances in the selected tasks is less than the number of instances in the subset. Something wrong.")

    print("Applying Taylor softmax on gains...")
    probs=taylor_softmax_v1(torch.from_numpy(np.array([gains])/TEMPERATURE)).numpy()[0]
    print("Creating a budget split according to the probabilities computed...")
    budgets=budget_split(num_instances, probs.tolist())

    task_budgets={}
    for i, b in enumerate(budgets):
        task_budgets[tasks[indices[i]]]=b
    print("Checking if we need to redistribute budget because some tasks may be assigned a budget more than the number of instances in the task")

    to_redistribute=0
    for task in task_budgets.keys():
        if task_totals[task]<task_budgets[task]:
            to_redistribute+=task_budgets[task]-task_totals[task]
            task_budgets[task]=task_totals[task]
    print(f"Need to redistribute a budget of {to_redistribute}. Redistributing...")
    retry=0
    while to_redistribute>0:
        for task in task_budgets.keys():
            if task_totals[task]>task_budgets[task]:
                task_budgets[task]+=1
                to_redistribute-=1
                if to_redistribute==0:
                    break
            retry+=1
            # if retry>1000000:
            #     raise Exception("Something went wrong while redistributing budget. Retry limit exceeded.")
    print("Setting the budget of the unselected tasks to 0...")
    unselected_tasks=[task for task in tasks if task not in task_budgets.keys()]
    for task in unselected_tasks:
        task_budgets[task]=0
    return task_budgets

def get_task_template_budgets(task_indices, task_budgets):
    print("Getting budget for each (task, template) pair...")
    CNT=0
    TOTAL_BUDGET=0
    task_template_budgets={}
    os.makedirs('template_task_budgets', exist_ok=True)
    for submixture in SUBMIXTURES:
        task_template_budgets[submixture]={}
        print(f"Processing {submixture}")
        submixture_task_indices=task_indices[submixture]
        pbar=tqdm(range(len(submixture_task_indices.keys())))
        for task in submixture_task_indices.keys():
            try:
                task_budget=task_budgets[task]
                TOTAL_BUDGET+=task_budget
                template_counts=[]
                for template_type in TEMPLATE_TYPES:
                    if template_type in submixture_task_indices[task]:
                        template_counts.append(len(submixture_task_indices[task][template_type]))
                    else:
                        template_counts.append(0)
                if sum(template_counts)<task_budget:
                    raise Exception(f"Task budget exceeds number of instances in the task({task})")
                is_fs_present=(template_counts[2]+template_counts[3]>0)
                template_budgets=[0, 0, 0, 0]
                if not is_fs_present:
                    template_budgets[2]=0
                    template_budgets[3]=0
                    if template_counts[0]==0:
                        if template_counts[1]==0:
                            raise Exception("No templates present?? Something Wrong")
                        else:
                            template_budgets[0]=0
                            template_budgets[1]=task_budget
                    elif template_counts[1]==0:
                        if template_counts[0]==0:
                            raise Exception("No templates present?? Something Wrong")
                        else:
                            template_budgets[0]=task_budget
                            template_budgets[1]=0
                    else:
                        zs_opt, zs_noopt=budget_split(task_budget, [template_counts[0], template_counts[1]])
                        template_budgets[0]=zs_opt
                        template_budgets[1]=zs_noopt
                else:
                    zs, fs=budget_split(task_budget, [template_counts[0]+template_counts[1], template_counts[2]+template_counts[3]])
                    if template_counts[0]==0:
                        if template_counts[1]==0:
                            raise Exception("No ZS templates present?? Something Wrong")
                        else:
                            template_budgets[0]=0
                            template_budgets[1]=zs
                    elif template_counts[1]==0:
                        if template_counts[0]==0:
                            raise Exception("No ZS templates present?? Something Wrong")
                        else:
                            template_budgets[0]=zs
                            template_budgets[1]=0
                    else:
                        zs_opt, zs_noopt=budget_split(zs, [template_counts[0], template_counts[1]])
                        template_budgets[0]=zs_opt
                        template_budgets[1]=zs_noopt
                    if template_counts[2]==0:
                        if template_counts[3]==0:
                            raise Exception("No FS templates present?? Something Wrong")
                        else:
                            template_budgets[2]=0
                            template_budgets[3]=fs
                    elif template_counts[3]==0:
                        if template_counts[2]==0:
                            raise Exception("No FS templates present?? Something Wrong")
                        else:
                            template_budgets[2]=fs
                            template_budgets[3]=0
                    else:
                        fs_opt, fs_noopt=budget_split(fs, [template_counts[2], template_counts[3]])
                        template_budgets[2]=fs_opt
                        template_budgets[3]=fs_noopt
                task_template_budgets[submixture][task]=template_budgets
                CNT+=sum(template_budgets)
                pbar.update(1)
            except Exception as e:
                print(f"Exception occurred while processing {task} in {submixture}")
                print(e)
        with open(f"template_task_budgets/{submixture}.pkl", 'wb') as f:
            pickle.dump(task_template_budgets[submixture], f)
    assert CNT==TOTAL_BUDGET
    return task_template_budgets

def load_instances_submodular_ordering(submod_fnc_instances):
    submod_ordering={}
    try:
        print(f"Trying to load submodular ordering of instances from {submod_fnc_instances}_ordering directory...")
        for submixture in SUBMIXTURES:
            with open(f"{submod_fnc_instances}_ordering/{submixture}.pkl", "rb") as f:
                submod_ordering[submixture]=pickle.load(f)
    except FileNotFoundError as e:
        raise Exception(f"Submodular ordering for instances not found for {submod_fnc_instances} in submixture {submixture}") from e
    return submod_ordering

def get_task_ordering(submod_fnc:str, embeddings:np.array, budget:int):
    # print("creating similarity kernel...")
    data_sijs=submodlib.helper.create_kernel(X=embeddings, metric='cosine', method='sklearn') #sijs = similarity kernel i.e. S_ij
    if submod_fnc=='fl':
        # print('Instantiating facility location function...')
        submod_obj=submod_fn.facilityLocation.FacilityLocationFunction(n=embeddings.shape[0], mode='dense', sijs=data_sijs, separate_rep=False)
    elif submod_fnc=='gc':
        # print("Instantiating graph cut function...")
        submod_obj=submod_fn.graphCut.GraphCutFunction(n=embeddings.shape[0], mode='dense', ggsijs=data_sijs, lambdaVal=0.4, separate_rep=False)
    elif submod_fnc=='logdet':
        # print('Instantiating log-determinant function...')
        submod_obj=submod_fn.logDeterminant.LogDeterminantFunction(n=embeddings.shape[0], mode='dense', sijs=data_sijs, lambdaVal=1)

    if budget > 0 and budget < embeddings.shape[0]:
        greedyList = submod_obj.maximize(budget=budget, optimizer="LazyGreedy", show_progress=True)
        return greedyList

    else: return None

    
def get_submodular_ordering(submod_fnc, template_task_budgets):
    
    submodular_ordering = {}

    try:
        for submixture in SUBMIXTURES:
            with open(f"submodular_ordering/{submixture}.pkl", "rb") as f:
                submodular_ordering[submixture] = pickle.load(f)
    except:
        print('Generating submodular ordering')

        os.makedirs("submodular_ordering", exist_ok=True)
        for submixture in SUBMIXTURES:
            submodular_ordering[submixture] = {}  # Initialize submixture level
            if not os.path.exists(f'submodular_ordering/{submixture}.pkl'):
                prompt_embeddings = np.load(f"prompts_embeddings/{submixture}.npy")
                task_indices = pickle.load(open(f'task_indices/{submixture}.pkl', 'rb'))
                pbar=tqdm(task_indices.keys(), desc=f"Submodular ordering for tasks in {submixture}")
                for task in task_indices.keys():
                    if task not in submodular_ordering[submixture]:
                        submodular_ordering[submixture][task] = {}  # Initialize task level
                    
                    for i, template_type in enumerate(TEMPLATE_TYPES):
                        if template_type not in submodular_ordering[submixture][task]:
                            submodular_ordering[submixture][task][template_type] = []  # Initialize template_type level
                        if template_type in task_indices[task].keys():
                            if task == 'dialog' or 'cot': # Take smaller portions so that it doesn't crash the computer.
                                indices = random.sample(task_indices[task][template_type], math.floor(0.6 * len(task_indices[task][template_type])))
                            else:
                                indices = task_indices[task][template_type]
                            index_mapping = {i: indices[i] for i in range(len(indices))}
                            embeddings = prompt_embeddings[indices]
                            
                            greedyList = get_task_ordering(submod_fnc=submod_fnc, embeddings=embeddings, budget=template_task_budgets[submixture][task][i])
                            if greedyList is not None:
                                greedyList_mapped = [(index_mapping[idx], score) for idx, score in greedyList]
                                sorted_indices = [idx for idx, _ in sorted(greedyList_mapped, key=lambda x: x[1], reverse=True)]
                                submodular_ordering[submixture][task][template_type] = sorted_indices
                            else:
                                continue
                            pbar.update(1)

                # Save the submodular ordering for this submixture
                with open(f"submodular_ordering/{submixture}.pkl", 'wb') as f:
                    pickle.dump(submodular_ordering[submixture], f)
    
    return submodular_ordering

def get_subset_indices(submodular_ordering, task_template_budgets, task_indices):
    print("Getting subset of instances from each task based on submodular ordering and task_template_budgets...")
    indices={}
    for submixture in SUBMIXTURES:
        indices[submixture]=[]
        submixture_ordering=submodular_ordering[submixture]
        submixture_task_template_budgets=task_template_budgets[submixture]
        submixture_task_indices=task_indices[submixture]
        for task in submixture_task_template_budgets.keys():
            for i, template_type in enumerate(TEMPLATE_TYPES):
                if template_type in submixture_ordering[task].keys():
                    # task_template_budget=submixture_task_template_budgets[task][i]
                    indices[submixture].extend(submixture_ordering[task][template_type])
                    # if task_template_budget==len(submixture_task_indices[task][template_type]):
                    #     indices[submixture].extend(submixture_task_indices[task][template_type])
                    # else:
                    #     indices[submixture].extend([idx for idx, _ in submixture_ordering[task][template_type][:task_template_budget]])
    return indices

def get_final_dataset(indices):
    print("Getting final dataset that is uploadable to hub...")
    submixture_train_datasets = []
    submixture_val_datasets = []
    
    for submixture in SUBMIXTURES:
        submixture_data = DatasetDict.load_from_disk(f"{dir}/flan2022/{submixture}")
        
        # Assuming we are working with the "train" split in each submixture
        train_data = submixture_data['train']
        
        submixture_train_datasets.append(
            train_data.select(indices[submixture]).remove_columns(["task_source", "task_name", "template_type"])
        )
        submixture_val_datasets.append(
            submixture_data['test'].remove_columns(["task_source", "task_name", "template_type"])
        )
    
    train_dataset = datasets.concatenate_datasets(submixture_train_datasets)
    train_dataset = train_dataset.shuffle(seed=23)
    train_dataset = train_dataset.rename_column("inputs", "prompt")
    train_dataset = train_dataset.rename_column("targets", "response")
    
    val_dataset = datasets.concatenate_datasets(submixture_val_datasets)
    val_dataset = val_dataset.rename_column("inputs", "prompt")
    val_dataset = val_dataset.rename_column("targets", "response")
    
    train_val_dataset = datasets.DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

    train_val_dataset.save_to_disk('final_dataset')

    return train_val_dataset

def get_entire_dataset(indices):
    print("Getting final dataset that is uploadable to hub...")
    submixture_train_datasets = []
    submixture_val_datasets = []
    
    for submixture in SUBMIXTURES:
        submixture_data = DatasetDict.load_from_disk(f"{dir}/flan2022/{submixture}")
        
        # Assuming we are working with the "train" split in each submixture
        train_data = submixture_data['train']
        
        submixture_train_datasets.append(
            train_data.remove_columns(["task_source", "task_name", "template_type"])
        )
        submixture_val_datasets.append(
            submixture_data['test'].remove_columns(["task_source", "task_name", "template_type"])
        )
    
    train_dataset = datasets.concatenate_datasets(submixture_train_datasets)
    train_dataset = train_dataset.shuffle(seed=23)
    train_dataset = train_dataset.rename_column("inputs", "prompt")
    train_dataset = train_dataset.rename_column("targets", "response")
    
    val_dataset = datasets.concatenate_datasets(submixture_val_datasets)
    val_dataset = val_dataset.rename_column("inputs", "prompt")
    val_dataset = val_dataset.rename_column("targets", "response")
    
    train_val_dataset = datasets.DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

    train_val_dataset.save_to_disk('entire_dataset')

    return train_val_dataset


def main():
    args=parse_args()

    HUB_TOKEN=args.HUB_TOKEN
    HUB_USERNAME=args.HUB_USERNAME

    # Download the FLAN 2022 dataset
    download_flan2022()

    # Compute embeddings for all prompts in FLAN 2022
    compute_prompt_embeddings()

    # Get (task, template_type) -> indices mapping
    task_indices=get_task_indices()

    # Get task embeddings
    tasks, embeddings= get_task_embeddings(task_indices)

    # Get task totals
    task_totals=get_task_totals(task_indices)

    # Get task budgets
    task_budgets=get_task_budgets(tasks, task_totals, embeddings, args.submod_fnc_tasks, args.num_tasks, args.num_instances)

    # Get (task, template) budgets
    task_template_budgets=get_task_template_budgets(task_indices, task_budgets)

    # Load or generate submodular ordering
    submod_ordering = get_submodular_ordering(args.submod_fnc_instances, task_template_budgets)

    # Load the submodular ordering of instances in each task
    # submod_ordering=load_instances_submodular_ordering(args.submod_fnc_instances)

    # get a list of indices to select based on task_template_budgets
    indices=get_subset_indices(submod_ordering, task_template_budgets, task_indices)

    # prepare final dataset based on indices
    dataset=get_final_dataset(indices)

    assert len(dataset["train"])==args.num_instances

    # push to hub
    # dataset.push_to_hub(
    #     f"{HUB_USERNAME}/flan2022-{CONTEXT_LEN}-{args.num_tasks}-tasks-{args.submod_fnc_tasks}-{args.num_instances}-instances-{args.submod_fnc_instances}",
    #     token=HUB_TOKEN,
    #     private=True
    # )

if __name__ == '__main__':
    main()