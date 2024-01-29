import numpy as np
import pandas as pd
import re
import json
import torch
import pickle as pkl
import os
import random
from ast import literal_eval
from torch.utils.data import Dataset
from pathlib import Path
from transformers import set_seed

def set_random():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    set_seed(42)

class SCRSDataset(Dataset):
    def __init__(
        self,
        path_dir: str,
        tokenizer,
        max_len_seq=60,
        sliding=False,
        mlm_probability=0.2,
        test_truncate=False,
        pos_only=False,
        ignore_sentiment=False,
        split_sentiment=True,
        use_standard_collator=False,
        train_truncate=False,
        evaluation_mode=False,
        ignore_role=False,
        user_only=True,
        dataset_name="redial",
        remove_prop = []
    ): 
        self.mlm_probability = mlm_probability
        self.test_truncate = test_truncate
        self.split_sentiment = split_sentiment
        self.max_len_seq = max_len_seq
        self.ignore_sentiment = ignore_sentiment
        self.ignore_role = ignore_role
        self.parent_folder = Path(__file__).parent.parent
        self.tokenizer = tokenizer
        self.sliding = sliding
        self.use_standard_collator = use_standard_collator
        self.pos_only = pos_only
        self.train_truncate = train_truncate
        self.evaluation_mode = evaluation_mode
        self.user_only = user_only
        self.dataset_name = dataset_name
        self.items_ = [v for (k, v) in self.tokenizer.vocab.items() if "I" in k]
        x = pd.read_csv(path_dir, index_col=0, converters={"movie_sent": literal_eval})
        x = x.dropna()
        x = x.reset_index()
        x = x.loc[:, ["conversationId", "sequence", "movie_sent"]]
        database_config = {key: value for key, value in locals().items() if key not in ["self", "kwargs", "args"]}
        movie_tokens = pkl.load(open(os.path.join(self.parent_folder, "data", self.dataset_name, "movie_ids.pkl"), "rb"))
        movie_tokens = ["[I"+str(x)+"]" for x in movie_tokens]
        self.candidates = [tokenizer.encode(tk)[0] for tk in movie_tokens if tokenizer.encode(tk)[0]!=0]
        self.candidates_num = len(self.candidates)
        if not self.split_sentiment:

            def aggregate_sent(sequence):
                sequence = re.sub("(\[NEG\]) (\w*)", "\g<1>\g<2>", sequence)
                sequence = re.sub("(\[POS\]) (\w*)", "\g<1>\g<2>", sequence)
                return sequence

            x.loc[:, "sequence"] = x.loc[:, "sequence"].map(aggregate_sent)

        if self.user_only:
            x = self.__drop_recommender_entity__(x)

        if self.pos_only:
            x = self.__drop_negative__(x)
            x = self.__drop_sentiment__(x)

        if self.ignore_sentiment:
            x = self.__drop_sentiment__(x)

        if self.ignore_role:
            x = self.__drop_role__(x)


        x = self.__trim_sequences__(x)
        x = self.__get_sequence_target__(x)
        self.dataset = x
        self.__tokenize__(x)

        if self.evaluation_mode:
            self.__prepare_eval_data__(x)
            return

        if self.sliding and self.mlm_probability > 0:
            self.__sliding_mask_collator__(x)
        elif self.sliding:
            self.__sliding_mask_one_collator__(x)
        else:
            self.__random_mask_items__(x)

        if self.test_truncate:
            self.dataset.loc[:, "train_sequence"] = self.dataset.loc[
                :, "masked_sequence"
            ]
            self.dataset.loc[:, "masked_sequence"] = self.dataset.loc[:, "truncated"]
        #set_random()
        #self.candidates = random.sample(self.items_, self.candidates_num)
        #self.candidates_num = len(self.candidates)

    def __drop_recommender_entity__(self, x):
        def find_rec_sequence(sequence):
            # Extract all substring that match the pattern
            return re.findall('\[REC\](.+?)\[SKR\]', sequence)
        x.loc[:, "recommender_sub_sequences"] = x.loc[:, "sequence"].map(find_rec_sequence)
        def find_rec_sequence(recommender_sub_sequences):
            # Extract all item from recommender part
            recommender_items = []
            for sub_seq in recommender_sub_sequences:
                 recommender_items.append([x.group() for x in re.finditer('\[[A-Z]{3,3}\] \[I[0-9]+\]', sub_seq)])
            return recommender_items
        x.loc[:, "recommender_items"] = x.loc[:, "recommender_sub_sequences"].map(find_rec_sequence)
        def replace_recommender_sub(row):
            recommender_items = row['recommender_items']
            recommender_sub_sequences = row['recommender_sub_sequences']
            sequence = row['sequence']
            for sub_sequence, items in zip(recommender_sub_sequences, recommender_items):
                if len(items) == 0:
                    sequence = sequence.replace('[REC]'+sub_sequence, "")
                    continue
                items_string = " ".join(items).strip()
                sequence = sequence.replace(sub_sequence, " "+items_string+" ")
            return sequence.strip()
        x.loc[:, "sequence"] = x.apply(replace_recommender_sub, axis=1)
        del x['recommender_items']
        del x['recommender_sub_sequences']
        return x


    def __remove_entities__(self, x, entity):
        map_entities = {
            "ACTORS": "EA",
            "GENRES": "EG",
            "DIRECTORS": "ED"
        }
        ent = map_entities[entity.upper()]
        def drop_ent(sequence):
            sequence = re.sub(f"\[{ent}[0-9]+\](\s*)", "\g<1>", sequence)
            sequence = " ".join(sequence.split())
            return sequence
        
        entity = entity.upper()
        

        x.loc[:, "sequence"] = x.loc[:, "sequence"].map(drop_ent)
        return x

    def __drop_sentiment__(self, x):
        def drop_sent(sequence):
            sequence = re.sub("\[POS\] ", "", sequence)
            sequence = re.sub("\[NEG\] ", "", sequence)
            return sequence

        x.loc[:, "sequence"] = x.loc[:, "sequence"].map(drop_sent)
        return x

    def __drop_role__(self, x):
        def drop_role(sequence):
            sequence = re.sub("\[SKR\]\s*", "", sequence)
            sequence = re.sub("\[REC\]\s*", "", sequence)
            return sequence

        x.loc[:, "sequence"] = x.loc[:, "sequence"].map(drop_role)
        return x

    def __drop_negative__(self, x):
        def drop_negative(sequence):
            sequence = re.sub("\s*\[NEG\] \w*\s*", " ", sequence)
            sequence = re.sub("\[POS\] ", "", sequence)
            return sequence

        x.loc[:, "sequence"] = x.loc[:, "sequence"].map(drop_negative)
        return x
 
    def __random_mask_items__(self, x):
        labels = []
        input_ids = []
        attention_mask_list = []
        self.batch = {}
        sequence_df = (
            x.groupby("conversationId", as_index=True)
            .agg(
                masked_sequence=("masked_sequence", "min"),
                sequence=("sequence", "min"),
                target=("target", list),
                len_tr=("len_tr", list),
            )
            .reset_index()
        )
        del sequence_df["masked_sequence"]
        del sequence_df["len_tr"]
        del sequence_df["target"]
        sequence_df["num_items"] = sequence_df.sequence.str.count("I")
        for i, row in sequence_df.iterrows():
            num_items = row["num_items"]
            num_items_to_mask = max(1, int(self.mlm_probability * num_items))
            masked_items_index = set()
            sequence = row["sequence"]
            if row["num_items"] > 1:
                for _ in range(num_items_to_mask):
                    casual_item_index = random.randrange(0, num_items - 1)
                    while casual_item_index in masked_items_index:
                        casual_item_index = random.randrange(0, num_items - 1)
                    masked_items_index.add(casual_item_index)
                matches = re.findall("\[I[0-9]*\]", sequence)  # find item
                masked_sequence = sequence
                targets = []
                for i, match in enumerate(matches):  # for each movie found
                    if i not in masked_items_index:
                        continue
                    masked_sequence = masked_sequence.replace(
                        match, self.tokenizer.mask_token, 1
                    )  # create the masked text for sentiment analysis
                    targets.append(match)
                # encode sequence, returns a dictionary
                sequence_dict = self.tokenizer(
                    masked_sequence,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_len_seq,
                )
                attention_mask = sequence_dict["attention_mask"]
                sequence_encoding = sequence_dict["input_ids"]
                # encode targets
                targets_encoding = [
                    self.tokenizer.encode(target)[0] for target in targets
                ]
                targets_index_in_sequence = np.where(
                    np.array(sequence_encoding) == self.tokenizer.mask_token_id
                )[0]
                target_sequence_encoding = [-100] * self.max_len_seq
                for index, target in zip(targets_index_in_sequence, targets_encoding):
                    target_sequence_encoding[index] = target
                target_sequence_encoding = np.array(target_sequence_encoding)
                positions = np.where(target_sequence_encoding != -100)
                masked_items = target_sequence_encoding[positions]
                # was input_ids.append(torch.Tensor(sequence_encoding).to(torch.long)), same thing with labels
                #input_ids.append(torch.tensor(sequence_encoding))
                #attention_mask_list.append(torch.tensor(attention_mask))
                #labels.append(torch.tensor(target_sequence_encoding))
            # Regular expression to match the last occurrence of an item
            sequence = masked_sequence
            regex = re.compile(r"^(.*)(\[I[0-9]+\])(?!.*\2)", re.DOTALL)
            last_target = matches[-1]
            # Replace the last occurrence of an item with [MASK]
            last_sequence = regex.sub(r"\1" + self.tokenizer.mask_token, sequence)
            masked_last_sequence = re.sub(
                r"\[MASK\][A-Za-z0-9 ]*", "[MASK]", last_sequence
            )
            last_sequence_dict = self.tokenizer(
                masked_last_sequence,
                truncation=True,
                padding="max_length",
                max_length=self.max_len_seq,
            )
            last_attention_mask = last_sequence_dict["attention_mask"]
            last_sequence_encoding = last_sequence_dict["input_ids"]

            last_target_encoding = self.tokenizer.encode(last_target)[0]
            last_target_sequence_encoding = [-100] * self.max_len_seq
            last_target_sequence_encoding[-1] = last_target_encoding
            for pos, item in zip(positions[0], masked_items):
                last_target_sequence_encoding[pos] = item
            input_ids.append(torch.tensor(last_sequence_encoding))
            attention_mask_list.append(torch.tensor(last_attention_mask))
            labels.append(torch.tensor(last_target_sequence_encoding))

        self.batch["input_ids"] = torch.stack(input_ids)
        self.batch["attention_mask"] = torch.stack(attention_mask_list)
        self.batch["labels"] = torch.stack(labels)

    def __prepare_eval_data__(self, x):
        self.__sliding_mask_one_collator__(x)

    def __sliding_mask_one_collator__(self, x):
        labels = []
        input_ids = []
        attention_mask_list = []
        self.batch = {}
        sequence_df = (
            x.groupby("conversationId", as_index=True)
            .agg(
                masked_sequence=("masked_sequence", "min"),
                sequence=("sequence", "min"),
                target=("target", list),
                len_tr=("len_tr", list),
            )
            .reset_index()
        )
        del sequence_df["masked_sequence"]
        del sequence_df["len_tr"]
        del sequence_df["target"]

        for i, row in sequence_df.iterrows():
            sequence = row.sequence
            matches = list(
                re.finditer("\[I[0-9]*\]", sequence)
            )  
            for match in reversed(matches):
                start_index = match.start()
                end_index = match.end()
                if self.train_truncate or self.evaluation_mode:
                    masked_sequence = sequence[:start_index] + self.tokenizer.mask_token
                else:
                    masked_sequence = (
                        sequence[:start_index]
                        + self.tokenizer.mask_token
                        + sequence[end_index:]
                    )
                target = match.group()
                sequence_dict = self.tokenizer(
                    masked_sequence,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_len_seq,
                )
                attention_mask = sequence_dict["attention_mask"]
                sequence_encoding = sequence_dict["input_ids"]
                target_encoding = self.tokenizer.encode(target)[0]
                targets_index_in_sequence = np.where(
                    np.array(sequence_encoding) == self.tokenizer.mask_token_id
                )[0][0]
                target_sequence_encoding = [-100] * self.max_len_seq
                target_sequence_encoding[targets_index_in_sequence] = target_encoding
                input_ids.append(torch.tensor(sequence_encoding))
                attention_mask_list.append(torch.tensor(attention_mask))
                labels.append(torch.tensor(target_sequence_encoding))
        self.batch["input_ids"] = torch.stack(input_ids)
        self.batch["attention_mask"] = torch.stack(attention_mask_list)
        self.batch["labels"] = torch.stack(labels)

    def __sliding_mask_collator__(self, x):
        labels = []
        input_ids = []
        attention_mask_list = []
        self.batch = {}
        sequence_df = (
            x.groupby("conversationId", as_index=True)
            .agg(
                masked_sequence=("masked_sequence", "min"),
                sequence=("sequence", "min"),
                target=("target", list),
                len_tr=("len_tr", list),
            )
            .reset_index()
        )
        del sequence_df["masked_sequence"]
        del sequence_df["len_tr"]
        del sequence_df["target"]
        for i, row in sequence_df.iterrows():
            sequence = row.sequence
            matches = list(
                re.finditer("\[I[0-9]*\]", sequence)
            )  
            num_items = len(
                matches
            )  
            all_targets_list = []
            for index_match, match in enumerate(reversed(matches)):
                start_index = match.start()
                end_index = match.end()

                if index_match == 0:
                    masked_sequence = sequence[:start_index] + self.tokenizer.mask_token
                else:
                    if self.train_truncate:
                        masked_sequence = (
                            sequence[:start_index] + self.tokenizer.mask_token
                        )
                    else:
                        masked_sequence = (
                            sequence[:start_index]
                            + self.tokenizer.mask_token
                            + sequence[end_index:]
                        )
                fixed_target = match.group()
                index_fixed_target = masked_sequence.split(" ").index(
                    self.tokenizer.mask_token
                )

                num_items_to_mask = max(1, int(self.mlm_probability * num_items))
                random_masked_items_index = set()
                sequence = row["sequence"]
                random_targets = []
                if num_items > 1:
                    for _ in range(num_items_to_mask):
                        casual_item_index = random.randrange(0, num_items - 1)
                        while casual_item_index in random_masked_items_index:
                            casual_item_index = random.randrange(0, num_items - 1)
                        random_masked_items_index.add(casual_item_index)
                    matches = re.findall("\[I[0-9]+\]", masked_sequence)
                    for i in range(len(matches)):  # for each movie found
                        if i not in random_masked_items_index:
                            continue
                        pattern = matches[i].replace("[", "\[").replace("]", "\]")
                        match = re.search(pattern, masked_sequence)
                        start_index = match.start()
                        end_index = match.end()
                        masked_sequence = (
                            masked_sequence[:start_index]
                            + "TEMP"
                            + masked_sequence[end_index:]
                        )
                        index_random_target = masked_sequence.split(" ").index("TEMP")
                        masked_sequence = re.sub(
                            "TEMP", self.tokenizer.mask_token, masked_sequence
                        )
                        random_targets.append(
                            (match.group().strip(), index_random_target)
                        )
                all_targets_list = random_targets
                all_targets_list.append((fixed_target, index_fixed_target))
                all_targets_list = sorted(all_targets_list, key=lambda x: x[1])
                all_targets_list = list(map(lambda x: x[0], all_targets_list))
                sequence_dict = self.tokenizer(
                    masked_sequence,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_len_seq,
                )
                attention_mask = sequence_dict["attention_mask"]
                sequence_encoding = sequence_dict["input_ids"]
                targets_encoding = [
                    self.tokenizer.encode(target)[0] for target in all_targets_list
                ]
                targets_index_in_sequence = np.where(
                    np.array(sequence_encoding) == self.tokenizer.mask_token_id
                )[0]
                target_sequence_encoding = [-100] * self.max_len_seq
                for index, target in zip(targets_index_in_sequence, targets_encoding):
                    target_sequence_encoding[index] = target
                input_ids.append(torch.tensor(sequence_encoding))
                attention_mask_list.append(torch.tensor(attention_mask))
                labels.append(torch.tensor(target_sequence_encoding))
        self.batch["input_ids"] = torch.stack(input_ids)
        self.batch["attention_mask"] = torch.stack(attention_mask_list)
        self.batch["labels"] = torch.stack(labels)

    def __trim_sequences__(self, x):
        for _, row in x.iterrows():
            sequence = row.sequence
            temp_seq = sequence.split(" ")
            while len(temp_seq) > self.max_len_seq:
                element_to_remove = len(temp_seq) - self.max_len_seq
                sequence = temp_seq[element_to_remove:]
                if len(sequence) <= self.max_len_seq and sequence[0] not in [
                    "[POS]",
                    "[NEG]",
                ]:
                    sequence = sequence[1:]
                temp_seq = sequence
                sequence = " ".join(sequence)
            x.loc[_, "sequence"] = sequence
        return x

    def __tokenize__(self, x):
        def tokenize(text):
            return self.tokenizer.encode(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len_seq,
                return_special_tokens_mask=True,
            )

        col = "masked_sequence" if self.sliding else "sequence"
        self.rawset = x[col].map(tokenize)
        temp_set = self.rawset.map(lambda x: str(x))
        indexs = temp_set.drop_duplicates().index
        return self.rawset[indexs]

    def __get_sequence_target__(self, x):
        new_x = []
        for _, row in x.iterrows():
            for id, sent, sugg in row.movie_sent:
                if sugg == "S" and id in row.sequence:
                    string_set = (
                        ""
                        if self.split_sentiment
                        else "[POS]"
                        if sent == 1
                        else "[NEG]"
                    )
                    index_start_last_item = row.sequence.index(string_set + "[I" + id)
                    index_end_last_item = index_start_last_item + len(
                        string_set + "[I" + id + "]"
                    )  # row.sequence.index("", index_start_last_item+len(string_set+"I"+id)-1)
                    element_to_predict = row.sequence[
                        index_start_last_item:index_end_last_item
                    ]
                    input_sequence = row.sequence.replace(element_to_predict, "[MASK]")
                    truncated = row.sequence[:index_end_last_item]
                    truncated = truncated.replace(element_to_predict, "[MASK]")
                    new_x.append(
                        {
                            "conversationId": row.conversationId,
                            "masked_sequence": input_sequence.strip(),
                            "target": element_to_predict.strip(),
                            "sequence": row.sequence.strip(),
                            "truncated": truncated.strip(),
                            "len_tr": len(truncated),
                        }
                    )
        x = pd.DataFrame(new_x)
        return x

    def __len__(self) -> int:
        return len(self.batch["labels"])

    def __getitem__(self, index: int):
        if self.evaluation_mode:
            candidates = torch.LongTensor(self.candidates)
            sequence_labels = self.batch["labels"][index]
            if sequence_labels[sequence_labels!=-100] not in candidates:
                remove_idx = random.randint(0, self.candidates_num-1)
                candidates[torch.tensor(remove_idx)] = sequence_labels[sequence_labels!=-100].item()
            if len(candidates)!=self.candidates_num:
                print(candidates)
                print(sequence_labels[sequence_labels!=-100])
                print(index)
                exit(0)

            lables_ = np.zeros(candidates.shape)
            lables_[candidates == sequence_labels[sequence_labels!=-100]] = 1.0
            return self.batch["input_ids"][index], torch.LongTensor(candidates), torch.LongTensor(lables_)
        else:
            sequence_labels = self.batch["labels"][index]
            sequence_labels[sequence_labels==-100] = 0
            return self.batch["input_ids"][index], sequence_labels

    def save_config(self, path_save):
        dict_properties = {
            "max_len_seq": self.max_len_seq,
            "split_sentiment": self.split_sentiment
            if not self.use_standard_collator
            else "",
            "ignore_sentiment": True
            if self.pos_only
            else self.ignore_sentiment
            if not self.use_standard_collator
            else "",
            "sliding": self.sliding if not self.use_standard_collator else "",
            "pos_only": self.pos_only if not self.use_standard_collator else "",
            "test_truncate": self.test_truncate,
            "mlm_probability": self.mlm_probability,
            "use_standard_collator": self.use_standard_collator,
            "train_truncate": self.train_truncate,
        }
        with open(path_save + self.dataset + "/dataset_config.json", "w") as outfile:
            json.dump(dict_properties, outfile)
    



