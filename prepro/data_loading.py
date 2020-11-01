from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
from nltk.tokenize import word_tokenize 
from nltk.tokenize import sent_tokenize
stop_words = set(stopwords.words('english')) 
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer
import re , unicodedata , string

from sklearn.model_selection import train_test_split

from transformers import BertTokenizer
import torch


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences



class Preprocessing_text:
    '''
    Preprocessing pipeline : sequence of indices representing the methods to be used while preprocessing each row in the df passed
    The index to method mapping is as follows:
    remove_URL:1, remove_non_ascii:2, remove_punctuation:3, remove_stopwords:4, to_lower:5, lemmatize_postags:6, 
    replace_nan:7, get_top_n_words:8, remove_common_words:9


    All methods can be used in apply() method for dataframes [except #8 which requires the entire corpus] 
    '''
    def __init__ (self,preprocessing_pipeline=[7,5,2,3,4],stopword_list=list(set(stopwords.words('english')))):
        self.stopword_list = stopword_list
        self.preprocess_methods = [self.blank,self.remove_URL,self.remove_non_ascii,self.remove_punctuation,self.remove_stopwords,
                                    self.to_lower,self.lemmatize_postags,self.replace_nan,self.get_top_n_words,self.remove_common_words]
        self.preprocessing_pipeline = preprocessing_pipeline
        self.words_frequent = []
        
    def blank(self,sample_str):
        return sample_str

    def remove_URL(self,sample_str):
        """Remove URLs from a sample string"""
        return re.sub(r"http\S+", "", sample_str)

    def remove_non_ascii(self,sample_str):
        """Remove non-ASCII characters from a sample string [sample_str]"""
        words = word_tokenize(sample_str)
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return ' '.join(new_words)

    def remove_punctuation(self,sample_str):
        """Remove punctuation from a sample string"""
        words = word_tokenize(sample_str)
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return ' '.join(new_words)

    def remove_stopwords(self,sample_str):
        """Remove stop words from a sample string"""
        words = word_tokenize(sample_str)
        new_words = []
        for word in words:
            if word not in self.stopword_list:
                new_words.append(word)
        return ' '.join(new_words)

    def to_lower(self,sample_str):
        """ Converting all words to lowercase in a sample string"""
        return sample_str.lower()


    def lemmatize_postags(self,sample_str):
        """Lemmatize verbs,adj and noun in a sample string"""
        words = word_tokenize(sample_str)
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            word = lemmatizer.lemmatize(word, pos='v')
            word = lemmatizer.lemmatize(word, pos='n')
            word = lemmatizer.lemmatize(word, pos='a')
            lemmas.append(lemma)
        return ' '.join(lemmas)

    def replace_nan(self,sample_str):
        """Replacing nan strings with empty strings - required for textrank"""        
        sample_str_new = re.sub('nan' , '' , str(sample_str))
        return sample_str_new

    def get_top_n_words(self,corpus, n=5):
        """
        List the top n words in a vocabulary according to occurrence in a text corpus and updates self.words_frequent to be used later

        get_top_n_words(["I love Python", "Python is a language programming", "Hello world", "I love the world"]) -> 
        [('python', 2),
        ('world', 2),
        ('love', 2),
        ('hello', 1),
        ('is', 1),
        ('programming', 1),
        ('the', 1),
        ('language', 1)]
        Repetitive words may be removed because they might not hold enough specific information to be assigned as keywords
        """
        vec = CountVectorizer().fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        self.words_frequent = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        self.words_frequent =sorted(words_freq, key = lambda x: x[1], reverse=True)
        self.words_frequent = [word for word,count in self.words_frequent]
        print(f'Most {len(words_freq)} common words in the given corpus are : \n')
        for i in range(n):
            print(words_freq[i])
        self.words_frequent = words_freq[:n]

    def remove_common_words(self,sample_str):
        """Removes the most frequent words from the sample string using words_frequent list provided by get_top_n_words"""
        words = word_tokenize(sample_str)
        filtered_words = []
        for word in words:
            if word.lower not in self.words_frequent:
                filtered_words.append(word)
        return " ".join(filtered_words)

    def main_df_preprocess(self,df,column_list):
        """Main function to apply all listed methods as specified"""
        for col in column_list:
            for index in self.preprocessing_pipeline:
                if index!=8 or index!=9:
                    df[col] = df[col].apply(self.preprocess_methods[index])
        return df


class Preprocess_dataloading_bert():
    def __init__(self,sentencesA,sentencesB,labels):
        self.sentencesA = sentencesA
        self.sentencesB = sentencesB
        self.labels = labels
    
    def tokenize(self,tokenizer,tokenizer_name="BertTokenizer",max_length=400,padding='post',truncation='post'):
        input_ids = []
        attention_ids = []
        sent_id = []
        print("Tokenizer name - {}".format(tokenizer_name))
        print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))


        for s1,s2 in zip(self.sentencesA , self.sentencesB):
            token_type_id = []
            input_id1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s1))
            input_id2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s2))

            while len(input_id1) + len(input_id2) > (max_length - 3):
                if len(input_id2) > len(input_id1):
                    input_id2.pop()
                if len(input_id1) > len(input_id2):
                    input_id1.pop()

            input_id3 = tokenizer.build_inputs_with_special_tokens(input_id1,input_id2)
            input_ids.append(input_id3)

            token_type_id = [0]*(len(input_id1)+2)
            token_type_id = token_type_id + [1]*(len(input_id2)+1)
            sent_id.append(token_type_id)

        input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", value=int(tokenizer.pad_token_id), truncating=truncation, padding=padding)
        sent_id = pad_sequences(sent_id, maxlen=max_length, dtype="long", value=int(tokenizer.pad_token_id), truncating="post", padding="post")

        for each_sent in input_ids:
            mask_id = [int(token_id>0) for token_id in each_sent]
            attention_ids.append(mask_id)

            # output = tokenizer.__call__(s1,s2,padding=padding,truncation=truncation,max_length=max_length,return_tensors='pt')

        print("Tokenizing of data and including special tokens is completed. ")

        return input_ids , attention_ids , sent_id
    
    def train_test_split_dataloading(self, input_ids , attention_ids , sent_id, test_size):
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, self.labels , 
                                                            random_state=2018, test_size=test_size)

        train_masks, validation_masks, _, _ = train_test_split(attention_ids, self.labels,
                                                    random_state=2018, test_size=test_size)

        train_segment, validation_segment, _, _ = train_test_split(sent_id, self.labels,
                                                    random_state=2018, test_size=test_size)
        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)
        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)
        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)
        train_segment = torch.tensor(train_segment)
        validation_segment = torch.tensor(validation_segment)

        train = [train_inputs, train_masks, train_labels, train_segment ]
        val = [validation_inputs, validation_masks, validation_labels, validation_segment]

        print("Train test split completed. Shape of train inputs: {} ".format(train_inputs.shape))
        return train , val
    
    def from_list_to_tensor(self,list_of_tensors):
        tensors_ = torch.tensor(list_of_tensors)
        del list_of_tensors
        return tensors_
    
    def from_num_to_tensor(self,label):
        label_tensors = torch.tensor(label)
        del label
        return label_tensors

    def dataloading(self,type_,batch_size, input, masks,labels,segment ):

        # Create the DataLoader for our training set.
        data = TensorDataset(input, masks, labels, segment)
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        print("{}".format(type_))
        print("dataloaders created for batch size : {}".format(batch_size))
        return dataloader 
