from transformers import BertModel, BertTokenizer
import multiprocessing as mp
import pandas as pd
import torch
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda')
bert_base_model_name = "bert-base-uncased"
bert_similarity_model_name = "sentence-transformers/all-MiniLM-L6-v2"


class FeatureEnhanceNLP:
    def __init__(self, file_path):
        # utf-8 encoding will fail
        df = pd.read_csv(file_path, sep=",", encoding="ANSI")
        df.dropna(axis=0, inplace=True)
        self.df = df.reset_index(drop=True)

        # create tokenizer to generate tokens for teh bert model
        self.base_tokenizer = BertTokenizer.from_pretrained(bert_base_model_name)
        self.similarity_tokenizer = BertTokenizer.from_pretrained(bert_similarity_model_name)

        # init bert model
        self.bert_base_model = BertModel.from_pretrained(bert_base_model_name).to(device)
        self.bert_similarity_model = BertModel.from_pretrained(bert_base_model_name).to(device)

        self.song_titles_columns = None
        self.album_names_columns = None
        self.terms_columns = None

    def embedding_serial(self):
        # get String value from 3 columns
        song_titles = self.df["title"]
        album_names = self.df["album_name"]
        terms = self.df["artist_terms"]

        # tokenization for 3 columns
        print("\tStart tokenization...")
        song_titles_tokenization = [torch.tensor(self.base_tokenizer.encode(each)).unsqueeze(0).to(device)
                                    for each in song_titles]
        album_names_tokenization = [torch.tensor(self.base_tokenizer.encode(each)).unsqueeze(0).to(device)
                                    for each in album_names]
        terms_tokenization = [torch.tensor(self.similarity_tokenizer.encode(each)).unsqueeze(0).to(device)
                                    for each in terms]

        # embeddings for 3 columns
        print("\tStart embedding...")
        song_titles_embeddings = [self.bert_base_model(t)[1].cpu().detach().numpy()[0]
                                  for t in song_titles_tokenization]
        print("\tsong_titles_embedding done!")
        album_names_embeddings = [self.bert_base_model(t)[1].cpu().detach().numpy()[0]
                                  for t in album_names_tokenization]
        print("\talbum_names_embedding done!")
        terms_embeddings = [self.bert_similarity_model(t)[1].cpu().detach().numpy()[0]
                            for t in terms_tokenization]
        print("\tterms embedding done!")

        # Convert numpy array to pandas DataFrame
        self.song_titles_columns = pd.DataFrame(song_titles_embeddings)
        self.album_names_columns = pd.DataFrame(album_names_embeddings)
        self.terms_columns = pd.DataFrame(terms_embeddings)

    def song_title_embedding(self):
        song_titles = self.df["title"]
        print("\ttokenization for song titles start!")
        song_titles_tokenization = [torch.tensor(self.base_tokenizer.encode(each)).unsqueeze(0).to(device)
                                    for each in song_titles]
        print("\tembedding for song titles start!")
        song_titles_embeddings = [self.bert_base_model(t)[1].cpu().detach().numpy()[0]
                                  for t in song_titles_tokenization]
        print("\tembedding for song titles done!")
        self.song_titles_columns = pd.DataFrame(song_titles_embeddings)

    def album_name_embedding(self):
        album_names = self.df["album_name"]
        print("\ttokenization for album names start!")
        album_names_tokenization = [torch.tensor(self.base_tokenizer.encode(each)).unsqueeze(0).to(device)
                                    for each in album_names]
        print("\tembedding for album names start!")
        album_names_embeddings = [self.bert_base_model(t)[1].cpu().detach().numpy()[0]
                                  for t in album_names_tokenization]
        print("\tembedding for album names done!")
        self.album_names_columns = pd.DataFrame(album_names_embeddings)

    def terms_embedding(self):
        terms = self.df["artist_terms"]
        print("\ttokenization for terms start!")
        terms_tokenization = [torch.tensor(self.similarity_tokenizer.encode(each)).unsqueeze(0).to(device)
                              for each in terms]
        print("\tembedding for terms start!")
        terms_embeddings = [self.bert_similarity_model(t)[1].cpu().detach().numpy()[0]
                            for t in terms_tokenization]
        print("\tembedding for terms done!")
        self.terms_columns = pd.DataFrame(terms_embeddings)

    def generate_features(self):
        nlp_features = pd.concat((self.song_titles_columns,
                                  self.album_names_columns,
                                  self.terms_columns), axis=1)
        whole_features = pd.concat((self.df, nlp_features), axis=1)

        print(whole_features)
        whole_features.to_csv("./processed/new_enhanced_data", index=False)
        print("Enhanced Data Saved!")


if __name__ == '__main__':
    my_bert_embeddings = FeatureEnhanceNLP("../processed/whole_data.csv")

    # =================== Parallel Embedding Start ===================
    print("Parallel Embedding Start!")
    start = time.perf_counter()
    # =================== define 3 different Process ===================
    song_title_process = mp.Process(target=my_bert_embeddings.song_title_embedding)
    album_name_process = mp.Process(target=my_bert_embeddings.album_name_embedding)
    terms_process = mp.Process(target=my_bert_embeddings.terms_embedding)

    # =================== start 3 different Process ===================
    song_title_process.start()
    album_name_process.start()
    terms_process.start()

    # =================== wait for 3 different Process ===================
    song_title_process.join()
    album_name_process.join()
    terms_process.join()
    # 162.08 seconds
    print("Parallel Embedding using time:", time.perf_counter() - start)
    # =================== Parallel Embedding Done ===================

    print("Serial Embedding Start!")
    start = time.perf_counter()
    my_bert_embeddings.embedding_serial()
    print("Serial Embedding using time:", time.perf_counter() - start)

    # print(my_bert_embeddings.song_titles_columns)
    # print(my_bert_embeddings.album_names_columns)
    # print(my_bert_embeddings.terms_columns)

    my_bert_embeddings.generate_features()
